"""
Custom batch generation utilities built on top of mlx_lm.generate.

This module provides:

- CustomBatchGenerator: a batch generator that supports:
  - max_kv_size: capping rotating KV cache size per layer.
  - prompt_caches: optional per-prompt prefix caches for batched generation.
- custom_batch_generate: a convenience wrapper similar to mlx_lm.batch_generate.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import mlx.core as mx

from mlx_lm.models import cache as cache_mod
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from mlx_lm.generate import (
    BatchStats,
    BatchResponse,
    Batch,
    _left_pad_prompts,
    generation_stream,
    wired_limit,
)

logger = logging.getLogger(__name__)


def _batchify_kv_layer(
    layer_caches: List[KVCache],
    left_padding: List[int],
) -> BatchKVCache:
    """
    Turn per-prompt KVCache layer caches into a single BatchKVCache.
    """
    prefix_lens = [c.offset for c in layer_caches]
    B = len(layer_caches)
    Tmax = max(prefix_lens) if prefix_lens else 0

    if Tmax == 0:
        return BatchKVCache(left_padding)

    k_list = []
    v_list = []
    for c, L in zip(layer_caches, prefix_lens):
        k, v = c.state  # [1, n_heads, L, D]
        k = k[0]
        v = v[0]
        pad = Tmax - L
        if pad > 0:
            pad_cfg = [(0, 0), (0, 0), (pad, 0), (0, 0)]
            k = mx.pad(k, pad_cfg)
            v = mx.pad(v, pad_cfg)
        k_list.append(k)
        v_list.append(v)

    k_b = mx.stack(k_list, axis=0)  # [B, n_heads, Tmax, D]
    v_b = mx.stack(v_list, axis=0)

    left_pad = [Tmax - L for L in prefix_lens]

    bc = BatchKVCache(left_pad)
    bc.keys = k_b
    bc.values = v_b
    bc.offset = mx.array(prefix_lens)
    bc._idx = Tmax
    return bc


def _make_batch_cache_from_prompt_caches(
    prompt_caches: List[Optional[List[Any]]],
    left_padding: List[int],
    max_kv_size: Optional[int],
    model,
) -> List[Any]:
    """
    Build a batched cache from per-prompt prompt caches.

    If all prompt_caches are None, falls back to the default _make_cache.
    """
    if all(c is None for c in prompt_caches):
        return _make_cache(model, left_padding, max_kv_size=max_kv_size)

    cache_lists = [c for c in prompt_caches if c is not None]
    num_layers = len(cache_lists[0])
    if any(len(c) != num_layers for c in cache_lists):
        raise ValueError("All prompt_caches must have same per-layer length.")

    batched: List[Any] = []
    for layer_idx in range(num_layers):
        layer_caches = [c[layer_idx] for c in cache_lists]

        if isinstance(layer_caches[0], KVCache):
            batched.append(_batchify_kv_layer(layer_caches, left_padding))
        else:
            raise ValueError(
                f"Per-prompt cache type {type(layer_caches[0])} "
                "not supported yet for batched different prompt caches."
            )

    return batched


def _make_cache(
    model,
    left_padding: List[int],
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Convert a list of regular caches into their corresponding batch-aware caches.
    """

    def to_batch_cache(c):
        if isinstance(c, KVCache):
            return BatchKVCache(left_padding)
        elif isinstance(c, ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            size = c.max_size
            if max_kv_size is not None:
                size = min(size, max_kv_size)
            return BatchRotatingKVCache(size, left_padding)
        elif isinstance(c, CacheList):
            return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        cache = cache_mod.make_prompt_cache(model, max_kv_size=max_kv_size)
        return [to_batch_cache(c) for c in cache]


class CustomBatchGenerator:
    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]

    def __init__(
        self,
        model,
        max_tokens: int = 128,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        max_kv_size: Optional[int] = None,
    ):
        self.model = model
        self.unprocessed_prompts: List[
            Tuple[int, List[int], int, Optional[List[Any]]]
        ] = []
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size
        self._stats = BatchStats()

        self.active_batch: Optional[Batch] = None
        self.max_kv_size = max_kv_size

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Union[List[int], int, None] = None,
        prompt_caches: Optional[List[Optional[List[Any]]]] = None,
    ) -> List[int]:
        uids: List[int] = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)
        if prompt_caches is None:
            prompt_caches = [None] * len(prompts)

        for p, m, c in zip(prompts, max_tokens, prompt_caches):
            self.unprocessed_prompts.append((self.uid_count, p, m, c))
            uids.append(self.uid_count)
            self.uid_count += 1

        self.unprocessed_prompts = sorted(
            self.unprocessed_prompts, key=lambda x: len(x[1])
        )
        return uids

    def _process_prompts_uniform(self, prompts):
        """
        Process a list of prompts where either
        - all prompt_caches are None, or
        - all prompt_caches are non-None.
        """
        if not prompts:
            return None

        uids, inputs, max_tokens, prompt_caches = zip(*prompts)
        lengths = [len(p) for p in inputs]
        max_length = max(lengths)
        self._stats.prompt_tokens += sum(lengths)
        left_padding = [max_length - l for l in lengths]
        inputs = _left_pad_prompts(inputs, max_length=max_length)

        prompt_cache = _make_batch_cache_from_prompt_caches(
            list(prompt_caches),
            left_padding,
            max_kv_size=self.max_kv_size,
            model=self.model,
        )

        while inputs.shape[1] > 1:
            n_to_process = min(self.prefill_step_size, inputs.shape[1] - 1)
            self.model(inputs[:, :n_to_process], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            inputs = inputs[:, n_to_process:]
            mx.clear_cache()

        y, logprobs = self._step(inputs, prompt_cache)
        mx.async_eval(y, logprobs)
        return Batch(
            list(uids),
            y,
            logprobs,
            list(max_tokens),
            [0] * len(uids),
            prompt_cache,
        )

    def _process_prompts(self, prompts):
        """
        Process prompts for a single prefill step.

        If some prompts have prefix caches and some do not, we
        process the two groups separately (all-cached, all-None)
        and then merge the resulting batches. This avoids issues
        in _make_batch_cache_from_prompt_caches, which assumes
        uniform cache presence within a batch.
        """
        if not prompts:
            return None

        # Split into prompts with/without caches if mixed.
        has_cache_flags = [p[3] is not None for p in prompts]
        if any(has_cache_flags) and not all(has_cache_flags):
            cached_prompts = [p for p in prompts if p[3] is not None]
            nocache_prompts = [p for p in prompts if p[3] is None]

            logger.debug(
                "CustomBatchGenerator: mixed batch with prefix caches: "
                "%d cached, %d without cache",
                len(cached_prompts),
                len(nocache_prompts),
            )

            batch_cached = self._process_prompts_uniform(cached_prompts)
            batch_nocache = self._process_prompts_uniform(nocache_prompts)

            if batch_cached is None:
                return batch_nocache
            if batch_nocache is None:
                return batch_cached

            batch_cached.extend(batch_nocache)
            return batch_cached

        # Uniform case: all have caches or all do not.
        if has_cache_flags and has_cache_flags[0]:
            logger.debug(
                "CustomBatchGenerator: batch uses prefix caches (n=%d)",
                len(prompts),
            )
        else:
            logger.debug(
                "CustomBatchGenerator: batch without prefix caches (n=%d)",
                len(prompts),
            )

        return self._process_prompts_uniform(prompts)

    def _step(self, input_tokens: mx.array, prompt_cache: List[Any]):
        logits = self.model(input_tokens, cache=prompt_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)
        return sampled, logprobs

    def stats(self):
        self._stats.prompt_tps = self._stats.prompt_tokens / max(
            self._stats.prompt_time, 1e-8
        )
        self._stats.generation_tps = self._stats.generation_tokens / max(
            self._stats.generation_time, 1e-8
        )
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def _next(self):
        import time

        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        while num_to_add >= self.prefill_batch_size:
            prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            # Finish processing the last examples of the last batch
            if len(prompts) == 0 and num_active > 0:
                break
            # No more prompts and no more completions, all done
            elif len(prompts) == 0:
                self.active_batch = None
                return []
            # Process prompts
            if batch is not None and not prompt_processing:
                # Finish any active completion tokens
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            batch = self._process_prompts(prompts)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.prefill_batch_size :
            ]
            prompt_processing = True
            # If there was no active batch, set it
            if self.active_batch is None:
                self.active_batch = batch
            else:
                self.active_batch.extend(batch)

            num_active = len(self.active_batch)
            num_to_add -= len(batch)

        batch = self.active_batch
        if batch is None:
            return []

        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

        y_list = y.tolist()
        toc = time.perf_counter()
        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic
        keep_idx = []
        end_idx = []
        responses: List[CustomBatchGenerator.Response] = []

        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y_list, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            responses.append(
                CustomBatchGenerator.Response(uid, t, logprobs[e], finish_reason)
            )

        # Remove any finished completions
        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def next(self) -> List["CustomBatchGenerator.Response"]:
        with mx.stream(generation_stream):
            return self._next()


def custom_batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    max_kv_size: Optional[int] = None,
    prompt_caches: Optional[List[Optional[List[Any]]]] = None,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts using CustomBatchGenerator.

    Args:
        model: The language model (nn.Module).
        tokenizer: The tokenizer with a `decode` method.
        prompts: List of token id sequences (suffix-only when using prompt_caches).
        max_tokens: Maximum number of output tokens (int or per-prompt list).
        verbose: If True, print progress and simple stats.
        max_kv_size: Optional cap on rotating KV cache size.
        prompt_caches: Optional list of per-prompt prefix caches.
        **kwargs: Passed through to CustomBatchGenerator (e.g. sampler, batch sizes).
    """

    gen = CustomBatchGenerator(
        model,
        stop_tokens=set(tokenizer.eos_token_ids),
        max_kv_size=max_kv_size,
        **kwargs,
    )

    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(
            f"[custom_batch_generate] Finished processing 0/{num_samples} ...",
            end="\r",
        )

    with wired_limit(model, [generation_stream]):
        uids = gen.insert(prompts, max_tokens, prompt_caches=prompt_caches)
        results = {uid: [] for uid in uids}
        while responses := gen.next():
            for r in responses:
                if verbose and r.finish_reason is not None:
                    fin += 1
                    print(
                        f"[custom_batch_generate] Finished processing {fin}/{num_samples} ...",
                        end="\r",
                    )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)

    if verbose:
        print(
            f"[custom_batch_generate] Finished processing {fin}/{num_samples}",
            end="\n",
        )

    texts = [tokenizer.decode(results[uid]) for uid in uids]
    stats = gen.stats()
    if verbose:
        print(
            f"[custom_batch_generate] Prompt: {stats.prompt_tokens} tokens, "
            f"{stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[custom_batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[custom_batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return BatchResponse(texts, stats)


def _main():
    """
    Simple CLI entry point to exercise CustomBatchGenerator directly.

    Example usage:

        python custom_batch.py \\
            --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
            --prompt "Hello" --prompt "How are you?" \\
            --max-tokens 32 --max-kv-size 4096

    To test prompt caches (KV-only models), you can also pass:

        --use-prefix-cache --prefix "System: you are a helpful assistant."
    """
    import argparse
    from mlx_lm.utils import load

    parser = argparse.ArgumentParser(description="Test CustomBatchGenerator.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model repo or local path (same as mlx_lm.generate --model).",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        action="append",
        dest="prompts",
        required=True,
        help="Prompt string. Can be specified multiple times.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=32,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Cap for rotating KV cache size.",
    )
    parser.add_argument(
        "--use-prefix-cache",
        action="store_true",
        help="Build per-prompt prefix caches from --prefix and test prompt_caches path.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix text used to build prompt caches when --use-prefix-cache is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timing and progress information.",
    )

    args = parser.parse_args()

    model, tokenizer = load(args.model)

    # Encode prompts
    raw_prompts = args.prompts

    prompt_caches = None
    token_prompts: List[List[int]]

    if args.use_prefix_cache and args.prefix:
        # Build per-prompt prefix caches; prompts become suffix tokens only.
        prefix_tokens = tokenizer.encode(args.prefix)
        prompt_caches = []
        for _ in raw_prompts:
            cache = cache_mod.make_prompt_cache(model, max_kv_size=args.max_kv_size)
            inputs = mx.array([prefix_tokens], dtype=mx.uint32)
            model(inputs, cache=cache)
            mx.eval([c.state for c in cache])
            prompt_caches.append(cache)

        token_prompts = [tokenizer.encode(p) for p in raw_prompts]
    else:
        # No prefix caches: prompts are full inputs.
        token_prompts = [tokenizer.encode(p) for p in raw_prompts]

    response = custom_batch_generate(
        model,
        tokenizer,
        prompts=token_prompts,
        max_tokens=args.max_tokens,
        max_kv_size=args.max_kv_size,
        prompt_caches=prompt_caches,
        verbose=args.verbose,
    )

    for i, text in enumerate(response.texts):
        print(f"\n=== Response {i} ===")
        print(text)


if __name__ == "__main__":
    _main()
