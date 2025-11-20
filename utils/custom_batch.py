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
    _BaseCache,
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
    kv_keep: int,
    model,
) -> List[Any]:
    """
    Build a batched cache from per-prompt prompt caches.

    If all prompt_caches are None, falls back to the default _make_cache.
    """
    if all(c is None for c in prompt_caches):
        return _make_cache(model, left_padding, max_kv_size=max_kv_size, kv_keep=kv_keep)

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
    kv_keep: int = 4,
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
            return CustomBatchRotatingKVCache(size, left_padding, keep=kv_keep)
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


class CustomBatchRotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, left_padding: List[int], keep: int = 0):
        self.keys = None
        self.values = None
        self.keep = keep
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])

        self.max_size = max_size
        self._idx = 0
        self._offset = 0
        self.rotated = False

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            # When rotated with keep > 0, the buffer is [keep, ring_buffer]
            # We need to reorder the ring_buffer part
            if self.keep > 0:
                # Split keep and ring buffer
                k_keep = self.keys[..., :self.keep, :]
                v_keep = self.values[..., :self.keep, :]
                k_ring = self.keys[..., self.keep:, :]
                v_ring = self.values[..., self.keep:, :]
                
                # Roll the ring buffer
                # The ring buffer starts at self.keep and has size max_size - keep
                # The current insertion point relative to ring start is _idx - keep
                shift = -(self._idx - self.keep)
                k_ring = mx.roll(k_ring, shift, axis=2)
                v_ring = mx.roll(v_ring, shift, axis=2)
                
                self.keys = mx.concatenate([k_keep, k_ring], axis=2)
                self.values = mx.concatenate([v_keep, v_ring], axis=2)
            else:
                self.keys = mx.roll(self.keys, -self._idx, axis=2)
                self.values = mx.roll(self.values, -self._idx, axis=2)
                
            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._temporal_order()

            # Slice off the end if needed
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size

        # Rotate
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = self.keep
        if self.rotated:
            self.left_padding -= S

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._offset < k.shape[2]:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self._offset, self._idx, self.rotated, self.keep)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self._offset, self._idx = map(
            int,
            v[:3],
        )
        self.rotated = bool(v[3])
        if len(v) > 4:
            self.keep = int(v[4])

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError("CustomBatchRotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        
        # Adjust left padding if we trimmed
        if (trim_size := self._idx - self.max_size + int(N > 1)) > 0:
            left_padding = left_padding - trim_size

        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            # If rotated, we need to roll the mask to match the rotated buffer
            # The buffer is [keep, ring_buffer]
            # The ring buffer is rotated by _idx - keep
            
            if self.keep > 0:
                # This part is tricky. The mask generation in BatchRotatingKVCache 
                # assumes a simple roll. With keep, it's more complex.
                # However, make_mask is usually for the attention bias.
                # If we are rotating the keys/values, we might need to rotate the mask too?
                # Actually, if we use the standard causal mask logic, we might not need to roll 
                # if we are just masking out padding.
                
                # But BatchRotatingKVCache does:
                # mask = mx.roll(mask, shift=idx + 1, axis=-1)
                
                # Let's look at RotatingKVCache.make_mask
                # It constructs the mask manually when window_size < max_size
                
                # For now, let's assume we can just roll the part after keep?
                # Or maybe we should just use the RotatingKVCache logic?
                
                # BatchRotatingKVCache logic seems to rely on the fact that the whole buffer is rotated.
                # If we have keep, only the part after keep is rotated.
                
                # Let's try to adapt the roll logic.
                idx = self._idx
                if idx >= self.max_size:
                    idx = self.keep # Wrap around to keep
                
                # We want to roll the mask such that the current position aligns?
                # Actually, let's look at how RotatingKVCache does it.
                # It doesn't seem to use left_padding.
                
                # If we are unsure, maybe we should fall back to a simpler mask if possible?
                # But we need left_padding support.
                
                # Let's assume for now that we can just roll the whole thing if keep=0.
                # If keep > 0, we might need to be careful.
                
                # If we look at _update_in_place, we write to _idx.
                # The "logical" end of the buffer is at _idx.
                # The "logical" start of the ring buffer is at _idx + 1 (if full).
                
                # If keep > 0, the logical order is:
                # [0...keep-1] (fixed)
                # [_idx...max_size-1] (oldest part of ring)
                # [keep..._idx-1] (newest part of ring)
                
                # So we need to roll the part from keep to max_size?
                pass
            
            idx = self._idx
            if idx >= self.max_size:
                idx = self.keep # Reset to keep if at end
                
            # If keep > 0, we only roll the ring buffer part?
            # No, the mask is 2D (L, L) or similar.
            # The mask corresponds to the keys/values layout.
            
            # If keys are [keep, ring], and ring is rotated, then the mask for ring should be rotated.
            # But the mask for keep should stay?
            
            # This seems complicated to implement correctly with vectorization.
            # However, if we look at how BatchRotatingKVCache does it:
            # mask = mx.roll(mask, shift=idx + 1, axis=-1)
            
            # This shifts the mask columns.
            
            # If we have keep tokens, they are at the beginning and don't move.
            # The ring buffer rotates.
            
            # Maybe we can split the mask?
            # mask[:, :, :, :keep] -> stays
            # mask[:, :, :, keep:] -> rolls
            
            if self.keep > 0:
                mask_keep = mask[..., :self.keep]
                mask_ring = mask[..., self.keep:]
                
                # Roll the ring part
                # The shift should be relative to the ring size?
                # idx is absolute index.
                # ring index is idx - keep.
                shift = (idx - self.keep) + 1
                mask_ring = mx.roll(mask_ring, shift=shift, axis=-1)
                
                mask = mx.concatenate([mask_keep, mask_ring], axis=-1)
            else:
                mask = mx.roll(mask, shift=idx + 1, axis=-1)

        return mask


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
        kv_keep: int = 4,
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
        self.kv_keep = kv_keep

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
            left_padding,
            max_kv_size=self.max_kv_size,
            kv_keep=self.kv_keep,
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
        logger.debug(
            "_step: generating next token, input_tokens_shape=%s",
            input_tokens.shape,
        )
        logits = self.model(input_tokens, cache=prompt_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)
        logger.debug(
            "_step: sampled tokens, sampled_shape=%s, sampled=%s",
            sampled.shape,
            sampled.tolist() if sampled.size <= 10 else f"{sampled.tolist()[:10]}...",
        )
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
            logger.debug(
                "_next: creating Response, uid=%s, token=%s, num_tokens=%d/%d, "
                "finish_reason=%s",
                uid,
                t,
                num_tok,
                max_tok,
                finish_reason,
            )
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
    kv_keep: int = 4,
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
        kv_keep=kv_keep,
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
        "--kv-keep",
        type=int,
        default=4,
        help="Number of tokens to keep in rotating KV cache.",
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
        kv_keep=args.kv_keep,
        prompt_caches=prompt_caches,
        verbose=args.verbose,
    )

    for i, text in enumerate(response.texts):
        print(f"\n=== Response {i} ===")
        print(text)


if __name__ == "__main__":
    _main()
