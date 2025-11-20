import logging
import os
import queue
import time
import threading
from typing import Literal, Generator, Optional
from uuid import uuid4

import mlx.core as mx

from mlx_lm.generate import wired_limit, generation_stream
from mlx_lm.models import cache as cache_mod
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

from api.types import (
    MessagesBody,
    MessagesResponse,
    Usage,
)
from token_parser.base_token_parser import BaseTokenParser
from token_parser.token_parser_factory import TokenParserFactory
from utils.chat_template_manager import ChatTemplateManager
from utils.custom_batch import CustomBatchGenerator
from utils.prompt_cache_helper import PromptCacheHelper

logger = logging.getLogger(__name__)


class Server:
    def __init__(
        self,
        model_name: str,
        token_parser_name: Literal["qwen3_moe"],
        message_converter_name: str = "openai",
        prefill_batch_size: int = 8,
        completion_batch_size: int = 32,
        trust_remote_code: bool = False,
        max_kv_size: Optional[int] = None,
        kv_keep: int = 4,
        cache_path: str = "caches",
        chat_template: Optional[str] = None
    ):
        self.loaded = False
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.queue_prompts = queue.Queue(maxsize=0)
        self.token_parser_name = token_parser_name
        self.token_parser: BaseTokenParser = TokenParserFactory.create(
            token_parser_name
        )

        self.active_generations = {}  # prompt_id -> generation data
        self.batch_thread = None
        self.running = False
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size
        self.trust_remote_code = trust_remote_code
        # Ensure max_kv_size is always an integer; this guards against
        # CLI argument parsing passing a string value through.
        self.max_kv_size = int(max_kv_size) if max_kv_size is not None else None
        self.kv_keep = kv_keep
        self.cache_path = cache_path
        self.prompt_cache_manager = PromptCacheHelper(cache_path)
        self.chat_template_input = chat_template
        self.message_converter_name = message_converter_name
        self.chat_template_manager = None

    def load(self):
        before_loading_time = time.time()
        model, tokenizer = load(self.model_name)
        after_loading_time = time.time()
        loading_time = after_loading_time - before_loading_time
        logger.info(f"Model loaded in {loading_time:.2f}s")
        self.model = model
        self.tokenizer = TokenizerWrapper(tokenizer)
        
        # Load custom chat template if provided
        if self.chat_template_input:
            self._load_custom_chat_template()

        self.chat_template_manager = ChatTemplateManager(self.tokenizer,
                                                         custom_chat_template=self.chat_template_input,
                                                         message_converter_name=self.message_converter_name,
                                                         trust_remote_code=self.trust_remote_code
                                                         )
        
        self.loaded = True
        
        # Log memory stats after model loading
        try:
            memory_gb = mx.get_active_memory() / 1e9
            peak_gb = mx.get_peak_memory() / 1e9
            logger.info(f"Model loaded | VRAM: {memory_gb:.2f}GB (peak: {peak_gb:.2f}GB)")
        except AttributeError:
            logger.info("Model loaded successfully")

    def unload(self):
        self.stop_batch_processing()
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def _load_custom_chat_template(self):
        """Load and validate custom chat template"""
        template_str = self.chat_template_input
        
        # Check if it's a file path
        if os.path.exists(template_str):
            logger.info(f"Loading custom chat template from file: {template_str}")
            try:
                with open(template_str, 'r') as f:
                    template_str = f.read()
            except Exception as e:
                logger.error(f"Failed to read chat template file: {e}")
                raise ValueError(f"Failed to read chat template file: {e}")
        else:
            logger.info("Using inline custom chat template")
        
        # Validate the template by attempting to use it
        try:
            # Test with a simple message
            test_messages = [{
                "role": "user",
                "content": "Hello"
            }]
            test_result = self.tokenizer.apply_chat_template(
                test_messages,
                chat_template=template_str,
                tokenize=False,
                trust_remote_code=self.trust_remote_code
            )
            logger.info(f"Custom chat template validated successfully. Test output length: {len(test_result)}")
            self.custom_chat_template = template_str
        except Exception as e:
            logger.error(f"Invalid custom chat template: {e}")
            raise ValueError(f"Invalid custom chat template: {e}")

    def chat_template(self, messages_body: MessagesBody, **kwargs):
        """Apply chat template with OpenAI-compatible format.

        By default, add_generation_prompt=True (to append the assistant
        generation stub), but callers can override this via kwargs.
        
        If a custom chat template is configured, it will be used instead
        of the tokenizer's default template.
        """
        return self.chat_template_manager.chat_template(messages_body, **kwargs)

    def count_input_tokens(self, messages_body: MessagesBody):
        """Count the number of input tokens for a given chat template"""
        prompt = self.chat_template(messages_body, tokenize=False)
        prompt_tokens = self.tokenizer.encode(prompt)
        token_count = len(prompt_tokens)
        return token_count

    def create_messages_response(
            self,
            generated_text: str,
            input_tokens: int,
            output_tokens: int,
            finish_reason: str = "end_turn"
    ) -> MessagesResponse:
        """Create a MessagesResponse from generated text"""
        # Use the token parser service to parse tool calls
        content, stop_reason = self.token_parser.parse_tool_calls(generated_text)

        # Use the finish_reason from generation if it indicates max_tokens
        if finish_reason == "length":
            stop_reason = "max_tokens"

        # Check if content contains tool calls and set stop_reason to "tool_use"
        from api.types import OutputToolContentItem
        if any(isinstance(item, OutputToolContentItem) for item in content):
            stop_reason = "tool_use"

        return MessagesResponse(
            id=f"msg_{uuid4().hex}",
            type="message",
            role="assistant",
            model=self.model_name,
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=input_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                output_tokens=output_tokens
            )
        )

    def start_batch_processing(self):
        """Start the continuous batch processing thread"""
        if self.batch_thread is not None and self.batch_thread.is_alive():
            logger.warning("Batch processing already running")
            return

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.running = True
        self.batch_thread = threading.Thread(target=self.continuous_batch_generate, daemon=True)
        self.batch_thread.start()
        logger.info("Started continuous batch processing thread")

    def stop_batch_processing(self):
        """Stop the continuous batch processing thread"""
        if self.batch_thread is None:
            return

        self.running = False
        if self.batch_thread.is_alive():
            self.batch_thread.join(timeout=5.0)
        logger.info("Stopped continuous batch processing thread")

    def add_to_batch(self, messages_body: MessagesBody) -> str:
        """
        Add a prompt to the batch queue.
        Returns prompt_id for tracking.
        """
        if not self.running:
            self.start_batch_processing()

        prompt = self.chat_template(messages_body, tokenize=False)
        prompt_tokens_full = self.tokenizer.encode(prompt)
        prompt_id = f"prompt_{uuid4().hex}"

        cache, cache_prompt = self.prompt_cache_manager.find_matching_cache_from_messages_body(
            # For cache lookup, use a canonical prompt that does NOT
            # include the final assistant generation stub. This ensures
            # that cached prefixes remain valid when new user turns are
            # appended before the stub in subsequent requests.
            prompt_builder=lambda body, tokenize=False: self.chat_template(
                body,
                tokenize=tokenize,
                add_generation_prompt=False,
            ),
            message_body=messages_body,
        )
        prompt_tokens = prompt_tokens_full
        prompt_cache_for_generator = None
        cache_prompt_tokens_len: int | None = None

        if cache is not None and cache_prompt is not None:
            cache_prompt_tokens = self.tokenizer.encode(cache_prompt)
            cache_prompt_tokens_len = len(cache_prompt_tokens)
            full_prompt_tokens_len = len(prompt_tokens_full)

            # Ensure the cached prompt is a real prefix (in token space)
            # of the full prompt; otherwise we cannot safely reuse it.
            prefix_len_ok = 1 < cache_prompt_tokens_len <= full_prompt_tokens_len
            prefix_tokens_match = (
                prefix_len_ok
                and prompt_tokens_full[:cache_prompt_tokens_len] == cache_prompt_tokens
            )

            if prefix_len_ok and prefix_tokens_match:
                cached_token_count = cache_prompt_tokens_len - 1

                # We must still send at least one token.
                if cached_token_count < len(prompt_tokens_full):
                    prompt_tokens = prompt_tokens_full[cached_token_count:]
                    prompt_cache_for_generator = cache
            else:
                pass

        # We consider it a cache hit when we can actually reuse a prefix
        # cache for this generation.
        cache_hit = prompt_cache_for_generator is not None


        # Initialize tracking
        self.active_generations[prompt_id] = {
            "tokens": [],
            "input_tokens": len(prompt_tokens_full),
            "completed": False,
            "finish_reason": None,
            "last_token_index": 0,  # Track which tokens have been yielded for streaming
            "prompt": prompt,
            "prompt_cache_hit": cache_hit,
            "messages_body": messages_body,
        }

        # Add to queue
        self.queue_prompts.put_nowait({
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "max_tokens": messages_body.max_tokens,
            "prompt_id": prompt_id,
            "input_tokens": len(prompt_tokens),
            "prompt_cache": prompt_cache_for_generator,
            "prompt_cache_hit": cache_hit,
        })

        return prompt_id

    def get_result(self, prompt_id: str, remove: bool = True) -> MessagesResponse | None:
        """
        Get the result for a specific prompt_id if completed.
        Returns None if not yet completed.

        Args:
            prompt_id: The prompt ID to check
            remove: Whether to remove the generation data after retrieving (default: True)
        """
        if prompt_id not in self.active_generations:
            return None

        gen_data = self.active_generations[prompt_id]
        if not gen_data["completed"]:
            return None

        # Generate response
        generated_text = self.tokenizer.decode(gen_data["tokens"])
        response = self.create_messages_response(
            generated_text,
            gen_data["input_tokens"],
            len(gen_data["tokens"]),
            gen_data["finish_reason"],
        )

        # Clean up if requested
        if remove:
            del self.active_generations[prompt_id]

        return response

    def wait_for_result(self, prompt_id: str, timeout: float = 300.0) -> MessagesResponse:
        """
        Block and wait for a result to be ready.

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            MessagesResponse when complete

        Raises:
            TimeoutError: If timeout is reached
            KeyError: If prompt_id doesn't exist
        """
        if prompt_id not in self.active_generations:
            raise KeyError(f"Prompt ID {prompt_id} not found")

        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_result(prompt_id, remove=True)
            if result is not None:
                return result
            time.sleep(0.01)  # Small sleep to avoid busy waiting

        raise TimeoutError(f"Timeout waiting for result {prompt_id}")

    def get_streaming_tokens(self, prompt_id: str) -> list[int]:
        """
        Get new tokens since last call for streaming.
        Returns list of new token IDs.
        """
        if prompt_id not in self.active_generations:
            return []

        gen_data = self.active_generations[prompt_id]
        last_index = gen_data["last_token_index"]
        new_tokens = gen_data["tokens"][last_index:]
        gen_data["last_token_index"] = len(gen_data["tokens"])


        return new_tokens

    def continuous_batch_generate(self):
        """
        Continuously process batch queue.
        This is the ONLY place where actual model generation happens.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        gen = CustomBatchGenerator(
            self.model,
            max_tokens=128000,
            stop_tokens=set(self.tokenizer.eos_token_ids),
            prefill_batch_size=self.prefill_batch_size,
            completion_batch_size=self.completion_batch_size,
            max_kv_size=self.max_kv_size,
            kv_keep=self.kv_keep,
        )
        active_uids = {}  # Maps batch uid to prompt_id
        
        # Log KV cache configuration at startup
        logger.info(
            "KV Cache Configuration: max_kv_size=%s, kv_keep=%d | prefill_batch=%d, completion_batch=%d",
            self.max_kv_size if self.max_kv_size else "unlimited",
            self.kv_keep,
            self.prefill_batch_size,
            self.completion_batch_size
        )

        logger.info("Starting continuous batch generation loop")

        with wired_limit(self.model, [generation_stream]):
            while self.running:
                while not self.queue_prompts.empty():
                    try:
                        queue_item = self.queue_prompts.get_nowait()
                        max_tokens = queue_item["max_tokens"]
                        prompt_id = queue_item["prompt_id"]

                        logger.info(f"Adding prompt {prompt_id} to batch")
                        uids = gen.insert(
                            [queue_item["prompt_tokens"]],
                            [max_tokens],
                            prompt_caches=[queue_item["prompt_cache"]],
                        )
                        active_uids[uids[0]] = prompt_id

                    except queue.Empty:
                        break

                # Process responses if there are active generations
                if active_uids:
                    responses = gen.next()
                    if responses:
                        for r in responses:
                            if r.uid in active_uids:
                                prompt_id = active_uids[r.uid]

                                if prompt_id not in self.active_generations:
                                    logger.warning(
                                        f"Prompt {prompt_id} not in active generations"
                                    )
                                    continue

                                gen_data = self.active_generations[prompt_id]

                                # Do not append EOS / stop tokens to the output;
                                # they are only used to signal termination.
                                if r.token is not None and r.finish_reason != "stop":
                                    gen_data["tokens"].append(r.token)

                                if r.finish_reason is not None:
                                    gen_data["completed"] = True
                                    gen_data["finish_reason"] = r.finish_reason
                                    # Handle prompt cache updates for this completed
                                    # generation on the background thread.
                                    self._on_generation_complete(prompt_id, gen_data)
                                    logger.info(
                                        "Completed generation for %s (reason: %s)",
                                        prompt_id,
                                        r.finish_reason,
                                    )
                                    # Remove from active tracking
                                    del active_uids[r.uid]
                else:
                    # No active generations, sleep briefly to avoid busy waiting
                    time.sleep(0.01)

        logger.info("Continuous batch generation loop ended")

    def _on_generation_complete(self, prompt_id: str, gen_data: dict) -> None:
        """
        Handle bookkeeping and prompt cache updates when a generation finishes.

        This method runs on the background generation thread inside the
        wired_limit context, so all model / MLX work it performs is
        serialized with the rest of generation.
        """
        original_body = gen_data.get("messages_body")
        prompt_text = gen_data.get("prompt", "")

        try:
            # Log generation completion with key stats including memory
            try:
                memory_gb = mx.get_active_memory() / 1e9
                logger.info(
                    "Generation complete: %s | %s | in=%d out=%d cache_hit=%s | VRAM=%.2fGB",
                    prompt_id[:8],
                    gen_data.get("finish_reason"),
                    gen_data.get("input_tokens"),
                    len(gen_data.get("tokens", [])),
                    gen_data.get("prompt_cache_hit"),
                    memory_gb,
                )
            except AttributeError:
                logger.info(
                    "Generation complete: %s | %s | in=%d out=%d cache_hit=%s",
                    prompt_id[:8],
                    gen_data.get("finish_reason"),
                    gen_data.get("input_tokens"),
                    len(gen_data.get("tokens", [])),
                    gen_data.get("prompt_cache_hit"),
                )

            # Cache for the current prompt (conversation up to this turn),
            # using a canonical prompt that does NOT include the assistant
            # generation stub. This keeps caches stable as the conversation
            # grows with new user turns.
            if original_body is not None:
                base_prompt_for_cache = self.chat_template(
                    original_body,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if self.prompt_cache_manager.load_cache(base_prompt_for_cache) is None:
                    logger.info("Building base cache for %s", prompt_id[:8])
                    self._build_and_save_prompt_cache(base_prompt_for_cache)
            elif prompt_text:
                # Fallback for non-chat usage: cache the raw prompt string.
                if self.prompt_cache_manager.load_cache(prompt_text) is None:
                    logger.info("Building base cache for %s", prompt_id[:8])
                    self._build_and_save_prompt_cache(prompt_text)

            # Cache for the extended "next turn" prompt, built by appending
            # this assistant response to the original MessagesBody.
            if original_body is not None:
                generated_text = self.tokenizer.decode(gen_data["tokens"])
                tmp_response = self.create_messages_response(
                    generated_text,
                    gen_data["input_tokens"],
                    len(gen_data["tokens"]),
                    gen_data["finish_reason"],
                )
                extended_body = self.prompt_cache_manager.build_body_with_response(
                    original_body,
                    tmp_response,
                )
                if extended_body is not None:
                    extended_prompt = self.chat_template(
                        extended_body,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    if (
                        self.prompt_cache_manager.load_cache(extended_prompt)
                        is None
                    ):
                        self._build_and_save_prompt_cache(extended_prompt)

                else:
                    logger.debug(
                        "on_generation_complete: extended body is None for %s; "
                        "skipping extended cache build",
                        prompt_id,
                    )
            else:
                logger.debug(
                    "on_generation_complete: no original messages_body stored "
                    "for %s; skipping extended cache build",
                    prompt_id,
                )
        except Exception:
            logger.exception(
                "Failed to build prompt cache for %s",
                prompt_id,
            )

    def _build_and_save_prompt_cache(self, prompt_text: str) -> None:
        """
        Build and persist a prompt cache for the given prompt string.

        The cache is built over all but the last token of the prompt so that
        future requests can reuse the prefix and only feed the remaining
        suffix tokens to reconstruct the full context.
        """
        prompt_tokens = self.tokenizer.encode(prompt_text)
        if len(prompt_tokens) <= 1:
            return

        prefix_tokens = prompt_tokens[:-1]
        logger.debug(
            "_build_and_save_prompt_cache: prompt_len_tokens=%d, "
            "prefix_len_tokens=%d, max_kv_size=%d",
            len(prompt_tokens),
            len(prefix_tokens),
            self.max_kv_size,
        )
        cache = cache_mod.make_prompt_cache(
            self.model,
            max_kv_size=self.max_kv_size,
        )
        
        # Log cache type being created
        if cache and len(cache) > 0:
            cache_type = type(cache[0]).__name__
            logger.info(
                "Building prompt cache | type=%s, layers=%d, max_kv_size=%s",
                cache_type,
                len(cache),
                self.max_kv_size if self.max_kv_size else "unlimited"
            )
        
        inputs = mx.array([prefix_tokens], dtype=mx.uint32)
        try:
            self.model(inputs, cache=cache)
            mx.eval([c.state for c in cache])
        except OverflowError:
            # If the underlying cache implementation overflows when trying
            # to compute an internal prime size (e.g. for very large
            # contexts), skip caching for this prompt instead of failing
            # the whole request.
            logger.exception(
                "Overflow while building prompt cache; skipping cache for this prompt"
            )
            return

        self.prompt_cache_manager.save_cache(prompt_text, cache)

    def generate(self, messages_body: MessagesBody, timeout: float = 300.0) -> MessagesResponse:
        """
        Generate a single response synchronously.
        Uses continuous_batch_generate under the hood.

        Args:
            messages_body: The messages to generate from
            timeout: Maximum time to wait in seconds

        Returns:
            MessagesResponse when complete
        """
        prompt_id = self.add_to_batch(messages_body)
        return self.wait_for_result(prompt_id, timeout=timeout)

    def generate_stream(
            self,
            messages_body: MessagesBody,
            timeout: float = 300.0
    ) -> Generator[dict, None, MessagesResponse]:
        """
        Generate a streaming response.
        Yields partial tokens as dicts with 'delta' key.
        Returns final MessagesResponse when complete.
        Uses continuous_batch_generate under the hood.

        Args:
            messages_body: The messages to generate from
            timeout: Maximum time to wait in seconds

        Yields:
            Dict with 'delta' key containing new token text

        Returns:
            MessagesResponse when complete
        """
        prompt_id = self.add_to_batch(messages_body)

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Get new tokens
            new_tokens = self.get_streaming_tokens(prompt_id)
            if new_tokens:
                # Decode and yield new tokens
                token_text = self.tokenizer.decode(new_tokens)
                if token_text:
                    yield {"delta": token_text}

            # Check if completed
            if prompt_id in self.active_generations:
                if self.active_generations[prompt_id]["completed"]:
                    # Get final result
                    result = self.get_result(prompt_id, remove=True)
                    return result
            else:
                # Generation was removed, likely completed
                break

            time.sleep(0.01)  # Small sleep to avoid busy waiting

        raise TimeoutError(f"Timeout waiting for streaming result {prompt_id}")

    def batch_generate(
            self,
            messages_bodies: list[MessagesBody],
            timeout: float = 300.0
    ) -> list[MessagesResponse]:
        """
        Generate responses for multiple prompts in a batch.
        Uses continuous_batch_generate under the hood.

        Args:
            messages_bodies: List of messages to generate from
            timeout: Maximum time to wait in seconds

        Returns:
            List of MessagesResponse objects in the same order as input
        """
        if not messages_bodies:
            return []

        # Add all to batch
        prompt_ids = [self.add_to_batch(msg_body) for msg_body in messages_bodies]

        # Wait for all to complete
        start_time = time.time()
        results = [None] * len(prompt_ids)
        completed = set()

        while time.time() - start_time < timeout:
            for idx, prompt_id in enumerate(prompt_ids):
                if idx in completed:
                    continue

                result = self.get_result(prompt_id, remove=False)
                if result is not None:
                    results[idx] = result
                    completed.add(idx)
                    # Now remove it
                    if prompt_id in self.active_generations:
                        del self.active_generations[prompt_id]

            # Check if all completed
            if len(completed) == len(prompt_ids):
                return results

            time.sleep(0.01)  # Small sleep to avoid busy waiting

        # Timeout - clean up remaining
        for idx, prompt_id in enumerate(prompt_ids):
            if idx not in completed and prompt_id in self.active_generations:
                del self.active_generations[prompt_id]

        raise TimeoutError(f"Timeout waiting for batch results. Completed {len(completed)}/{len(prompt_ids)}")
