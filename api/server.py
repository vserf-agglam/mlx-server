import logging
import queue
import time
import json
import re
import threading
from typing import Literal, Generator
from uuid import uuid4

from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

from api.types import (
    ToolType,
    MessagesBody,
    MessagesResponse,
    OutputTextContentItem,
    OutputToolContentItem,
    Usage
)
from token_parser.base_token_parser import BaseTokenParser
from token_parser.token_parser_factory import TokenParserFactory

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, model_name: str, token_parser_name: Literal["qwen3_moe"]):
        self.loaded = False
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.queue_prompts = queue.Queue(maxsize=0)
        self.token_parser_name = token_parser_name
        self.token_parser: BaseTokenParser = TokenParserFactory.create(token_parser_name)
        self.active_generations = {}  # prompt_id -> generation data
        self.batch_thread = None
        self.running = False

    def load(self):
        before_loading_time = time.time()
        model, tokenizer = load(self.model_name)
        after_loading_time = time.time()
        loading_time = after_loading_time - before_loading_time
        logger.info(f"Model loaded in {loading_time:.2f}s")
        self.model = model
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.loaded = True




    def unload(self):
        self.stop_batch_processing()
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def chat_template(self, messages_body: MessagesBody, **kwargs):
        """Apply chat template with OpenAI-compatible format"""
        openai_messages = messages_body.get_openai_compatible_messages()
        openai_tools = messages_body.get_openai_compatible_tools()

        return self.tokenizer.apply_chat_template(
            openai_messages,
            tools=openai_tools,
            add_generation_prompt=True,
            **kwargs
        )


    def count_input_tokens(self, messages_body: MessagesBody):
        """Count the number of input tokens for a given chat template"""
        prompt = self.chat_template(messages_body, tokenize=False)
        prompt_tokens = self.tokenizer.encode(prompt)
        return len(prompt_tokens)

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

        return MessagesResponse(
            id=f"msg_{uuid4().hex}",
            type="message",
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=input_tokens,
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
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_id = f"prompt_{uuid4().hex}"

        # Initialize tracking
        self.active_generations[prompt_id] = {
            "tokens": [],
            "input_tokens": len(prompt_tokens),
            "completed": False,
            "finish_reason": None,
            "last_token_index": 0  # Track which tokens have been yielded for streaming
        }

        # Add to queue
        self.queue_prompts.put_nowait({
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "max_tokens": messages_body.max_tokens,
            "prompt_id": prompt_id,
            "input_tokens": len(prompt_tokens)
        })

        logger.debug(f"Added prompt {prompt_id} to batch queue")
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
            gen_data["finish_reason"]
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

        gen = BatchGenerator(self.model, stop_tokens=set(self.tokenizer.eos_token_ids))
        active_uids = {}  # Maps batch uid to prompt_id

        logger.info("Starting continuous batch generation loop")

        with wired_limit(self.model, [generation_stream]):
            while self.running:
                # Add new prompts from queue
                new_prompts_added = False
                while not self.queue_prompts.empty():
                    try:
                        queue_item = self.queue_prompts.get_nowait()
                        prompt = queue_item["prompt"]
                        max_tokens = queue_item["max_tokens"]
                        prompt_id = queue_item["prompt_id"]

                        logger.info(f"Adding prompt {prompt_id} to batch")
                        uids = gen.insert([queue_item["prompt_tokens"]], [max_tokens])
                        active_uids[uids[0]] = prompt_id
                        new_prompts_added = True

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
                                    logger.warning(f"Prompt {prompt_id} not in active generations")
                                    continue

                                gen_data = self.active_generations[prompt_id]

                                if r.token is not None:
                                    gen_data["tokens"].append(r.token)

                                if r.finish_reason is not None:
                                    gen_data["completed"] = True
                                    gen_data["finish_reason"] = r.finish_reason
                                    logger.info(f"Completed generation for {prompt_id} (reason: {r.finish_reason})")
                                    # Remove from active tracking
                                    del active_uids[r.uid]
                else:
                    # No active generations, sleep briefly to avoid busy waiting
                    time.sleep(0.01)

        logger.info("Continuous batch generation loop ended")

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