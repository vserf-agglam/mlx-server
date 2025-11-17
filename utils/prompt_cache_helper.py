import logging
import os
from hashlib import md5

from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

from api.types import (
    MessagesBody,
    MessagesResponse,
    InputTextOrImageMessage,
    InputToolUseMessage,
    OutputTextContentItem,
    OutputToolContentItem,
)

logger = logging.getLogger(__name__)


class PromptCacheHelper:
    def __init__(self, cache_path: str) -> None:
        self.cache_path = cache_path

    def get_file_name(self, prompt: str) -> str:
        hash = md5(prompt.encode()).hexdigest() + ".safetensors"
        return os.path.join(self.cache_path, hash)

    def find_matching_cache_from_messages_body(self, prompt_builder, message_body: MessagesBody):
        current_messages_length = len(message_body.messages)
        # Try to find the longest prefix of the current messages list
        # for which we already have a saved prompt cache.
        #
        # We iterate from the full list down to a single-message prefix
        # and return as soon as we find a hit.
        for i in range(current_messages_length, 0, -1):
            messages_slice = message_body.messages[0:i]
            copied_messages_body = MessagesBody(**message_body.model_dump())
            copied_messages_body.messages = messages_slice
            prompt = prompt_builder(copied_messages_body, tokenize=False)
            cache = self.load_cache(prompt)

            if cache:
                logger.debug(
                    "Found cached prompt cache for messages body with i=%s", i
                )
                # Return both the cache object and the prompt string it was
                # built for so the caller can compute the suffix tokens.
                return cache, prompt

        return None, None

    def build_body_with_response(
        self,
        original_body: MessagesBody,
        response: MessagesResponse,
    ) -> MessagesBody | None:
        """
        Create a new MessagesBody that represents the conversation after
        the assistant reply has been added.

        This is used to precompute and cache the prompt for the next turn,
        so that future requests can reuse a longer cached prefix.
        """
        # Work on a shallow copy to avoid mutating the original request body.
        new_body = MessagesBody(**original_body.model_dump())

        # Reconstruct how this assistant turn would appear in the messages list,
        # including both text content and tool calls, so that the resulting
        # prompt string matches what a client is expected to send on the next
        # turn.
        current_text_parts: list[str] = []
        appended_any = False

        for item in response.content:
            if isinstance(item, OutputTextContentItem):
                current_text_parts.append(item.text)
            elif isinstance(item, OutputToolContentItem):
                # Flush any accumulated text as an assistant message before
                # appending the tool call message, to keep ordering reasonable.
                if current_text_parts:
                    new_body.messages.append(
                        InputTextOrImageMessage(
                            role="assistant",
                            content="\n\n".join(current_text_parts),
                            type="text",
                        )
                    )
                    appended_any = True
                    current_text_parts = []

                new_body.messages.append(
                    InputToolUseMessage(
                        type="tool_use",
                        id=item.id,
                        name=item.name,
                        input=item.input,
                    )
                )
                appended_any = True

        # Flush any remaining assistant text at the end.
        if current_text_parts:
            new_body.messages.append(
                InputTextOrImageMessage(
                    role="assistant",
                    content="\n\n".join(current_text_parts),
                    type="text",
                )
            )
            appended_any = True

        if not appended_any:
            # No text or tool calls to append; nothing to cache for the next turn.
            return None

        return new_body


    def save_cache(self, prompt: str, cache) -> None:
        """
        Save a prompt cache for the given prompt string.

        The prompt string is only used to derive a deterministic file name;
        the actual cache object is passed through to mlx_lm.save_prompt_cache.
        """
        file_name = self.get_file_name(prompt)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        logger.debug("PromptCacheHelper: saving cache to %s", file_name)
        save_prompt_cache(file_name, cache)

    def load_cache(self, prompt: str):
        file_name = self.get_file_name(prompt)
        if os.path.exists(file_name):
            logger.debug("PromptCacheHelper: cache hit for %s", file_name)
            return load_prompt_cache(file_name)

        logger.debug("PromptCacheHelper: cache miss for %s", file_name)
        return None
