import logging
import os
from hashlib import md5

from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

from api.types import MessagesBody

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
