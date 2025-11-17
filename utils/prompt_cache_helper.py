import os
from hashlib import md5

from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache


class PromptCacheHelper:
    def __init__(self, cache_path: str) -> None:
        self.cache_path = cache_path

    def get_file_name(self, prompt: str) -> str:
        hash = md5(prompt.encode()).hexdigest() + ".safetensors"
        return os.path.join(self.cache_path, hash)

    def save_cache(self, prompt):
        file_name = self.get_file_name(prompt)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        save_prompt_cache(file_name, prompt)

    def load_cache(self, prompt):
        file_name = self.get_file_name(prompt)
        if os.path.exists(file_name):
            return load_prompt_cache(file_name)

        return None