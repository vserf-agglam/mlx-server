import json
from functools import lru_cache

from api.types import MessagesBody
from message_converter import BaseMessageConverter, MessageConverterFactory


class ChatTemplateManager:  # Assuming this is inside your class
    def __init__(self, tokenizer,
                 custom_chat_template: str = None,
                 message_converter_name: str = "openai",
                 trust_remote_code: bool = True,
                 ):
        self.custom_chat_template = custom_chat_template
        self.message_converter_name = message_converter_name
        self.message_converter: BaseMessageConverter = MessageConverterFactory.create(
            message_converter_name
        )
        self.trust_remote_code = trust_remote_code
        self.tokenizer = tokenizer


    def chat_template(self, messages_body: MessagesBody, **kwargs):
        """Apply chat template with OpenAI-compatible format with Caching."""

        # 1. Extract raw data
        openai_messages = messages_body.get_openai_compatible_messages()
        openai_tools = messages_body.get_openai_compatible_tools()

        # 2. Pre-process kwargs (Apply defaults BEFORE caching to ensure key consistency)
        kwargs.setdefault("add_generation_prompt", True)
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template

        # 3. Create a stable Cache Key
        # We serialize inputs to JSON because lists/dicts are not hashable.
        # sort_keys=True guarantees that the order of keys doesn't cause a cache miss.
        # default=str handles any non-standard objects gracefully.
        cache_key = json.dumps({
            "msgs": openai_messages,
            "tools": openai_tools,
            "kwargs": kwargs
        }, sort_keys=True, default=str)

        # 4. Call the cached internal method
        return self._apply_template_cached(cache_key)


    @lru_cache(maxsize=1000)  # Stores the last 1000 unique requests
    def _apply_template_cached(self, cache_key_json: str):
        """Internal method to perform the actual processing, cached by lru_cache."""
        print("Cache MISS: Computing chat template...")

        # Deserialize the data
        data = json.loads(cache_key_json)
        openai_messages = data["msgs"]
        openai_tools = data["tools"]
        run_kwargs = data["kwargs"]

        # Convert messages and tools to model-specific format
        # Note: We access 'self' directly here
        converted_messages = self.message_converter.convert_messages(openai_messages)
        converted_tools = self.message_converter.convert_tools(openai_tools)

        return self.tokenizer.apply_chat_template(
            converted_messages,
            tools=converted_tools,
            trust_remote_code=self.trust_remote_code,
            **run_kwargs
        )
