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


        # 2. Pre-process kwargs (Apply defaults BEFORE caching to ensure key consistency)
        kwargs.setdefault("add_generation_prompt", True)
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template


        # 4. Call the cached internal method
        return self._apply_template_cached( messages_body, **kwargs)


    def _apply_template_cached(self, messages_body: MessagesBody, **kwargs):
        """Internal method to perform the actual processing, cached by lru_cache."""
        print("Cache MISS: Computing chat template...")

        openai_messages = messages_body.messages
        openai_tools = messages_body.get_openai_compatible_tools()
        run_kwargs = kwargs

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
