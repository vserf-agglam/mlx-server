"""
Factory for creating message converters.
"""

from message_converter.base_message_converter import BaseMessageConverter, logger
from message_converter.openai_message_converter import OpenAIMessageConverter
from message_converter.qwen3_message_converter import Qwen3MessageConverter
from message_converter.anthropic_message_converter import AnthropicMessageConverter


class MessageConverterFactory:
    """Factory for creating message converters"""

    _converters: dict[str, type[BaseMessageConverter]] = {
        "openai": OpenAIMessageConverter,
        "qwen3": Qwen3MessageConverter,
        "anthropic": AnthropicMessageConverter,
    }

    @classmethod
    def create(cls, converter_name: str) -> BaseMessageConverter:
        """
        Create a message converter instance.

        Args:
            converter_name: Name of the converter to create

        Returns:
            Message converter instance

        Raises:
            ValueError: If converter_name is not supported
        """
        converter_class = cls._converters.get(converter_name)
        if converter_class is None:
            supported = ", ".join(cls._converters.keys())
            raise ValueError(
                f"Unsupported message converter: {converter_name}. "
                f"Supported converters: {supported}"
            )

        logger.info(f"Creating message converter: {converter_name}")
        return converter_class()

    @classmethod
    def register_converter(cls, name: str, converter_class: type[BaseMessageConverter]):
        """
        Register a custom message converter.

        Args:
            name: Name to register the converter under
            converter_class: Converter class to register
        """
        cls._converters[name] = converter_class
        logger.info(f"Registered custom message converter: {name}")

    @classmethod
    def list_converters(cls) -> list[str]:
        """Get list of available converter names"""
        return list(cls._converters.keys())