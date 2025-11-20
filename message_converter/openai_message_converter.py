"""
OpenAI message converter - pass-through converter for OpenAI-compatible models.
"""

from typing import Any

from message_converter.base_message_converter import BaseMessageConverter, logger


class OpenAIMessageConverter(BaseMessageConverter):
    """
    Message converter for OpenAI-compatible models.
    This is a pass-through converter since messages are already in OpenAI format.
    """

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Pass through messages without conversion (already in OpenAI format).

        Args:
            messages: List of messages in OpenAI-compatible format

        Returns:
            Same list of messages (unchanged)
        """
        logger.debug(f"OpenAI converter: passing through {len(messages)} messages")
        return messages

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """
        Pass through tools without conversion (already in OpenAI format).

        Args:
            tools: List of tools in OpenAI-compatible format or None

        Returns:
            Same list of tools (unchanged) or None
        """
        if tools:
            logger.debug(f"OpenAI converter: passing through {len(tools)} tools")
        return tools