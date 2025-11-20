"""
Base message converter for different model formats.
Handles conversion of messages and tools to model-specific formats.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseMessageConverter(ABC):
    """Abstract base class for message converters"""

    @abstractmethod
    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert messages to model-specific format.

        Args:
            messages: List of messages in OpenAI-compatible format

        Returns:
            List of messages in model-specific format
        """
        pass

    @abstractmethod
    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """
        Convert tools to model-specific format.

        Args:
            tools: List of tools in OpenAI-compatible format or None

        Returns:
            List of tools in model-specific format or None
        """
        pass