"""
Token parsers for different model formats.
Handles parsing tool calls and other special tokens from generated text.
"""

import logging
from abc import ABC, abstractmethod

from api.types import OutputTextContentItem, OutputToolContentItem

logger = logging.getLogger(__name__)


class BaseTokenParser(ABC):
    """Abstract base class for token parsers"""

    @abstractmethod
    def parse_tool_calls(
            self,
            text: str
    ) -> tuple[list[OutputTextContentItem | OutputToolContentItem], str]:
        """
        Parse tool calls from generated text.

        Args:
            text: The generated text to parse

        Returns:
            Tuple of (content_items, stop_reason)
        """
        pass

    @abstractmethod
    def get_stop_tokens(self) -> list[str]:
        """
        Get stop tokens for this parser format.

        Returns:
            List of stop token strings
        """
        pass
