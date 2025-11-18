import json
import re

from api.types import OutputToolContentItem, OutputTextContentItem
from token_parser.base_token_parser import BaseTokenParser, logger
from utils.tool_stream_helper import make_tool_call_id


class Qwen3MoeParser(BaseTokenParser):
    """
    Token parser for Qwen3 MoE models.

    Format: <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
    """

    def __init__(self):
        # Hints used by streaming helpers to avoid emitting partial blocks.
        self.tool_call_open_tag = "<tool_call>"
        self.tool_call_close_tag = "</tool_call>"
        self.tool_call_pattern = r"<tool_call>(.*?)</tool_call>"

    def parse_tool_calls(
            self,
            text: str
    ) -> tuple[list[OutputTextContentItem | OutputToolContentItem], str]:
        """
        Parse tool calls from Qwen3 MoE generated text.

        Args:
            text: The generated text to parse

        Returns:
            Tuple of (content_items, stop_reason)
        """
        content = []
        stop_reason = "end_turn"

        # Find all tool call matches
        matches = list(re.finditer(self.tool_call_pattern, text, re.DOTALL))

        if not matches:
            # No tool calls found, return entire text
            content.append(OutputTextContentItem(type="text", text=text))
            return content, stop_reason

        last_end = 0
        for match in matches:
            # Add text before tool call
            text_before = text[last_end:match.start()].strip()
            if text_before:
                content.append(OutputTextContentItem(type="text", text=text_before))

            # Parse tool call JSON
            try:
                raw_tool_json = match.group(1)
                tool_data = json.loads(raw_tool_json)
                content.append(OutputToolContentItem(
                    id=make_tool_call_id(raw_tool_json),
                    name=tool_data.get("name", ""),
                    input=tool_data.get("arguments", {})
                ))
                logger.debug(f"Parsed tool call: {tool_data.get('name')}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON: {e}")
                logger.error(f"Raw tool call content: {match.group(1)}")
                # Add as text if parsing fails
                content.append(OutputTextContentItem(type="text", text=match.group(0)))

            last_end = match.end()

        # Add remaining text after last tool call
        text_after = text[last_end:].strip()
        if text_after:
            content.append(OutputTextContentItem(type="text", text=text_after))

        # If we found tool calls, keep stop_reason as end_turn
        if any(isinstance(item, OutputToolContentItem) for item in content):
            stop_reason = "end_turn"

        return content, stop_reason

    def get_stop_tokens(self) -> list[str]:
        """Get stop tokens for Qwen3 MoE"""
        return ["</tool_call>", "<|endoftext|>", "<|im_end|>"]
