import json
import re

from api.types import OutputToolContentItem, OutputTextContentItem
from token_parser.base_token_parser import BaseTokenParser, logger
from utils.tool_stream_helper import make_tool_call_id


class Qwen3MoeParser(BaseTokenParser):
    """
    Token parser for Qwen3 MoE models.

    Format: 
    <tool_call>
    <function=function_name>
    <parameter=param_name>
    value
    </parameter>
    </function>
    </tool_call>
    """

    def __init__(self):
        # Hints used by streaming helpers to avoid emitting partial blocks.
        self.tool_call_open_tag = "<tool_call>"
        self.tool_call_close_tag = "</tool_call>"
        self.tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        self.function_pattern = r"<function=([^>]+)>(.*?)</function>"
        self.parameter_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"

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
        logger.debug(
            "parse_tool_calls: parsing text, text_len=%d, text_preview=%r",
            len(text),
            text[:200] if len(text) > 200 else text,
        )
        content = []
        stop_reason = "end_turn"

        # Find all tool call matches
        matches = list(re.finditer(self.tool_call_pattern, text, re.DOTALL))
        logger.debug(
            "parse_tool_calls: found %d tool_call matches",
            len(matches),
        )

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

            # Parse tool call with XML-like format
            try:
                tool_content = match.group(1).strip()
                logger.debug(
                    "parse_tool_calls: parsing tool_call block, "
                    "tool_content_len=%d, tool_content=%r",
                    len(tool_content),
                    tool_content[:500] if len(tool_content) > 500 else tool_content,
                )

                # Extract function name and parameters
                function_match = re.search(self.function_pattern, tool_content, re.DOTALL)
                if function_match:
                    function_name = function_match.group(1)
                    function_content = function_match.group(2)
                    logger.debug(
                        "parse_tool_calls: extracted function_name=%s, "
                        "function_content_len=%d",
                        function_name,
                        len(function_content),
                    )
                    
                    # Extract parameters
                    parameters = {}
                    param_matches = list(re.finditer(self.parameter_pattern, function_content, re.DOTALL))
                    logger.debug(
                        "parse_tool_calls: found %d parameter matches",
                        len(param_matches),
                    )
                    for param_match in param_matches:
                        param_name = param_match.group(1)
                        param_value = param_match.group(2).strip()

                        # Try to parse as JSON if it looks like JSON
                        if (param_value.startswith('{') and param_value.endswith('}')) or \
                           (param_value.startswith('[') and param_value.endswith(']')):
                            try:
                                param_value = json.loads(param_value)
                            except json.JSONDecodeError:
                                # Keep as string if JSON parsing fails
                                pass

                        parameters[param_name] = param_value
                        logger.debug(
                            "parse_tool_calls: extracted parameter, param_name=%s, "
                            "param_value_preview=%r",
                            param_name,
                            str(param_value)[:200] if len(str(param_value)) > 200 else param_value,
                        )
                    
                    tool_call_id = make_tool_call_id(tool_content)
                    content.append(OutputToolContentItem(
                        type="tool_use",
                        id=tool_call_id,
                        name=function_name,
                        input=parameters
                    ))
                    logger.debug(
                        "parse_tool_calls: created OutputToolContentItem, "
                        "tool_call_id=%s, function_name=%s, parameters=%s",
                        tool_call_id,
                        function_name,
                        parameters,
                    )
                else:
                    # If no function tag found, log error and add as text
                    logger.error(f"No function tag found in tool call")
                    logger.error(f"Raw tool call content: {tool_content}")
                    content.append(OutputTextContentItem(type="text", text=match.group(0)))
                    
            except Exception as e:
                logger.error(f"Failed to parse tool call: {e}")
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
