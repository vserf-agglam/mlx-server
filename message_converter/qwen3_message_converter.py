import json
from typing import Any
from message_converter.base_message_converter import BaseMessageConverter, logger


class Qwen3MessageConverter(BaseMessageConverter):
    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logger.debug(f"Qwen3 converter: processing {len(messages)} messages")
        converted_messages = []

        for msg in messages:
            new_msg = msg.copy()

            # =================================================================
            # FIX: Handle existing tool_calls (from InputToolUseMessage)
            # =================================================================
            if "tool_calls" in new_msg and isinstance(new_msg["tool_calls"], list):
                for tool_call in new_msg["tool_calls"]:
                    if "function" in tool_call:
                        args = tool_call["function"].get("arguments")
                        # If arguments is a JSON string, parse it back to a dict
                        if isinstance(args, str):
                            try:
                                tool_call["function"]["arguments"] = json.loads(args)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tool arguments: {args}")

            # =================================================================
            # EXISTING LOGIC: Handle Tool Use hidden in "content" string
            # =================================================================
            content = new_msg.get("content")
            parsed_content = None
            is_json_content = False

            # Attempt to parse content if it looks like JSON
            if isinstance(content, str) and content.strip().startswith('{'):
                try:
                    parsed_content = json.loads(content)
                    is_json_content = True
                except json.JSONDecodeError:
                    pass

            # Case: content looks like '{"type": "tool_use", ...}'
            if (new_msg["role"] == "assistant" and
                    is_json_content and
                    parsed_content.get("type") == "tool_use"):

                func_name = parsed_content.get("name")
                arguments = parsed_content.get("input", {})

                qwen3_tool_call = {
                    "function": {
                        "name": func_name,
                        # Ensure this is a dict (parsed_content.get returns dict usually)
                        "arguments": arguments
                    }
                }

                new_msg["tool_calls"] = [qwen3_tool_call]
                new_msg["content"] = ""  # Clear content to avoid duplication

            # Case: content looks like '{"type": "tool_result", ...}'
            elif (is_json_content and
                  parsed_content.get("type") == "tool_result"):

                new_msg["role"] = "tool"
                inner_content = parsed_content.get("content", "")
                new_msg["content"] = inner_content

            converted_messages.append(new_msg)

        return converted_messages

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        return tools