import json
from typing import Any
from message_converter.base_message_converter import BaseMessageConverter, logger

class Qwen3MessageConverter(BaseMessageConverter):
    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logger.debug(f"Qwen3 converter: processing {len(messages)} messages")
        converted_messages = []

        for msg in messages:
            new_msg = msg.copy()
            content = new_msg.get("content")

            # --- 1. Parse JSON content if it exists ---
            # Your input wraps tool usage inside a stringified JSON in 'content'.
            # We need to unpack this first.
            parsed_content = None
            is_json_content = False
            if isinstance(content, str) and content.strip().startswith('{'):
                try:
                    parsed_content = json.loads(content)
                    is_json_content = True
                except json.JSONDecodeError:
                    pass

            # --- 2. Handle Assistant "Tool Use" ---
            # Case: content looks like '{"type": "tool_use", "name": "...", "input": ...}'
            if (new_msg["role"] == "assistant" and
                is_json_content and
                parsed_content.get("type") == "tool_use"):

                # Extract the tool details
                func_name = parsed_content.get("name")
                # Note: Your input uses "input", standard OpenAI uses "arguments"
                arguments = parsed_content.get("input", {})

                # Construct the structure Qwen3 template expects
                qwen3_tool_call = {
                    "function": {
                        "name": func_name,
                        "arguments": arguments  # Pass as dict, template handles serialization
                    }
                }

                # Qwen3 expects this in a list under 'tool_calls'
                new_msg["tool_calls"] = [qwen3_tool_call]

                # Clear the raw JSON content so it doesn't print to the user
                new_msg["content"] = ""

            # --- 3. Handle "Tool Result" (Output) ---
            # Case: content looks like '{"type": "tool_result", "content": "..."}'
            # AND strictly change role from 'user' to 'tool' so Qwen handles it correctly
            elif (is_json_content and
                  parsed_content.get("type") == "tool_result"):

                # Switch role to 'tool' (Qwen template requires this specific role for results)
                new_msg["role"] = "tool"

                # Extract the actual inner content (the DB schema result)
                # Sometimes the inner content is ALSO a stringified JSON (your example shows this)
                inner_content = parsed_content.get("content", "")
                new_msg["content"] = inner_content

            converted_messages.append(new_msg)

        return converted_messages

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        return tools