import json
from typing import Any
from api.types import InputMessageType
from message_converter.base_message_converter import BaseMessageConverter, logger


class Qwen3MessageConverter(BaseMessageConverter):
    def convert_messages(self, messages: list[InputMessageType]):
        converted_messages = []

        for msg in messages:
            if isinstance(msg.content, list):
                text_parts = ""
                tool_calls = []
                tool_results = []

                for content in msg.content:
                    # 1. Handle Text
                    if content.type == "text":
                        text_parts += content.text

                    # 2. Handle Tool Calls (Assistant side)
                    if content.type == "tool_call" or content.type == "tool_use":
                        tool_calls.append({
                            "function": {
                                "name": content.name,
                                "arguments": json.loads(content.input) if isinstance(content.input,
                                                                                     str) else content.input
                            }
                        })

                    # 3. Handle Tool Results
                    # FIX: Don't transform this into a dict yet; keep the object
                    # so we can access .tool_use_id and .content later.
                    if content.type == "tool_result":
                        tool_results.append(content)

                # Append Assistant/User Message (Text + Tool Calls)
                if len(tool_calls) > 0 or text_parts != "":
                    msg_body = {
                        "role": msg.role,
                        "content": text_parts,
                    }

                    if len(tool_calls) > 0:
                        msg_body["tool_calls"] = tool_calls

                    converted_messages.append(msg_body)

                # Append Tool Result Messages
                # FIX: Iterate over the stored objects and format correctly for Qwen
                if tool_results:
                    for tool_result in tool_results:
                        converted_messages.append({
                            "role": "tool",
                            # Map the tool_use_id correctly
                            "tool_call_id": tool_result.tool_use_id,
                            # Ensure content is a string
                            "content": tool_result.content if isinstance(tool_result.content, str) else json.dumps(
                                tool_result.content)
                        })

            else:
                # Handle simple string content
                converted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return converted_messages

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        return tools