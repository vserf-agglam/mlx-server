from __future__ import annotations

import json
from typing import Any, Literal, Union, Optional

from pydantic import BaseModel, Field


class Usage(BaseModel):
    input_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int


class OutputToolContentItem(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict


class OutputTextContentItem(BaseModel):
    type: Literal["text"]
    text: str


class InputSchema(BaseModel):
    type: Literal["object"]
    properties: dict[str, dict[str, str | list | object]]
    required: list[str] = Field(default_factory=list)


class OpenaiCompitableToolType(BaseModel):
    name: str
    description: Optional[str] = None
    properties: InputSchema


class ToolType(BaseModel):
    name: str
    description:Optional[str] = None
    input_schema: InputSchema

    def get_openai_compitable(self) -> dict[str, Any]:
        """Convert Anthropic tool format to OpenAI tool format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_dump()
            }
        }


class Source(BaseModel):
    type: Literal["base64", "url"]
    media_type: str
    data: str | None = None
    url: str | None = None


class InputContentItem(BaseModel):
    """
    Unified model handling Text, Images, Tool Uses (calls), and Tool Results.
    Fields are optional to accommodate different message types.
    """
    type: Literal["text", "image", "tool_use", "tool_call", "tool_result"] = "text"

    # Common fields
    id: Optional[str] = None  # Used for tool_use_id or tool_call_id

    # Content fields
    text: Optional[str] = None  # Helper for direct text input in dicts
    content: Union[str, list, dict, None] = None
    source: Optional[Source] = None

    # Tool specific fields
    name: Optional[str] = None
    input: Optional[dict] = None  # Anthropic style arguments
    tool_use_id: Optional[str] = None  # Specific to tool_result

    def get_openai_compatible(self) -> dict[str, Any]:
        """
        Converts this specific item into an OpenAI-compatible dictionary component.
        Note: This returns the inner content structure, not the full message wrapper
        unless it is a standalone tool result.
        """
        # 1. Handle Text
        if self.type == "text":
            # Handle edge case where content is in 'text' field or 'content' field
            text_content = self.text or self.content
            if isinstance(text_content, list):
                # Flatten list if necessary, though usually text is str
                return {"type": "text", "text": str(text_content)}
            return {"type": "text", "text": str(text_content) if text_content else ""}

        # 2. Handle Images
        elif self.type == "image" and self.source:
            url_data = self.source.url
            if self.source.type == "base64":
                url_data = f"data:{self.source.media_type};base64,{self.source.data}"

            return {
                "type": "image_url",
                "image_url": {"url": url_data}
            }

        # 3. Handle Tool Use / Tool Call (Assistant requesting execution)
        elif self.type in ["tool_use", "tool_call"]:
            # Determine arguments (handle JSON string vs dict)
            args = self.input if self.input is not None else self.arguments
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                args_str = str(args) if args else "{}"

            return {
                "id": self.id,
                "type": "function",
                "function": {
                    "name": self.name,
                    "arguments": args_str
                }
            }

        # 4. Handle Tool Result (Output of execution)
        elif self.type == "tool_result":
            content_str = self.content
            if isinstance(self.content, (dict, list)):
                content_str = json.dumps(self.content)

            # Tool results must be distinct messages in OpenAI, so we return the full message structure
            # This requires the parent processor to handle this return type differently
            return {
                "role": "tool",
                "tool_call_id": self.tool_use_id or self.id,
                "content": str(content_str)
            }

        return {"type": "text", "text": ""}


class InputMessageType(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    # Content is now a list of the Unified InputContentItem
    content: Union[str, list[InputContentItem]]

    def get_openai_compatible(self) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """
        Convert Anthropic-style message(s) to OpenAI format.
        Handles merging text/images and separating tool calls/results.
        """

        # 1. Handle Simple String Content
        if isinstance(self.content, str):
            return {
                "role": self.role,
                "content": self.content
            }

        # 2. Handle List Content
        openai_messages = []

        # Buffers for merging content within a single message
        current_text_image_content = []
        current_tool_calls = []

        for item in self.content:
            converted = item.get_openai_compatible()

            if item.type == "tool_result":
                # OpenAI requires tool results to be individual messages with role='tool'
                # We return these directly.
                # If we have buffered content, valid or not, we treat this as a break.
                openai_messages.append(converted)

            elif item.type in ["tool_use", "tool_call"]:
                # Accumulate tool calls (OpenAI allows multiple calls in one message)
                current_tool_calls.append(converted)

            else:
                # Text or Image
                current_text_image_content.append(converted)

        # Construct the final message(s) based on accumulated buffers

        # Case A: Standard User/System/Assistant Text/Image Message
        if current_text_image_content and not current_tool_calls:
            return {
                "role": self.role,
                "content": current_text_image_content
            }

        # Case B: Assistant Message with Tool Calls (and potentially text)
        if current_tool_calls:
            msg = {
                "role": "assistant",  # Tool calls are always from assistant
                "tool_calls": current_tool_calls
            }
            # OpenAI allows content (thought process) alongside tool_calls
            if current_text_image_content:
                # If it's just text, flatten it to a string if preferred,
                # but keeping it as a list of content blocks is safer for multimodal.
                msg["content"] = current_text_image_content if len(current_text_image_content) > 1 else \
                current_text_image_content[0].get('text', '')
            else:
                msg["content"] = None

            return msg

        # Case C: Only Tool Results were processed
        if openai_messages:
            if len(openai_messages) == 1:
                return openai_messages[0]
            return openai_messages

        # Fallback
        return {"role": self.role, "content": ""}

class ToolChoice(BaseModel):
    type: Literal["auto"]


class MessagesBody(BaseModel):
    model: str
    # Accept both plain text/image messages and tool-related messages.
    # We intentionally do not use a discriminated union here so that
    # callers can omit the top-level "type" field for normal text messages.
    messages: list[InputMessageType]
    tools: list[ToolType] | None = None
    max_tokens: int = 16000
    stop_sequence: list[str] | None = None
    tool_choice: ToolChoice | None = None
    top_k: int | None = None
    top_p: float | None = None
    temperature: float | None = None
    stream: bool = False

    def get_openai_compatible_messages(self) -> list[dict[str, Any]]:
        """Convert all messages to OpenAI-compatible format"""
        result = []
        for msg in self.messages:
                result.append(msg.get_openai_compatible())
        return result

    def get_openai_compatible_tools(self) -> list[dict[str, Any]] | None:
        """Convert all tools to OpenAI-compatible format"""
        if not self.tools:
            return None
        return [tool.get_openai_compitable() for tool in self.tools]


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["user", "assistant", "system"]
    model: str
    content: list[OutputTextContentItem | OutputToolContentItem]
    stop_reason: Literal["end_turn", "max_tokens", "tool_use"]
    stop_sequence: str | None
    usage: Usage
