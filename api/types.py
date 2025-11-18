from __future__ import annotations

import json
from typing import Any, Literal, Union, Optional

from pydantic import BaseModel, Field, Discriminator
from typing import Annotated


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class OutputToolContentItem(BaseModel):
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
    data: str | None
    url: str | None


class InputToolUseMessage(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict

    def get_openai_compatible(self) -> dict[str, Any]:
        """Convert Anthropic tool_use to OpenAI tool_calls format"""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "arguments": json.dumps(self.input)  # OpenAI expects JSON string
                    }
                }
            ]
        }


class InputToolCallMessage(BaseModel):
    """OpenAI-style tool call message that maps to Anthropic's tool_use"""
    type: Literal["tool_call"]
    id: str
    name: str
    arguments: dict | str  # Can be either dict or JSON string
    
    def get_anthropic_compatible(self) -> InputToolUseMessage:
        """Convert OpenAI tool_call to Anthropic tool_use format"""
        # Handle arguments that might be JSON strings
        input_data = self.arguments
        if isinstance(self.arguments, str):
            input_data = json.loads(self.arguments)
            
        return InputToolUseMessage(
            type="tool_use",
            id=self.id,
            name=self.name,
            input=input_data
        )
    
    def get_openai_compatible(self) -> dict[str, Any]:
        """Return OpenAI-compatible format (already in correct format)"""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "arguments": json.dumps(self.arguments) if isinstance(self.arguments, dict) else self.arguments
                    }
                }
            ]
        }


class InputToolResultMessage(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | dict | list

    def get_openai_compatible(self) -> dict[str, Any]:
        """Convert Anthropic tool_result to OpenAI tool message format"""
        # Convert content to string if it's not already
        content_str = self.content
        if isinstance(self.content, (dict, list)):
            content_str = json.dumps(self.content)

        return {
            "role": "tool",
            "tool_call_id": self.tool_use_id,
            "content": content_str
        }


class InputTextOrImageMessage(BaseModel):
    type: Literal["text", "image"]
    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]] | None = None
    source: Source | None = None

    def get_openai_compatible(self) -> dict[str, Any]:
        """Convert Anthropic text/image message to OpenAI format"""
        if self.type == "text":
            content_value: str | None = self.content
            if isinstance(self.content, list):
                text_parts: list[str] = []
                for part in self.content:
                    if isinstance(part, dict):
                        text_value = part.get("text")
                        if isinstance(text_value, str):
                            text_parts.append(text_value)
                        else:
                            text_parts.append(json.dumps(part))
                    else:
                        text_parts.append(str(part))
                content_value = "\n\n".join(text_parts)

            return {
                "role": self.role,
                "content": content_value
            }
        elif self.type == "image" and self.source:
            # OpenAI format for images
            if self.source.type == "url":
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": self.source.url}
                }
            else:  # base64
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self.source.media_type};base64,{self.source.data}"
                    }
                }

            return {
                "role": self.role,
                "content": [image_content]
            }


class ToolChoice(BaseModel):
    type: Literal["auto"]


class MessagesBody(BaseModel):
    model: str
    messages: list[Annotated[InputTextOrImageMessage | InputToolUseMessage | InputToolCallMessage | InputToolResultMessage, Field(discriminator='type')]]
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
            if isinstance(msg, InputToolCallMessage):
                # Tool call messages are already in OpenAI format, convert to tool_use for processing
                result.append(msg.get_anthropic_compatible().get_openai_compatible())
            else:
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
    content: list[OutputTextContentItem | OutputToolContentItem]
    stop_reason: Literal["end_turn", "max_tokens"]
    stop_sequence: str | None
    usage: Usage
