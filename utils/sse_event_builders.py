"""
SSE (Server-Sent Events) event builders for streaming responses.

This module provides helper functions to build SSE-formatted events
for the Anthropic Messages API streaming response format.
"""

import json
from typing import Any


def build_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """
    Build a Server-Sent Event string.

    Args:
        event_type: The SSE event type (e.g., "message_start", "content_block_delta")
        data: The event data dictionary to be JSON-encoded

    Returns:
        Formatted SSE event string with event type and data

    Example:
        >>> build_sse_event("ping", {"type": "ping"})
        'event: ping\\ndata: {"type": "ping"}\\n\\n'
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def build_text_delta_event(text: str, index: int = 0) -> dict[str, Any]:
    """
    Build a content_block_delta event with text_delta.

    Args:
        text: The text content to include in the delta
        index: The content block index (default: 0)

    Returns:
        Dictionary representing a content_block_delta event with text_delta

    Example:
        >>> build_text_delta_event("Hello")
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        }
    """
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "text_delta",
            "text": text
        }
    }


def build_input_json_delta_event(partial_json: str, index: int = 0) -> dict[str, Any]:
    """
    Build a content_block_delta event with input_json_delta for tool input streaming.

    Args:
        partial_json: The partial JSON string chunk to include in the delta
        index: The content block index (default: 0)

    Returns:
        Dictionary representing a content_block_delta event with input_json_delta

    Example:
        >>> build_input_json_delta_event('{"location": "San', 1)
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"location": "San'}
        }
    """
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "input_json_delta",
            "partial_json": partial_json
        }
    }


def build_tool_use_block_start_event(index: int, tool_id: str, tool_name: str) -> dict[str, Any]:
    """
    Build a content_block_start event for tool_use content blocks.

    Args:
        index: The content block index
        tool_id: The tool call ID (e.g., "toolu_01A09q90qw90lq917835lq9")
        tool_name: The name of the tool being called

    Returns:
        Dictionary representing a content_block_start event with tool_use type

    Example:
        >>> build_tool_use_block_start_event(1, "toolu_123", "get_weather")
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {}
            }
        }
    """
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": {}
        }
    }


def build_message_start_event(message_id: str, model: str, input_tokens: int) -> dict[str, Any]:
    """
    Build a message_start event.

    Args:
        message_id: The message ID
        model: The model name
        input_tokens: The number of input tokens

    Returns:
        Dictionary representing a message_start event

    Example:
        >>> build_message_start_event("msg_123", "claude-3", 100)
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 0}
            }
        }
    """
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0
            }
        }
    }


def build_content_block_start_event(index: int = 0) -> dict[str, Any]:
    """
    Build a content_block_start event for text content blocks.

    For tool_use content blocks, use build_tool_use_block_start_event() instead.

    Args:
        index: The content block index (default: 0)

    Returns:
        Dictionary representing a content_block_start event for text

    Example:
        >>> build_content_block_start_event()
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }
    """
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "text",
            "text": ""
        }
    }


def build_content_block_stop_event(index: int = 0) -> dict[str, Any]:
    """
    Build a content_block_stop event.

    Args:
        index: The content block index (default: 0)

    Returns:
        Dictionary representing a content_block_stop event

    Example:
        >>> build_content_block_stop_event()
        {"type": "content_block_stop", "index": 0}
    """
    return {
        "type": "content_block_stop",
        "index": index
    }


def build_message_delta_event(
    stop_reason: str | None,
    stop_sequence: str | None,
    input_tokens: int,
    output_tokens: int
) -> dict[str, Any]:
    """
    Build a message_delta event with usage information.

    Args:
        stop_reason: The reason generation stopped
        stop_sequence: The stop sequence that triggered stopping (if any)
        input_tokens: The number of input tokens
        output_tokens: The number of output tokens

    Returns:
        Dictionary representing a message_delta event

    Example:
        >>> build_message_delta_event("end_turn", None, 100, 50)
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
    """
    return {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    }


def build_message_stop_event() -> dict[str, Any]:
    """
    Build a message_stop event.

    Returns:
        Dictionary representing a message_stop event

    Example:
        >>> build_message_stop_event()
        {"type": "message_stop"}
    """
    return {"type": "message_stop"}


def build_ping_event() -> dict[str, Any]:
    """
    Build a ping event for keeping SSE connections alive during long-running streams.

    Returns:
        Dictionary representing a ping event

    Example:
        >>> build_ping_event()
        {"type": "ping"}
    """
    return {"type": "ping"}


def build_error_event(error_type: str, message: str) -> dict[str, Any]:
    """
    Build an error event.

    Args:
        error_type: The type of error (e.g., "timeout_error", "api_error")
        message: The error message

    Returns:
        Dictionary representing an error event

    Example:
        >>> build_error_event("timeout_error", "Request timeout")
        {
            "type": "error",
            "error": {"type": "timeout_error", "message": "Request timeout"}
        }
    """
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }
