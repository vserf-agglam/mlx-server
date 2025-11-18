"""
Unit tests for SSE (Server-Sent Events) event builders.

Tests all 9 event builder functions to ensure they return correct
structures and properly format SSE events.
"""

import pytest
import json
from utils.sse_event_builders import (
    build_sse_event,
    build_text_delta_event,
    build_input_json_delta_event,
    build_tool_use_block_start_event,
    build_ping_event,
    build_message_start_event,
    build_content_block_start_event,
    build_content_block_stop_event,
    build_message_delta_event,
    build_message_stop_event,
    build_error_event,
)


class TestBuildSSEEvent:
    """Tests for build_sse_event function."""

    def test_basic_event(self):
        """Test basic SSE event formatting."""
        result = build_sse_event("ping", {"type": "ping"})
        assert result == 'event: ping\ndata: {"type": "ping"}\n\n'

    def test_complex_event(self):
        """Test SSE event with complex data."""
        data = {
            "type": "message_start",
            "message": {"id": "123", "role": "assistant"}
        }
        result = build_sse_event("message_start", data)
        assert result.startswith("event: message_start\ndata: ")
        assert result.endswith("\n\n")

        # Verify JSON is valid
        json_part = result.split("data: ")[1].rstrip("\n")
        parsed = json.loads(json_part)
        assert parsed == data

    def test_empty_data(self):
        """Test with empty dictionary."""
        result = build_sse_event("test", {})
        assert result == 'event: test\ndata: {}\n\n'

    def test_unicode_content(self):
        """Test with unicode characters."""
        data = {"text": "Hello 世界 🌍"}
        result = build_sse_event("text", data)
        # JSON can escape unicode (e.g., \u4e16\u754c) which is valid
        # Verify the data can be parsed back correctly
        import json
        lines = result.split("\n")
        data_line = [l for l in lines if l.startswith("data: ")][0]
        parsed_data = json.loads(data_line.replace("data: ", ""))
        assert parsed_data["text"] == "Hello 世界 🌍"

    def test_special_characters(self):
        """Test with special JSON characters."""
        data = {"text": 'Quote: " Newline: \n Tab: \t'}
        result = build_sse_event("test", data)
        # Should be properly escaped
        assert '\\"' in result  # Escaped quote


class TestBuildTextDeltaEvent:
    """Tests for build_text_delta_event function."""

    def test_basic_text_delta(self):
        """Test basic text delta event."""
        result = build_text_delta_event("Hello")
        assert result == {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        }

    def test_custom_index(self):
        """Test text delta with custom index."""
        result = build_text_delta_event("World", index=2)
        assert result["index"] == 2
        assert result["delta"]["text"] == "World"

    def test_empty_text(self):
        """Test with empty string."""
        result = build_text_delta_event("")
        assert result["delta"]["text"] == ""

    def test_multiline_text(self):
        """Test with multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        result = build_text_delta_event(text)
        assert result["delta"]["text"] == text

    def test_unicode_text(self):
        """Test with unicode text."""
        text = "こんにちは 🎌"
        result = build_text_delta_event(text)
        assert result["delta"]["text"] == text


class TestBuildInputJsonDeltaEvent:
    """Tests for build_input_json_delta_event function."""

    def test_basic_json_delta(self):
        """Test basic input JSON delta event."""
        result = build_input_json_delta_event('{"location": "San', index=1)
        assert result == {
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"location": "San'
            }
        }

    def test_default_index(self):
        """Test with default index."""
        result = build_input_json_delta_event('{"key": ')
        assert result["index"] == 0

    def test_complete_json_field(self):
        """Test with complete JSON field value."""
        result = build_input_json_delta_event('"San Francisco, CA"', index=2)
        assert result["delta"]["partial_json"] == '"San Francisco, CA"'

    def test_empty_json(self):
        """Test with empty JSON string."""
        result = build_input_json_delta_event('', index=1)
        assert result["delta"]["partial_json"] == ''

    def test_unicode_in_json(self):
        """Test with unicode characters in JSON."""
        result = build_input_json_delta_event('{"city": "東京"}', index=1)
        assert result["delta"]["partial_json"] == '{"city": "東京"}'


class TestBuildToolUseBlockStartEvent:
    """Tests for build_tool_use_block_start_event function."""

    def test_basic_tool_use_start(self):
        """Test basic tool_use block start event."""
        result = build_tool_use_block_start_event(1, "toolu_123", "get_weather")
        assert result == {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {}
            }
        }

    def test_different_index(self):
        """Test with different index values."""
        result = build_tool_use_block_start_event(5, "toolu_abc", "search")
        assert result["index"] == 5
        assert result["content_block"]["id"] == "toolu_abc"
        assert result["content_block"]["name"] == "search"

    def test_empty_input(self):
        """Test that input starts empty."""
        result = build_tool_use_block_start_event(0, "toolu_1", "test")
        assert result["content_block"]["input"] == {}

    def test_unicode_tool_name(self):
        """Test with unicode in tool name."""
        result = build_tool_use_block_start_event(1, "toolu_x", "検索")
        assert result["content_block"]["name"] == "検索"


class TestBuildPingEvent:
    """Tests for build_ping_event function."""

    def test_ping_event(self):
        """Test ping event structure."""
        result = build_ping_event()
        assert result == {"type": "ping"}

    def test_ping_event_in_sse(self):
        """Test ping event in SSE format."""
        sse = build_sse_event("ping", build_ping_event())
        assert sse == 'event: ping\ndata: {"type": "ping"}\n\n'


class TestBuildMessageStartEvent:
    """Tests for build_message_start_event function."""

    def test_basic_message_start(self):
        """Test basic message start event."""
        result = build_message_start_event("msg_123", "qwen2.5", 100)

        assert result["type"] == "message_start"
        assert result["message"]["id"] == "msg_123"
        assert result["message"]["model"] == "qwen2.5"
        assert result["message"]["role"] == "assistant"
        assert result["message"]["type"] == "message"
        assert result["message"]["content"] == []
        assert result["message"]["stop_reason"] is None
        assert result["message"]["stop_sequence"] is None
        assert result["message"]["usage"]["input_tokens"] == 100
        assert result["message"]["usage"]["output_tokens"] == 0

    def test_zero_input_tokens(self):
        """Test with zero input tokens."""
        result = build_message_start_event("msg_0", "model", 0)
        assert result["message"]["usage"]["input_tokens"] == 0

    def test_large_input_tokens(self):
        """Test with large token count."""
        result = build_message_start_event("msg_big", "model", 100000)
        assert result["message"]["usage"]["input_tokens"] == 100000


class TestBuildContentBlockStartEvent:
    """Tests for build_content_block_start_event function."""

    def test_default_content_block_start(self):
        """Test default content block start."""
        result = build_content_block_start_event()
        assert result == {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }

    def test_custom_index(self):
        """Test with custom index."""
        result = build_content_block_start_event(index=3)
        assert result["index"] == 3



class TestBuildContentBlockStopEvent:
    """Tests for build_content_block_stop_event function."""

    def test_default_content_block_stop(self):
        """Test default content block stop."""
        result = build_content_block_stop_event()
        assert result == {
            "type": "content_block_stop",
            "index": 0
        }

    def test_custom_index(self):
        """Test with custom index."""
        result = build_content_block_stop_event(index=5)
        assert result["index"] == 5


class TestBuildMessageDeltaEvent:
    """Tests for build_message_delta_event function."""

    def test_basic_message_delta(self):
        """Test basic message delta event."""
        result = build_message_delta_event(
            "end_turn",
            None,
            100,
            50
        )

        assert result["type"] == "message_delta"
        assert result["delta"]["stop_reason"] == "end_turn"
        assert result["delta"]["stop_sequence"] is None
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50

    def test_with_stop_sequence(self):
        """Test message delta with stop sequence."""
        result = build_message_delta_event(
            "stop_sequence",
            "</s>",
            200,
            150
        )
        assert result["delta"]["stop_sequence"] == "</s>"

    def test_max_tokens_stop(self):
        """Test message delta with max_tokens stop reason."""
        result = build_message_delta_event(
            "max_tokens",
            None,
            1000,
            2000
        )
        assert result["delta"]["stop_reason"] == "max_tokens"

    def test_zero_tokens(self):
        """Test with zero output tokens."""
        result = build_message_delta_event("end_turn", None, 50, 0)
        assert result["usage"]["output_tokens"] == 0


class TestBuildMessageStopEvent:
    """Tests for build_message_stop_event function."""

    def test_message_stop(self):
        """Test message stop event."""
        result = build_message_stop_event()
        assert result == {"type": "message_stop"}


class TestBuildErrorEvent:
    """Tests for build_error_event function."""

    def test_timeout_error(self):
        """Test timeout error event."""
        result = build_error_event("timeout_error", "Request timeout")
        assert result == {
            "type": "error",
            "error": {
                "type": "timeout_error",
                "message": "Request timeout"
            }
        }

    def test_api_error(self):
        """Test API error event."""
        result = build_error_event("api_error", "Internal server error")
        assert result["error"]["type"] == "api_error"
        assert result["error"]["message"] == "Internal server error"

    def test_validation_error(self):
        """Test validation error."""
        result = build_error_event(
            "invalid_request_error",
            "Missing required field: model"
        )
        assert result["error"]["type"] == "invalid_request_error"

    def test_unicode_error_message(self):
        """Test error with unicode message."""
        result = build_error_event("error", "Failed: 失敗")
        assert result["error"]["message"] == "Failed: 失敗"


class TestIntegration:
    """Integration tests combining multiple builders."""

    def test_full_sse_stream_sequence(self):
        """Test a complete SSE event sequence with Anthropic-compatible format."""
        # message_start
        msg_start = build_sse_event(
            "message_start",
            build_message_start_event("msg_1", "model", 10)
        )
        assert "event: message_start" in msg_start

        # content_block_start for text
        text_block_start = build_sse_event(
            "content_block_start",
            build_content_block_start_event(index=0)
        )
        assert "event: content_block_start" in text_block_start
        assert '"type": "text"' in text_block_start

        # content_block_delta for text
        text_delta = build_sse_event(
            "content_block_delta",
            build_text_delta_event("Hello", index=0)
        )
        assert "event: content_block_delta" in text_delta
        assert '"text_delta"' in text_delta

        # content_block_stop for text
        text_block_stop = build_sse_event(
            "content_block_stop",
            build_content_block_stop_event(index=0)
        )
        assert "event: content_block_stop" in text_block_stop

        # content_block_start for tool_use
        tool_block_start = build_sse_event(
            "content_block_start",
            build_tool_use_block_start_event(1, "toolu_123", "get_weather")
        )
        assert "event: content_block_start" in tool_block_start
        assert '"type": "tool_use"' in tool_block_start

        # content_block_delta for tool input JSON
        json_delta = build_sse_event(
            "content_block_delta",
            build_input_json_delta_event('{"location": "San Francisco"}', index=1)
        )
        assert "event: content_block_delta" in json_delta
        assert '"input_json_delta"' in json_delta

        # content_block_stop for tool
        tool_block_stop = build_sse_event(
            "content_block_stop",
            build_content_block_stop_event(index=1)
        )
        assert "event: content_block_stop" in tool_block_stop

        # ping event
        ping = build_sse_event("ping", build_ping_event())
        assert "event: ping" in ping

        # message_delta
        msg_delta = build_sse_event(
            "message_delta",
            build_message_delta_event("tool_use", None, 10, 25)
        )
        assert "event: message_delta" in msg_delta

        # message_stop
        msg_stop = build_sse_event(
            "message_stop",
            build_message_stop_event()
        )
        assert "event: message_stop" in msg_stop

    def test_error_event_stream(self):
        """Test error event in SSE format."""
        error = build_sse_event(
            "error",
            build_error_event("timeout_error", "Timeout")
        )
        assert "event: error" in error
        assert "timeout_error" in error
