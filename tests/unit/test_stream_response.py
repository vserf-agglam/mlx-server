"""
Integration tests for stream_response function.

Tests the complete SSE streaming flow with mocked server,
including text streaming, tool calls, and error handling.
"""

import pytest
import json
from unittest.mock import Mock, patch
from api.types import MessagesBody, MessagesResponse, Usage, OutputTextContentItem, OutputToolContentItem


# We need to mock the server globally before importing app
@pytest.fixture(autouse=True)
def mock_global_server(mock_server):
    """Mock the global server in app.py."""
    with patch('app.server', mock_server):
        yield mock_server


class TestStreamResponseBasic:
    """Basic stream_response tests."""

    @pytest.mark.asyncio
    async def test_basic_text_stream(self, sample_messages_body, sample_final_response):
        """Test basic text streaming without tools."""
        # Mock server to yield simple text chunks
        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=100)
        mock_server.token_parser = None

        def mock_stream():
            yield {"delta": "Hello "}
            yield {"delta": "world"}
            return sample_final_response

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            # Collect all SSE events
            events = []
            async for event in stream_response(sample_messages_body):
                events.append(event)

            # Verify event sequence
            assert len(events) > 0

            # Parse SSE events
            event_types = []
            for event in events:
                if event.startswith("event: "):
                    event_type = event.split("\n")[0].replace("event: ", "")
                    event_types.append(event_type)

            # Should have: message_start, content_block_start, deltas, content_block_stop, message_delta, message_stop
            assert "message_start" in event_types
            assert "content_block_start" in event_types
            assert "content_block_delta" in event_types
            assert "content_block_stop" in event_types
            assert "message_delta" in event_types
            assert "message_stop" in event_types

    @pytest.mark.asyncio
    async def test_event_order(self, sample_messages_body, sample_final_response):
        """Test events are in correct order."""
        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=50)
        mock_server.token_parser = None

        def mock_stream():
            yield {"delta": "Test"}
            return sample_final_response

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            event_types = []
            async for event in stream_response(sample_messages_body):
                if event.startswith("event: "):
                    event_type = event.split("\n")[0].replace("event: ", "")
                    event_types.append(event_type)

            # Verify order
            expected_order = [
                "message_start",
                "content_block_start",
                "content_block_delta",  # At least one
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]

            # Check that expected events appear in order
            indices = []
            for expected in expected_order:
                try:
                    idx = event_types.index(expected)
                    indices.append(idx)
                except ValueError:
                    pytest.fail(f"Missing expected event: {expected}")

            # Indices should be increasing (maintaining order)
            assert indices == sorted(indices), f"Events out of order: {event_types}"


class TestStreamResponseWithTools:
    """Tests for stream_response with tool calls."""

    @pytest.mark.asyncio
    async def test_tool_call_streaming(self, sample_messages_body_with_tools, sample_final_response_with_tool):
        """Test streaming with tool calls."""
        from token_parser.qwen3_mode_parser import Qwen3MoeParser

        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=100)
        mock_server.token_parser = Qwen3MoeParser()

        def mock_stream():
            yield {"delta": "Let me check. "}
            yield {"delta": "<tool_call>"}
            yield {"delta": '{"name": "get_weather", "arguments": {"location": "NYC"}}'}
            yield {"delta": "</tool_call>"}
            return sample_final_response_with_tool

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            events = []
            async for event in stream_response(sample_messages_body_with_tools):
                events.append(event)

            # Should have tool_call event
            event_types = []
            for event in events:
                if event.startswith("event: "):
                    event_type = event.split("\n")[0].replace("event: ", "")
                    event_types.append(event_type)

            assert "tool_call" in event_types

            # Find and parse tool_call event
            tool_events = [e for e in events if "event: tool_call" in e]
            assert len(tool_events) >= 1

            # Parse the tool call data
            for tool_event in tool_events:
                lines = tool_event.split("\n")
                data_line = [l for l in lines if l.startswith("data: ")][0]
                data_json = data_line.replace("data: ", "")
                data = json.loads(data_json)

                assert data["type"] == "tool_call"
                assert "id" in data
                assert "name" in data

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_stream(self, sample_messages_body_with_tools, sample_final_response_with_tool):
        """Test streaming with both text and tool calls."""
        from token_parser.qwen3_mode_parser import Qwen3MoeParser

        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=100)
        mock_server.token_parser = Qwen3MoeParser()

        def mock_stream():
            yield {"delta": "I'll help you. "}
            yield {"delta": "<tool_call>"}
            yield {"delta": '{"name": "test", "arguments": {}}'}
            yield {"delta": "</tool_call>"}
            yield {"delta": " Done!"}
            return sample_final_response_with_tool

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            events = []
            async for event in stream_response(sample_messages_body_with_tools):
                events.append(event)

            # Should have both text deltas and tool calls
            event_types = [
                event.split("\n")[0].replace("event: ", "")
                for event in events
                if event.startswith("event: ")
            ]

            assert "content_block_delta" in event_types
            assert "tool_call" in event_types


class TestStreamResponseErrors:
    """Tests for error handling in stream_response."""

    @pytest.mark.asyncio
    async def test_timeout_error(self, sample_messages_body):
        """Test timeout error handling."""
        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=100)
        mock_server.generate_stream = Mock(side_effect=TimeoutError("Timeout"))

        with patch('app.server', mock_server):
            from app import stream_response

            events = []
            async for event in stream_response(sample_messages_body):
                events.append(event)

            # Should have error event
            error_events = [e for e in events if "event: error" in e]
            assert len(error_events) >= 1

            # Parse error event
            error_event = error_events[0]
            lines = error_event.split("\n")
            data_line = [l for l in lines if l.startswith("data: ")][0]
            data = json.loads(data_line.replace("data: ", ""))

            assert data["type"] == "error"
            assert data["error"]["type"] == "timeout_error"

    @pytest.mark.asyncio
    async def test_generic_exception(self, sample_messages_body):
        """Test generic exception handling."""
        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=100)
        mock_server.generate_stream = Mock(side_effect=Exception("Test error"))

        with patch('app.server', mock_server):
            from app import stream_response

            events = []
            async for event in stream_response(sample_messages_body):
                events.append(event)

            # Should have error event
            error_events = [e for e in events if "event: error" in e]
            assert len(error_events) >= 1

            # Parse error event
            error_event = error_events[0]
            lines = error_event.split("\n")
            data_line = [l for l in lines if l.startswith("data: ")][0]
            data = json.loads(data_line.replace("data: ", ""))

            assert data["type"] == "error"
            assert data["error"]["type"] == "api_error"
            assert "Test error" in data["error"]["message"]


class TestStreamResponseSSEFormat:
    """Tests for SSE format compliance."""

    @pytest.mark.asyncio
    async def test_sse_format_valid(self, sample_messages_body, sample_final_response):
        """Test all events follow SSE format."""
        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=50)

        def mock_stream():
            yield {"delta": "Test"}
            return sample_final_response

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            async for event in stream_response(sample_messages_body):
                # Each event should follow SSE format
                if event.strip():  # Skip empty lines
                    assert event.startswith("event: ")
                    assert "\ndata: " in event
                    assert event.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_json_data_valid(self, sample_messages_body, sample_final_response):
        """Test all data fields contain valid JSON."""
        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=50)

        def mock_stream():
            yield {"delta": "Test"}
            return sample_final_response

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            async for event in stream_response(sample_messages_body):
                if "data: " in event:
                    lines = event.split("\n")
                    data_line = [l for l in lines if l.startswith("data: ")][0]
                    json_str = data_line.replace("data: ", "")

                    # Should be valid JSON
                    try:
                        data = json.loads(json_str)
                        assert isinstance(data, dict)
                        assert "type" in data
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON in event: {json_str}")


class TestStreamResponseRealisticScenarios:
    """Realistic end-to-end streaming scenarios."""

    @pytest.mark.asyncio
    async def test_weather_query_complete_flow(self, sample_messages_body_with_tools):
        """Realistic scenario: Complete weather query flow."""
        from token_parser.qwen3_mode_parser import Qwen3MoeParser

        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=120)
        mock_server.token_parser = Qwen3MoeParser()

        final_response = MessagesResponse(
            id="msg_weather",
            type="message",
            role="assistant",
            content=[
                OutputTextContentItem(type="text", text="Let me check the weather."),
                OutputToolContentItem(
                    id="toolu_weather_1",
                    name="get_weather",
                    input={"location": "NYC", "unit": "fahrenheit"}
                ),
            ],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=120, output_tokens=80)
        )

        def mock_stream():
            # Simulate realistic LLM streaming
            yield {"delta": "Let me "}
            yield {"delta": "check the "}
            yield {"delta": "weather for "}
            yield {"delta": "you. "}
            yield {"delta": "<tool_call>"}
            yield {"delta": '{"name": "get_weather", '}
            yield {"delta": '"arguments": {"location": "NYC", "unit": "fahrenheit"}}'}
            yield {"delta": "</tool_call>"}
            return final_response

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            all_text = []
            tool_calls = []

            async for event in stream_response(sample_messages_body_with_tools):
                if "event: content_block_delta" in event:
                    # Extract text
                    lines = event.split("\n")
                    data_line = [l for l in lines if l.startswith("data: ")][0]
                    data = json.loads(data_line.replace("data: ", ""))
                    if "delta" in data and "text" in data["delta"]:
                        all_text.append(data["delta"]["text"])

                elif "event: tool_call" in event:
                    # Extract tool call
                    lines = event.split("\n")
                    data_line = [l for l in lines if l.startswith("data: ")][0]
                    data = json.loads(data_line.replace("data: ", ""))
                    tool_calls.append(data)

            # Verify we got text
            combined_text = "".join(all_text)
            assert "Let me check" in combined_text or "weather" in combined_text

            # Verify we got tool call
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "get_weather"
            assert tool_calls[0]["arguments"]["location"] == "NYC"

    @pytest.mark.asyncio
    async def test_plain_text_conversation(self, sample_messages_body):
        """Realistic scenario: Plain text conversation."""
        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=80)
        mock_server.token_parser = None

        final_response = MessagesResponse(
            id="msg_chat",
            type="message",
            role="assistant",
            content=[
                OutputTextContentItem(
                    type="text",
                    text="The capital of France is Paris. It's a beautiful city."
                ),
            ],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=80, output_tokens=60)
        )

        def mock_stream():
            yield {"delta": "The capital "}
            yield {"delta": "of France "}
            yield {"delta": "is Paris. "}
            yield {"delta": "It's a "}
            yield {"delta": "beautiful "}
            yield {"delta": "city."}
            return final_response

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            all_text = []
            async for event in stream_response(sample_messages_body):
                if "event: content_block_delta" in event:
                    lines = event.split("\n")
                    data_line = [l for l in lines if l.startswith("data: ")][0]
                    data = json.loads(data_line.replace("data: ", ""))
                    if "delta" in data and "text" in data["delta"]:
                        all_text.append(data["delta"]["text"])

            combined_text = "".join(all_text)
            assert "The capital of France is Paris" in combined_text

    @pytest.mark.asyncio
    async def test_multiple_tools_scenario(self, sample_messages_body_with_tools):
        """Realistic scenario: Multiple tool calls."""
        from token_parser.qwen3_mode_parser import Qwen3MoeParser

        mock_server = Mock()
        mock_server.count_input_tokens = Mock(return_value=100)
        mock_server.token_parser = Qwen3MoeParser()

        final_response = MessagesResponse(
            id="msg_multi",
            type="message",
            role="assistant",
            content=[
                OutputTextContentItem(type="text", text="I'll search for both."),
                OutputToolContentItem(id="toolu_1", name="search", input={"query": "Python"}),
                OutputToolContentItem(id="toolu_2", name="search", input={"query": "JavaScript"}),
            ],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=100, output_tokens=100)
        )

        def mock_stream():
            yield {"delta": "I'll search for both. "}
            yield {"delta": "<tool_call>"}
            yield {"delta": '{"name": "search", "arguments": {"query": "Python"}}'}
            yield {"delta": "</tool_call> "}
            yield {"delta": "<tool_call>"}
            yield {"delta": '{"name": "search", "arguments": {"query": "JavaScript"}}'}
            yield {"delta": "</tool_call>"}
            return final_response

        mock_server.generate_stream = Mock(return_value=mock_stream())

        with patch('app.server', mock_server):
            from app import stream_response

            tool_calls = []
            async for event in stream_response(sample_messages_body_with_tools):
                if "event: tool_call" in event:
                    lines = event.split("\n")
                    data_line = [l for l in lines if l.startswith("data: ")][0]
                    data = json.loads(data_line.replace("data: ", ""))
                    tool_calls.append(data)

            # Should have 2 tool calls
            assert len(tool_calls) == 2
            assert tool_calls[0]["arguments"]["query"] == "Python"
            assert tool_calls[1]["arguments"]["query"] == "JavaScript"
