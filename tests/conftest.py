"""
Shared test fixtures and configuration for pytest.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from api.types import MessagesBody, MessagesResponse, Usage, OutputTextContentItem, OutputToolContentItem
from token_parser.qwen3_mode_parser import Qwen3MoeParser


@pytest.fixture
def mock_qwen_parser():
    """Mock Qwen3MoeParser for testing tool call parsing."""
    parser = Qwen3MoeParser()
    return parser


@pytest.fixture
def mock_simple_parser():
    """Mock parser without tool support."""
    parser = Mock()
    parser.tool_call_pattern = None
    parser.tool_call_open_tag = None
    parser.parse_tool_calls = Mock(return_value=(
        [OutputTextContentItem(type="text", text="Hello")],
        "end_turn"
    ))
    return parser


@pytest.fixture
def sample_messages_body():
    """Sample MessagesBody for testing."""
    return MessagesBody(
        model="qwen2.5-7b",
        messages=[
            {"role": "user", "content": "What's the weather in NYC?"}
        ],
        max_tokens=1000,
        stream=True
    )


@pytest.fixture
def sample_messages_body_with_tools():
    """Sample MessagesBody with tool definitions."""
    return MessagesBody(
        model="qwen2.5-7b",
        messages=[
            {"role": "user", "content": "What's the weather in NYC?"}
        ],
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ],
        max_tokens=1000,
        stream=True
    )


@pytest.fixture
def sample_final_response():
    """Sample MessagesResponse for testing."""
    return MessagesResponse(
        id="msg_123",
        type="message",
        role="assistant",
        content=[
            OutputTextContentItem(type="text", text="The weather in NYC is sunny.")
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=100, output_tokens=50)
    )


@pytest.fixture
def sample_final_response_with_tool():
    """Sample MessagesResponse with tool call."""
    return MessagesResponse(
        id="msg_123",
        type="message",
        role="assistant",
        content=[
            OutputTextContentItem(type="text", text="Let me check the weather for you."),
            OutputToolContentItem(
                id="toolu_abc123",
                name="get_weather",
                input={"location": "NYC"}
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=100, output_tokens=75)
    )


@pytest.fixture
def mock_server(sample_final_response):
    """Mock Server instance for testing."""
    server = Mock()
    server.tokenizer = Mock()
    server.tokenizer.decode = Mock(return_value="decoded")
    server.token_parser = Qwen3MoeParser()
    server.count_input_tokens = Mock(return_value=100)

    # Mock generate_stream as a generator
    def mock_stream():
        yield {"delta": "Hello "}
        yield {"delta": "world"}
        return sample_final_response

    server.generate_stream = Mock(return_value=mock_stream())
    return server


@pytest.fixture
def mock_server_with_tool_stream(sample_final_response_with_tool):
    """Mock Server that yields tool call chunks."""
    server = Mock()
    server.tokenizer = Mock()
    server.token_parser = Qwen3MoeParser()
    server.count_input_tokens = Mock(return_value=100)

    # Mock stream with tool call
    def mock_stream():
        yield {"delta": "Let me check "}
        yield {"delta": "<tool_call>"}
        yield {"delta": '{"name": "get_weather", '}
        yield {"delta": '"arguments": {"location": "NYC"}}'}
        yield {"delta": "</tool_call>"}
        return sample_final_response_with_tool

    server.generate_stream = Mock(return_value=mock_stream())
    return server


# Test data for realistic scenarios

@pytest.fixture
def weather_query_chunks():
    """Realistic LLM response: text → tool call → text."""
    return [
        {"delta": "Let me check "},
        {"delta": "the weather "},
        {"delta": "for you. "},
        {"delta": "<tool_call>"},
        {"delta": '{"name": "get_weather", '},
        {"delta": '"arguments": {"location": "NYC", "unit": "fahrenheit"}}'},
        {"delta": "</tool_call>"},
        {"delta": " I'll get that "},
        {"delta": "information now."},
    ]


@pytest.fixture
def multi_tool_chunks():
    """Realistic LLM response: multiple tool calls."""
    return [
        {"delta": "I'll search for both. "},
        {"delta": "<tool_call>"},
        {"delta": '{"name": "search", "arguments": {"query": "Python"}}'},
        {"delta": "</tool_call>"},
        {"delta": " "},
        {"delta": "<tool_call>"},
        {"delta": '{"name": "search", "arguments": {"query": "JavaScript"}}'},
        {"delta": "</tool_call>"},
    ]


@pytest.fixture
def interrupted_tool_chunks():
    """Realistic LLM response: tool call split into tiny chunks."""
    return [
        {"delta": "Here's the data: "},
        {"delta": "<"},
        {"delta": "tool"},
        {"delta": "_call>"},
        {"delta": "{"},
        {"delta": '"name"'},
        {"delta": ': "'},
        {"delta": "calc"},
        {"delta": '", '},
        {"delta": '"arguments"'},
        {"delta": ': {'},
        {"delta": '"expr"'},
        {"delta": ': "'},
        {"delta": '2+2'},
        {"delta": '"}}'},
        {"delta": "</tool_call>"},
    ]


@pytest.fixture
def plain_text_chunks():
    """Realistic LLM response: just plain text."""
    return [
        {"delta": "The capital "},
        {"delta": "of France "},
        {"delta": "is Paris. "},
        {"delta": "It's a beautiful "},
        {"delta": "city with "},
        {"delta": "rich history."},
    ]
