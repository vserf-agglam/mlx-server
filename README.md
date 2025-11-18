# MLX Chat Completions API Server

An Anthropic Messages API-compatible chat completions server powered by MLX with continuous batching support for maximum throughput.

## Features

- **Anthropic Messages API Compatible**: Drop-in replacement for Anthropic's Messages API
- **Tool Calling Support**: Full support for function/tool calling with Qwen3 MoE parser
- **Continuous Batching**: Dynamically adds new requests to running batches for optimal GPU utilization
- **Streaming Support**: Real-time token streaming with Server-Sent Events (SSE)
- **Batch Processing**: Process multiple requests efficiently with dedicated batch endpoint
- **Prompt Caching**: Automatic caching of prompts for improved performance
- **Concurrent Request Handling**: Efficiently handles multiple simultaneous requests
- **Production Ready**: Comprehensive error handling, logging, and graceful shutdown

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Starting the Server

```bash
# Basic usage
python app.py Qwen/Qwen3-Coder-25B-mlx

# With custom configuration
python app.py Qwen/Qwen3-Coder-25B-mlx \
  --host 0.0.0.0 \
  --port 8000 \
  --prefill-batch-size 8 \
  --completion-batch-size 32 \
  --max-kv-size 8096 \
  --verbose

# With trust remote code (required for some models)
python app.py Qwen/Qwen3-Coder-25B-mlx --trust_remote_code
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `model_name_or_path` | Hugging Face model name or local path (required) | - |
| `--host` | Host to bind to | `0.0.0.0` |
| `--port` | Port to bind to | `8000` |
| `--token-parser` | Token parser for tool calling | `qwen3_moe` |
| `--prefill-batch-size` | Number of messages to prefill in batch | `8` |
| `--completion-batch-size` | Number of messages to complete in batch | `32` |
| `--max-kv-size` | Maximum size of key-value cache | `8096` |
| `--trust_remote_code` | Trust remote code when loading model | `False` |
| `--verbose` | Enable verbose logging | `False` |
| `--reload` | Enable auto-reload for development | `False` |

## API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Root Endpoint

Get server information and status.

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "MLX Anthropic API",
  "version": "2.0.0",
  "model": "Qwen/Qwen3-Coder-25B-mlx",
  "status": "ready"
}
```

#### 2. List Models

Get available models (Anthropic-compatible).

**Endpoint:** `GET /v1/models`

**Request:**
```bash
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "data": [
    {
      "id": "Qwen/Qwen3-Coder-25B-mlx",
      "object": "model",
      "created": 0,
      "owned_by": "local"
    }
  ]
}
```

#### 3. Create Message (Non-Streaming)

Generate a chat completion response.

**Endpoint:** `POST /v1/messages`

**Request:**
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Coder-25B-mlx",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'
```

**Request Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `messages` | array | Yes | Array of message objects |
| `max_tokens` | integer | No | Maximum tokens to generate (default: 16000) |
| `temperature` | float | No | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | Nucleus sampling probability |
| `top_k` | integer | No | Top-k sampling parameter |
| `stop_sequence` | array | No | Sequences where generation should stop |
| `stream` | boolean | No | Enable streaming (default: false) |
| `tools` | array | No | Array of tool definitions |
| `tool_choice` | object | No | Tool choice configuration |

**Response:**
```json
{
  "id": "msg_a1b2c3d4",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "The capital of France is Paris."
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 15,
    "output_tokens": 8
  }
}
```

#### 4. Create Message (Streaming)

Stream tokens as they are generated.

**Endpoint:** `POST /v1/messages`

**Request:**
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Coder-25B-mlx",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Tell me a short story"
      }
    ]
  }'
```

**Streaming Response (Server-Sent Events):**
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_stream_model","type":"message","role":"assistant","content":[],"model":"model","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Once"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" upon"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":12,"output_tokens":156}}

event: message_stop
data: {"type":"message_stop"}
```

#### 5. Batch Messages

Process multiple messages in a single request.

**Endpoint:** `POST /v1/messages/batch`

**Request:**
```bash
curl -X POST http://localhost:8000/v1/messages/batch \
  -H "Content-Type: application/json" \
  -d '{
    [
      {
        "model": "Qwen/Qwen3-Coder-25B-mlx",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What is 2+2?"}]
      },
      {
        "model": "Qwen/Qwen3-Coder-25B-mlx",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What is the sky blue?"}]
      }
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "id": "msg_batch1",
      "type": "message",
      "role": "assistant",
      "content": [{"type": "text", "text": "2+2 equals 4."}],
      "stop_reason": "end_turn",
      "stop_sequence": null,
      "usage": {"input_tokens": 10, "output_tokens": 6}
    },
    {
      "id": "msg_batch2",
      "type": "message",
      "role": "assistant",
      "content": [{"type": "text", "text": "The sky appears blue due to Rayleigh scattering..."}],
      "stop_reason": "end_turn",
      "stop_sequence": null,
      "usage": {"input_tokens": 12, "output_tokens": 45}
    }
  ]
}
```

#### 6. Health Check

Check server health and status.

**Endpoint:** `GET /health`

**Request:**
```bash
curl http://localhost:8000/health
```

**Response (Healthy):**
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen3-Coder-25B-mlx",
  "batch_processing": true
}
```

**Response (Unhealthy):**
```json
{
  "status": "unhealthy",
  "message": "Server not ready"
}
```

## Advanced Usage

### Tool Calling

The server supports Anthropic-style tool calling with automatic parsing.

**Request with Tools:**
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Coder-25B-mlx",
    "max_tokens": 1024,
    "tools": [
      {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"]
        }
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Paris?"
      }
    ]
  }'
```

**Response with Tool Use:**
```json
{
  "id": "msg_tool123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "id": "toolu_abc123",
      "name": "get_weather",
      "input": {
        "location": "Paris",
        "unit": "celsius"
      }
    }
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 150,
    "output_tokens": 45
  }
}
```

**Continuing Conversation with Tool Result:**
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Coder-25B-mlx",
    "max_tokens": 1024,
    "tools": [...],
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Paris?"
      },
      {
        "type": "tool_use",
        "id": "toolu_abc123",
        "name": "get_weather",
        "input": {
          "location": "Paris",
          "unit": "celsius"
        }
      },
      {
        "type": "tool_result",
        "tool_use_id": "toolu_abc123",
        "content": "The weather in Paris is 18°C and partly cloudy."
      }
    ]
  }'
```

### Multi-Modal Messages (Images)

Support for image inputs in messages.

**Request with Image (Base64):**
```json
{
  "model": "Qwen/Qwen3-Coder-25B-mlx",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "iVBORw0KGgoAAAANSUhEUgAA..."
      }
    },
    {
      "role": "user",
      "type": "text",
      "content": "What's in this image?"
    }
  ]
}
```

**Request with Image (URL):**
```json
{
  "model": "Qwen/Qwen3-Coder-25B-mlx",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "type": "image",
      "source": {
        "type": "url",
        "url": "https://example.com/image.jpg"
      }
    },
    {
      "role": "user",
      "type": "text",
      "content": "Describe this image"
    }
  ]
}
```

### System Messages

```json
{
  "model": "Qwen/Qwen3-Coder-25B-mlx",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding assistant."
    },
    {
      "role": "user",
      "content": "Write a Python function to reverse a string"
    }
  ]
}
```

## Message Types

### Input Message Types

1. **Text Message**
```json
{
  "role": "user",
  "type": "text",
  "content": "Hello!"
}
```

2. **Image Message**
```json
{
  "role": "user",
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": "..."
  }
}
```

3. **Tool Use Message**
```json
{
  "type": "tool_use",
  "id": "toolu_123",
  "name": "function_name",
  "input": {"param": "value"}
}
```

4. **Tool Result Message**
```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_123",
  "content": "Result from tool"
}
```

### Output Content Types

1. **Text Content**
```json
{
  "type": "text",
  "text": "Response text"
}
```

2. **Tool Use Content**
```json
{
  "id": "toolu_123",
  "name": "function_name",
  "input": {"param": "value"}
}
```

## Error Handling

The API returns errors in Anthropic-compatible format:

**Invalid Request (400):**
```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Empty batch request"
  }
}
```

**Timeout (408):**
```json
{
  "error": {
    "type": "timeout_error",
    "message": "Request timeout"
  }
}
```

**Server Error (500):**
```json
{
  "error": {
    "type": "api_error",
    "message": "An internal server error occurred"
  }
}
```

**Service Unavailable (503):**
```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Server not ready"
  }
}
```

## Architecture

### Continuous Batching

The server uses a sophisticated continuous batching system:

1. **Request Queue**: Incoming requests are added to a queue
2. **Batch Processor**: Background thread continuously processes batches
3. **Dynamic Batching**: New requests are added to running batches
4. **Token Streaming**: Tokens are streamed back as they're generated
5. **Prompt Caching**: Frequently used prompts are cached for performance

### Components

- **`app.py`**: FastAPI application and main entry point
- **`api/server.py`**: Core server logic with continuous batching
- **`api/types.py`**: Pydantic models for API types
- **`token_parser/`**: Token parsers for tool calling
  - `base_token_parser.py`: Base parser interface
  - `qwen3_mode_parser.py`: Qwen3 MoE-specific parser
  - `token_parser_factory.py`: Parser factory
- **`utils/`**: Utility modules
  - `custom_batch.py`: Custom batch generator
  - `prompt_cache_helper.py`: Prompt caching utilities

## Performance Optimization

### Prompt Caching

The server automatically caches prompt prefixes to improve performance:

- Caches are built for prompts after first use
- Future requests with the same prefix reuse the cache
- Reduces computation for repeated conversation contexts
- Automatically creates "next turn" caches

### Batch Size Tuning

Adjust batch sizes based on your hardware:

```bash
# For smaller GPUs (e.g., M1/M2)
python app.py model --prefill-batch-size 4 --completion-batch-size 16

# For larger GPUs (e.g., M3 Max, M4)
python app.py model --prefill-batch-size 16 --completion-batch-size 64
```

### KV Cache Size

Adjust maximum KV cache size for longer contexts:

```bash
# For long conversations
python app.py model --max-kv-size 16384

# For shorter conversations (saves memory)
python app.py model --max-kv-size 4096
```

## Python Client Example

```python
import anthropic

# Initialize client with custom base URL
client = anthropic.Anthropic(
    api_key="not-needed",  # Server doesn't require API key
    base_url="http://localhost:8000"
)

# Non-streaming request
message = client.messages.create(
    model="Qwen/Qwen3-Coder-25B-mlx",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is MLX?"}
    ]
)
print(message.content[0].text)

# Streaming request
with client.messages.stream(
    model="Qwen/Qwen3-Coder-25B-mlx",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Count to 10"}
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Tool calling
message = client.messages.create(
    model="Qwen/Qwen3-Coder-25B-mlx",
    max_tokens=1024,
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
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ]
)
```

## Logging

The server uses structured logging:

- **INFO**: General server operations, request completions
- **DEBUG**: Detailed request/response (use `--verbose`)
- **WARNING**: Timeouts, cache issues
- **ERROR**: Exceptions and critical errors

View logs with timestamps:
```
2025-01-18 10:30:45 - INFO - Model loaded in 3.45s
2025-01-18 10:30:45 - INFO - Started continuous batch processing thread
2025-01-18 10:30:50 - INFO - Adding prompt prompt_abc123 to batch
2025-01-18 10:30:52 - INFO - Completed generation for prompt_abc123 (reason: end_turn)
```

## Development

### Running in Development Mode

```bash
python app.py model --reload --verbose
```

### Project Structure

```
mlx-server/
├── app.py                          # FastAPI application
├── api/
│   ├── server.py                   # Server with continuous batching
│   └── types.py                    # API type definitions
├── token_parser/
│   ├── __init__.py
│   ├── base_token_parser.py        # Base parser interface
│   ├── qwen3_mode_parser.py        # Qwen3 parser implementation
│   └── token_parser_factory.py     # Parser factory
├── utils/
│   ├── __init__.py
│   ├── custom_batch.py             # Batch generator
│   ├── logging.py                  # Logging utilities
│   └── prompt_cache_helper.py      # Prompt caching
├── caches/                         # Prompt cache storage (auto-created)
├── requirements.txt
└── README.md
```

## Troubleshooting

### Server Won't Start

**Issue**: Model fails to load
```
Solution: Ensure model path is correct and model is compatible with MLX
```

**Issue**: Port already in use
```bash
# Use different port
python app.py model --port 8001
```

### Memory Issues

**Issue**: Out of memory errors
```bash
# Reduce batch sizes and KV cache
python app.py model --prefill-batch-size 2 --completion-batch-size 8 --max-kv-size 2048
```

### Timeout Errors

**Issue**: Requests timing out
```
Solution: The default timeout is 300s. For very long generations, this is working as expected.
Check the server logs for more details on what's taking time.
```

## License

This project follows the same license as the MLX framework.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- Compatible with [Anthropic's Messages API](https://docs.anthropic.com/en/api/messages)
- Uses FastAPI for high-performance async request handling
