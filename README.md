# MLX Chat Completions API Server

An OpenAI-compatible chat completions API server powered by MLX with continuous batching support for maximum throughput.

## Features

- **OpenAI v1 API Compatibility**: Drop-in replacement for OpenAI's chat completions API
- **Continuous Batching**: Dynamically adds new requests to running batches for optimal GPU utilization
- **Streaming Support**: Real-time token streaming with Server-Sent Events
- **Concurrent Request Handling**: Efficiently handles multiple simultaneous requests
- **Performance Monitoring**: Built-in statistics endpoint for monitoring throughput
- **Production Ready**: Comprehensive error handling, logging, and graceful shutdown

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
# Basic usage
python app.py /path/to/your/mlx/model

# With options
python app.py /path/to/your/mlx/model --host 0.0.0.0 --port 8000 --verbose

# Example with a specific model
python app.py /Users/vahit/.lmstudio/models/az13770129/Qwen3-Coder-REAP-25B-A3B-mlx-4Bit
```

### Command Line Options

- `model_name_or_path`: Path to the MLX model (required)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--workers`: Number of worker processes (default: 1)
- `--reload`: Enable auto-reload for development
- `--verbose`: Enable verbose logging

### API Endpoints

#### Chat Completions
```bash
# Non-streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "stream": false
  }'

# Streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model", 
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### List Models
```bash
curl http://localhost:8000/v1/models
```

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Server Statistics
```bash
curl http://localhost:8000/stats
```

## Testing

Run the comprehensive test suite:

```bash
# Basic tests
python test_api.py

# With stress testing
python test_api.py --stress

# Test against a different server
python test_api.py --url http://your-server:8000
```

The test suite includes:
- Health and endpoint checks
- Non-streaming completions
- Streaming completions
- Concurrent request handling
- Continuous batching verification
- Stress testing (optional)

## Architecture

The server uses a multi-threaded architecture optimized for continuous batching:

1. **Main Thread**: FastAPI server handling HTTP requests
2. **Processing Thread**: Monitors the prompt queue and initiates batch processing
3. **Batch Thread**: Runs the MLX batch generator for each batch
4. **Results Thread**: Routes generated tokens to the appropriate request queues

This design allows new requests to be dynamically added to running batches, maximizing GPU utilization and throughput.

## Performance

The continuous batching implementation provides several performance benefits:

- **Higher Throughput**: Multiple requests processed simultaneously
- **Better GPU Utilization**: Keeps the GPU busy with dynamic batching
- **Lower Latency**: Requests don't wait for previous batches to complete
- **Efficient Memory Usage**: Shared model weights across all requests

## API Compatibility

The server implements the OpenAI v1 chat completions API specification, making it compatible with:

- OpenAI Python/JavaScript SDKs (with base_url override)
- LangChain
- LlamaIndex
- Any tool that supports OpenAI's API format

## Development

### Project Structure

```
mlx/
├── app.py                    # FastAPI application and main entry point
├── api/
│   ├── server_handler.py     # Core server logic with continuous batching
│   └── generator.py          # Batch generation implementation
├── test_api.py               # Comprehensive test suite
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### Logging

The server uses structured logging with different levels:
- `INFO`: General server operations
- `DEBUG`: Detailed request/response information (--verbose flag)
- `WARNING`: Timeout and error conditions
- `ERROR`: Exceptions and critical errors

## License

This project follows the same license as the MLX framework.