#!/usr/bin/env python3
"""
MLX Advanced Server
A high-performance inference server with continuous batching support.
"""

import argparse
import atexit
import signal
import sys
import logging
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.types import MessagesBody, MessagesResponse
from api.server import Server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration"""
    model_name: str
    token_parser_name: Literal["qwen3_moe"]
    host: str = "0.0.0.0"
    port: int = 8000
    verbose: bool = False
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    trust_remote_code : bool = False,
    max_kv_size: int = 4096

# Global instances
config: ServerConfig | None = None
server: Server | None = None


def cleanup():
    """Cleanup function to unload model and stop server"""
    global server

    if server is not None:
        logger.info("Cleaning up server resources...")
        try:
            server.stop_batch_processing()
            server.unload()
            logger.info("Server cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            server = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    signal_name = signal.Signals(signum).name
    logger.info(f"Received signal {signal_name}, initiating graceful shutdown...")

    # Cleanup
    cleanup()

    # Exit
    logger.info("Shutdown complete")
    sys.exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global server, config

    # Startup
    logger.info(f"Loading model: {config.model_name}")
    server = Server(
        model_name=config.model_name,
        token_parser_name=config.token_parser_name,
        prefill_batch_size=config.prefill_batch_size,
        completion_batch_size=config.completion_batch_size,
        trust_remote_code=config.trust_remote_code,
    )
    server.load()
    server.start_batch_processing()
    logger.info("Server started and ready to accept requests")

    yield

    # Shutdown
    logger.info("Shutting down server via lifespan...")
    cleanup()


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="MLX Chat Completions API",
        description="Anthropic-compatible chat completions API powered by MLX with continuous batching",
        version="2.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions in Anthropic format"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "message": exc.detail,
                }
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "api_error",
                    "message": "An internal server error occurred",
                }
            }
        )

    # API endpoints
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "MLX Anthropic API",
            "version": "2.0.0",
            "model": config.model_name if config else "unknown",
            "status": "ready" if server and server.loaded else "loading"
        }

    @app.get("/v1/models")
    async def list_models():
        """List available models (Anthropic-compatible)"""
        if not server or not server.loaded:
            raise HTTPException(status_code=503, detail="Server not ready")

        return {
            "data": [
                {
                    "id": config.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local"
                }
            ]
        }

    @app.post("/v1/messages", response_model=None)
    async def messages(request: MessagesBody):
        """
        Chat completion endpoint with continuous batching support.
        Anthropic Messages API compatible.
        """
        if not server or not server.loaded:
            raise HTTPException(status_code=503, detail="Server not ready")

        try:
            # Check if streaming is requested
            if request.stream:
                return StreamingResponse(
                    stream_response(request),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                response = server.generate(request, timeout=300.0)
                return JSONResponse(content=response.model_dump())

        except TimeoutError as e:
            logger.error(f"Request timeout: {e}")
            raise HTTPException(status_code=408, detail="Request timeout")
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/messages/batch", response_model=None)
    async def messages_batch(requests: list[MessagesBody]):
        """
        Batch chat completion endpoint.
        Processes multiple requests efficiently using continuous batching.
        """
        if not server or not server.loaded:
            raise HTTPException(status_code=503, detail="Server not ready")

        if not requests:
            raise HTTPException(status_code=400, detail="Empty batch request")

        # Batch requests cannot be streamed
        for req in requests:
            if req.stream:
                raise HTTPException(
                    status_code=400,
                    detail="Streaming is not supported in batch requests"
                )

        try:
            responses = server.batch_generate(requests, timeout=300.0)
            return JSONResponse(
                content={
                    "results": [resp.model_dump() for resp in responses]
                }
            )

        except TimeoutError as e:
            logger.error(f"Batch request timeout: {e}")
            raise HTTPException(status_code=408, detail="Batch request timeout")
        except Exception as e:
            logger.error(f"Error processing batch request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        if not server or not server.loaded:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Server not ready"}
            )

        return {
            "status": "healthy",
            "model": config.model_name,
            "batch_processing": server.running
        }

    return app


async def stream_response(request: MessagesBody):
    """
    Stream response generator for Server-Sent Events (SSE).
    Follows Anthropic's streaming format.
    """
    global server
    input_tokens = server.count_input_tokens(request)  # You'll need to implement this

    try:
        # Send message_start event
        message_start = {
            "type": "message_start",
            "message": {
                "id": f"msg_stream_{request.model}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send content_block_start event
        content_block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }
        yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

        # Stream the actual content
        generator = server.generate_stream(request, timeout=300.0)

        for chunk in generator:
            if isinstance(chunk, dict) and "delta" in chunk:
                # Send content_block_delta event
                delta = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": chunk["delta"]
                    }
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
            elif isinstance(chunk, MessagesResponse):
                # This is the final response
                final_response = chunk

                # Send content_block_stop event
                content_block_stop = {
                    "type": "content_block_stop",
                    "index": 0
                }
                yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

                # Send message_delta event
                message_delta = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": final_response.stop_reason,
                        "stop_sequence": final_response.stop_sequence
                    },
                    "usage": {
                        "input_tokens": final_response.usage.input_tokens,
                        "output_tokens": final_response.usage.output_tokens
                    }
                }
                yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

                # Send message_stop event
                message_stop = {
                    "type": "message_stop"
                }
                yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"

                break

    except TimeoutError as e:
        logger.error(f"Stream timeout: {e}")
        error_event = {
            "type": "error",
            "error": {
                "type": "timeout_error",
                "message": "Request timeout"
            }
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        error_event = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": str(e)
            }
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"


def main():
    """Main entry point"""
    global config

    parser = argparse.ArgumentParser(
        description='MLX Advanced Server - Anthropic Compatible API with Continuous Batching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('model_name_or_path', help='Huggingface model name or path')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument("--max-kv-size", type=int, default=8096, help="Maximum size of key-value cache (default: 8096)")
    parser.add_argument("--prefill-batch-size", type=int, default=8, help="Number of messages to prefill batch (default: 8)")
    parser.add_argument("--completion-batch-size", type=int, default=32, help="Number of messages to complete batch (default: 8)")
    parser.add_argument("--trust_remote_code", default=False, action='store_true', help='Trust remote code')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument(
        '--token-parser',
        default='qwen3_moe',
        choices=['qwen3_moe'],
        help='Token parser to use for tool calling (default: qwen3_moe)'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create config
    config = ServerConfig(
        model_name=args.model_name_or_path,
        token_parser_name=args.token_parser,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
        completion_batch_size=args.completion_batch_size,
        prefill_batch_size=args.prefill_batch_size,
        trust_remote_code=args.trust_remote_code,
        max_kv_size=args.max_kv_size

    )

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

    # Register cleanup function to run on exit
    atexit.register(cleanup)

    logger.info(f"Starting MLX Advanced Server")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Host: {config.host}:{config.port}")
    logger.info(f"Token Parser: {config.token_parser_name}")

    # Create app
    app = create_app()

    # Run server
    # Check if running in PyCharm debug mode
    import sys
    if 'pydevd' in sys.modules:
        # In debug mode, run without loop_factory to avoid compatibility issues
        import asyncio
        asyncio.run(uvicorn.Server(uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            log_level="debug" if config.verbose else "info"
        )).serve())
    else:
        # Normal execution
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="debug" if config.verbose else "info"
        )


if __name__ == "__main__":
    main()
