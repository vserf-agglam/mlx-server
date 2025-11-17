#!/usr/bin/env python3
"""
MLX Advanced Server - OpenAI Compatible API
A high-performance inference server with continuous batching support.
"""

import argparse
import signal
import sys
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from mlx_lm.utils import load

from api.types import MessagesBody, MessagesResponse

# Create FastAPI app
app = FastAPI(
    title="MLX Chat Completions API",
    description="OpenAI-compatible chat completions API powered by MLX with continuous batching",
    version="2.0.0",
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
    """Handle HTTP exceptions in OpenAI format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": exc.status_code
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
                "message": "An internal error occurred",
                "type": "internal_error",
                "code": 500
            }
        }
    )


# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MLX Anthropic  API",
        "version": "2.0.0",

    }

@app.post("/messages")
def messages(request: MessagesBody) -> MessagesResponse:
    """
    Chat completion endpoint with continuous batching support.
    """
    model_name = request.model
    messages = request.messages
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p
    n = request.n
    stop = request.stop




def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='MLX Advanced Server - OpenAI Compatible API with Continuous Batching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('model_name_or_path', help='Huggingface model name or path')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes (default: 1)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info" if args.verbose else "warning"
    )


if __name__ == "__main__":
    main()