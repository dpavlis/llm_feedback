#!/usr/bin/env python3
"""
LLM Feedback Chat Application - Startup Script

Usage:
    python run.py [--host HOST] [--port PORT] [--reload] [--workers N]

Configuration is loaded from .env file. See .env.example for all options.
Command-line arguments override .env settings.
"""

import argparse
import logging
import sys

import uvicorn

from app.config import settings


def setup_logging():
    """Configure logging based on settings."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the LLM Feedback Chat server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                          # Run with defaults from .env
    python run.py --port 9000              # Override port
    python run.py --reload                 # Development mode with auto-reload
    python run.py --workers 2              # Multiple workers (CPU inference only)

Configuration:
    Copy .env.example to .env and edit as needed.
    Environment variables override .env settings.
    Command-line arguments override environment variables.
        """,
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"Host to bind to (default: {settings.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind to (default: {settings.port})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.debug,
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help=f"Number of worker processes (default: {settings.workers})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=settings.debug,
        help="Enable debug mode",
    )

    args = parser.parse_args()

    setup_logging()

    # Build model info string
    model_source = str(settings.model_path) if settings.model_path else settings.model_name
    device_info = settings.model_device or "auto-detect"
    if settings.cuda_visible_devices:
        device_info += f" (CUDA devices: {settings.cuda_visible_devices})"

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    {settings.app_name:^52} ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Model:   {model_source:<64} ║
║  Device:  {device_info:<64} ║
║  Server:  http://{args.host}:{args.port:<57} ║
║  Workers: {args.workers:<64} ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Uvicorn configuration
    uvicorn_kwargs = {
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "log_level": settings.log_level.lower(),
    }

    # Only use workers if not reloading (reload is incompatible with workers)
    if not args.reload and args.workers > 1:
        uvicorn_kwargs["workers"] = args.workers
        print(f"Warning: Using {args.workers} workers. Ensure you have enough GPU memory!")

    uvicorn.run("app.main:app", **uvicorn_kwargs)


if __name__ == "__main__":
    main()
