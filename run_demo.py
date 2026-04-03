from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from pipeline import DEFAULT_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the moderation demo with the current Python interpreter.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto reload for local development.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["MODEL_DIR"] = args.model_dir

    uvicorn.run(
        "serve_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(Path(__file__).resolve().parent),
    )


if __name__ == "__main__":
    main()
