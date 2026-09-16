"""Command-line entrypoint for the LiveKit voice frontend."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .remote import RemoteGraphConfig
from .tracing import configure_environment


def _load_local_environment() -> None:
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)


def main() -> None:
    _load_local_environment()
    parser = argparse.ArgumentParser(
        description="Run a LiveKit console voice agent backed by LangGraph Server."
    )
    parser.add_argument(
        "--langgraph-url",
        default=os.getenv("LANGGRAPH_URL", "http://127.0.0.1:2024"),
    )
    parser.add_argument(
        "--assistant-id", default=os.getenv("LANGGRAPH_ASSISTANT_ID", "voice_agent")
    )
    parser.add_argument(
        "--project",
        default=os.getenv("LANGSMITH_PROJECT", "voice-separate-langgraph"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("LANGGRAPH_TIMEOUT_SECONDS", "45")),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("langsmith.integrations").setLevel(logging.DEBUG)

    configure_environment(args.project)
    config = RemoteGraphConfig(
        url=args.langgraph_url,
        assistant_id=args.assistant_id,
        project_name=args.project,
        api_key=os.getenv("LANGGRAPH_API_KEY") or None,
        timeout_seconds=args.timeout,
    )

    # Import after tracing environment is finalized and before LiveKit plugins.
    from .livekit_agent import run

    run(config)


if __name__ == "__main__":
    main()
