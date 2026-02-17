#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""
Stop Hook - Session End Handler

This hook runs when Claude stops and:
1. Logs session data (stop.json, optional chat.json)
2. Writes a "stop" signal for the observer daemon to finalize the session summary

The observer daemon handles all memory extraction (narrative summaries) via
the /api/session/summarize endpoint. This hook just signals it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import ensure_session_log_dir
from utils.hook_logger import log_hook_event


def read_transcript(transcript_path: str) -> list[dict] | None:
    """Read transcript from .jsonl file and return as list of messages."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        messages = []
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return messages if messages else None
    except Exception:
        return None


def write_signal(session_id: str, signal_type: str, tool: str = "claude") -> None:
    """Write a signal file for the observer daemon to pick up.

    Inlined from cems.observer.signals — hooks run standalone via uv and
    cannot import from the cems package.
    """
    signals_dir = Path.home() / ".cems" / "observer" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signal_file = signals_dir / f"{session_id}.json"

    data = {"type": signal_type, "ts": time.time(), "tool": tool}
    try:
        tmp = signal_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.rename(signal_file)
    except OSError:
        pass


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--chat", action="store_true", help="Copy transcript to chat.json"
        )
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract required fields
        session_id = input_data.get("session_id", "")
        transcript_path = input_data.get("transcript_path", "")
        cwd = input_data.get("cwd", os.getcwd())

        log_hook_event("Stop", session_id, {"cwd": cwd}, input_data=input_data)

        # --- Session logging ---
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "stop.json"

        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        log_data.append(input_data)

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Handle --chat switch
        if args.chat and transcript_path:
            transcript = read_transcript(transcript_path)
            if transcript:
                chat_file = log_dir / "chat.json"
                with open(chat_file, "w") as f:
                    json.dump(transcript, f, indent=2)

        # --- Signal the observer daemon to finalize the session summary ---
        if session_id:
            write_signal(session_id, "stop", "claude")

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
