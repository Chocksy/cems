#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""
Stop Hook - Session Logging Handler

This hook runs when Claude stops (fires after EVERY assistant turn) and:
1. Logs session data (stop.json, optional chat.json)

NOTE: We intentionally do NOT write stop signals here. Claude Code fires
the Stop hook after every assistant turn, not just session exit. The observer
daemon's staleness detection handles session finalization.
"""

import argparse
import json
import os
import sys
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

        # NOTE: We intentionally do NOT write a "stop" signal here.
        # Claude Code fires the Stop hook after EVERY assistant turn, not just
        # session exit. Writing a stop signal would permanently mark the session
        # as done (is_done=True) after the first response. The observer daemon's
        # staleness detection handles finalization when the session truly goes idle.

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
