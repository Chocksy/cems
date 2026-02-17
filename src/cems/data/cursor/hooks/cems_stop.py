#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS Stop Hook - Cursor Session Cleanup

This hook runs when a Cursor session ends. It:
1. Reads the accumulated transcript from afterAgentResponse hook
2. Cleans up the transcript file

Session summaries are handled by the observer daemon, not by the hook.

Configuration (environment variables):
  CEMS_SESSION_ID - Session ID (set by sessionStart hook)
"""

import json
import os
import sys


# Transcript storage location (must match cems_agent_response.py)
TRANSCRIPT_DIR = "/tmp/cems-transcripts"


def get_transcript_path(session_id: str) -> str:
    """Get the path to the transcript file for a session."""
    return os.path.join(TRANSCRIPT_DIR, f"{session_id}.jsonl")


def cleanup_transcript(session_id: str):
    """Remove the transcript file after processing."""
    transcript_path = get_transcript_path(session_id)
    try:
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
    except IOError:
        pass


def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Get session ID from environment
        session_id = os.getenv("CEMS_SESSION_ID", "")

        if not session_id:
            print(json.dumps({}))
            sys.exit(0)

        # Clean up transcript file
        cleanup_transcript(session_id)

        print(json.dumps({}))
        sys.exit(0)

    except json.JSONDecodeError:
        print(json.dumps({}))
        sys.exit(0)
    except Exception:
        print(json.dumps({}))
        sys.exit(0)


if __name__ == "__main__":
    main()
