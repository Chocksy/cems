#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS Agent Response Hook - Transcript Accumulation

This hook runs after each agent response and accumulates the transcript
for later analysis by the stop hook.

Configuration (environment variables):
  CEMS_SESSION_ID - Session ID (set by sessionStart hook)
"""

import json
import os
import sys
from datetime import datetime

# Transcript storage location
TRANSCRIPT_DIR = "/tmp/cems-transcripts"


def get_transcript_path(session_id: str) -> str:
    """Get the path to the transcript file for a session."""
    return os.path.join(TRANSCRIPT_DIR, f"{session_id}.jsonl")


def ensure_transcript_dir():
    """Ensure the transcript directory exists."""
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def append_to_transcript(session_id: str, text: str):
    """Append a response to the session transcript."""
    if not session_id or not text:
        return
    
    ensure_transcript_dir()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": "assistant",
        "text": text[:10000],  # Limit to 10K chars per response
    }
    
    transcript_path = get_transcript_path(session_id)
    
    try:
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except IOError:
        pass  # Silently fail - don't disrupt the session


def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        text = input_data.get("text", "")
        
        # Get session ID from environment (set by sessionStart hook)
        session_id = os.getenv("CEMS_SESSION_ID", "")
        
        if not session_id:
            # No session ID, skip
            sys.exit(0)
        
        # Skip very short responses
        if len(text) < 50:
            sys.exit(0)
        
        # Append to transcript
        append_to_transcript(session_id, text)
        
        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
