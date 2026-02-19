#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS Stop Hook — Cursor Session End

Writes an observer signal so the daemon can finalize the session summary.
Does NOT delete transcripts (let the observer daemon handle cleanup).

Signal location: ~/.cems/observer/signals/{conversation_id}.json
"""

import json
import sys
import time
from pathlib import Path


def write_signal(session_id: str, signal_type: str) -> None:
    """Write a signal file for the observer daemon."""
    signals_dir = Path.home() / ".cems" / "observer" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signal_file = signals_dir / f"{session_id}.json"

    data = {"type": signal_type, "ts": time.time(), "tool": "cursor"}
    try:
        tmp = signal_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.rename(signal_file)
    except OSError:
        pass


def main():
    try:
        input_data = json.load(sys.stdin)

        # Cursor stop hook provides conversation_id
        session_id = input_data.get("conversation_id", "")
        status = input_data.get("status", "completed")

        if not session_id:
            return

        # Write observer signal for session finalization
        write_signal(session_id, "stop")

    except json.JSONDecodeError:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
