#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""
PreCompact Hook - Context compaction handler.

Fires when Claude Code auto-compacts (context window fills up).
Writes a "compact" signal for the observer daemon (triggers epoch bump).

Matcher: "auto" only (manual /compact is user-controlled).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.hook_logger import log_hook_event


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
        input_data = json.load(sys.stdin)

        session_id = input_data.get("session_id", "")
        trigger = input_data.get("trigger", "unknown")
        cwd = input_data.get("cwd", os.getcwd())

        log_hook_event("PreCompact", session_id, {
            "trigger": trigger,
            "cwd": cwd,
        }, input_data=input_data)

        # Signal the observer daemon about compaction (triggers epoch bump)
        if session_id:
            write_signal(session_id, "compact", "claude")

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
