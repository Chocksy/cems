#!/usr/bin/env python3
"""Central hook event logger — one JSONL file for all hook activity.

Two files per session:
  ~/.claude/hooks/logs/hook_events.jsonl     — lean central log (tail -f friendly)
  ~/.claude/hooks/logs/verbose/{sid}.jsonl   — full input_data payloads per session
"""

from __future__ import annotations

import json
import time
from pathlib import Path

LOG_DIR = Path.home() / ".claude" / "hooks" / "logs"
LOG_FILE = LOG_DIR / "hook_events.jsonl"
VERBOSE_DIR = LOG_DIR / "verbose"


def log_hook_event(
    event: str,
    session_id: str,
    extra: dict | None = None,
    input_data: dict | None = None,
) -> None:
    """Log a hook event to both the central log and verbose per-session file.

    Args:
        event: Hook event name (e.g. "SessionStart", "PreCompact", "Stop")
        session_id: Claude Code session ID
        extra: Optional dict merged into the lean log entry
        input_data: Full stdin payload from Claude Code — written to verbose log
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        short_sid = session_id[:12] if session_id else ""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        # --- Verbose per-session log (full payload) ---
        verbose_ref = None
        if input_data and short_sid:
            VERBOSE_DIR.mkdir(parents=True, exist_ok=True)
            verbose_path = VERBOSE_DIR / f"{short_sid}.jsonl"

            # Scrub large fields to keep files manageable
            scrubbed = _scrub_payload(input_data)
            verbose_entry = {"ts": ts, "event": event, **scrubbed}

            with open(verbose_path, "a") as f:
                f.write(json.dumps(verbose_entry, default=str) + "\n")

            verbose_ref = str(verbose_path)

        # --- Lean central log ---
        entry = {"ts": ts, "event": event, "session_id": short_sid}
        if extra:
            entry.update(extra)
        if verbose_ref:
            entry["verbose"] = verbose_ref

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    except Exception:
        pass  # Never break a hook because of logging


def _scrub_payload(data: dict) -> dict:
    """Return a copy with oversized fields truncated."""
    scrubbed = {}
    for k, v in data.items():
        if k in ("transcript_path",):
            # Keep path but don't inline transcript content
            scrubbed[k] = v
        elif k in ("tool_response", "tool_input"):
            # Truncate large tool I/O to keep verbose logs readable
            s = json.dumps(v, default=str) if not isinstance(v, str) else v
            scrubbed[k] = s[:2000] + "..." if len(s) > 2000 else v
        elif k == "prompt":
            # Keep first 500 chars of user prompt
            scrubbed[k] = v[:500] if isinstance(v, str) else v
        elif k == "custom_instructions":
            # Skip — huge and not useful for debugging
            scrubbed[k] = f"[{len(v)} chars]" if isinstance(v, str) else v
        else:
            scrubbed[k] = v
    return scrubbed
