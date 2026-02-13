"""Signal-based IPC between hooks and the observer daemon.

Hooks write tiny signal files to tell the daemon about lifecycle events
(compact, stop). The daemon reads and clears them each cycle.

Signal files are single JSON objects (overwrite, not append) stored at:
    ~/.cems/observer/signals/{session_id}.json
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SIGNALS_DIR = Path.home() / ".cems" / "observer" / "signals"


@dataclass
class Signal:
    """A lifecycle signal from a hook to the daemon."""
    type: str    # "compact" | "stop"
    ts: float    # unix timestamp
    tool: str    # "claude" | "codex" | "cursor"


def write_signal(session_id: str, signal_type: str, tool: str = "claude") -> None:
    """Write a signal file for the daemon to pick up.

    Overwrites any existing signal for this session (last write wins).

    Args:
        session_id: Session UUID.
        signal_type: "compact" or "stop".
        tool: Tool that generated the signal.
    """
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    signal_file = SIGNALS_DIR / f"{session_id}.json"

    data = {
        "type": signal_type,
        "ts": time.time(),
        "tool": tool,
    }

    try:
        # Atomic write: write to temp then rename
        tmp = signal_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.rename(signal_file)
    except OSError as e:
        logger.error(f"Could not write signal for {session_id}: {e}")


def read_signal(session_id: str) -> Signal | None:
    """Read a signal file if one exists.

    Args:
        session_id: Session UUID.

    Returns:
        Signal object, or None if no signal file exists.
    """
    signal_file = SIGNALS_DIR / f"{session_id}.json"

    if not signal_file.exists():
        return None

    try:
        data = json.loads(signal_file.read_text())
        return Signal(
            type=data["type"],
            ts=data["ts"],
            tool=data.get("tool", "claude"),
        )
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning(f"Could not read signal for {session_id}: {e}")
        return None


def clear_signal(session_id: str) -> None:
    """Remove a signal file after the daemon has processed it.

    Args:
        session_id: Session UUID.
    """
    signal_file = SIGNALS_DIR / f"{session_id}.json"

    try:
        signal_file.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Could not clear signal for {session_id}: {e}")
