"""Per-session observation state tracking.

Stores state in ~/.cems/observer/{session-uuid}.json so the daemon
knows what has already been observed per session.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

OBSERVER_STATE_DIR = Path.home() / ".cems" / "observer"


@dataclass
class ObservationState:
    """Tracks observation progress for a single session."""
    session_id: str
    project_id: str | None = None
    source_ref: str | None = None
    last_observed_bytes: int = 0
    last_observed_at: float = 0.0
    observation_count: int = 0
    session_started: float = field(default_factory=time.time)
    # Epoch model: bumped on compact signal, each epoch gets its own document
    epoch: int = 0
    last_finalized_at: float = 0.0
    # Staleness detection: tracks when the file last grew
    last_growth_seen_at: float = 0.0
    is_done: bool = False


def session_tag(session_id: str, epoch: int = 0) -> str:
    """Build the session tag used to identify documents per epoch.

    Epoch 0 uses backwards-compatible format: session:{id[:8]}
    Epoch N>0 appends epoch suffix: session:{id[:8]}:e{N}

    Args:
        session_id: Session UUID.
        epoch: Epoch number.

    Returns:
        Session tag string.
    """
    tag = f"session:{session_id[:8]}"
    if epoch > 0:
        tag += f":e{epoch}"
    return tag


def load_state(session_id: str) -> ObservationState:
    """Load observation state for a session, or create new state.

    Args:
        session_id: Session UUID.

    Returns:
        ObservationState (loaded or fresh).
    """
    state_file = OBSERVER_STATE_DIR / f"{session_id}.json"

    if state_file.exists():
        try:
            with open(state_file) as f:
                data = json.load(f)
            return ObservationState(**{
                k: v for k, v in data.items()
                if k in ObservationState.__dataclass_fields__
            })
        except (json.JSONDecodeError, TypeError, OSError) as e:
            logger.warning(f"Could not load state for {session_id}: {e}")

    return ObservationState(session_id=session_id)


def save_state(state: ObservationState) -> None:
    """Persist observation state to disk.

    Uses atomic tmp+rename to prevent corruption if the process
    crashes mid-write (corrupted state → fresh state → re-send from byte 0).

    Args:
        state: ObservationState to save.
    """
    OBSERVER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = OBSERVER_STATE_DIR / f"{state.session_id}.json"

    try:
        tmp = state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(state), f, indent=2)
        tmp.rename(state_file)
    except OSError as e:
        logger.error(f"Could not save state for {state.session_id}: {e}")


def cleanup_old_states(max_age_days: int = 7) -> int:
    """Remove state files for sessions older than max_age_days.

    Args:
        max_age_days: Delete state files older than this many days.

    Returns:
        Number of files cleaned up.
    """
    if not OBSERVER_STATE_DIR.exists():
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    removed = 0

    for state_file in OBSERVER_STATE_DIR.glob("*.json"):
        try:
            if state_file.stat().st_mtime < cutoff:
                state_file.unlink()
                removed += 1
        except OSError:
            continue

    if removed:
        logger.info(f"Cleaned up {removed} old observer state files")
    return removed
