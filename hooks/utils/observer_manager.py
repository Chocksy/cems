#!/usr/bin/env python3
"""Observer daemon lifecycle management for Claude hooks.

Provides ensure_daemon_running() which hooks call to auto-start the
observer daemon if it's not running. Uses a PID file for process tracking
and a cooldown mechanism to avoid spam-spawning on repeated failures.

PID file:      ~/.claude/observer/daemon.pid
Cooldown file:  ~/.claude/observer/.spawn_cooldown
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from utils.credentials import get_cems_key, get_credentials_env

OBSERVER_DIR = Path.home() / ".claude" / "observer"
PID_FILE = OBSERVER_DIR / "daemon.pid"
COOLDOWN_FILE = OBSERVER_DIR / ".spawn_cooldown"

# Don't retry spawn for 10 minutes after a failure
SPAWN_COOLDOWN_SECONDS = 600

# Only check daemon health every 5 minutes (avoid overhead on every prompt)
HEALTH_CHECK_INTERVAL = 300


def _read_pid() -> int | None:
    """Read PID from file, return None if missing or invalid."""
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            return pid if pid > 0 else None
    except (ValueError, OSError):
        pass
    return None


def _is_process_alive(pid: int) -> bool:
    """Check if a process with given PID exists."""
    try:
        os.kill(pid, 0)  # Signal 0 = existence check, no actual signal sent
        return True
    except (ProcessLookupError, PermissionError):
        return False


def is_daemon_running() -> bool:
    """Check if the observer daemon is currently running.

    Reads the PID file and verifies the process exists.

    Returns:
        True if daemon is running, False otherwise.
    """
    pid = _read_pid()
    if pid is None:
        return False
    if _is_process_alive(pid):
        return True
    # Stale PID file — process is dead
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def _is_in_cooldown() -> bool:
    """Check if we're in spawn cooldown (recent failure)."""
    try:
        if COOLDOWN_FILE.exists():
            age = time.time() - COOLDOWN_FILE.stat().st_mtime
            if age < SPAWN_COOLDOWN_SECONDS:
                return True
            # Cooldown expired, clean up
            COOLDOWN_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def _set_cooldown() -> None:
    """Set spawn cooldown after a failure."""
    try:
        OBSERVER_DIR.mkdir(parents=True, exist_ok=True)
        COOLDOWN_FILE.touch()
    except OSError:
        pass


def _should_check() -> bool:
    """Rate-limit health checks using PID file mtime.

    Returns True if enough time has passed since last check.
    Called from user_prompts_submit to avoid overhead on every prompt.
    """
    try:
        if PID_FILE.exists():
            age = time.time() - PID_FILE.stat().st_mtime
            if age < HEALTH_CHECK_INTERVAL:
                return False
        # Touch PID file to reset the timer (even if daemon isn't running)
        # We'll create/update it properly when spawning
        return True
    except OSError:
        return True


def _spawn_daemon() -> bool:
    """Spawn the observer daemon as a detached background process.

    Uses the project's Python to run `python -m cems.observer`.
    The daemon writes its own PID file on startup.

    Returns:
        True if spawn succeeded, False otherwise.
    """
    # Find the CEMS project root (where the daemon code lives)
    # The hooks know CEMS_API_URL/KEY from env, but need the venv python
    cems_root = os.getenv("CEMS_PROJECT_ROOT", "")

    # Try to find it from common locations
    if not cems_root:
        candidates = [
            Path.home() / "Development" / "cems",
            Path.home() / "projects" / "cems",
            Path.home() / "code" / "cems",
        ]
        for candidate in candidates:
            if (candidate / "src" / "cems" / "observer").exists():
                cems_root = str(candidate)
                break

    if not cems_root:
        return False

    cems_path = Path(cems_root)
    venv_python = cems_path / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        venv_python = cems_path / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return False

    # Check required env vars (env or ~/.cems/credentials)
    if not get_cems_key():
        return False

    try:
        OBSERVER_DIR.mkdir(parents=True, exist_ok=True)

        # Build env with credentials from file fallback
        spawn_env = get_credentials_env()
        spawn_env["PYTHONPATH"] = str(cems_path / "src")

        # Spawn daemon as fully detached process
        # stdout/stderr go to a log file for debugging
        log_file = OBSERVER_DIR / "daemon.log"
        with open(log_file, "a") as logf:
            proc = subprocess.Popen(
                [str(venv_python), "-m", "cems.observer"],
                cwd=str(cems_path),
                stdout=logf,
                stderr=logf,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent process group
                env=spawn_env,
            )

        # Write PID file
        PID_FILE.write_text(str(proc.pid))

        # Give it a moment to crash if it's going to
        time.sleep(0.3)
        if _is_process_alive(proc.pid):
            return True

        # Process died immediately
        PID_FILE.unlink(missing_ok=True)
        return False

    except (OSError, subprocess.SubprocessError):
        return False


def ensure_daemon_running(force_check: bool = False) -> bool:
    """Ensure the observer daemon is running, spawn if needed.

    This is the main entry point for hooks. It:
    1. Checks if daemon is alive (rate-limited unless force_check)
    2. If dead, checks cooldown
    3. If no cooldown, spawns daemon
    4. On spawn failure, sets cooldown to avoid retrying for 10 min

    Args:
        force_check: Skip rate limiting (use from session_start).
            When False, only checks every HEALTH_CHECK_INTERVAL seconds.

    Returns:
        True if daemon is running (already or newly spawned).
    """
    # Rate-limit checks from frequent hooks (user_prompts_submit)
    if not force_check and not _should_check():
        return True  # Assume running, checked recently

    # Check if already running
    if is_daemon_running():
        # Touch PID file to reset health check timer
        try:
            PID_FILE.touch()
        except OSError:
            pass
        return True

    # Not running — check cooldown
    if _is_in_cooldown():
        return False

    # Try to spawn
    success = _spawn_daemon()
    if not success:
        _set_cooldown()
    return success
