#!/usr/bin/env python3
"""Observer daemon lifecycle management for Claude hooks.

Provides ensure_daemon_running() which hooks call to auto-start the
observer daemon if it's not running. Uses a PID file for process tracking
and a cooldown mechanism to avoid spam-spawning on repeated failures.

PID file:      ~/.cems/observer/daemon.pid
Cooldown file:  ~/.cems/observer/.spawn_cooldown
"""

import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

from utils.credentials import get_cems_key, get_credentials_env

OBSERVER_DIR = Path.home() / ".cems" / "observer"
PID_FILE = OBSERVER_DIR / "daemon.pid"
COOLDOWN_FILE = OBSERVER_DIR / ".spawn_cooldown"
SPAWN_LOCK_FILE = OBSERVER_DIR / ".spawn.lock"

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


def _any_observer_process_exists() -> bool:
    """Fallback check: is ANY cems-observer process running?

    Uses pgrep to detect orphaned daemons that lost their PID file.
    This prevents spawning duplicates even when the PID file is missing.
    """
    try:
        # Match both "cems-observer" (installed) and "cems.observer" (python -m)
        result = subprocess.run(
            ["pgrep", "-f", "cems[.-]observer"],
            capture_output=True, timeout=2,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_daemon_running() -> bool:
    """Check if the observer daemon is currently running.

    Primary: reads PID file and verifies process exists.
    Fallback: uses pgrep to catch orphaned daemons without PID files.

    Returns:
        True if daemon is running, False otherwise.
    """
    pid = _read_pid()
    if pid is not None:
        if _is_process_alive(pid):
            return True
        # Stale PID file — process is dead
        try:
            PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    # Fallback: check for any cems-observer process (catches orphans)
    return _any_observer_process_exists()


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

    Pure predicate: returns True if enough time has passed since last check.
    The caller (ensure_daemon_running) is responsible for touching the PID
    file to reset the timer after a successful check.

    Called from user_prompts_submit to avoid overhead on every prompt.
    """
    try:
        if PID_FILE.exists():
            age = time.time() - PID_FILE.stat().st_mtime
            if age < HEALTH_CHECK_INTERVAL:
                return False
        return True
    except OSError:
        return True


def _find_observer_command() -> list[str] | None:
    """Find the best way to run the observer daemon.

    Tries in order:
    1. `cems-observer` command (installed via uv tool install / pip install)
    2. Source tree fallback via CEMS_PROJECT_ROOT env var (development only)

    Returns:
        Command list for subprocess.Popen, or None if not found.
    """
    # 1. Installed command (works after `uv tool install cems` or `pip install cems`)
    cems_observer_cmd = shutil.which("cems-observer")
    if cems_observer_cmd:
        return [cems_observer_cmd]

    # 2. Development fallback: source tree via env var
    cems_root = os.getenv("CEMS_PROJECT_ROOT", "")
    if cems_root:
        cems_path = Path(cems_root)
        if (cems_path / "src" / "cems" / "observer").exists():
            # Find a working python in the project venv
            for python_name in ("python3", "python"):
                venv_python = cems_path / ".venv" / "bin" / python_name
                if venv_python.exists():
                    return [str(venv_python), "-m", "cems.observer"]

    return None


def _spawn_daemon() -> bool:
    """Spawn the observer daemon as a detached background process.

    Uses a file lock to prevent two hooks from spawning simultaneously
    (which would cause a PID file race even though flock in __main__.py
    ensures only one daemon survives).

    Tries `cems-observer` command first (installed package), then falls
    back to source tree via CEMS_PROJECT_ROOT (development only).

    Returns:
        True if spawn succeeded, False otherwise.
    """
    # Check required credentials (env or ~/.cems/credentials)
    if not get_cems_key():
        return False

    cmd = _find_observer_command()
    if not cmd:
        return False

    try:
        OBSERVER_DIR.mkdir(parents=True, exist_ok=True)

        # Acquire spawn lock to prevent concurrent hook spawning
        lock_fd = os.open(str(SPAWN_LOCK_FILE), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(lock_fd)
            # Another hook is already spawning — check if daemon appeared
            time.sleep(0.5)
            return is_daemon_running()

        try:
            # Re-check after acquiring lock (daemon may have started)
            if is_daemon_running():
                return True

            # Build env for daemon: strip parent CEMS_API_KEY/URL so daemon
            # reads from ~/.cems/credentials (avoids inheriting stale session keys)
            spawn_env = {k: v for k, v in os.environ.items()
                         if k not in ("CEMS_API_KEY", "CEMS_API_URL")}

            # Spawn daemon as fully detached process
            # stdout/stderr go to a log file for debugging
            log_file = OBSERVER_DIR / "daemon.log"
            with open(log_file, "a") as logf:
                proc = subprocess.Popen(
                    cmd,
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

        finally:
            os.close(lock_fd)  # Releases flock

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
