"""Entry point for the CEMS Observer Daemon.

Usage:
    python -m cems.observer          # Run continuously
    python -m cems.observer --once   # Run one cycle and exit
"""

import argparse
import atexit
import fcntl
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

OBSERVER_DIR = Path.home() / ".claude" / "observer"
PID_FILE = OBSERVER_DIR / "daemon.pid"
LOCK_FILE = OBSERVER_DIR / "daemon.lock"


def _acquire_lock() -> "int | None":
    """Acquire an exclusive file lock to ensure only one daemon runs.

    Returns the lock file descriptor on success, None if another daemon
    already holds the lock.
    """
    OBSERVER_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        os.close(fd)
        return None


def _kill_stale_daemons() -> int:
    """Kill any pre-existing observer processes (except ourselves).

    This cleans up zombie daemons from before the flock singleton was added.
    Called after acquiring the lock, so we know WE are the rightful daemon.

    Returns:
        Number of stale processes killed.
    """
    my_pid = os.getpid()
    killed = 0
    try:
        # Find all observer processes (both cems-observer and python -m cems.observer)
        result = subprocess.run(
            ["pgrep", "-f", "cems[.-]observer"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                try:
                    pid = int(line.strip())
                    if pid != my_pid:
                        os.kill(pid, signal.SIGTERM)
                        killed += 1
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return killed


def _write_pid_file() -> None:
    """Write current PID to file so hooks can find us."""
    OBSERVER_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _cleanup_pid_file() -> None:
    """Remove PID file on clean exit."""
    try:
        if PID_FILE.exists():
            stored_pid = int(PID_FILE.read_text().strip())
            if stored_pid == os.getpid():
                PID_FILE.unlink()
    except (ValueError, OSError):
        pass


def main():
    parser = argparse.ArgumentParser(description="CEMS Observer Daemon")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Singleton enforcement: only one daemon can run at a time
    if not args.once:
        lock_fd = _acquire_lock()
        if lock_fd is None:
            print("Another cems-observer daemon is already running. Exiting.", file=sys.stderr)
            sys.exit(0)
        # Lock is held for the lifetime of the process (released on exit)

        # Kill any zombie daemons from before singleton enforcement
        killed = _kill_stale_daemons()
        if killed > 0:
            logger = logging.getLogger(__name__)
            logger.info(f"Killed {killed} stale observer daemon(s)")

    # Load credentials: env vars first, then ~/.cems/credentials fallback
    _creds_file = Path.home() / ".cems" / "credentials"
    _file_creds: dict[str, str] = {}
    try:
        if _creds_file.exists():
            for _line in _creds_file.read_text().splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    _k, _v = _k.strip(), _v.strip().strip("'\"")
                    if _k and _v:
                        _file_creds[_k] = _v
    except OSError:
        pass

    # Credentials file takes priority for daemon (env vars may be stale/session-specific)
    api_url = _file_creds.get("CEMS_API_URL") or os.getenv("CEMS_API_URL", "https://cems.chocksy.com")
    api_key = _file_creds.get("CEMS_API_KEY") or os.getenv("CEMS_API_KEY", "")

    if not api_key:
        print("Error: Set CEMS_API_KEY in ~/.cems/credentials or environment", file=sys.stderr)
        sys.exit(1)

    from cems.observer.daemon import run_cycle, run_daemon

    if args.once:
        triggered = run_cycle(api_url, api_key)
        print(f"Observations triggered: {triggered}")
    else:
        _write_pid_file()
        atexit.register(_cleanup_pid_file)
        # SIGTERM is handled in daemon.py — raises SystemExit(0)
        # which triggers atexit → _cleanup_pid_file()
        run_daemon(api_url, api_key)


if __name__ == "__main__":
    main()
