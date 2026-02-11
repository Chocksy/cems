"""Entry point for the CEMS Observer Daemon.

Usage:
    python -m cems.observer          # Run continuously
    python -m cems.observer --once   # Run one cycle and exit
"""

import argparse
import atexit
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PID_FILE = Path.home() / ".claude" / "observer" / "daemon.pid"


def _write_pid_file() -> None:
    """Write current PID to file so hooks can find us."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _cleanup_pid_file() -> None:
    """Remove PID file on clean exit."""
    try:
        if PID_FILE.exists():
            # Only remove if it's our PID (another instance might have taken over)
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

    api_url = os.getenv("CEMS_API_URL", "https://cems.chocksy.com")
    api_key = os.getenv("CEMS_API_KEY", "")

    if not api_key:
        print("Error: CEMS_API_KEY environment variable is required", file=sys.stderr)
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
