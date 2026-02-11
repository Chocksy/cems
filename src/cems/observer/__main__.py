"""Entry point for the CEMS Observer Daemon.

Usage:
    python -m cems.observer          # Run continuously
    python -m cems.observer --once   # Run one cycle and exit
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


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
        run_daemon(api_url, api_key)


if __name__ == "__main__":
    main()
