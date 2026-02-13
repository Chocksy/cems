#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""
Stop Hook - Session End Handler

This hook runs when Claude stops and:
1. Logs session data (stop.json, optional chat.json)
2. Writes a "stop" signal for the observer daemon to finalize the session summary
3. Sends transcript to CEMS for learning extraction (separate from summarization)

The observer daemon handles session summarization via the signal — this hook
no longer does its own transcript extraction or summarization.
"""

import argparse
import json
import os
import re
import sys
import subprocess
import time
from pathlib import Path

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import ensure_session_log_dir
from utils.credentials import get_cems_key, get_cems_url
from utils.hook_logger import log_hook_event

# CEMS configuration — env vars first, then ~/.cems/credentials fallback
CEMS_API_URL = get_cems_url()
CEMS_API_KEY = get_cems_key()


def get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo')."""
    if not cwd:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix('.git')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def read_transcript(transcript_path: str) -> list[dict] | None:
    """Read transcript from .jsonl file and return as list of messages."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        messages = []
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return messages if messages else None
    except Exception:
        return None


def analyze_session(
    transcript: list[dict],
    session_id: str,
    working_dir: str | None = None,
    project: str | None = None,
) -> bool:
    """Send transcript to CEMS API for learning extraction.

    Returns True if the API call succeeded.
    """
    if not CEMS_API_KEY:
        return False

    try:
        import urllib.request
        import urllib.error

        payload = {
            "transcript": transcript,
            "session_id": session_id,
            "working_dir": working_dir,
        }
        if project:
            payload["source_ref"] = f"project:{project}"

        data = json.dumps(payload).encode('utf-8')

        req = urllib.request.Request(
            f"{CEMS_API_URL}/api/session/analyze",
            data=data,
            headers={
                "Authorization": f"Bearer {CEMS_API_KEY}",
                "Content-Type": "application/json"
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status == 200

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False
    except Exception:
        return False


def write_signal(session_id: str, signal_type: str, tool: str = "claude") -> None:
    """Write a signal file for the observer daemon to pick up.

    Inlined from cems.observer.signals — hooks run standalone via uv and
    cannot import from the cems package.
    """
    signals_dir = Path.home() / ".claude" / "observer" / "signals"
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
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--chat", action="store_true", help="Copy transcript to chat.json"
        )
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract required fields
        session_id = input_data.get("session_id", "")
        transcript_path = input_data.get("transcript_path", "")
        cwd = input_data.get("cwd", os.getcwd())

        log_hook_event("Stop", session_id, {"cwd": cwd}, input_data=input_data)

        # --- Session logging ---
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "stop.json"

        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        log_data.append(input_data)

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Handle --chat switch
        if args.chat and transcript_path:
            transcript = read_transcript(transcript_path)
            if transcript:
                chat_file = log_dir / "chat.json"
                with open(chat_file, "w") as f:
                    json.dump(transcript, f, indent=2)

        # --- Signal the observer daemon to finalize the session summary ---
        if session_id:
            write_signal(session_id, "stop", "claude")

        # --- Learning extraction (daemon doesn't handle this) ---
        if transcript_path:
            transcript = read_transcript(transcript_path)
            if transcript and len(transcript) > 2:
                project = get_project_id(cwd)
                analyze_session(
                    transcript=transcript,
                    session_id=session_id,
                    working_dir=cwd,
                    project=project,
                )

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
