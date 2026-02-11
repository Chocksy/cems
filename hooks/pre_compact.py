#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""
PreCompact Hook - Capture transcript before context compaction.

Fires when Claude Code auto-compacts (context window fills up).
Sends the full transcript to CEMS for learning extraction BEFORE
the transcript is lost to compaction.

Matcher: "auto" only (manual /compact is user-controlled).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.credentials import get_cems_key, get_cems_url
from utils.hook_logger import log_hook_event

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
    """Send transcript to CEMS API for analysis. Returns True on success."""
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


def main():
    try:
        input_data = json.load(sys.stdin)

        session_id = input_data.get("session_id", "")
        transcript_path = input_data.get("transcript_path", "")
        trigger = input_data.get("trigger", "unknown")
        cwd = input_data.get("cwd", os.getcwd())

        log_hook_event("PreCompact", session_id, {
            "trigger": trigger,
            "cwd": cwd,
        }, input_data=input_data)

        # Read and send transcript to CEMS for learning extraction
        if transcript_path:
            transcript = read_transcript(transcript_path)
            if transcript and len(transcript) > 2:
                project = get_project_id(cwd)
                success = analyze_session(
                    transcript=transcript,
                    session_id=session_id,
                    working_dir=cwd,
                    project=project,
                )
                log_hook_event("PreCompact:analyze", session_id, {
                    "success": success,
                    "message_count": len(transcript),
                })

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
