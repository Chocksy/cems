#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS Stop Hook - Session Summary Storage

This hook runs when Claude stops and auto-stores a session summary to CEMS
if significant work (git commits) was done during the session.

Configuration (environment variables):
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta

# CEMS configuration from environment
CEMS_API_URL = os.getenv("CEMS_API_URL", "")
CEMS_API_KEY = os.getenv("CEMS_API_KEY", "")


def get_recent_commits(cwd: str, since_minutes: int = 120) -> list[dict]:
    """
    Get commits made in the last N minutes from the current working directory.
    Returns list of {hash, message, files} dicts.
    """
    try:
        since_time = (datetime.now() - timedelta(minutes=since_minutes)).strftime("%Y-%m-%d %H:%M:%S")

        result = subprocess.run(
            ['git', 'log', f'--since={since_time}', '--pretty=format:%H|%s', '--name-only'],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return []

        commits = []
        current_commit = None

        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                if current_commit:
                    commits.append(current_commit)
                hash_val, message = line.split('|', 1)
                current_commit = {
                    'hash': hash_val[:8],
                    'message': message,
                    'files': []
                }
            elif line.strip() and current_commit:
                current_commit['files'].append(line.strip())

        if current_commit:
            commits.append(current_commit)

        return commits

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return []


def store_cems(content: str, category: str = "sessions") -> bool:
    """Store memory using CEMS API."""
    if not CEMS_API_URL or not CEMS_API_KEY:
        return False

    try:
        data = json.dumps({
            "content": content,
            "category": category,
            "scope": "personal"
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{CEMS_API_URL}/api/memory/add",
            data=data,
            headers={
                "Authorization": f"Bearer {CEMS_API_KEY}",
                "Content-Type": "application/json"
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def store_session_summary(session_id: str, commits: list[dict]) -> bool:
    """
    Store session summary to CEMS.
    Returns True if stored successfully.
    """
    if not commits:
        return False

    # Build summary
    commit_summaries = []
    for c in commits[:5]:  # Limit to 5 most recent
        files_str = ', '.join(c['files'][:3])
        if len(c['files']) > 3:
            files_str += f" (+{len(c['files']) - 3} more)"
        commit_summaries.append(f"- {c['message']} ({files_str})")

    summary = f"Session {session_id[:8]} commits:\n" + '\n'.join(commit_summaries)

    return store_cems(summary, "sessions")


def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        session_id = input_data.get("session_id", "")
        cwd = input_data.get("cwd", os.getcwd())

        # Auto-store session summary to CEMS if commits were made
        commits = get_recent_commits(cwd, since_minutes=120)
        if commits:
            store_session_summary(session_id, commits)

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
