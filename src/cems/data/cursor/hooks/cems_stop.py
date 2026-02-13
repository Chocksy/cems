#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS Stop Hook - Session Analysis and Learning Extraction

This hook runs when a Cursor session ends. It:
1. Reads the accumulated transcript from afterAgentResponse hook
2. Gets recent git commits
3. Calls CEMS session_analyze to extract and store learnings
4. Cleans up the transcript file

Configuration (environment variables):
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
  CEMS_SESSION_ID - Session ID (set by sessionStart hook)
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta

def _read_credentials() -> tuple[str, str]:
    """Read CEMS credentials from env vars, falling back to ~/.cems/credentials."""
    url = os.getenv("CEMS_API_URL", "")
    key = os.getenv("CEMS_API_KEY", "")
    if url and key:
        return url, key
    try:
        creds_file = os.path.join(os.path.expanduser("~"), ".cems", "credentials")
        if os.path.exists(creds_file):
            with open(creds_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, _, v = line.partition("=")
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k == "CEMS_API_URL" and not url:
                            url = v
                        elif k == "CEMS_API_KEY" and not key:
                            key = v
    except OSError:
        pass
    return url, key


# CEMS configuration from environment or credentials file
CEMS_API_URL, CEMS_API_KEY = _read_credentials()

# Transcript storage location (must match cems_agent_response.py)
TRANSCRIPT_DIR = "/tmp/cems-transcripts"


def get_transcript_path(session_id: str) -> str:
    """Get the path to the transcript file for a session."""
    return os.path.join(TRANSCRIPT_DIR, f"{session_id}.jsonl")


def read_transcript(session_id: str) -> str:
    """Read the accumulated transcript for a session."""
    transcript_path = get_transcript_path(session_id)
    
    if not os.path.exists(transcript_path):
        return ""
    
    try:
        lines = []
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    text = entry.get("text", "")
                    timestamp = entry.get("timestamp", "")
                    lines.append(f"[{timestamp}] Assistant:\n{text}\n")
                except json.JSONDecodeError:
                    continue
        
        return "\n---\n".join(lines)
    except IOError:
        return ""


def cleanup_transcript(session_id: str):
    """Remove the transcript file after processing."""
    transcript_path = get_transcript_path(session_id)
    try:
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
    except IOError:
        pass


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


def format_commits(commits: list[dict]) -> str:
    """Format commits for the transcript."""
    if not commits:
        return ""
    
    lines = ["## Git Commits This Session"]
    for c in commits[:10]:
        files_str = ', '.join(c['files'][:5])
        if len(c['files']) > 5:
            files_str += f" (+{len(c['files']) - 5} more)"
        lines.append(f"- {c['hash']}: {c['message']}")
        if files_str:
            lines.append(f"  Files: {files_str}")
    
    return '\n'.join(lines)


def call_session_analyze(transcript: str, session_id: str, working_dir: str) -> bool:
    """Call CEMS session_analyze endpoint."""
    if not CEMS_API_URL or not CEMS_API_KEY:
        return False
    
    if not transcript or len(transcript) < 100:
        return False
    
    try:
        data = json.dumps({
            "transcript": transcript,
            "session_id": session_id,
            "working_dir": working_dir,
        }).encode('utf-8')

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


def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        status = input_data.get("status", "")
        
        # Get session ID from environment
        session_id = os.getenv("CEMS_SESSION_ID", "")
        cwd = os.getcwd()
        
        if not session_id:
            # Output empty response
            print(json.dumps({}))
            sys.exit(0)
        
        # Read accumulated transcript
        transcript = read_transcript(session_id)
        
        # Get recent commits
        commits = get_recent_commits(cwd)
        commits_text = format_commits(commits)
        
        # Combine transcript and commits
        full_transcript = transcript
        if commits_text:
            full_transcript = f"{transcript}\n\n{commits_text}"
        
        # Call session_analyze if we have content
        if full_transcript and len(full_transcript) > 100:
            call_session_analyze(full_transcript, session_id, cwd)
        
        # Clean up transcript file
        cleanup_transcript(session_id)
        
        # Output response (no followup needed)
        print(json.dumps({}))
        sys.exit(0)

    except json.JSONDecodeError:
        print(json.dumps({}))
        sys.exit(0)
    except Exception:
        print(json.dumps({}))
        sys.exit(0)


if __name__ == "__main__":
    main()
