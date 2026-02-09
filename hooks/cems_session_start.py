#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///

"""
CEMS SessionStart Hook - Profile Injection

Injects user profile context at session start:
1. User preferences and guidelines
2. Recent relevant memories (last 24h)
3. Gate rules summary
4. Project-specific context

Configuration:
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import httpx

CEMS_API_URL = os.getenv("CEMS_API_URL", "")
CEMS_API_KEY = os.getenv("CEMS_API_KEY", "")


def get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo').

    Parses the git remote origin URL to extract the org/repo identifier.
    Works with both SSH and HTTPS formats.

    Args:
        cwd: Current working directory (project root)

    Returns:
        Project ID like 'org/repo' or None if not a git repo
    """
    if not cwd:
        return None

    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # SSH: git@github.com:org/repo.git -> org/repo
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                # HTTPS: https://github.com/org/repo.git -> org/repo
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix('.git')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def fetch_profile(project: str | None = None, token_budget: int = 2500) -> dict:
    """Fetch profile context from CEMS /api/memory/profile endpoint.

    Args:
        project: Optional project ID (org/repo) for project-scoped context
        token_budget: Maximum tokens for context (default 2500)

    Returns:
        Profile dict with 'context' field, or empty dict on error
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return {}

    try:
        params = {"token_budget": str(token_budget)}
        if project:
            params["project"] = project

        response = httpx.get(
            f"{CEMS_API_URL}/api/memory/profile",
            params=params,
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=8.0,  # Allow more time for profile aggregation
        )

        if response.status_code == 200:
            return response.json()
        return {}
    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError) as e:
        print(f"CEMS profile fetch error: {e}", file=sys.stderr)
        return {}


def main():
    try:
        input_data = json.load(sys.stdin)
        session_id = input_data.get("session_id", "")
        source = input_data.get("source", "startup")
        is_background_agent = input_data.get("is_background_agent", False)
        cwd = input_data.get("cwd", "")

        # Skip for background agents and resume/compact (avoid redundant injection)
        if is_background_agent or source in ("resume", "compact"):
            sys.exit(0)

        # Skip if CEMS is not configured
        if not CEMS_API_URL or not CEMS_API_KEY:
            sys.exit(0)

        project = get_project_id(cwd) if cwd else None
        profile = fetch_profile(project)

        if not profile.get("success") or not profile.get("context"):
            sys.exit(0)

        # Format context for injection
        context = f"""<cems-profile>
{profile['context']}
</cems-profile>"""

        # SessionStart: stdout text is added as context for Claude
        # Use hookSpecificOutput for structured injection
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context
            }
        }))

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as e:
        print(f"CEMS SessionStart error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
