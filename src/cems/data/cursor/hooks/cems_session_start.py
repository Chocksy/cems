#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///

"""
CEMS Session Start Hook - Profile + Foundation Injection (Cursor)

Injects user context at session start:
1. User preferences and guidelines (via /api/memory/profile)
2. Foundation guidelines (via /api/memory/foundation, cached 15min)
3. Project-specific context

Configuration (environment variables or ~/.cems/credentials):
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx


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


def fetch_profile(project: str | None = None, token_budget: int = 2500) -> dict:
    """Fetch profile context from CEMS /api/memory/profile endpoint."""
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
            timeout=8.0,
        )

        if response.status_code == 200:
            return response.json()
        return {}
    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return {}


# =============================================================================
# Foundation Guidelines Cache
# =============================================================================

FOUNDATION_CACHE_DIR = Path.home() / ".cems" / "cache" / "foundation"
FOUNDATION_CACHE_TTL = 900  # 15 minutes


def _get_foundation_cache_path(project: str | None) -> Path:
    """Get cache file path for foundation guidelines."""
    FOUNDATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if project:
        safe_name = project.replace("/", "_").replace("\\", "_")
        return FOUNDATION_CACHE_DIR / f"{safe_name}.json"
    return FOUNDATION_CACHE_DIR / "global.json"


def fetch_foundation(project: str | None = None) -> list[dict]:
    """Fetch foundation guidelines from CEMS, with local cache.

    Uses /api/memory/foundation endpoint. Caches locally for 15 minutes.
    On server error, falls back to stale cache if available.
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return []

    cache_path = _get_foundation_cache_path(project)

    # Return cached if fresh
    if cache_path.exists():
        try:
            age = cache_path.stat().st_mtime
            if time.time() - age < FOUNDATION_CACHE_TTL:
                return json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    # Fetch from API
    try:
        url = f"{CEMS_API_URL}/api/memory/foundation"
        if project:
            url += f"?project={project}"

        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=5.0,
        )

        if response.status_code != 200:
            if cache_path.exists():
                try:
                    return json.loads(cache_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            return []

        data = response.json()
        guidelines = data.get("guidelines", [])

        try:
            cache_path.write_text(json.dumps(guidelines, indent=2))
        except OSError:
            pass

        return guidelines

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return []


def format_foundation(guidelines: list[dict]) -> str:
    """Format foundation guidelines for context injection."""
    if not guidelines:
        return ""
    lines = []
    for g in guidelines:
        content = g.get("content", "")
        if content:
            lines.append(f"- {content}")
    if not lines:
        return ""
    return "## Foundation Guidelines\n" + "\n".join(lines)


def main():
    try:
        input_data = json.load(sys.stdin)

        session_id = input_data.get("session_id", "")
        is_background = input_data.get("is_background_agent", False)

        # Skip for background agents
        if is_background:
            print(json.dumps({"continue": True}))
            return

        # Skip if CEMS is not configured
        if not CEMS_API_URL or not CEMS_API_KEY:
            print(json.dumps({"continue": True}))
            return

        cwd = os.getcwd()
        project = get_project_id(cwd)
        profile = fetch_profile(project)
        foundation = fetch_foundation(project)

        context_parts = []

        if profile.get("success") and profile.get("context"):
            context_parts.append(f"""<cems-profile>
{profile['context']}
</cems-profile>""")

        if foundation:
            foundation_text = format_foundation(foundation)
            if foundation_text:
                context_parts.append(f"""<cems-foundation>
{foundation_text}

These are foundational principles. Follow them throughout this session.
</cems-foundation>""")

        response = {
            "continue": True,
            "env": {
                "CEMS_SESSION_ID": session_id,
            },
        }

        if context_parts:
            response["additional_context"] = "\n\n".join(context_parts)

        print(json.dumps(response))

    except json.JSONDecodeError:
        print(json.dumps({"continue": True}))
    except Exception:
        print(json.dumps({"continue": True}))
        sys.exit(0)


if __name__ == "__main__":
    main()
