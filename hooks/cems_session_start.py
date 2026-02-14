#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///

"""
CEMS SessionStart Hook - Profile + Foundation Injection

Injects user context at session start:
1. User preferences and guidelines (via /api/memory/profile)
2. Foundation guidelines (via /api/memory/foundation, cached 15min)
3. Recent relevant memories (last 24h)
4. Gate rules summary
5. Project-specific context

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
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from utils.credentials import get_cems_key, get_cems_url
from utils.hook_logger import log_hook_event

CEMS_API_URL = get_cems_url()
CEMS_API_KEY = get_cems_key()


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

    Returns:
        List of foundation guideline dicts with 'content' and 'tags'
    """
    import time

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
            # On error, return stale cache if available
            if cache_path.exists():
                try:
                    return json.loads(cache_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            return []

        data = response.json()
        guidelines = data.get("guidelines", [])

        # Write cache (even if empty — avoids re-fetching when no guidelines exist)
        try:
            cache_path.write_text(json.dumps(guidelines, indent=2))
        except OSError:
            pass

        return guidelines

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        # On network error, return stale cache if available
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


AUTO_UPDATE_INTERVAL = 300  # 5 minutes — background no-op if already latest


def _maybe_auto_update() -> None:
    """Check if CEMS should auto-update (every 5min, background, non-blocking).

    Checks ~/.cems/.last_update_check timestamp. If older than 5min,
    spawns `cems update` in the background. Respects CEMS_AUTO_UPDATE=0.
    """
    import shutil
    import time

    # Check if auto-update is disabled
    auto_update = os.environ.get("CEMS_AUTO_UPDATE", "").strip()
    if auto_update == "0":
        return

    # Also check credentials file for the setting
    if not auto_update:
        creds_file = Path.home() / ".cems" / "credentials"
        try:
            if creds_file.exists():
                for line in creds_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("CEMS_AUTO_UPDATE="):
                        val = line.partition("=")[2].strip().strip("'\"")
                        if val == "0":
                            return
        except OSError:
            pass

    marker = Path.home() / ".cems" / ".last_update_check"
    now = time.time()

    # Check if we updated recently
    try:
        if marker.exists():
            last_check = marker.stat().st_mtime
            if now - last_check < AUTO_UPDATE_INTERVAL:
                return  # Too soon
    except OSError:
        pass

    # Touch the marker BEFORE updating (prevents concurrent updates)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(int(now)))
    except OSError:
        return

    # Find cems binary
    cems_bin = shutil.which("cems")
    if not cems_bin:
        return

    # Spawn update in background (completely detached, fire-and-forget)
    try:
        subprocess.Popen(
            [cems_bin, "update"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        pass


def main():
    try:
        input_data = json.load(sys.stdin)
        session_id = input_data.get("session_id", "")
        source = input_data.get("source", "startup")
        is_background_agent = input_data.get("is_background_agent", False)
        cwd = input_data.get("cwd", "")

        log_hook_event("SessionStart", session_id, {
            "source": source,
            "is_background_agent": is_background_agent,
        }, input_data=input_data)

        # Skip for background agents and resume (avoid redundant injection)
        if is_background_agent or source == "resume":
            sys.exit(0)

        # Skip if CEMS is not configured
        if not CEMS_API_URL or not CEMS_API_KEY:
            sys.exit(0)

        # Auto-update check (background, non-blocking)
        try:
            _maybe_auto_update()
        except Exception:
            pass  # Auto-update is best-effort, never block session start

        # Ensure observer daemon is running (spawns if dead)
        try:
            from utils.observer_manager import ensure_daemon_running
            ensure_daemon_running(force_check=True)
        except Exception:
            pass  # Observer is nice-to-have, never block session start

        project = get_project_id(cwd) if cwd else None
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

        if not context_parts:
            sys.exit(0)

        context = "\n\n".join(context_parts)

        # SessionStart: use hookSpecificOutput for structured injection
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
