#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///

"""
CEMS Session Start Hook - Memory Context Injection

This hook runs when a Cursor session starts and injects relevant CEMS memories
as additional context for the session.

Configuration (environment variables):
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
"""

import json
import os
import sys

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


def get_project_context() -> str:
    """
    Get context about the current project from working directory.
    """
    cwd = os.getcwd()
    project_name = os.path.basename(cwd)
    
    # Try to get more context from git
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            remote_url = result.stdout.strip()
            # Extract repo name from URL
            repo_name = remote_url.split('/')[-1].replace('.git', '')
            return f"{repo_name} ({project_name})"
    except Exception:
        pass
    
    return project_name


def search_cems(query: str, limit: int = 5) -> list[dict] | None:
    """
    Search CEMS for relevant memories.
    Returns list of memories or None if search fails.
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return None

    if not query or len(query) < 3:
        return None

    try:
        response = httpx.post(
            f"{CEMS_API_URL}/api/memory/search",
            json={"query": query, "limit": limit, "scope": "both"},
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=5.0,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        if not data.get("success") or not data.get("results"):
            return None

        return data["results"]

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return None


def format_memories(memories: list[dict]) -> str:
    """Format memories for injection as context."""
    if not memories:
        return ""
    
    lines = ["## Relevant CEMS Memories", ""]
    for i, m in enumerate(memories, 1):
        content = m.get("content", m.get("memory", ""))
        category = m.get("category", "general")
        mem_id = m.get("memory_id", m.get("id", ""))[:8] if m.get("memory_id") or m.get("id") else ""
        lines.append(f"{i}. [{category}] {content}")
        if mem_id:
            lines.append(f"   (memory_id: {mem_id})")
    
    lines.append("")
    lines.append("Use these memories as context. Call memory_search for more specific queries.")
    
    return "\n".join(lines)


def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        session_id = input_data.get("session_id", "")
        is_background = input_data.get("is_background_agent", False)
        
        # Skip for background agents
        if is_background:
            print(json.dumps({"continue": True}))
            return
        
        # Get project context for search
        project_context = get_project_context()
        
        # Search for relevant memories
        memories = search_cems(project_context, limit=5)
        
        # Build response
        response = {
            "continue": True,
            "env": {
                "CEMS_SESSION_ID": session_id,
            },
        }
        
        # Add memory context if found
        if memories:
            context = format_memories(memories)
            response["additional_context"] = context
        
        print(json.dumps(response))

    except json.JSONDecodeError:
        # On error, allow session to continue
        print(json.dumps({"continue": True}))
    except Exception as e:
        # Log error but don't block session
        print(json.dumps({"continue": True}), file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
