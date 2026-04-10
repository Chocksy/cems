"""Auto-detect current session ID from Claude Code or Codex context.

Used by MCP server and CLI to automatically enrich search queries
with session context — no manual input needed.

Resolver ladder:
1. CEMS_SESSION_ID env var (explicit override / testing)
2. Claude Code PPID session file (~/.claude/sessions/{ppid}.json)
3. CODEX_COMPANION_SESSION_ID env var
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def detect_session_id() -> str:
    """Auto-detect current session ID.

    Returns session UUID string, or empty string if not detectable.
    """
    # 1. Explicit override (testing / manual)
    sid = os.environ.get("CEMS_SESSION_ID", "")
    if sid:
        return sid

    # 2. Claude Code: each session spawns its own MCP/CLI process
    # PPID → ~/.claude/sessions/{ppid}.json → sessionId
    session_file = Path.home() / ".claude" / "sessions" / f"{os.getppid()}.json"
    try:
        if session_file.exists():
            data = json.loads(session_file.read_text())
            sid = data.get("sessionId", "")
            if sid:
                return sid
    except (json.JSONDecodeError, OSError):
        pass

    # 3. Codex companion
    return os.environ.get("CODEX_COMPANION_SESSION_ID", "")
