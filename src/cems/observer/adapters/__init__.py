"""Multi-tool session adapters for the observer daemon.

Each adapter knows how to discover sessions, extract transcript text,
and enrich metadata for a specific tool (Claude Code, Codex CLI, Cursor IDE).
"""

from cems.observer.adapters.base import SessionAdapter, SessionInfo
from cems.observer.adapters.claude import ClaudeAdapter
from cems.observer.adapters.codex import CodexAdapter
from cems.observer.adapters.cursor import CursorAdapter

__all__ = [
    "SessionAdapter",
    "SessionInfo",
    "ClaudeAdapter",
    "CodexAdapter",
    "CursorAdapter",
    "get_adapters",
]


def get_adapters() -> list[SessionAdapter]:
    """Return all available adapters."""
    return [ClaudeAdapter(), CodexAdapter(), CursorAdapter()]
