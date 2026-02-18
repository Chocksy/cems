"""Base adapter protocol for multi-tool session discovery and extraction.

Each tool (Claude Code, Codex CLI, Cursor IDE) implements this protocol
to let the daemon discover sessions, extract transcript text, and enrich
metadata in a uniform way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class SessionInfo:
    """Metadata about a discovered session, tool-agnostic."""
    path: Path
    session_id: str
    tool: str                      # "claude" | "codex" | "cursor" | "goose"
    file_size: int = 0
    modified_at: float = 0.0
    # Enriched fields (populated by adapter.enrich_metadata)
    cwd: str = ""
    git_branch: str = ""
    project_id: str | None = None      # e.g., "chocksy/cems"
    source_ref: str | None = None      # e.g., "project:chocksy/cems"
    # Tool-specific extra context
    extra: dict = field(default_factory=dict)


class SessionAdapter(Protocol):
    """Interface each tool adapter implements."""

    tool_name: str  # "claude" | "codex" | "cursor" | "goose"

    def discover_sessions(self, max_age_hours: int = 2) -> list[SessionInfo]:
        """Find active sessions for this tool.

        Args:
            max_age_hours: Only return sessions modified within this many hours.

        Returns:
            List of SessionInfo with basic file metadata populated.
        """
        ...

    def extract_text(self, session: SessionInfo, from_byte: int) -> str | None:
        """Extract transcript text starting from a byte offset.

        Args:
            session: Session to read from.
            from_byte: Byte offset to start reading.

        Returns:
            Formatted transcript text, or None if no new content.
        """
        ...

    def enrich_metadata(self, session: SessionInfo) -> SessionInfo:
        """Populate cwd, git_branch, project_id, source_ref from session data.

        Mutates and returns the session object.

        Args:
            session: SessionInfo with path set.

        Returns:
            The same SessionInfo with enriched metadata.
        """
        ...
