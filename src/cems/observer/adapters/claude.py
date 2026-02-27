"""Claude Code session adapter.

Discovers and extracts from Claude Code JSONL session files at:
    ~/.claude/projects/{project-dir}/{session-uuid}.jsonl
"""

import logging
import time
from pathlib import Path

from cems.observer.adapters.base import SessionInfo
from cems.observer.session import populate_session_metadata

logger = logging.getLogger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


class ClaudeAdapter:
    """Adapter for Claude Code session transcripts."""

    tool_name = "claude"

    def discover_sessions(self, max_age_hours: int = 2) -> list[SessionInfo]:
        """Find Claude Code sessions modified within the last N hours."""
        if not CLAUDE_PROJECTS_DIR.exists():
            return []

        cutoff = time.time() - (max_age_hours * 3600)
        sessions = []

        for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue

            for jsonl_file in project_dir.glob("*.jsonl"):
                try:
                    stat = jsonl_file.stat()
                    if stat.st_mtime < cutoff:
                        continue

                    sessions.append(SessionInfo(
                        path=jsonl_file,
                        session_id=jsonl_file.stem,
                        tool="claude",
                        file_size=stat.st_size,
                        modified_at=stat.st_mtime,
                        extra={"project_dir": project_dir.name},
                    ))
                except OSError:
                    continue

        sessions.sort(key=lambda s: s.modified_at, reverse=True)
        return sessions

    def extract_text(self, session: SessionInfo, from_byte: int) -> str | None:
        """Extract transcript text from Claude Code JSONL starting at byte offset."""
        from cems.observer.transcript import extract_transcript_from_bytes

        try:
            file_size = session.path.stat().st_size
            if file_size <= from_byte:
                return None

            with open(session.path, "rb") as f:
                f.seek(from_byte)
                new_bytes = f.read()

            return extract_transcript_from_bytes(new_bytes, compact=True)

        except OSError as e:
            logger.debug(f"Could not read delta from {session.path}: {e}")
            return None

    def enrich_metadata(self, session: SessionInfo) -> SessionInfo:
        """Read the first JSONL entry to extract cwd, git_branch, project_id."""
        return populate_session_metadata(session)
