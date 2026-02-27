"""Cursor IDE session adapter.

Discovers and extracts from Cursor IDE agent transcript files at:
    ~/.cursor/projects/{project}/agent-transcripts/{uuid}.txt
"""

import logging
import re
import time
from pathlib import Path

from cems.observer.adapters.base import SessionInfo

logger = logging.getLogger(__name__)

CURSOR_PROJECTS_DIR = Path.home() / ".cursor" / "projects"

# UUID pattern for Cursor transcript filenames
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.txt$")


class CursorAdapter:
    """Adapter for Cursor IDE agent transcripts."""

    tool_name = "cursor"

    def discover_sessions(self, max_age_hours: int = 2) -> list[SessionInfo]:
        """Find Cursor IDE sessions modified within the last N hours."""
        if not CURSOR_PROJECTS_DIR.exists():
            return []

        cutoff = time.time() - (max_age_hours * 3600)
        sessions = []

        for project_dir in CURSOR_PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue

            transcripts_dir = project_dir / "agent-transcripts"
            if not transcripts_dir.exists():
                continue

            for txt_file in transcripts_dir.glob("*.txt"):
                if not _UUID_RE.match(txt_file.name):
                    continue

                try:
                    stat = txt_file.stat()
                    if stat.st_mtime < cutoff:
                        continue

                    session_id = txt_file.stem  # UUID without .txt
                    sessions.append(SessionInfo(
                        path=txt_file,
                        session_id=session_id,
                        tool="cursor",
                        file_size=stat.st_size,
                        modified_at=stat.st_mtime,
                        extra={"project_dir": project_dir.name},
                    ))
                except OSError:
                    continue

        sessions.sort(key=lambda s: s.modified_at, reverse=True)
        return sessions

    def extract_text(self, session: SessionInfo, from_byte: int) -> str | None:
        """Extract transcript text from Cursor transcript starting at byte offset."""
        from cems.observer.cursor_transcript import extract_cursor_transcript_from_bytes

        try:
            file_size = session.path.stat().st_size
            if file_size <= from_byte:
                return None

            with open(session.path, "rb") as f:
                f.seek(from_byte)
                new_bytes = f.read()

            return extract_cursor_transcript_from_bytes(new_bytes)

        except OSError as e:
            logger.debug(f"Could not read delta from {session.path}: {e}")
            return None

    def enrich_metadata(self, session: SessionInfo) -> SessionInfo:
        """Derive project context from directory structure.

        Cursor transcripts don't contain git info directly, so we
        derive the project name from the parent directory. Cursor encodes
        the full path as the directory name, replacing "/" with "-".

        Note: This is best-effort — paths with hyphens in directory names
        (e.g., "my-project") will be misinterpreted. We validate by
        checking if the reconstructed path exists on disk.
        """
        project_dir_name = session.extra.get("project_dir", "")
        if project_dir_name:
            # Try to reconstruct path: "Users-razvan-Development-cems" → "/Users/razvan/Development/cems"
            candidate = Path("/" + project_dir_name.replace("-", "/"))
            if candidate.is_dir():
                session.cwd = str(candidate)
                session.project_id = candidate.name
            else:
                # Fallback: use the raw directory name as-is
                session.cwd = project_dir_name
                parts = project_dir_name.rsplit("-", 1)
                if parts:
                    session.project_id = parts[-1]

        return session
