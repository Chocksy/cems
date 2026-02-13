"""Claude Code session adapter.

Discovers and extracts from Claude Code JSONL session files at:
    ~/.claude/projects/{project-dir}/{session-uuid}.jsonl
"""

import json
import logging
import re
import subprocess
import time
from pathlib import Path

from cems.observer.adapters.base import SessionInfo

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
        try:
            with open(session.path, "r") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return session
                entry = json.loads(first_line)

            session.cwd = entry.get("cwd", "")
            session.git_branch = entry.get("gitBranch", "")

            if session.cwd:
                session.project_id = _get_project_id(session.cwd)
                if session.project_id:
                    session.source_ref = f"project:{session.project_id}"

        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.debug(f"Could not read metadata from {session.path}: {e}")

        return session


def _get_project_id(cwd: str) -> str | None:
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
