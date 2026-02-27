"""Session discovery and metadata extraction.

Finds active Claude Code sessions by scanning ~/.claude/projects/
and extracts project metadata from JSONL entries.
"""

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


@dataclass
class SessionInfo:
    """Metadata about a discovered session."""
    path: Path
    project_dir: str        # e.g., "-Users-razvan-Development-cems"
    session_id: str         # UUID from filename
    file_size: int = 0
    modified_at: float = 0.0
    cwd: str = ""
    git_branch: str = ""
    project_id: str | None = None   # e.g., "chocksy/cems"
    source_ref: str | None = None   # e.g., "project:chocksy/cems"


def discover_active_sessions(max_age_hours: int = 2) -> list[SessionInfo]:
    """Find sessions modified within the last N hours.

    Scans ~/.claude/projects/*/ for JSONL files (session transcripts)
    that have been modified recently.

    Args:
        max_age_hours: Only return sessions modified within this many hours.

    Returns:
        List of SessionInfo with basic file metadata populated.
    """
    if not CLAUDE_PROJECTS_DIR.exists():
        logger.debug(f"Claude projects dir not found: {CLAUDE_PROJECTS_DIR}")
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
                    project_dir=project_dir.name,
                    session_id=jsonl_file.stem,
                    file_size=stat.st_size,
                    modified_at=stat.st_mtime,
                ))
            except OSError:
                continue

    # Sort by modification time (most recent first)
    sessions.sort(key=lambda s: s.modified_at, reverse=True)
    return sessions


def populate_session_metadata(session: SessionInfo) -> SessionInfo:
    """Read the first JSONL entry to populate cwd, gitBranch, project_id.

    Mutates the session object in place and returns it.

    Args:
        session: SessionInfo with path set.

    Returns:
        The same SessionInfo with cwd, git_branch, project_id, source_ref populated.
    """
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
    """Extract project ID from git remote (e.g., 'org/repo').

    Same logic as hooks/stop.py:get_project_id().
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
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix('.git')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def read_content_delta(session: SessionInfo, from_byte: int) -> str | None:
    """Read new transcript content from a session file starting at byte offset.

    Extracts user messages, assistant text, and tool action summaries.
    Skips progress, system, file-history-snapshot, tool_result entries.

    Args:
        session: SessionInfo with path set.
        from_byte: Byte offset to start reading from.

    Returns:
        Formatted transcript text, or None if no new content.
    """
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
