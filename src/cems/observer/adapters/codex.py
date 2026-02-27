"""Codex CLI session adapter.

Discovers and extracts from Codex CLI JSONL session files at:
    ~/.codex/sessions/YYYY/MM/DD/rollout-{timestamp}-{uuid}.jsonl
"""

import json
import logging
import re
import time
from pathlib import Path

from cems.observer.adapters.base import SessionInfo

logger = logging.getLogger(__name__)

CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"

# UUID pattern in Codex filenames: rollout-{timestamp}-{uuid}.jsonl
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class CodexAdapter:
    """Adapter for Codex CLI session transcripts."""

    tool_name = "codex"

    def discover_sessions(self, max_age_hours: int = 2) -> list[SessionInfo]:
        """Find Codex CLI sessions modified within the last N hours."""
        if not CODEX_SESSIONS_DIR.exists():
            return []

        cutoff = time.time() - (max_age_hours * 3600)
        sessions = []

        for jsonl_file in CODEX_SESSIONS_DIR.rglob("*.jsonl"):
            try:
                stat = jsonl_file.stat()
                if stat.st_mtime < cutoff:
                    continue

                session_id = _extract_session_id(jsonl_file.name)
                if not session_id:
                    continue

                sessions.append(SessionInfo(
                    path=jsonl_file,
                    session_id=session_id,
                    tool="codex",
                    file_size=stat.st_size,
                    modified_at=stat.st_mtime,
                ))
            except OSError:
                continue

        sessions.sort(key=lambda s: s.modified_at, reverse=True)
        return sessions

    def extract_text(self, session: SessionInfo, from_byte: int) -> str | None:
        """Extract transcript text from Codex JSONL starting at byte offset."""
        from cems.observer.codex_transcript import extract_codex_transcript_from_bytes

        try:
            file_size = session.path.stat().st_size
            if file_size <= from_byte:
                return None

            with open(session.path, "rb") as f:
                f.seek(from_byte)
                new_bytes = f.read()

            return extract_codex_transcript_from_bytes(new_bytes)

        except OSError as e:
            logger.debug(f"Could not read delta from {session.path}: {e}")
            return None

    def enrich_metadata(self, session: SessionInfo) -> SessionInfo:
        """Read session_meta from first JSONL line for cwd, git info."""
        try:
            with open(session.path, "r") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return session
                record = json.loads(first_line)

            # New format: session_meta with payload
            if record.get("type") == "session_meta":
                payload = record.get("payload", {})
                session.cwd = payload.get("cwd", "")

                git_info = payload.get("git", {})
                if git_info:
                    session.git_branch = git_info.get("branch", "")
                    repo_url = git_info.get("repository_url", "")
                    if repo_url:
                        project_id = _extract_project_from_url(repo_url)
                        if project_id:
                            session.project_id = project_id
                            session.source_ref = f"project:{project_id}"

            # Old format: direct fields
            elif "cwd" in record or "git" in record:
                session.cwd = record.get("cwd", "")
                git_info = record.get("git", {})
                if git_info:
                    session.git_branch = git_info.get("branch", "")

        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.debug(f"Could not read metadata from {session.path}: {e}")

        return session


def _extract_session_id(filename: str) -> str | None:
    """Extract UUID from Codex filename.

    Example: rollout-2026-02-05T10-36-02-019c2cf1-aed9-7560-933d-874296a5e2a7.jsonl
    """
    match = _UUID_RE.search(filename)
    return match.group(0) if match else None


def _extract_project_from_url(url: str) -> str | None:
    """Extract org/repo from a git URL."""
    if not url:
        return None
    # git@github.com:org/repo.git
    if url.startswith("git@"):
        match = re.search(r":(.+?)(?:\.git)?$", url)
        if match:
            return match.group(1)
    # https://github.com/org/repo.git
    match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    if match:
        return match.group(1)
    return None
