"""Goose session adapter.

Discovers and extracts from Goose's SQLite session database at:
    ~/.local/share/goose/sessions/sessions.db
"""

import json
import logging
import re
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cems.observer.adapters.base import SessionInfo

logger = logging.getLogger(__name__)

GOOSE_DB_PATH = Path.home() / ".local" / "share" / "goose" / "sessions" / "sessions.db"


class GooseAdapter:
    """Adapter for Goose session transcripts stored in SQLite."""

    tool_name = "goose"

    def discover_sessions(self, max_age_hours: int = 2) -> list[SessionInfo]:
        """Find Goose sessions modified within the last N hours."""
        if not GOOSE_DB_PATH.exists():
            return []

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        sessions = []

        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{GOOSE_DB_PATH}?mode=ro", uri=True, timeout=5
            )
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT s.id, s.working_dir, s.updated_at,
                       SUM(LENGTH(m.content_json)) as total_content_size,
                       MAX(m.id) as max_message_id
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.updated_at > ?
                GROUP BY s.id
                """,
                (cutoff,),
            ).fetchall()

            for row in rows:
                try:
                    updated_at = datetime.fromisoformat(row["updated_at"])
                    modified_ts = updated_at.timestamp()
                except (ValueError, TypeError):
                    modified_ts = 0.0

                sessions.append(
                    SessionInfo(
                        path=Path(GOOSE_DB_PATH),
                        session_id=row["id"],
                        tool="goose",
                        file_size=row["total_content_size"] or 0,
                        modified_at=modified_ts,
                        extra={
                            "working_dir": row["working_dir"] or "",
                            "max_message_id": row["max_message_id"] or 0,
                        },
                    )
                )

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.debug(f"Could not read Goose DB: {e}")
        finally:
            if conn:
                conn.close()

        sessions.sort(key=lambda s: s.modified_at, reverse=True)
        return sessions

    def extract_text(self, session: SessionInfo, from_byte: int) -> str | None:
        """Extract transcript text from Goose SQLite DB using message-ID watermark.

        The from_byte parameter is part of the protocol but ignored for Goose.
        We use session.extra["last_observed_message_id"] as the watermark instead.
        """
        watermark = session.extra.get("last_observed_message_id", 0)

        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{GOOSE_DB_PATH}?mode=ro", uri=True, timeout=5
            )
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT id, role, content_json
                FROM messages
                WHERE session_id = ? AND id > ?
                ORDER BY id
                """,
                (session.session_id, watermark),
            ).fetchall()

            if not rows:
                return None

            lines = []
            max_id = watermark

            for row in rows:
                msg_id = row["id"]
                if msg_id > max_id:
                    max_id = msg_id

                role = row["role"] or "unknown"
                content_json = row["content_json"] or "[]"

                try:
                    blocks = json.loads(content_json)
                except (json.JSONDecodeError, TypeError):
                    blocks = []

                for block in blocks:
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get("type", "")

                    if block_type == "text":
                        label = role.upper()
                        text = block.get("text", "")
                        if text.strip():
                            lines.append(f"[{label}]: {text}")

                    elif block_type == "toolRequest":
                        tool_call = block.get("toolCall", {})
                        tool_name = tool_call.get("name", "unknown")
                        args = tool_call.get("arguments", {})
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                        # Truncate long args
                        if len(args_str) > 500:
                            args_str = args_str[:500] + "..."
                        lines.append(f"[TOOL]: {tool_name}({args_str})")

                    elif block_type == "toolResponse":
                        result = block.get("result", "")
                        if isinstance(result, list):
                            # Extract text from result blocks
                            texts = [
                                r.get("text", "")
                                for r in result
                                if isinstance(r, dict) and r.get("type") == "text"
                            ]
                            result = "\n".join(texts)
                        result_str = str(result)
                        if len(result_str) > 1000:
                            result_str = result_str[:1000] + "..."
                        lines.append(f"[TOOL RESULT]: {result_str}")

            # Update max message ID in session.extra for the daemon to persist
            session.extra["max_message_id"] = max_id

            return "\n".join(lines) if lines else None

        except sqlite3.OperationalError as e:
            logger.debug(f"Could not read Goose messages: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def enrich_metadata(self, session: SessionInfo) -> SessionInfo:
        """Populate cwd and project_id from Goose session working_dir."""
        working_dir = session.extra.get("working_dir", "")
        if working_dir:
            session.cwd = working_dir
            project_id = _get_project_id(working_dir)
            if project_id:
                session.project_id = project_id
                session.source_ref = f"project:{project_id}"

        return session


def _get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo')."""
    if not cwd:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix(".git")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
