"""Tests for the Goose observer adapter."""

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from cems.observer.adapters.base import SessionInfo
from cems.observer.adapters.goose import GooseAdapter
from cems.observer.state import ObservationState, load_state, save_state, session_tag


def _create_test_db(db_path: Path) -> sqlite3.Connection:
    """Create a Goose-compatible SQLite DB with sessions + messages tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        working_dir TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content_json TEXT,
        created_timestamp INTEGER
    )""")
    conn.commit()
    return conn


class TestGooseDiscovery:
    """Tests for GooseAdapter.discover_sessions."""

    def test_discover_finds_recent_sessions(self, tmp_path):
        """Should find sessions modified within max_age_hours."""
        db_path = tmp_path / "sessions.db"
        conn = _create_test_db(db_path)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, working_dir, updated_at) VALUES (?, ?, ?)",
            ("sess-abc-123", "/Users/test/myproject", now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
            ("sess-abc-123", "user", json.dumps([{"type": "text", "text": "hello"}])),
        )
        conn.commit()
        conn.close()

        adapter = GooseAdapter()
        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            sessions = adapter.discover_sessions(max_age_hours=1)

        assert len(sessions) == 1
        assert sessions[0].session_id == "sess-abc-123"
        assert sessions[0].tool == "goose"
        assert sessions[0].file_size > 0
        assert sessions[0].extra["working_dir"] == "/Users/test/myproject"
        assert sessions[0].extra["db_max_message_id"] == 1

    def test_discover_skips_old_sessions(self, tmp_path):
        """Should skip sessions with old updated_at timestamps."""
        db_path = tmp_path / "sessions.db"
        conn = _create_test_db(db_path)

        old_time = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, working_dir, updated_at) VALUES (?, ?, ?)",
            ("old-session", "/tmp/old", old_time),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
            ("old-session", "user", json.dumps([{"type": "text", "text": "old"}])),
        )
        conn.commit()
        conn.close()

        adapter = GooseAdapter()
        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            sessions = adapter.discover_sessions(max_age_hours=2)

        assert len(sessions) == 0

    def test_discover_handles_missing_db(self, tmp_path):
        """Should return empty list if the DB file doesn't exist."""
        adapter = GooseAdapter()
        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", tmp_path / "nonexistent.db"):
            sessions = adapter.discover_sessions()

        assert sessions == []

    def test_discover_handles_locked_db(self, tmp_path):
        """Should handle sqlite3.OperationalError gracefully."""
        db_path = tmp_path / "sessions.db"
        # Create an invalid file that will cause OperationalError
        db_path.write_text("not a database")

        adapter = GooseAdapter()
        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            sessions = adapter.discover_sessions()

        assert sessions == []


class TestGooseExtractText:
    """Tests for GooseAdapter.extract_text."""

    def test_extract_text_formats_user_assistant(self, tmp_path):
        """Should format user and assistant messages with [USER] and [ASSISTANT] labels."""
        db_path = tmp_path / "sessions.db"
        conn = _create_test_db(db_path)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, working_dir, updated_at) VALUES (?, ?, ?)",
            ("sess-1", "/tmp", now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
            ("sess-1", "user", json.dumps([{"type": "text", "text": "How do I fix this bug?"}])),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
            ("sess-1", "assistant", json.dumps([{"type": "text", "text": "Check the logs for errors."}])),
        )
        conn.commit()
        conn.close()

        adapter = GooseAdapter()
        session = SessionInfo(
            path=db_path, session_id="sess-1", tool="goose",
            extra={"last_observed_message_id": 0},
        )

        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            text = adapter.extract_text(session, from_byte=0)

        assert text is not None
        assert "[USER]: How do I fix this bug?" in text
        assert "[ASSISTANT]: Check the logs for errors." in text

    def test_extract_text_respects_watermark(self, tmp_path):
        """Should only return messages with ID > watermark."""
        db_path = tmp_path / "sessions.db"
        conn = _create_test_db(db_path)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, working_dir, updated_at) VALUES (?, ?, ?)",
            ("sess-2", "/tmp", now),
        )
        for i in range(5):
            conn.execute(
                "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
                ("sess-2", "user", json.dumps([{"type": "text", "text": f"Message {i+1}"}])),
            )
        conn.commit()
        conn.close()

        adapter = GooseAdapter()
        session = SessionInfo(
            path=db_path, session_id="sess-2", tool="goose",
            extra={"last_observed_message_id": 3},
        )

        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            text = adapter.extract_text(session, from_byte=0)

        assert text is not None
        assert "Message 4" in text
        assert "Message 5" in text
        assert "Message 1" not in text
        assert "Message 2" not in text
        assert "Message 3" not in text

    def test_extract_text_handles_tool_requests(self, tmp_path):
        """Should format toolRequest blocks with [TOOL] label."""
        db_path = tmp_path / "sessions.db"
        conn = _create_test_db(db_path)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, working_dir, updated_at) VALUES (?, ?, ?)",
            ("sess-3", "/tmp", now),
        )
        tool_block = {
            "type": "toolRequest",
            "toolCall": {
                "name": "Read",
                "arguments": {"file_path": "/src/main.py"},
            },
        }
        conn.execute(
            "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
            ("sess-3", "assistant", json.dumps([tool_block])),
        )
        conn.commit()
        conn.close()

        adapter = GooseAdapter()
        session = SessionInfo(
            path=db_path, session_id="sess-3", tool="goose",
            extra={"last_observed_message_id": 0},
        )

        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            text = adapter.extract_text(session, from_byte=0)

        assert text is not None
        assert "[TOOL]: Read(" in text
        assert "/src/main.py" in text

    def test_extract_text_returns_none_when_no_new(self, tmp_path):
        """Should return None when watermark is past all messages."""
        db_path = tmp_path / "sessions.db"
        conn = _create_test_db(db_path)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, working_dir, updated_at) VALUES (?, ?, ?)",
            ("sess-4", "/tmp", now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
            ("sess-4", "user", json.dumps([{"type": "text", "text": "hello"}])),
        )
        conn.commit()
        conn.close()

        adapter = GooseAdapter()
        session = SessionInfo(
            path=db_path, session_id="sess-4", tool="goose",
            extra={"last_observed_message_id": 999},
        )

        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            text = adapter.extract_text(session, from_byte=0)

        assert text is None

    def test_extract_text_updates_last_observed_message_id(self, tmp_path):
        """Should update session.extra['last_observed_message_id'] after extraction."""
        db_path = tmp_path / "sessions.db"
        conn = _create_test_db(db_path)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, working_dir, updated_at) VALUES (?, ?, ?)",
            ("sess-5", "/tmp", now),
        )
        for i in range(3):
            conn.execute(
                "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
                ("sess-5", "user", json.dumps([{"type": "text", "text": f"msg {i}"}])),
            )
        conn.commit()
        conn.close()

        adapter = GooseAdapter()
        session = SessionInfo(
            path=db_path, session_id="sess-5", tool="goose",
            extra={"last_observed_message_id": 0},
        )

        with patch("cems.observer.adapters.goose.GOOSE_DB_PATH", db_path):
            adapter.extract_text(session, from_byte=0)

        assert session.extra["last_observed_message_id"] == 3


class TestGooseEnrichMetadata:
    """Tests for GooseAdapter.enrich_metadata."""

    def test_enrich_sets_cwd_from_working_dir(self):
        """Should set session.cwd from extra['working_dir']."""
        adapter = GooseAdapter()
        session = SessionInfo(
            path=Path("/fake/db"),
            session_id="sess-enrich",
            tool="goose",
            extra={"working_dir": "/Users/test/myproject"},
        )

        with patch("cems.observer.adapters.goose._get_project_id", return_value="test/myproject"):
            result = adapter.enrich_metadata(session)

        assert result.cwd == "/Users/test/myproject"
        assert result.project_id == "test/myproject"
        assert result.source_ref == "project:test/myproject"

    def test_enrich_handles_missing_working_dir(self):
        """Should handle missing working_dir gracefully."""
        adapter = GooseAdapter()
        session = SessionInfo(
            path=Path("/fake/db"),
            session_id="sess-no-wd",
            tool="goose",
            extra={},
        )

        result = adapter.enrich_metadata(session)

        assert result.cwd == ""
        assert result.project_id is None
        assert result.source_ref is None


class TestSessionTagFix:
    """Tests for session_tag with Goose-style session IDs."""

    def test_session_tag_uuid_at_12_chars(self):
        """UUID session IDs should work with [:12] truncation."""
        tag = session_tag("550e8400-e29b-41d4-a716-446655440000", epoch=0)
        assert tag == "session:550e8400-e29"
        assert tag.startswith("session:")

    def test_session_tag_goose_ids_no_collision(self):
        """Different Goose session IDs should produce different tags at [:12]."""
        tag1 = session_tag("20260218_1934", epoch=0)
        tag2 = session_tag("20260218_1835", epoch=0)
        assert tag1 != tag2

    def test_session_tag_short_ids(self):
        """Short session IDs shorter than 12 chars should still work."""
        tag = session_tag("short", epoch=0)
        assert tag == "session:short"

    def test_session_tag_backwards_compatible(self):
        """Epoch 0 should use base format, epoch N>0 should append :eN."""
        base = session_tag("abcdefghijkl-extra", epoch=0)
        assert base == "session:abcdefghijkl"

        e2 = session_tag("abcdefghijkl-extra", epoch=2)
        assert e2 == "session:abcdefghijkl:e2"


class TestObservationStateNewField:
    """Tests for ObservationState compatibility with message-ID watermarking."""

    def test_state_has_last_observed_bytes(self):
        """ObservationState should have last_observed_bytes defaulting to 0."""
        state = ObservationState(session_id="test")
        assert state.last_observed_bytes == 0

    def test_state_roundtrip_preserves_fields(self, tmp_path):
        """Save state with custom fields, load it back, verify preserved."""
        state = ObservationState(
            session_id="goose-sess",
            project_id="org/repo",
            last_observed_bytes=42,
            observation_count=5,
        )

        with patch("cems.observer.state.OBSERVER_STATE_DIR", tmp_path):
            save_state(state)
            loaded = load_state("goose-sess")

        assert loaded.session_id == "goose-sess"
        assert loaded.project_id == "org/repo"
        assert loaded.last_observed_bytes == 42
        assert loaded.observation_count == 5

    def test_old_state_file_loads_without_new_fields(self, tmp_path):
        """A state file missing newer fields should load with defaults."""
        state_file = tmp_path / "legacy-sess.json"
        # Simulate old state format without epoch/is_done fields
        old_state = {
            "session_id": "legacy-sess",
            "last_observed_bytes": 100,
            "observation_count": 2,
            "last_observed_at": 1000.0,
            "session_started": 900.0,
        }
        state_file.write_text(json.dumps(old_state))

        with patch("cems.observer.state.OBSERVER_STATE_DIR", tmp_path):
            loaded = load_state("legacy-sess")

        assert loaded.session_id == "legacy-sess"
        assert loaded.last_observed_bytes == 100
        assert loaded.epoch == 0  # default
        assert loaded.is_done is False  # default
