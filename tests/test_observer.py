"""Tests for the CEMS Observer Daemon."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSessionDiscovery:
    """Tests for session discovery from ~/.claude/projects/."""

    def test_discover_finds_recent_sessions(self, tmp_path):
        """Should find JSONL files modified recently."""
        from cems.observer.session import discover_active_sessions

        # Create a fake project dir with session files
        project_dir = tmp_path / "-Users-test-Development-myproject"
        project_dir.mkdir()

        session_file = project_dir / "abc-123.jsonl"
        session_file.write_text('{"type":"user","cwd":"/tmp","gitBranch":"main"}\n')

        with patch("cems.observer.session.CLAUDE_PROJECTS_DIR", tmp_path):
            sessions = discover_active_sessions(max_age_hours=1)

        assert len(sessions) == 1
        assert sessions[0].session_id == "abc-123"
        assert sessions[0].project_dir == project_dir.name

    def test_discover_skips_old_sessions(self, tmp_path):
        """Should skip JSONL files older than max_age_hours."""
        import os
        from cems.observer.session import discover_active_sessions

        project_dir = tmp_path / "-Users-test-Development-old"
        project_dir.mkdir()

        session_file = project_dir / "old-session.jsonl"
        session_file.write_text('{"type":"user"}\n')
        # Set mtime to 10 hours ago
        old_time = time.time() - (10 * 3600)
        os.utime(session_file, (old_time, old_time))

        with patch("cems.observer.session.CLAUDE_PROJECTS_DIR", tmp_path):
            sessions = discover_active_sessions(max_age_hours=2)

        assert len(sessions) == 0

    def test_discover_handles_missing_dir(self, tmp_path):
        """Should return empty list if projects dir doesn't exist."""
        from cems.observer.session import discover_active_sessions

        with patch("cems.observer.session.CLAUDE_PROJECTS_DIR", tmp_path / "nonexistent"):
            sessions = discover_active_sessions()

        assert sessions == []


class TestSessionMetadata:
    """Tests for session metadata extraction."""

    def test_enrich_reads_first_entry(self, tmp_path):
        """Should read cwd and gitBranch from first JSONL line."""
        from cems.observer.session import SessionInfo, enrich_session_metadata

        session_file = tmp_path / "test.jsonl"
        entry = {"type": "user", "cwd": "/Users/test/myproject", "gitBranch": "main"}
        session_file.write_text(json.dumps(entry) + "\n")

        session = SessionInfo(
            path=session_file,
            project_dir="test",
            session_id="test-id",
        )

        with patch("cems.observer.session._get_project_id", return_value="test/myproject"):
            result = enrich_session_metadata(session)

        assert result.cwd == "/Users/test/myproject"
        assert result.git_branch == "main"
        assert result.project_id == "test/myproject"
        assert result.source_ref == "project:test/myproject"

    def test_enrich_handles_empty_file(self, tmp_path):
        """Should handle empty session file gracefully."""
        from cems.observer.session import SessionInfo, enrich_session_metadata

        session_file = tmp_path / "empty.jsonl"
        session_file.write_text("")

        session = SessionInfo(
            path=session_file,
            project_dir="test",
            session_id="empty-id",
        )

        result = enrich_session_metadata(session)
        assert result.cwd == ""
        assert result.project_id is None


class TestContentDelta:
    """Tests for reading new content from session files."""

    def test_read_delta_extracts_user_assistant(self, tmp_path):
        """Should only extract user and assistant messages."""
        from cems.observer.session import SessionInfo, read_content_delta

        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"content": "Please deploy the application to production"}}),
            json.dumps({"type": "progress", "data": {"type": "hook"}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "I will deploy the application to production now"}]}}),
            json.dumps({"type": "system", "message": {"content": "System msg"}}),
        ]
        session_file.write_text("\n".join(lines) + "\n")

        session = SessionInfo(
            path=session_file,
            project_dir="test",
            session_id="test-id",
        )

        content = read_content_delta(session, from_byte=0)
        assert content is not None
        assert "[USER]: Please deploy the application" in content
        assert "[ASSISTANT]: I will deploy the application" in content
        assert "System msg" not in content
        assert "hook" not in content

    def test_read_delta_from_offset(self, tmp_path):
        """Should only read content after the byte offset."""
        from cems.observer.session import SessionInfo, read_content_delta

        session_file = tmp_path / "test.jsonl"
        line1 = json.dumps({"type": "user", "message": {"content": "First message that is long enough to be captured"}})
        line2 = json.dumps({"type": "user", "message": {"content": "Second message that is also long enough to pass"}})
        full_content = line1 + "\n" + line2 + "\n"
        session_file.write_text(full_content)

        session = SessionInfo(
            path=session_file,
            project_dir="test",
            session_id="test-id",
        )

        # Read from after first line
        offset = len(line1.encode("utf-8")) + 1  # +1 for newline
        content = read_content_delta(session, from_byte=offset)
        assert content is not None
        assert "Second message that is also long enough" in content
        assert "First message" not in content

    def test_read_delta_handles_content_blocks(self, tmp_path):
        """Should extract text and tool_use from content block arrays."""
        from cems.observer.session import SessionInfo, read_content_delta

        session_file = tmp_path / "test.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Here is my response to your question about the config"},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/src/config.py"}},
                ]
            }
        }
        session_file.write_text(json.dumps(entry) + "\n")

        session = SessionInfo(
            path=session_file,
            project_dir="test",
            session_id="test-id",
        )

        content = read_content_delta(session, from_byte=0)
        assert content is not None
        assert "Here is my response" in content
        assert "[ACTIVITY] Read /src/config.py" in content

    def test_read_delta_returns_none_when_no_new_content(self, tmp_path):
        """Should return None if file hasn't grown past offset."""
        from cems.observer.session import SessionInfo, read_content_delta

        session_file = tmp_path / "test.jsonl"
        session_file.write_text('{"type":"user","message":{"content":"hi"}}\n')

        session = SessionInfo(
            path=session_file,
            project_dir="test",
            session_id="test-id",
        )

        # Offset at or past file size
        content = read_content_delta(session, from_byte=99999)
        assert content is None


class TestObservationState:
    """Tests for per-session state tracking."""

    def test_load_nonexistent_creates_fresh(self, tmp_path):
        """Should create fresh state for unknown session."""
        from cems.observer.state import load_state

        with patch("cems.observer.state.OBSERVER_STATE_DIR", tmp_path):
            state = load_state("new-session")

        assert state.session_id == "new-session"
        assert state.last_observed_bytes == 0
        assert state.observation_count == 0

    def test_save_and_load_roundtrip(self, tmp_path):
        """Should persist and reload state correctly."""
        from cems.observer.state import ObservationState, load_state, save_state

        state = ObservationState(
            session_id="test-session",
            project_id="org/repo",
            source_ref="project:org/repo",
            last_observed_bytes=50000,
            observation_count=3,
        )

        with patch("cems.observer.state.OBSERVER_STATE_DIR", tmp_path):
            save_state(state)
            loaded = load_state("test-session")

        assert loaded.session_id == "test-session"
        assert loaded.project_id == "org/repo"
        assert loaded.last_observed_bytes == 50000
        assert loaded.observation_count == 3

    def test_cleanup_removes_old_files(self, tmp_path):
        """Should remove state files older than max_age_days."""
        import os
        from cems.observer.state import cleanup_old_states

        # Create an old state file
        old_file = tmp_path / "old-session.json"
        old_file.write_text('{"session_id": "old"}')
        old_time = time.time() - (10 * 86400)  # 10 days ago
        os.utime(old_file, (old_time, old_time))

        # Create a fresh state file
        fresh_file = tmp_path / "fresh-session.json"
        fresh_file.write_text('{"session_id": "fresh"}')

        with patch("cems.observer.state.OBSERVER_STATE_DIR", tmp_path):
            removed = cleanup_old_states(max_age_days=7)

        assert removed == 1
        assert not old_file.exists()
        assert fresh_file.exists()


class TestDaemon:
    """Tests for the daemon run_cycle function."""

    def test_run_cycle_processes_sessions(self, tmp_path):
        """run_cycle should discover sessions and process them."""
        from cems.observer.daemon import run_cycle

        # Create a session with enough content
        project_dir = tmp_path / "-Users-test-Development-proj"
        project_dir.mkdir()

        session_file = project_dir / "test-session.jsonl"
        # Write enough content to trigger observation (> 50KB)
        lines = []
        for i in range(200):
            lines.append(json.dumps({
                "type": "user",
                "cwd": "/tmp/proj",
                "gitBranch": "main",
                "message": {"content": f"Message {i}: " + "x" * 200},
            }))
        session_file.write_text("\n".join(lines) + "\n")

        state_dir = tmp_path / "observer_state"

        with patch("cems.observer.session.CLAUDE_PROJECTS_DIR", tmp_path), \
             patch("cems.observer.state.OBSERVER_STATE_DIR", state_dir), \
             patch("cems.observer.daemon.send_observation", return_value=True) as mock_send, \
             patch("cems.observer.session._get_project_id", return_value="test/proj"):

            triggered = run_cycle("http://localhost:8765", "test-key")

        assert triggered == 1
        mock_send.assert_called_once()

        # Verify state was saved
        state_file = state_dir / "test-session.json"
        assert state_file.exists()

    def test_run_cycle_skips_small_sessions(self, tmp_path):
        """run_cycle should skip sessions with too little new content."""
        from cems.observer.daemon import run_cycle

        project_dir = tmp_path / "-Users-test-Development-small"
        project_dir.mkdir()

        session_file = project_dir / "small-session.jsonl"
        session_file.write_text('{"type":"user","message":{"content":"hi"}}\n')

        with patch("cems.observer.session.CLAUDE_PROJECTS_DIR", tmp_path), \
             patch("cems.observer.state.OBSERVER_STATE_DIR", tmp_path / "states"), \
             patch("cems.observer.daemon.send_observation") as mock_send:

            triggered = run_cycle("http://localhost:8765", "test-key")

        assert triggered == 0
        mock_send.assert_not_called()
