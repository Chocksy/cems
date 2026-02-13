"""Tests for multi-tool session adapters."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cems.observer.adapters.base import SessionInfo


class TestClaudeAdapter:
    """Tests for ClaudeAdapter."""

    def test_discover_sessions(self, tmp_path):
        """Should find recently modified JSONL files."""
        from cems.observer.adapters.claude import ClaudeAdapter

        project_dir = tmp_path / "-Users-test-Development-proj"
        project_dir.mkdir()

        session_file = project_dir / "abc12345-6789-0def-ghij-klmnopqrstuv.jsonl"
        session_file.write_text('{"type":"user","message":{"content":"hello"}}\n')

        with patch("cems.observer.adapters.claude.CLAUDE_PROJECTS_DIR", tmp_path):
            adapter = ClaudeAdapter()
            sessions = adapter.discover_sessions(max_age_hours=1)

        assert len(sessions) == 1
        assert sessions[0].tool == "claude"
        assert sessions[0].session_id == session_file.stem

    def test_discover_ignores_old_sessions(self, tmp_path):
        """Should skip sessions older than max_age_hours."""
        from cems.observer.adapters.claude import ClaudeAdapter
        import os

        project_dir = tmp_path / "-Users-test-Development-old"
        project_dir.mkdir()

        session_file = project_dir / "old-session.jsonl"
        session_file.write_text('{"type":"user"}\n')
        # Set mtime to 3 hours ago
        old_time = time.time() - 3 * 3600
        os.utime(session_file, (old_time, old_time))

        with patch("cems.observer.adapters.claude.CLAUDE_PROJECTS_DIR", tmp_path):
            adapter = ClaudeAdapter()
            sessions = adapter.discover_sessions(max_age_hours=2)

        assert len(sessions) == 0

    def test_extract_text(self, tmp_path):
        """Should extract formatted transcript text."""
        from cems.observer.adapters.claude import ClaudeAdapter

        session_file = tmp_path / "test.jsonl"
        lines = [json.dumps({
            "type": "user",
            "message": {"content": "Hello, I need help with authentication"},
        })]
        session_file.write_text("\n".join(lines) + "\n")

        session = SessionInfo(
            path=session_file,
            session_id="test",
            tool="claude",
            file_size=session_file.stat().st_size,
        )

        adapter = ClaudeAdapter()
        text = adapter.extract_text(session, 0)

        assert text is not None
        assert "[USER]:" in text
        assert "authentication" in text

    def test_enrich_metadata(self, tmp_path):
        """Should extract cwd and gitBranch from first JSONL line."""
        from cems.observer.adapters.claude import ClaudeAdapter

        session_file = tmp_path / "test.jsonl"
        session_file.write_text(json.dumps({
            "type": "user",
            "cwd": "/Users/test/project",
            "gitBranch": "main",
            "message": {"content": "hello"},
        }) + "\n")

        session = SessionInfo(
            path=session_file,
            session_id="test",
            tool="claude",
        )

        with patch("cems.observer.adapters.claude._get_project_id", return_value="test/project"):
            adapter = ClaudeAdapter()
            adapter.enrich_metadata(session)

        assert session.cwd == "/Users/test/project"
        assert session.git_branch == "main"
        assert session.project_id == "test/project"
        assert session.source_ref == "project:test/project"


class TestCodexAdapter:
    """Tests for CodexAdapter."""

    def test_discover_sessions(self, tmp_path):
        """Should find JSONL files with UUID in filename."""
        from cems.observer.adapters.codex import CodexAdapter

        # Create nested date directory structure
        date_dir = tmp_path / "2026" / "02" / "05"
        date_dir.mkdir(parents=True)

        session_file = date_dir / "rollout-2026-02-05T10-36-02-019c2cf1-aed9-7560-933d-874296a5e2a7.jsonl"
        session_file.write_text('{"type":"session_meta","payload":{"cwd":"/tmp"}}\n')

        with patch("cems.observer.adapters.codex.CODEX_SESSIONS_DIR", tmp_path):
            adapter = CodexAdapter()
            sessions = adapter.discover_sessions(max_age_hours=1)

        assert len(sessions) == 1
        assert sessions[0].tool == "codex"
        assert sessions[0].session_id == "019c2cf1-aed9-7560-933d-874296a5e2a7"

    def test_extract_text_new_format(self, tmp_path):
        """Should extract text from new Codex format (2026+)."""
        from cems.observer.adapters.codex import CodexAdapter

        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"type": "session_meta", "payload": {"cwd": "/tmp"}}),
            json.dumps({
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Fix the authentication bug in login"},
            }),
            json.dumps({
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "I'll investigate the JWT token expiration configuration and fix the issue."},
            }),
        ]
        session_file.write_text("\n".join(lines) + "\n")

        session = SessionInfo(
            path=session_file,
            session_id="test",
            tool="codex",
            file_size=session_file.stat().st_size,
        )

        adapter = CodexAdapter()
        text = adapter.extract_text(session, 0)

        assert text is not None
        assert "[USER]:" in text
        assert "authentication" in text
        assert "[ASSISTANT]:" in text

    def test_enrich_metadata_new_format(self, tmp_path):
        """Should extract metadata from session_meta payload."""
        from cems.observer.adapters.codex import CodexAdapter

        session_file = tmp_path / "test.jsonl"
        session_file.write_text(json.dumps({
            "type": "session_meta",
            "payload": {
                "cwd": "/Users/test/project",
                "git": {
                    "branch": "main",
                    "repository_url": "git@github.com:org/repo.git",
                },
            },
        }) + "\n")

        session = SessionInfo(
            path=session_file,
            session_id="test",
            tool="codex",
        )

        adapter = CodexAdapter()
        adapter.enrich_metadata(session)

        assert session.cwd == "/Users/test/project"
        assert session.git_branch == "main"
        assert session.project_id == "org/repo"
        assert session.source_ref == "project:org/repo"

    def test_extract_function_calls(self, tmp_path):
        """Should extract function_call and function_call_output records."""
        from cems.observer.adapters.codex import CodexAdapter

        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"type": "session_meta", "payload": {"cwd": "/tmp"}}),
            json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"command": "ls -la"}',
                },
            }),
            json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "output": "total 42\ndrwxr-xr-x ...",
                },
            }),
        ]
        session_file.write_text("\n".join(lines) + "\n")

        session = SessionInfo(
            path=session_file,
            session_id="test",
            tool="codex",
            file_size=session_file.stat().st_size,
        )

        adapter = CodexAdapter()
        text = adapter.extract_text(session, 0)

        assert text is not None
        assert "[TOOL] exec_command:" in text
        assert "[TOOL RESULT]:" in text


class TestCursorAdapter:
    """Tests for CursorAdapter."""

    def test_discover_sessions(self, tmp_path):
        """Should find UUID-named .txt files in agent-transcripts dirs."""
        from cems.observer.adapters.cursor import CursorAdapter

        project_dir = tmp_path / "Users-test-Development-proj"
        transcripts = project_dir / "agent-transcripts"
        transcripts.mkdir(parents=True)

        session_file = transcripts / "ce7b1dfa-c9fb-47a1-97ad-5b2b4adc2b82.txt"
        session_file.write_text("user:\n<user_query>hello</user_query>\n")

        with patch("cems.observer.adapters.cursor.CURSOR_PROJECTS_DIR", tmp_path):
            adapter = CursorAdapter()
            sessions = adapter.discover_sessions(max_age_hours=1)

        assert len(sessions) == 1
        assert sessions[0].tool == "cursor"
        assert sessions[0].session_id == "ce7b1dfa-c9fb-47a1-97ad-5b2b4adc2b82"

    def test_extract_text(self, tmp_path):
        """Should extract user/assistant text from Cursor transcript."""
        from cems.observer.adapters.cursor import CursorAdapter

        session_file = tmp_path / "test.txt"
        session_file.write_text(
            "user:\n"
            "<user_query>\n"
            "Help me fix the authentication bug in the login flow\n"
            "</user_query>\n\n"
            "assistant:\n"
            "I'll investigate the JWT token configuration to fix this issue.\n"
            "[Tool call] Read\n"
            "  path: /src/auth/config.py\n"
            "[Tool result] Read\n"
            "\n"
        )

        session = SessionInfo(
            path=session_file,
            session_id="test",
            tool="cursor",
            file_size=session_file.stat().st_size,
        )

        adapter = CursorAdapter()
        text = adapter.extract_text(session, 0)

        assert text is not None
        assert "[USER]:" in text
        assert "authentication" in text
        assert "[ASSISTANT]:" in text
        assert "[TOOL] Read:" in text

    def test_ignores_thinking_blocks(self, tmp_path):
        """Should skip <think> blocks in assistant turns."""
        from cems.observer.adapters.cursor import CursorAdapter

        session_file = tmp_path / "test.txt"
        session_file.write_text(
            "user:\n"
            "<user_query>\n"
            "What is the status?\n"
            "</user_query>\n\n"
            "assistant:\n"
            "<think>Let me think about this carefully...</think>"
            "The status is good. Everything is working correctly and passing all tests.\n"
        )

        session = SessionInfo(
            path=session_file,
            session_id="test",
            tool="cursor",
            file_size=session_file.stat().st_size,
        )

        adapter = CursorAdapter()
        text = adapter.extract_text(session, 0)

        assert text is not None
        assert "think about this" not in text
        assert "status is good" in text

    def test_enrich_metadata(self, tmp_path):
        """Should derive project from directory name when path exists."""
        from cems.observer.adapters.cursor import CursorAdapter

        session = SessionInfo(
            path=tmp_path / "test.txt",
            session_id="test",
            tool="cursor",
            extra={"project_dir": "Users-razvan-Development-cems"},
        )

        adapter = CursorAdapter()
        # The reconstructed path "/Users/razvan/Development/cems" exists on this machine
        with patch("os.path.isdir", return_value=True):
            adapter.enrich_metadata(session)

        assert session.cwd == "/Users/razvan/Development/cems"
        assert session.project_id == "cems"


class TestEpochTags:
    """Tests for epoch-aware session tagging."""

    def test_epoch_0_backwards_compatible(self):
        """Epoch 0 should use backwards-compatible tag format."""
        from cems.observer.state import session_tag

        tag = session_tag("abcdefgh-1234-5678-9abc-def012345678", epoch=0)
        assert tag == "session:abcdefgh"
        assert ":e" not in tag

    def test_epoch_n_has_suffix(self):
        """Epoch N>0 should include :eN suffix."""
        from cems.observer.state import session_tag

        tag = session_tag("abcdefgh-1234-5678-9abc-def012345678", epoch=1)
        assert tag == "session:abcdefgh:e1"

        tag = session_tag("abcdefgh-1234-5678-9abc-def012345678", epoch=5)
        assert tag == "session:abcdefgh:e5"

    def test_epoch_default_is_0(self):
        """Default epoch should be 0."""
        from cems.observer.state import session_tag

        tag = session_tag("abcdefgh-1234")
        assert tag == "session:abcdefgh"


class TestStaleness:
    """Tests for staleness detection."""

    def test_staleness_triggers_after_threshold(self):
        """Should detect staleness after STALE_THRESHOLD seconds."""
        from cems.observer.daemon import check_staleness
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test",
            observation_count=1,
            last_growth_seen_at=time.time() - 400,  # 400s ago, threshold is 300s
        )
        assert check_staleness(state) is True

    def test_no_staleness_without_observations(self):
        """Should not trigger staleness if no observations were made."""
        from cems.observer.daemon import check_staleness
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test",
            observation_count=0,
            last_growth_seen_at=time.time() - 400,
        )
        assert check_staleness(state) is False

    def test_growth_resets_staleness(self):
        """Recent growth should prevent staleness detection."""
        from cems.observer.daemon import check_staleness
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test",
            observation_count=3,
            last_growth_seen_at=time.time() - 10,  # 10s ago, well under threshold
        )
        assert check_staleness(state) is False

    def test_no_staleness_on_first_cycle(self):
        """Should not trigger staleness if last_growth_seen_at is 0 (first cycle)."""
        from cems.observer.daemon import check_staleness
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test",
            observation_count=1,
            last_growth_seen_at=0,
        )
        assert check_staleness(state) is False


class TestHandleSignal:
    """Tests for signal-driven lifecycle events in the daemon."""

    def test_compact_signal_bumps_epoch(self, tmp_path):
        """Compact signal should finalize current epoch and bump to next."""
        from cems.observer.daemon import handle_signal
        from cems.observer.signals import Signal
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test-session-001",
            observation_count=2,
            epoch=0,
            last_observed_bytes=5000,
        )

        session = SessionInfo(
            path=tmp_path / "test.jsonl",
            session_id="test-session-001",
            tool="claude",
            file_size=10000,
        )

        sig = Signal(type="compact", ts=time.time(), tool="claude")
        adapter = MagicMock()
        adapter.extract_text.return_value = "[USER]: some content here"
        adapter.enrich_metadata.return_value = session

        state_dir = tmp_path / "states"
        signals_dir = tmp_path / "signals"
        # Create the signal file so clear_signal can remove it
        signals_dir.mkdir(parents=True)
        (signals_dir / "test-session-001.json").write_text('{"type":"compact"}')

        with patch("cems.observer.daemon.send_summary", return_value=True), \
             patch("cems.observer.state.OBSERVER_STATE_DIR", state_dir), \
             patch("cems.observer.signals.SIGNALS_DIR", signals_dir):
            handle_signal(sig, session, state, adapter, "http://localhost:8765", "key")

        assert state.epoch == 1
        assert state.is_done is False  # compact doesn't end session

    def test_stop_signal_finalizes_and_marks_done(self, tmp_path):
        """Stop signal should finalize current epoch and mark session done."""
        from cems.observer.daemon import handle_signal
        from cems.observer.signals import Signal
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test-session-002",
            observation_count=3,
            epoch=1,
            last_observed_bytes=8000,
        )

        session = SessionInfo(
            path=tmp_path / "test.jsonl",
            session_id="test-session-002",
            tool="claude",
            file_size=15000,
        )

        sig = Signal(type="stop", ts=time.time(), tool="claude")
        adapter = MagicMock()
        adapter.extract_text.return_value = "[USER]: final content"
        adapter.enrich_metadata.return_value = session

        state_dir = tmp_path / "states"
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir(parents=True)
        (signals_dir / "test-session-002.json").write_text('{"type":"stop"}')

        with patch("cems.observer.daemon.send_summary", return_value=True) as mock_send, \
             patch("cems.observer.state.OBSERVER_STATE_DIR", state_dir), \
             patch("cems.observer.signals.SIGNALS_DIR", signals_dir):
            handle_signal(sig, session, state, adapter, "http://localhost:8765", "key")

        assert state.is_done is True
        assert state.epoch == 1  # stop doesn't bump epoch
        # Should have called send_summary with mode="finalize" and correct epoch
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert call_kwargs.kwargs.get("mode") or call_kwargs[1].get("mode") == "finalize"

    def test_stop_signal_skips_finalize_for_unobserved_session(self, tmp_path):
        """Stop signal on never-observed session should not send finalize."""
        from cems.observer.daemon import handle_signal
        from cems.observer.signals import Signal
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test-session-003",
            observation_count=0,  # never observed
            epoch=0,
        )

        session = SessionInfo(
            path=tmp_path / "test.jsonl",
            session_id="test-session-003",
            tool="claude",
            file_size=100,
        )

        sig = Signal(type="stop", ts=time.time(), tool="claude")
        adapter = MagicMock()

        state_dir = tmp_path / "states"
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir(parents=True)
        (signals_dir / "test-session-003.json").write_text('{"type":"stop"}')

        with patch("cems.observer.daemon.send_summary") as mock_send, \
             patch("cems.observer.state.OBSERVER_STATE_DIR", state_dir), \
             patch("cems.observer.signals.SIGNALS_DIR", signals_dir):
            handle_signal(sig, session, state, adapter, "http://localhost:8765", "key")

        assert state.is_done is True
        mock_send.assert_not_called()  # no content to finalize

    def test_compact_signal_clears_signal_file(self, tmp_path):
        """Signal file should be removed after processing."""
        from cems.observer.daemon import handle_signal
        from cems.observer.signals import Signal
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test-session-004",
            observation_count=1,
            epoch=0,
        )

        session = SessionInfo(
            path=tmp_path / "test.jsonl",
            session_id="test-session-004",
            tool="claude",
            file_size=5000,
        )

        sig = Signal(type="compact", ts=time.time(), tool="claude")
        adapter = MagicMock()
        adapter.extract_text.return_value = "[USER]: content"
        adapter.enrich_metadata.return_value = session

        state_dir = tmp_path / "states"
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir(parents=True)
        signal_file = signals_dir / "test-session-004.json"
        signal_file.write_text('{"type":"compact"}')

        with patch("cems.observer.daemon.send_summary", return_value=True), \
             patch("cems.observer.state.OBSERVER_STATE_DIR", state_dir), \
             patch("cems.observer.signals.SIGNALS_DIR", signals_dir):
            handle_signal(sig, session, state, adapter, "http://localhost:8765", "key")

        assert not signal_file.exists()


class TestHandleFinalize:
    """Tests for staleness-triggered finalization."""

    def test_finalize_marks_session_done(self, tmp_path):
        """Auto-finalize should mark session as done."""
        from cems.observer.daemon import handle_finalize
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test-finalize-001",
            observation_count=2,
            epoch=0,
        )

        session = SessionInfo(
            path=tmp_path / "test.jsonl",
            session_id="test-finalize-001",
            tool="codex",
            file_size=5000,
        )

        adapter = MagicMock()
        adapter.enrich_metadata.return_value = session

        state_dir = tmp_path / "states"

        with patch("cems.observer.daemon.send_summary", return_value=True) as mock_send, \
             patch("cems.observer.state.OBSERVER_STATE_DIR", state_dir):
            handle_finalize(session, state, adapter, "http://localhost:8765", "key")

        assert state.is_done is True
        assert state.last_finalized_at > 0
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args.kwargs.get("mode") == "finalize"

    def test_finalize_sends_correct_epoch(self, tmp_path):
        """Finalize should use the current epoch value."""
        from cems.observer.daemon import handle_finalize
        from cems.observer.state import ObservationState

        state = ObservationState(
            session_id="test-finalize-002",
            observation_count=5,
            epoch=3,  # already in epoch 3
        )

        session = SessionInfo(
            path=tmp_path / "test.jsonl",
            session_id="test-finalize-002",
            tool="cursor",
            file_size=5000,
        )

        adapter = MagicMock()
        adapter.enrich_metadata.return_value = session

        state_dir = tmp_path / "states"

        with patch("cems.observer.daemon.send_summary", return_value=True) as mock_send, \
             patch("cems.observer.state.OBSERVER_STATE_DIR", state_dir):
            handle_finalize(session, state, adapter, "http://localhost:8765", "key")

        call_args = mock_send.call_args
        assert call_args.kwargs.get("epoch") == 3


class TestHookSignalIntegration:
    """Tests for hook signal writing (matching daemon signal reading)."""

    def test_stop_hook_signal_readable_by_daemon(self, tmp_path):
        """Signal written by stop hook should be readable by daemon's read_signal."""
        from cems.observer.signals import read_signal

        # Simulate what the stop hook does (inlined write_signal)
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir(parents=True)
        signal_file = signals_dir / "test-hook-session.json"
        data = {"type": "stop", "ts": time.time(), "tool": "claude"}
        tmp_file = signal_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(data))
        tmp_file.rename(signal_file)

        # Daemon reads it
        with patch("cems.observer.signals.SIGNALS_DIR", signals_dir):
            sig = read_signal("test-hook-session")

        assert sig is not None
        assert sig.type == "stop"
        assert sig.tool == "claude"

    def test_compact_hook_signal_readable_by_daemon(self, tmp_path):
        """Signal written by pre-compact hook should be readable by daemon."""
        from cems.observer.signals import read_signal

        signals_dir = tmp_path / "signals"
        signals_dir.mkdir(parents=True)
        signal_file = signals_dir / "test-compact-session.json"
        data = {"type": "compact", "ts": time.time(), "tool": "claude"}
        tmp_file = signal_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(data))
        tmp_file.rename(signal_file)

        with patch("cems.observer.signals.SIGNALS_DIR", signals_dir):
            sig = read_signal("test-compact-session")

        assert sig is not None
        assert sig.type == "compact"
        assert sig.tool == "claude"


class TestCursorPathReconstruction:
    """Tests for Cursor adapter path reconstruction edge cases."""

    def test_enrich_with_valid_path(self, tmp_path):
        """Should reconstruct path when it exists on disk."""
        from cems.observer.adapters.cursor import CursorAdapter
        import os

        # Create a real directory that matches the reconstructed path
        real_dir = tmp_path / "project"
        real_dir.mkdir()

        session = SessionInfo(
            path=tmp_path / "test.txt",
            session_id="test",
            tool="cursor",
            extra={"project_dir": "project"},
        )

        adapter = CursorAdapter()
        # Patch os.path.isdir to simulate the path existing
        with patch("os.path.isdir", side_effect=lambda p: p == "/project"):
            adapter.enrich_metadata(session)

        assert session.cwd == "/project"
        assert session.project_id == "project"

    def test_enrich_with_invalid_path_uses_fallback(self, tmp_path):
        """Should fallback when reconstructed path doesn't exist."""
        from cems.observer.adapters.cursor import CursorAdapter

        session = SessionInfo(
            path=tmp_path / "test.txt",
            session_id="test",
            tool="cursor",
            extra={"project_dir": "nonexistent-path-here"},
        )

        adapter = CursorAdapter()
        # os.path.isdir will return False for the reconstructed path
        adapter.enrich_metadata(session)

        # Since /nonexistent/path/here doesn't exist, should use fallback
        assert session.cwd == "nonexistent-path-here"
        assert session.project_id == "here"
