"""Tests for the observer signal module."""

import json
import time
from unittest.mock import patch

import pytest

from cems.observer.signals import (
    SIGNALS_DIR,
    Signal,
    clear_signal,
    read_signal,
    write_signal,
)


class TestSignals:
    """Tests for signal read/write/clear operations."""

    def test_write_read_clear_roundtrip(self, tmp_path):
        """Signal should survive write → read → clear cycle."""
        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            write_signal("test-session-001", "stop", "claude")

            sig = read_signal("test-session-001")
            assert sig is not None
            assert sig.type == "stop"
            assert sig.tool == "claude"
            assert sig.ts > 0

            clear_signal("test-session-001")
            assert read_signal("test-session-001") is None

    def test_read_nonexistent_signal(self, tmp_path):
        """Reading a nonexistent signal returns None."""
        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            assert read_signal("nonexistent-session") is None

    def test_signal_overwrite(self, tmp_path):
        """Last write wins — overwriting a signal replaces it."""
        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            write_signal("test-session-002", "compact", "claude")
            write_signal("test-session-002", "stop", "codex")

            sig = read_signal("test-session-002")
            assert sig is not None
            assert sig.type == "stop"
            assert sig.tool == "codex"

    def test_write_creates_directory(self, tmp_path):
        """write_signal should create the signals directory if needed."""
        signals_dir = tmp_path / "nested" / "signals"
        with patch("cems.observer.signals.SIGNALS_DIR", signals_dir):
            write_signal("test-session-003", "compact", "cursor")
            assert (signals_dir / "test-session-003.json").exists()

    def test_clear_missing_signal_is_noop(self, tmp_path):
        """Clearing a nonexistent signal should not raise."""
        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            clear_signal("nonexistent-session")  # Should not raise

    def test_signal_file_format(self, tmp_path):
        """Signal file should be valid JSON with expected fields."""
        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            write_signal("test-session-004", "compact", "claude")

            signal_file = tmp_path / "test-session-004.json"
            data = json.loads(signal_file.read_text())

            assert data["type"] == "compact"
            assert data["tool"] == "claude"
            assert isinstance(data["ts"], float)

    def test_default_tool_is_claude(self, tmp_path):
        """write_signal should default to 'claude' tool."""
        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            write_signal("test-session-005", "stop")

            sig = read_signal("test-session-005")
            assert sig.tool == "claude"
