"""Tests for the observer daemon lifecycle manager (hooks/utils/observer_manager.py).

These tests verify PID file management, process health checks,
cooldown logic, and the ensure_daemon_running() entry point.

Run: .venv/bin/python3 -m pytest tests/test_observer_manager.py -x -v
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import observer_manager from hooks/utils/
HOOKS_DIR = Path(__file__).parent.parent / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from utils.observer_manager import (
    HEALTH_CHECK_INTERVAL,
    SPAWN_COOLDOWN_SECONDS,
    _is_in_cooldown,
    _is_process_alive,
    _read_pid,
    _should_check,
    _spawn_daemon,
    ensure_daemon_running,
    is_daemon_running,
)


class TestReadPid:
    """Tests for _read_pid()."""

    def test_returns_none_when_no_file(self, tmp_path):
        with patch("utils.observer_manager.PID_FILE", tmp_path / "daemon.pid"):
            assert _read_pid() is None

    def test_reads_valid_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("12345")
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert _read_pid() == 12345

    def test_returns_none_for_invalid_content(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("not_a_number")
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert _read_pid() is None

    def test_returns_none_for_zero_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("0")
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert _read_pid() is None

    def test_returns_none_for_negative_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("-1")
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert _read_pid() is None


class TestIsProcessAlive:
    """Tests for _is_process_alive()."""

    def test_current_process_is_alive(self):
        assert _is_process_alive(os.getpid()) is True

    def test_nonexistent_process(self):
        # PID 99999999 almost certainly doesn't exist
        assert _is_process_alive(99999999) is False


class TestIsDaemonRunning:
    """Tests for is_daemon_running()."""

    def test_returns_false_no_pid_file(self, tmp_path):
        with patch("utils.observer_manager.PID_FILE", tmp_path / "daemon.pid"):
            assert is_daemon_running() is False

    def test_returns_true_for_running_process(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text(str(os.getpid()))
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert is_daemon_running() is True

    def test_returns_false_and_cleans_stale_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("99999999")  # non-existent process
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert is_daemon_running() is False
            # Stale PID file should be cleaned up
            assert not pid_file.exists()


class TestCooldown:
    """Tests for cooldown logic."""

    def test_not_in_cooldown_no_file(self, tmp_path):
        with patch("utils.observer_manager.COOLDOWN_FILE", tmp_path / ".spawn_cooldown"):
            assert _is_in_cooldown() is False

    def test_in_cooldown_recent_file(self, tmp_path):
        cooldown_file = tmp_path / ".spawn_cooldown"
        cooldown_file.touch()
        with patch("utils.observer_manager.COOLDOWN_FILE", cooldown_file):
            assert _is_in_cooldown() is True

    def test_cooldown_expired(self, tmp_path):
        cooldown_file = tmp_path / ".spawn_cooldown"
        cooldown_file.touch()
        # Set mtime to past the cooldown period
        old_time = time.time() - SPAWN_COOLDOWN_SECONDS - 60
        os.utime(cooldown_file, (old_time, old_time))
        with patch("utils.observer_manager.COOLDOWN_FILE", cooldown_file):
            assert _is_in_cooldown() is False
            # Expired cooldown file should be cleaned up
            assert not cooldown_file.exists()


class TestShouldCheck:
    """Tests for rate-limited health checking."""

    def test_should_check_no_pid_file(self, tmp_path):
        with patch("utils.observer_manager.PID_FILE", tmp_path / "daemon.pid"):
            assert _should_check() is True

    def test_should_not_check_recent_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("12345")
        # Fresh file (just created) — should skip check
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert _should_check() is False

    def test_should_check_old_pid(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("12345")
        old_time = time.time() - HEALTH_CHECK_INTERVAL - 60
        os.utime(pid_file, (old_time, old_time))
        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert _should_check() is True


class TestSpawnDaemon:
    """Tests for _spawn_daemon()."""

    def test_fails_without_cems_root(self, tmp_path):
        # Point CEMS_PROJECT_ROOT to nonexistent dir and also prevent fallback candidates
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        with patch("utils.observer_manager.OBSERVER_DIR", tmp_path / "observer"), \
             patch("pathlib.Path.home", return_value=fake_home), \
             patch.dict(os.environ, {"CEMS_PROJECT_ROOT": "", "CEMS_API_KEY": "test"}, clear=False):
            result = _spawn_daemon()
            assert result is False

    def test_fails_without_api_key(self, tmp_path):
        import utils.credentials
        with patch.dict(os.environ, {"CEMS_API_KEY": "", "CEMS_CREDENTIALS_FILE": "/dev/null"}, clear=False):
            utils.credentials._cache = None  # Force reload with new path
            result = _spawn_daemon()
            assert result is False
            utils.credentials._cache = None  # Clean up

    def test_finds_cems_root_from_env(self, tmp_path):
        """Should use CEMS_PROJECT_ROOT env var when set and attempt Popen."""
        from unittest.mock import ANY

        cems_root = tmp_path / "cems"
        cems_root.mkdir()
        (cems_root / "src" / "cems" / "observer").mkdir(parents=True)
        venv_bin = cems_root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / "python3"
        fake_python.write_text("#!/bin/bash\nexit 0")
        fake_python.chmod(0o755)

        observer_dir = tmp_path / "observer"

        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with patch("utils.observer_manager.OBSERVER_DIR", observer_dir), \
             patch("utils.observer_manager.PID_FILE", observer_dir / "daemon.pid"), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("utils.observer_manager._is_process_alive", return_value=True), \
             patch.dict(os.environ, {
                 "CEMS_PROJECT_ROOT": str(cems_root),
                 "CEMS_API_KEY": "test-key",
             }, clear=False):
            result = _spawn_daemon()
            assert result is True
            # Verify it called Popen with the correct python path
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            assert cmd[0] == str(fake_python)
            assert cmd[1:] == ["-m", "cems.observer"]


class TestEnsureDaemonRunning:
    """Tests for the main entry point ensure_daemon_running()."""

    def test_returns_true_when_already_running(self, tmp_path):
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text(str(os.getpid()))

        with patch("utils.observer_manager.PID_FILE", pid_file):
            assert ensure_daemon_running(force_check=True) is True

    def test_skips_check_when_rate_limited(self, tmp_path):
        """When not force_check and recently checked, assumes running."""
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("12345")  # Fresh file = recently checked

        with patch("utils.observer_manager.PID_FILE", pid_file):
            # Not force_check, PID file is fresh — should skip and return True
            assert ensure_daemon_running(force_check=False) is True

    def test_force_check_bypasses_rate_limit(self, tmp_path):
        """force_check=True should check even with fresh PID file."""
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text(str(os.getpid()))  # Current process — alive

        with patch("utils.observer_manager.PID_FILE", pid_file):
            result = ensure_daemon_running(force_check=True)
            assert result is True

    def test_respects_cooldown_after_failure(self, tmp_path):
        """Should not attempt spawn during cooldown."""
        pid_file = tmp_path / "daemon.pid"
        cooldown_file = tmp_path / ".spawn_cooldown"
        cooldown_file.touch()  # Active cooldown

        with patch("utils.observer_manager.PID_FILE", pid_file), \
             patch("utils.observer_manager.COOLDOWN_FILE", cooldown_file), \
             patch("utils.observer_manager._spawn_daemon") as mock_spawn:
            result = ensure_daemon_running(force_check=True)
            assert result is False
            mock_spawn.assert_not_called()

    def test_spawns_when_daemon_dead(self, tmp_path):
        """Should attempt spawn when daemon is not running and no cooldown."""
        pid_file = tmp_path / "daemon.pid"
        cooldown_file = tmp_path / ".spawn_cooldown"
        observer_dir = tmp_path / "observer"

        with patch("utils.observer_manager.PID_FILE", pid_file), \
             patch("utils.observer_manager.COOLDOWN_FILE", cooldown_file), \
             patch("utils.observer_manager.OBSERVER_DIR", observer_dir), \
             patch("utils.observer_manager._spawn_daemon", return_value=True) as mock_spawn:
            result = ensure_daemon_running(force_check=True)
            assert result is True
            mock_spawn.assert_called_once()

    def test_sets_cooldown_on_spawn_failure(self, tmp_path):
        """Should set cooldown file when spawn fails."""
        pid_file = tmp_path / "daemon.pid"
        cooldown_file = tmp_path / ".spawn_cooldown"
        observer_dir = tmp_path / "observer"

        with patch("utils.observer_manager.PID_FILE", pid_file), \
             patch("utils.observer_manager.COOLDOWN_FILE", cooldown_file), \
             patch("utils.observer_manager.OBSERVER_DIR", observer_dir), \
             patch("utils.observer_manager._spawn_daemon", return_value=False):
            result = ensure_daemon_running(force_check=True)
            assert result is False
            assert cooldown_file.exists()
