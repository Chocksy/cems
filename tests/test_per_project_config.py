"""Tests for per-project CEMS configuration (walk-up credential resolution)."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestResolveCredentials:
    """Tests for hooks/utils/credentials.py resolve_credentials()."""

    def _get_resolve(self):
        """Import resolve_credentials fresh (avoids module-level cache issues)."""
        # Reset the module-level cache before each test
        import hooks.utils.credentials as creds_mod
        creds_mod._cache = {}
        return creds_mod.resolve_credentials

    def test_finds_project_credentials(self, tmp_path):
        """Walk-up finds .cems/credentials in project root."""
        # Create project credentials
        cems_dir = tmp_path / ".cems"
        cems_dir.mkdir()
        (cems_dir / "credentials").write_text(
            "CEMS_API_URL=https://project.cems.io\n"
            "CEMS_API_KEY=project_key_123\n"
        )

        resolve = self._get_resolve()
        with patch.dict(os.environ, {"CEMS_CREDENTIALS_FILE": "/dev/null"}, clear=True):
            result = resolve(cwd=str(tmp_path))

        assert result["CEMS_API_URL"] == "https://project.cems.io"
        assert result["CEMS_API_KEY"] == "project_key_123"

    def test_finds_credentials_in_parent(self, tmp_path):
        """Walk-up finds .cems/credentials in ancestor directory."""
        # Create project credentials at root
        cems_dir = tmp_path / ".cems"
        cems_dir.mkdir()
        (cems_dir / "credentials").write_text(
            "CEMS_API_URL=https://parent.cems.io\n"
            "CEMS_API_KEY=parent_key\n"
        )

        # Create a nested subdirectory
        sub_dir = tmp_path / "src" / "deep" / "nested"
        sub_dir.mkdir(parents=True)

        resolve = self._get_resolve()
        with patch.dict(os.environ, {}, clear=True):
            result = resolve(cwd=str(sub_dir))

        assert result["CEMS_API_URL"] == "https://parent.cems.io"

    def test_stops_before_home(self, tmp_path, monkeypatch):
        """Walk-up does NOT find ~/.cems/credentials during walk — that's the fallback."""
        # Create a directory under home simulation
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()

        # Put credentials at "home" level (should NOT be found by walk-up)
        (fake_home / ".cems").mkdir()
        (fake_home / ".cems" / "credentials").write_text(
            "CEMS_API_URL=https://global.cems.io\n"
            "CEMS_API_KEY=global_key\n"
        )

        # Work directory under home (no project .cems)
        work_dir = fake_home / "Development" / "myproject"
        work_dir.mkdir(parents=True)

        import hooks.utils.credentials as creds_mod
        creds_mod._cache = {}

        global_creds = str(fake_home / ".cems" / "credentials")
        with patch.dict(os.environ, {"CEMS_CREDENTIALS_FILE": global_creds}, clear=True), \
             patch.object(creds_mod, "_HOME", str(fake_home)):
            result = creds_mod.resolve_credentials(cwd=str(work_dir))

        # Walk-up stopped at fake_home, then fallback reads global via CEMS_CREDENTIALS_FILE
        assert result.get("CEMS_API_URL") == "https://global.cems.io"

    def test_env_vars_override_project(self, tmp_path):
        """Environment variables take priority over project .cems/credentials."""
        # Create project credentials
        cems_dir = tmp_path / ".cems"
        cems_dir.mkdir()
        (cems_dir / "credentials").write_text(
            "CEMS_API_URL=https://project.cems.io\n"
            "CEMS_API_KEY=project_key\n"
        )

        resolve = self._get_resolve()
        with patch.dict(os.environ, {
            "CEMS_API_URL": "https://env.cems.io",
            "CEMS_API_KEY": "env_key",
        }):
            result = resolve(cwd=str(tmp_path))

        assert result["CEMS_API_URL"] == "https://env.cems.io"
        assert result["CEMS_API_KEY"] == "env_key"

    def test_no_cwd_uses_global(self, tmp_path):
        """resolve_credentials(None) returns global credentials."""
        # Create a global credentials file
        global_creds = tmp_path / "global_creds"
        global_creds.write_text(
            "CEMS_API_URL=https://global.cems.io\n"
            "CEMS_API_KEY=global_key\n"
        )

        import hooks.utils.credentials as creds_mod
        creds_mod._cache = {}

        with patch.dict(os.environ, {"CEMS_CREDENTIALS_FILE": str(global_creds)}, clear=True):
            result = creds_mod.resolve_credentials(cwd=None)

        assert result["CEMS_API_URL"] == "https://global.cems.io"

    def test_backward_compat_no_cwd(self, tmp_path):
        """get_cems_url() and get_cems_key() without args work as before."""
        global_creds = tmp_path / "global_creds"
        global_creds.write_text(
            "CEMS_API_URL=https://legacy.cems.io\n"
            "CEMS_API_KEY=legacy_key\n"
        )

        import hooks.utils.credentials as creds_mod
        creds_mod._cache = {}

        with patch.dict(os.environ, {"CEMS_CREDENTIALS_FILE": str(global_creds)}, clear=True):
            assert creds_mod.get_cems_url() == "https://legacy.cems.io"
            assert creds_mod.get_cems_key() == "legacy_key"


class TestCEMSClient:
    """Tests for CEMSClient.from_cwd()."""

    def test_from_cwd_returns_client_when_configured(self, tmp_path):
        """CEMSClient.from_cwd() returns a client with resolved credentials."""
        cems_dir = tmp_path / ".cems"
        cems_dir.mkdir()
        (cems_dir / "credentials").write_text(
            "CEMS_API_URL=https://test.cems.io\n"
            "CEMS_API_KEY=test_key\n"
        )

        import hooks.utils.credentials as creds_mod
        creds_mod._cache = {}

        with patch.dict(os.environ, {}, clear=True):
            client = creds_mod.CEMSClient.from_cwd(str(tmp_path))

        assert client is not None
        assert client.url == "https://test.cems.io"
        assert client.key == "test_key"

    def test_from_cwd_returns_none_when_unconfigured(self, tmp_path):
        """CEMSClient.from_cwd() returns None if no credentials found."""
        import hooks.utils.credentials as creds_mod
        creds_mod._cache = {}

        with patch.dict(os.environ, {}, clear=True), \
             patch.dict(os.environ, {"CEMS_CREDENTIALS_FILE": "/dev/null"}):
            client = creds_mod.CEMSClient.from_cwd(str(tmp_path))

        assert client is None


class TestCredentialResolver:
    """Tests for the daemon's CredentialResolver."""

    def _make_resolver(self, global_url="https://global.cems.io", global_key="gkey"):
        """Create a resolver that returns fixed global credentials."""
        from cems.observer.daemon import CredentialResolver
        r = CredentialResolver.__new__(CredentialResolver)
        r._cache = {}
        r.default_url = global_url
        r.default_key = global_key
        return r

    def test_resolve_without_cwd_returns_global(self):
        """Empty CWD returns global credentials."""
        resolver = self._make_resolver()
        url, key = resolver.resolve("")
        assert url == "https://global.cems.io"
        assert key == "gkey"

    def test_resolve_with_project_cwd(self, tmp_path):
        """CWD in a project with .cems/credentials returns project creds."""
        resolver = self._make_resolver()

        # Create project credentials under tmp_path (walk-up will find them
        # since tmp_path is below $HOME)
        project = tmp_path / "projects" / "hubstaff"
        project.mkdir(parents=True)
        (project / ".cems").mkdir()
        (project / ".cems" / "credentials").write_text(
            "CEMS_API_URL=https://hubstaff.cems.io\n"
            "CEMS_API_KEY=hubstaff_key\n"
        )

        with patch.dict(os.environ, {}, clear=True):
            url, key = resolver.resolve(str(project / "src" / "deep"))
        assert url == "https://hubstaff.cems.io"
        assert key == "hubstaff_key"

    def test_resolve_caches_per_cwd(self):
        """Same CWD returns cached result without re-walking."""
        resolver = self._make_resolver()

        url1, _ = resolver.resolve("/some/path")
        url2, _ = resolver.resolve("/some/path")
        assert url1 == url2
        assert "/some/path" in resolver._cache

    def test_per_url_isolation(self, tmp_path):
        """Different projects resolve to different servers."""
        resolver = self._make_resolver()

        # Project A
        proj_a = tmp_path / "projects" / "alpha"
        proj_a.mkdir(parents=True)
        (proj_a / ".cems").mkdir()
        (proj_a / ".cems" / "credentials").write_text(
            "CEMS_API_URL=https://alpha.cems.io\nCEMS_API_KEY=akey\n"
        )

        # Project B
        proj_b = tmp_path / "projects" / "beta"
        proj_b.mkdir(parents=True)
        (proj_b / ".cems").mkdir()
        (proj_b / ".cems" / "credentials").write_text(
            "CEMS_API_URL=https://beta.cems.io\nCEMS_API_KEY=bkey\n"
        )

        with patch.dict(os.environ, {}, clear=True):
            url_a, _ = resolver.resolve(str(proj_a))
            url_b, _ = resolver.resolve(str(proj_b))

        assert url_a == "https://alpha.cems.io"
        assert url_b == "https://beta.cems.io"


class TestSignalCwd:
    """Tests for CWD in observer signals."""

    def test_signal_includes_cwd(self, tmp_path):
        """write_signal includes cwd field in JSON."""
        from cems.observer.signals import write_signal, read_signal, SIGNALS_DIR
        import json

        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            write_signal("test-123", "compact", "claude", cwd="/path/to/project")

        signal_file = tmp_path / "test-123.json"
        assert signal_file.exists()

        data = json.loads(signal_file.read_text())
        assert data["cwd"] == "/path/to/project"

    def test_signal_read_parses_cwd(self, tmp_path):
        """read_signal parses cwd field from JSON."""
        import json
        from cems.observer.signals import read_signal

        signal_file = tmp_path / "test-456.json"
        signal_file.write_text(json.dumps({
            "type": "stop",
            "ts": 1234567890.0,
            "tool": "claude",
            "cwd": "/my/project",
        }))

        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            sig = read_signal("test-456")

        assert sig is not None
        assert sig.cwd == "/my/project"

    def test_signal_read_handles_missing_cwd(self, tmp_path):
        """read_signal handles old signal format without cwd field."""
        import json
        from cems.observer.signals import read_signal

        signal_file = tmp_path / "test-789.json"
        signal_file.write_text(json.dumps({
            "type": "compact",
            "ts": 1234567890.0,
            "tool": "claude",
        }))

        with patch("cems.observer.signals.SIGNALS_DIR", tmp_path):
            sig = read_signal("test-789")

        assert sig is not None
        assert sig.cwd == ""  # Default empty string
