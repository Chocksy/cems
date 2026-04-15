"""Tests for cems setup command — MCP registration and config writing."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestRegisterClaudeMcpServer:
    """Tests for _register_claude_mcp_server."""

    def test_registers_full_path_to_cems_mcp(self, tmp_path):
        """MCP config should use the full path to cems-mcp, not bare name."""
        from cems.commands.setup import _register_claude_mcp_server

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text("{}")

        with patch("cems.commands.setup.Path.home", return_value=tmp_path), \
             patch("cems.commands.setup.shutil.which", return_value="/usr/local/bin/cems-mcp"):
            _register_claude_mcp_server("https://example.com")

        config = json.loads(claude_json.read_text())
        cmd = config["mcpServers"]["cems"]["command"]
        assert cmd == "/usr/local/bin/cems-mcp", f"Expected full path, got: {cmd}"

    def test_falls_back_to_venv_bin(self, tmp_path):
        """When cems-mcp not on PATH, check sys.prefix/bin."""
        from cems.commands.setup import _register_claude_mcp_server

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text("{}")

        # Create fake binary in sys.prefix
        fake_bin = tmp_path / "venv" / "bin" / "cems-mcp"
        fake_bin.parent.mkdir(parents=True)
        fake_bin.touch()

        with patch("cems.commands.setup.Path.home", return_value=tmp_path), \
             patch("cems.commands.setup.shutil.which", return_value=None), \
             patch("cems.commands.setup.sys") as mock_sys:
            mock_sys.prefix = str(tmp_path / "venv")
            _register_claude_mcp_server("https://example.com")

        config = json.loads(claude_json.read_text())
        cmd = config["mcpServers"]["cems"]["command"]
        assert str(fake_bin) == cmd, f"Expected venv path, got: {cmd}"

    def test_warns_when_not_found(self, tmp_path, capsys):
        """When cems-mcp is nowhere, register bare name but warn."""
        from cems.commands.setup import _register_claude_mcp_server

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text("{}")

        with patch("cems.commands.setup.Path.home", return_value=tmp_path), \
             patch("cems.commands.setup.shutil.which", return_value=None), \
             patch("cems.commands.setup.sys") as mock_sys:
            mock_sys.prefix = "/nonexistent"
            _register_claude_mcp_server("https://example.com")

        config = json.loads(claude_json.read_text())
        cmd = config["mcpServers"]["cems"]["command"]
        assert cmd == "cems-mcp", "Should fall back to bare name"

    def test_overwrites_old_http_config(self, tmp_path):
        """Old HTTP/SSE config should be replaced with stdio."""
        from cems.commands.setup import _register_claude_mcp_server

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({
            "mcpServers": {
                "cems": {
                    "type": "sse",
                    "url": "http://old-server:8766/sse",
                    "env": {"CEMS_API_KEY": "old-key"},
                }
            }
        }))

        with patch("cems.commands.setup.Path.home", return_value=tmp_path), \
             patch("cems.commands.setup.shutil.which", return_value="/usr/bin/cems-mcp"):
            _register_claude_mcp_server("https://example.com")

        config = json.loads(claude_json.read_text())
        cems = config["mcpServers"]["cems"]
        assert cems["command"] == "/usr/bin/cems-mcp"
        assert "url" not in cems, "Old URL should be gone"
        assert "env" not in cems, "Old env overrides should be gone"
        assert "type" not in cems, "Old type should be gone"


    def test_registers_http_transport(self, tmp_path):
        """HTTP transport should use type:http with URL and auth headers."""
        from cems.commands.setup import _register_claude_mcp_server

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text("{}")

        with patch("cems.commands.setup.Path.home", return_value=tmp_path), \
             patch("cems.commands.setup._discover_mcp_url", return_value="https://mcp-cems.example.com/mcp"), \
             patch("cems.commands.setup._read_credentials", return_value={"CEMS_API_KEY": "test-key"}):
            _register_claude_mcp_server("https://cems.example.com", transport="http", api_key="test-key")

        config = json.loads(claude_json.read_text())
        cems = config["mcpServers"]["cems"]
        assert cems["type"] == "http"
        assert cems["url"] == "https://mcp-cems.example.com/mcp"
        assert cems["headers"]["Authorization"] == "Bearer test-key"
        assert "X-Team-Id" not in cems.get("headers", {}), "No team headers should exist"
        assert "command" not in cems, "HTTP mode should not have command"


class TestDiscoverMcpUrl:
    """Tests for _discover_mcp_url."""

    def test_credentials_override(self):
        """CEMS_MCP_URL in credentials takes priority."""
        from cems.commands.setup import _discover_mcp_url

        with patch("cems.commands.setup._read_credentials", return_value={"CEMS_MCP_URL": "https://custom.com/mcp"}):
            assert _discover_mcp_url("https://cems.example.com") == "https://custom.com/mcp"

    def test_localhost_fallback(self):
        """Localhost API URL should derive port 8766 MCP URL."""
        from cems.commands.setup import _discover_mcp_url

        with patch("cems.commands.setup._read_credentials", return_value={}):
            result = _discover_mcp_url("http://localhost:8765")
            assert result == "http://localhost:8766/mcp"

    def test_remote_fallback(self):
        """Remote API URL should derive mcp- prefixed URL."""
        from cems.commands.setup import _discover_mcp_url

        with patch("cems.commands.setup._read_credentials", return_value={}):
            result = _discover_mcp_url("https://cems.chocksy.com")
            assert result == "https://mcp-cems.chocksy.com/mcp"


class TestAppendCemsInstructions:
    """Tests for _append_cems_instructions."""

    def test_creates_file_if_missing(self, tmp_path):
        """Should create the config file if it doesn't exist."""
        from cems.commands.setup import _append_cems_instructions

        config_file = tmp_path / "CLAUDE.md"
        _append_cems_instructions(config_file)

        assert config_file.exists()
        content = config_file.read_text()
        assert "## CEMS" in content
        assert "mcp__cems__memory_search" in content

    def test_appends_to_existing(self, tmp_path):
        """Should append to existing file without overwriting."""
        from cems.commands.setup import _append_cems_instructions

        config_file = tmp_path / "CLAUDE.md"
        config_file.write_text("# My Instructions\n\nExisting content.\n")

        _append_cems_instructions(config_file)

        content = config_file.read_text()
        assert content.startswith("# My Instructions")
        assert "## CEMS" in content
        assert "mcp__cems__memory_search" in content

    def test_idempotent(self, tmp_path):
        """Running twice should not duplicate instructions."""
        from cems.commands.setup import _append_cems_instructions

        config_file = tmp_path / "CLAUDE.md"
        _append_cems_instructions(config_file)
        first = config_file.read_text()

        _append_cems_instructions(config_file)
        second = config_file.read_text()

        assert first == second, "Should be idempotent"

    def test_warns_about_wrong_tools(self, tmp_path):
        """Instructions should explicitly warn against using apigcp/Context tools."""
        from cems.commands.setup import _append_cems_instructions

        config_file = tmp_path / "AGENTS.md"
        _append_cems_instructions(config_file)

        content = config_file.read_text()
        assert "apigcp" in content.lower() or "Context" in content, \
            "Should warn against using wrong MCP tools"
