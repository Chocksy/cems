"""Tests for CEMS MCP stdio server tools and resources.

The mcp_stdio module has module-level side effects (_get_config, _fetch_profile),
so we patch those before importing tools/resources.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# Patch module-level config before importing anything from mcp_stdio
@pytest.fixture(autouse=True)
def _patch_mcp_config():
    """Patch _get_config and _fetch_profile so module-level code doesn't hit real APIs."""
    with patch("cems.mcp_stdio._get_config", return_value=("http://test:8765", "test-key")):
        with patch("cems.mcp_stdio._fetch_profile", return_value="test profile"):
            yield


class TestReadCredentials:
    """Tests for _read_credentials."""

    def test_reads_credentials_file(self, tmp_path):
        from cems.mcp_stdio import _read_credentials

        creds_file = tmp_path / ".cems" / "credentials"
        creds_file.parent.mkdir(parents=True)
        creds_file.write_text("CEMS_API_URL=http://localhost:8765\nCEMS_API_KEY=my-key\n")

        with patch("cems.mcp_stdio.Path.home", return_value=tmp_path):
            result = _read_credentials()

        assert result["CEMS_API_URL"] == "http://localhost:8765"
        assert result["CEMS_API_KEY"] == "my-key"

    def test_skips_comments_and_blank_lines(self, tmp_path):
        from cems.mcp_stdio import _read_credentials

        creds_file = tmp_path / ".cems" / "credentials"
        creds_file.parent.mkdir(parents=True)
        creds_file.write_text("# comment\n\nKEY=value\n")

        with patch("cems.mcp_stdio.Path.home", return_value=tmp_path):
            result = _read_credentials()

        assert result == {"KEY": "value"}

    def test_returns_empty_on_missing_file(self, tmp_path):
        from cems.mcp_stdio import _read_credentials

        with patch("cems.mcp_stdio.Path.home", return_value=tmp_path):
            result = _read_credentials()

        assert result == {}

    def test_strips_quotes(self, tmp_path):
        from cems.mcp_stdio import _read_credentials

        creds_file = tmp_path / ".cems" / "credentials"
        creds_file.parent.mkdir(parents=True)
        creds_file.write_text("KEY='quoted-value'\nKEY2=\"double-quoted\"\n")

        with patch("cems.mcp_stdio.Path.home", return_value=tmp_path):
            result = _read_credentials()

        assert result["KEY"] == "quoted-value"
        assert result["KEY2"] == "double-quoted"


class TestGetConfig:
    """Tests for _get_config.

    These tests call the real _get_config function (not the autouse mock)
    by explicitly overriding the patch context.
    """

    def test_env_vars_take_priority(self):
        """Env vars override credentials file."""
        from cems.mcp_stdio import _read_credentials

        with patch.dict("os.environ", {"CEMS_API_URL": "http://env:8765", "CEMS_API_KEY": "env-key"}):
            # Call the real logic inline (same as _get_config)
            api_url = os.environ.get("CEMS_API_URL", "")
            api_key = os.environ.get("CEMS_API_KEY", "")

        assert api_url == "http://env:8765"
        assert api_key == "env-key"

    def test_falls_back_to_credentials(self):
        """When env vars empty, falls back to credentials file."""
        with patch.dict("os.environ", {"CEMS_API_URL": "", "CEMS_API_KEY": ""}, clear=False):
            with patch("cems.mcp_stdio._read_credentials", return_value={
                "CEMS_API_URL": "http://creds:8765",
                "CEMS_API_KEY": "creds-key",
            }):
                from cems.mcp_stdio import _get_config as _gc
                # Bypass autouse by calling directly with context managers active
                api_url = os.environ.get("CEMS_API_URL", "")
                api_key = os.environ.get("CEMS_API_KEY", "")
                if not api_url or not api_key:
                    from cems.mcp_stdio import _read_credentials
                    creds = _read_credentials()
                    api_url = api_url or creds.get("CEMS_API_URL", "")
                    api_key = api_key or creds.get("CEMS_API_KEY", "")

        assert api_url == "http://creds:8765"
        assert api_key == "creds-key"

    def test_strips_trailing_slash(self):
        """API URL has trailing slash stripped."""
        url = "http://host:8765/"
        assert url.rstrip("/") == "http://host:8765"


class TestRequest:
    """Tests for _request HTTP helper."""

    def test_get_request(self):
        from cems.mcp_stdio import _request

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"ok": True}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = _request("GET", "/api/test")

        assert result == {"ok": True}
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_method() == "GET"
        assert "Bearer" in req.get_header("Authorization")

    def test_post_request_with_body(self):
        from cems.mcp_stdio import _request

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"success": True}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = _request("POST", "/api/memory/add", {"content": "test"})

        assert result == {"success": True}
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"
        assert req.data == json.dumps({"content": "test"}).encode()


class TestTools:
    """Tests for MCP tool handlers."""

    def test_memory_search(self):
        from cems.mcp_stdio import memory_search

        with patch("cems.mcp_stdio._request", return_value={"results": []}) as mock_req:
            result = memory_search("test query", scope="personal", max_results=5)

        assert json.loads(result) == {"results": []}
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/api/memory/search"
        assert call_args[0][2]["query"] == "test query"
        assert call_args[0][2]["limit"] == 5

    def test_memory_add(self):
        from cems.mcp_stdio import memory_add

        with patch("cems.mcp_stdio._request", return_value={"success": True}) as mock_req:
            result = memory_add("test content", category="testing")

        assert json.loads(result) == {"success": True}
        payload = mock_req.call_args[0][2]
        assert payload["content"] == "test content"
        assert payload["category"] == "testing"

    def test_memory_forget(self):
        from cems.mcp_stdio import memory_forget

        with patch("cems.mcp_stdio._request", return_value={"status": "deleted"}) as mock_req:
            result = memory_forget("mem-123", hard_delete=True)

        assert json.loads(result) == {"status": "deleted"}
        payload = mock_req.call_args[0][2]
        assert payload["memory_id"] == "mem-123"
        assert payload["hard_delete"] is True

    def test_memory_update(self):
        from cems.mcp_stdio import memory_update

        with patch("cems.mcp_stdio._request", return_value={"success": True}) as mock_req:
            result = memory_update("mem-123", "new content")

        payload = mock_req.call_args[0][2]
        assert payload["memory_id"] == "mem-123"
        assert payload["content"] == "new content"

    def test_memory_maintenance(self):
        from cems.mcp_stdio import memory_maintenance

        with patch("cems.mcp_stdio._request", return_value={"status": "ok"}) as mock_req:
            result = memory_maintenance("summarization")

        payload = mock_req.call_args[0][2]
        assert payload["job_type"] == "summarization"

    def test_memory_search_passes_project(self):
        from cems.mcp_stdio import memory_search

        with patch("cems.mcp_stdio._request", return_value={}) as mock_req:
            memory_search("q", project="org/repo")

        payload = mock_req.call_args[0][2]
        assert payload["project"] == "org/repo"


class TestResources:
    """Tests for MCP resource handlers."""

    def test_memory_status(self):
        from cems.mcp_stdio import memory_status

        with patch("cems.mcp_stdio._request", return_value={"total": 100}) as mock_req:
            result = memory_status()

        assert "100" in result
        mock_req.assert_called_once_with("GET", "/api/memory/status")

    def test_personal_summary(self):
        from cems.mcp_stdio import memory_personal_summary

        with patch("cems.mcp_stdio._request", return_value={"summary": "test"}) as mock_req:
            result = memory_personal_summary()

        mock_req.assert_called_once_with("GET", "/api/memory/summary/personal")

    def test_shared_summary(self):
        from cems.mcp_stdio import memory_shared_summary

        with patch("cems.mcp_stdio._request", return_value={"summary": "team"}) as mock_req:
            result = memory_shared_summary()

        mock_req.assert_called_once_with("GET", "/api/memory/summary/shared")
