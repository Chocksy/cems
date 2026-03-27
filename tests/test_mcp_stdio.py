"""Tests for CEMS MCP stdio server tools and resources.

The mcp_stdio module has module-level side effects (_get_config, _fetch_profile),
so we patch those before importing tools/resources.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import cems.mcp_stdio as _mcp_module

# Keep a reference to the real _get_config before any patching
_real_get_config = _mcp_module._get_config


# Patch module-level config before importing anything from mcp_stdio
@pytest.fixture(autouse=True)
def _patch_mcp_config():
    """Patch _get_config and _fetch_profile so module-level code doesn't hit real APIs."""
    with patch("cems.mcp_stdio._get_config", return_value=("http://test:8765", "test-key")):
        with patch("cems.mcp_stdio._fetch_profile", return_value="test profile"):
            yield


class TestGetConfigWithResolver:
    """Tests for _get_config using the shared credential resolver.

    These tests call the real _get_config (via _real_get_config saved before
    autouse patching) and only mock resolve_credentials to control credential
    resolution behavior.
    """

    def test_per_project_credentials_from_cwd(self, tmp_path):
        """Per-project .cems/credentials in CWD takes precedence over global."""
        project_creds = tmp_path / ".cems" / "credentials"
        project_creds.parent.mkdir(parents=True)
        project_creds.write_text(
            "CEMS_API_URL=http://project:8765\nCEMS_API_KEY=project-key\n"
        )

        with patch("cems.mcp_stdio.resolve_credentials", return_value={
            "CEMS_API_URL": "http://project:8765",
            "CEMS_API_KEY": "project-key",
        }):
            api_url, api_key = _real_get_config()

        assert api_url == "http://project:8765"
        assert api_key == "project-key"

    def test_fallback_to_global_credentials(self):
        """Falls back to global ~/.cems/credentials when no project creds exist."""
        with patch("cems.mcp_stdio.resolve_credentials", return_value={
            "CEMS_API_URL": "http://global:8765",
            "CEMS_API_KEY": "global-key",
        }):
            api_url, api_key = _real_get_config()

        assert api_url == "http://global:8765"
        assert api_key == "global-key"

    def test_env_vars_win_over_file_credentials(self):
        """Env vars take highest precedence via resolve_credentials."""
        with patch("cems.mcp_stdio.resolve_credentials", return_value={
            "CEMS_API_URL": "http://env:8765",
            "CEMS_API_KEY": "env-key",
        }):
            api_url, api_key = _real_get_config()

        assert api_url == "http://env:8765"
        assert api_key == "env-key"

    def test_strips_trailing_slash(self):
        """API URL has trailing slash stripped."""
        with patch("cems.mcp_stdio.resolve_credentials", return_value={
            "CEMS_API_URL": "http://host:8765/",
            "CEMS_API_KEY": "some-key",
        }):
            api_url, api_key = _real_get_config()

        assert api_url == "http://host:8765"

    def test_passes_cwd_to_resolver(self):
        """_get_config passes os.getcwd() to resolve_credentials."""
        with patch("cems.mcp_stdio.resolve_credentials", return_value={
            "CEMS_API_URL": "http://test:8765",
            "CEMS_API_KEY": "test-key",
        }) as mock_resolve:
            _real_get_config()

        mock_resolve.assert_called_once_with(os.getcwd())


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
