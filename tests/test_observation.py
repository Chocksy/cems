"""Tests for observation extraction and API endpoint."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestObservationExtraction:
    """Tests for extract_observations and _parse_observations."""

    def test_parse_valid_observations(self):
        """Parse well-formed JSON array of observations."""
        from cems.llm.observation_extraction import _parse_observations

        response = """[
            {
                "content": "User is building a memory system for AI coding agents",
                "priority": "high",
                "category": "observation"
            },
            {
                "content": "User prefers PostgreSQL over SQLite for production databases",
                "priority": "high",
                "category": "observation"
            }
        ]"""

        result = _parse_observations(response)
        assert len(result) == 2
        assert result[0]["content"] == "User is building a memory system for AI coding agents"
        assert result[0]["priority"] == "high"
        assert result[0]["category"] == "observation"
        assert result[1]["priority"] == "high"

    def test_parse_observations_with_markdown_wrapper(self):
        """Parse observations wrapped in markdown code blocks."""
        from cems.llm.observation_extraction import _parse_observations

        response = """```json
[
    {
        "content": "User decided to use Tailwind CSS for the frontend styling",
        "priority": "high",
        "category": "observation"
    }
]
```"""

        result = _parse_observations(response)
        assert len(result) == 1
        assert "Tailwind CSS" in result[0]["content"]

    def test_parse_observations_skips_short_content(self):
        """Observations shorter than 30 chars are filtered out."""
        from cems.llm.observation_extraction import _parse_observations

        response = """[
            {"content": "Too short", "priority": "high", "category": "observation"},
            {"content": "User is building a comprehensive memory system for coding agents", "priority": "high", "category": "observation"}
        ]"""

        result = _parse_observations(response)
        assert len(result) == 1
        assert "comprehensive memory system" in result[0]["content"]

    def test_parse_observations_caps_long_content(self):
        """Content longer than 300 chars is truncated."""
        from cems.llm.observation_extraction import _parse_observations

        long_content = "A" * 400
        response = f'[{{"content": "{long_content}", "priority": "high", "category": "observation"}}]'

        result = _parse_observations(response)
        assert len(result) == 1
        assert len(result[0]["content"]) == 300  # 297 + "..."

    def test_parse_observations_normalizes_priority(self):
        """Invalid priority values default to medium."""
        from cems.llm.observation_extraction import _parse_observations

        response = '[{"content": "User is deploying the app to production servers now", "priority": "critical", "category": "observation"}]'

        result = _parse_observations(response)
        assert len(result) == 1
        assert result[0]["priority"] == "medium"

    def test_parse_observations_normalizes_category(self):
        """Non-observation categories get normalized via canonical list."""
        from cems.llm.observation_extraction import _parse_observations

        response = '[{"content": "User prefers Docker deployments over manual server setup", "priority": "high", "category": "docker"}]'

        result = _parse_observations(response)
        assert len(result) == 1
        assert result[0]["category"] == "deployment"  # docker -> deployment alias

    def test_parse_observations_max_cap(self):
        """No more than MAX_OBSERVATIONS returned."""
        from cems.llm.observation_extraction import MAX_OBSERVATIONS, _parse_observations

        obs = [
            {"content": f"Observation number {i} about an important project decision", "priority": "high", "category": "observation"}
            for i in range(10)
        ]
        import json
        response = json.dumps(obs)

        result = _parse_observations(response)
        assert len(result) == MAX_OBSERVATIONS

    def test_parse_empty_response(self):
        """Empty or invalid response returns empty list."""
        from cems.llm.observation_extraction import _parse_observations

        assert _parse_observations("") == []
        assert _parse_observations("not json") == []
        assert _parse_observations("null") == []

    def test_extract_observations_short_content_skipped(self):
        """Content shorter than 200 chars is skipped entirely."""
        from cems.llm.observation_extraction import extract_observations

        result = extract_observations("short", project_context="test")
        assert result == []

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_extract_observations_calls_llm(self, mock_openai_class):
        """extract_observations calls LLM with correct model and prompt."""
        from cems.llm.observation_extraction import extract_observations

        # Reset cached client
        import cems.llm.client
        cems.llm.client._client = None

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"content": "User is testing the observation system end to end", "priority": "high", "category": "observation"}]'
        mock_client.chat.completions.create.return_value = mock_response

        content = "A" * 300  # Must be > 200 chars
        result = extract_observations(content, project_context="test/project (main)")

        assert len(result) == 1
        assert "observation system" in result[0]["content"]

        # Verify model used
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "google/gemini-2.5-flash"

        # Cleanup
        cems.llm.client._client = None


class TestNormalizeCategory:
    """Tests for the normalize_category function."""

    def test_canonical_passthrough(self):
        """Canonical categories pass through unchanged."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("observation") == "observation"
        assert normalize_category("database") == "database"
        assert normalize_category("deployment") == "deployment"

    def test_alias_mapping(self):
        """Aliases map to canonical categories."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("docker") == "deployment"
        assert normalize_category("rails") == "ruby"
        assert normalize_category("postgres") == "database"

    def test_case_insensitive(self):
        """Category normalization is case-insensitive."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("Database") == "database"
        assert normalize_category("DEPLOYMENT") == "deployment"

    def test_separator_normalization(self):
        """Spaces and underscores convert to hyphens."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("error handling") == "error-handling"
        assert normalize_category("error_handling") == "error-handling"

    def test_unknown_defaults_to_general(self):
        """Unknown categories default to 'general'."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("something-random") == "general"
        assert normalize_category("") == "general"


class TestObservationAPIHandler:
    """Tests for the api_session_observe endpoint."""

    @pytest.mark.asyncio
    async def test_observe_requires_content(self):
        """POST /api/session/observe without content returns 400."""
        from cems.api.handlers.observation import api_session_observe

        request = MagicMock()
        request.json = AsyncMock(return_value={"session_id": "test"})

        response = await api_session_observe(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    @patch("cems.api.handlers.observation.extract_observations")
    @patch("cems.api.handlers.observation.get_memory")
    async def test_observe_stores_observations(self, mock_get_memory, mock_extract):
        """POST /api/session/observe stores extracted observations."""
        from cems.api.handlers.observation import api_session_observe

        mock_extract.return_value = [
            {"content": "User is building a test system", "priority": "high", "category": "observation"},
        ]

        mock_memory = MagicMock()
        mock_memory.add_async = AsyncMock(return_value={
            "results": [{"id": "mem-123"}]
        })
        mock_get_memory.return_value = mock_memory

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "content": "test transcript content...",
            "session_id": "test-session",
            "source_ref": "project:test/repo",
            "project_context": "test/repo (main)",
        })

        response = await api_session_observe(request)
        assert response.status_code == 200

        import json
        body = json.loads(response.body)
        assert body["success"] is True
        assert body["observations_stored"] == 1
        assert body["observations"][0]["memory_id"] == "mem-123"

        # Verify memory.add_async was called with correct args
        mock_memory.add_async.assert_called_once()
        call_kwargs = mock_memory.add_async.call_args[1]
        assert call_kwargs["category"] == "observation"
        assert call_kwargs["source_ref"] == "project:test/repo"
        assert "observation" in call_kwargs["tags"]

    @pytest.mark.asyncio
    @patch("cems.api.handlers.observation.extract_observations")
    async def test_observe_no_observations_returns_empty(self, mock_extract_obs):
        """POST /api/session/observe with no observations returns success with 0 stored."""
        from cems.api.handlers.observation import api_session_observe

        mock_extract_obs.return_value = []

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "content": "short session with nothing interesting",
        })

        response = await api_session_observe(request)
        assert response.status_code == 200

        import json
        body = json.loads(response.body)
        assert body["observations_stored"] == 0
