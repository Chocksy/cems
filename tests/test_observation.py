"""Tests for observation extraction and category normalization."""

import os
from unittest.mock import MagicMock, patch

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

    def test_parse_observations_preserves_long_content(self):
        """Long content is preserved without truncation."""
        from cems.llm.observation_extraction import _parse_observations

        long_content = "A" * 400
        response = f'[{{"content": "{long_content}", "priority": "high", "category": "observation"}}]'

        result = _parse_observations(response)
        assert len(result) == 1
        assert len(result[0]["content"]) == 400  # No truncation

    def test_parse_observations_normalizes_priority(self):
        """Invalid priority values default to medium."""
        from cems.llm.observation_extraction import _parse_observations

        response = '[{"content": "User is deploying the app to production servers now", "priority": "critical", "category": "observation"}]'

        result = _parse_observations(response)
        assert len(result) == 1
        assert result[0]["priority"] == "medium"

    def test_parse_observations_normalizes_category(self):
        """Non-observation categories get cleaned (lowercased, hyphenated)."""
        from cems.llm.observation_extraction import _parse_observations

        response = '[{"content": "User prefers Docker deployments over manual server setup", "priority": "high", "category": "Docker Config"}]'

        result = _parse_observations(response)
        assert len(result) == 1
        assert result[0]["category"] == "docker-config"  # lowercased + hyphenated

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

    def test_passthrough(self):
        """Categories pass through as lowercase-hyphenated."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("observation") == "observation"
        assert normalize_category("database") == "database"
        assert normalize_category("deployment") == "deployment"

    def test_free_text_passthrough(self):
        """LLM-generated categories pass through as-is (lowercased)."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("docker") == "docker"
        assert normalize_category("rails") == "rails"
        assert normalize_category("Payload CMS") == "payload-cms"

    def test_case_insensitive(self):
        """Category normalization is case-insensitive."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("Database") == "database"
        assert normalize_category("DEPLOYMENT") == "deployment"

    def test_separator_normalization(self):
        """Spaces, underscores, and slashes convert to hyphens."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("error handling") == "error-handling"
        assert normalize_category("error_handling") == "error-handling"
        assert normalize_category("cems/observer") == "cems-observer"

    def test_empty_defaults_to_general(self):
        """Empty string defaults to 'general'."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("") == "general"
