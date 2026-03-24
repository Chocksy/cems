"""Tests for observation parsing and category normalization."""

import pytest


class TestObservationExtraction:
    """Tests for _parse_observations."""

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
        """Non-observation categories get mapped to canonical categories."""
        from cems.llm.observation_extraction import _parse_observations

        response = '[{"content": "User prefers Docker deployments over manual server setup", "priority": "high", "category": "Docker Config"}]'

        result = _parse_observations(response)
        assert len(result) == 1
        assert result[0]["category"] == "infrastructure"  # "Docker Config" → "infrastructure" via alias

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

class TestNormalizeCategory:
    """Tests for the normalize_category function."""

    def test_passthrough(self):
        """Categories pass through as lowercase-hyphenated."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("observation") == "observation"
        assert normalize_category("database") == "database"
        assert normalize_category("deployment") == "deployment"

    def test_alias_mapping(self):
        """LLM-generated categories map to canonical categories via aliases."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("docker") == "infrastructure"
        assert normalize_category("rails") == "general"  # no alias, falls through
        assert normalize_category("ai") == "general"
        assert normalize_category("sql") == "database"

    def test_case_insensitive(self):
        """Category normalization is case-insensitive."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("Database") == "database"
        assert normalize_category("DEPLOYMENT") == "deployment"

    def test_prefix_and_suffix_matching(self):
        """Compound categories match via prefix or suffix to canonical."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("database-migration") == "database"
        assert normalize_category("testing-config") == "testing"
        assert normalize_category("docker-config") == "infrastructure"
        assert normalize_category("svelte-frontend") == "frontend"

    def test_unrecognized_defaults_to_general(self):
        """Unrecognized categories and empty strings default to 'general'."""
        from cems.llm.learning_extraction import normalize_category

        assert normalize_category("") == "general"
        assert normalize_category("zsh-completion") == "general"
        assert normalize_category("Payload CMS") == "general"
