"""Tests for observation reflection and reflector job."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestReflectObservations:
    """Tests for the reflect_observations LLM function."""

    def test_returns_input_if_fewer_than_3(self):
        """Should return input as-is when fewer than 3 observations."""
        from cems.llm.observation_reflection import reflect_observations

        obs = [
            {"content": "User deploys to Coolify on Hetzner", "priority": "high"},
            {"content": "User prefers Docker for local dev", "priority": "medium"},
        ]
        result = reflect_observations(obs)
        assert result == obs

    @patch("cems.llm.observation_reflection.get_client")
    def test_calls_llm_with_formatted_observations(self, mock_get_client):
        """Should format observations with numbered list and priorities."""
        from cems.llm.observation_reflection import reflect_observations

        mock_client = MagicMock()
        mock_client.complete.return_value = """[
            {"content": "User deploys CEMS to Coolify on Hetzner, uses Docker locally", "priority": "high", "category": "observation"}
        ]"""
        mock_get_client.return_value = mock_client

        obs = [
            {"content": "User deploys to Coolify on Hetzner", "priority": "high"},
            {"content": "User prefers Docker for local dev", "priority": "medium"},
            {"content": "User deployed CEMS to production via Coolify", "priority": "high"},
        ]

        result = reflect_observations(obs, project_context="chocksy/cems")
        assert len(result) == 1
        assert "Coolify" in result[0]["content"]
        assert "Docker" in result[0]["content"]

        # Verify LLM was called with numbered list
        call_args = mock_client.complete.call_args
        prompt = call_args.kwargs.get("prompt", call_args[1].get("prompt", ""))
        assert "1. [high]" in prompt
        assert "2. [medium]" in prompt
        assert "3. [high]" in prompt

    @patch("cems.llm.observation_reflection.get_client")
    def test_handles_llm_failure(self, mock_get_client):
        """Should return empty list when LLM call fails."""
        from cems.llm.observation_reflection import reflect_observations

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        obs = [
            {"content": f"Observation number {i} with enough content to pass validation", "priority": "medium"}
            for i in range(5)
        ]

        result = reflect_observations(obs)
        assert result == []

    @patch("cems.llm.observation_reflection.get_client")
    def test_handles_empty_response(self, mock_get_client):
        """Should return empty list when LLM returns empty."""
        from cems.llm.observation_reflection import reflect_observations

        mock_client = MagicMock()
        mock_client.complete.return_value = ""
        mock_get_client.return_value = mock_client

        obs = [
            {"content": f"Observation number {i} with enough content to pass validation", "priority": "medium"}
            for i in range(5)
        ]

        result = reflect_observations(obs)
        assert result == []

    @patch("cems.llm.observation_reflection.get_client")
    def test_uses_gemini_flash_by_default(self, mock_get_client):
        """Should use Gemini 2.5 Flash and fast_route=False."""
        from cems.llm.observation_reflection import reflect_observations

        mock_client = MagicMock()
        mock_client.complete.return_value = "[]"
        mock_get_client.return_value = mock_client

        obs = [
            {"content": f"Observation number {i} with enough content to pass validation", "priority": "medium"}
            for i in range(4)
        ]

        reflect_observations(obs)

        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["model"] == "google/gemini-2.5-flash"
        assert call_kwargs["fast_route"] is False


class TestParseReflected:
    """Tests for _parse_reflected validation."""

    def test_parse_valid_json(self):
        """Should parse well-formed JSON array."""
        from cems.llm.observation_reflection import _parse_reflected

        response = """[
            {"content": "User deploys CEMS to Coolify on Hetzner via Tailscale", "priority": "high", "category": "observation"},
            {"content": "User prefers Docker Compose for local development", "priority": "medium", "category": "observation"}
        ]"""

        result = _parse_reflected(response, max_count=5)
        assert len(result) == 2
        assert result[0]["priority"] == "high"
        assert result[1]["priority"] == "medium"

    def test_filters_short_content(self):
        """Should filter observations shorter than 30 chars."""
        from cems.llm.observation_reflection import _parse_reflected

        response = """[
            {"content": "Too short", "priority": "high"},
            {"content": "User deploys CEMS to production using Coolify platform", "priority": "high"}
        ]"""

        result = _parse_reflected(response, max_count=5)
        assert len(result) == 1
        assert "Coolify" in result[0]["content"]

    def test_truncates_long_content(self):
        """Should truncate observations longer than 300 chars."""
        from cems.llm.observation_reflection import _parse_reflected

        long_content = "x" * 350
        response = f'[{{"content": "{long_content}", "priority": "high"}}]'

        result = _parse_reflected(response, max_count=5)
        assert len(result) == 1
        assert len(result[0]["content"]) == 300
        assert result[0]["content"].endswith("...")

    def test_normalizes_invalid_priority(self):
        """Should default to 'medium' for invalid priorities."""
        from cems.llm.observation_reflection import _parse_reflected

        response = '[{"content": "User uses PostgreSQL for all production databases", "priority": "critical"}]'

        result = _parse_reflected(response, max_count=5)
        assert len(result) == 1
        assert result[0]["priority"] == "medium"

    def test_handles_markdown_wrapper(self):
        """Should handle JSON wrapped in markdown code blocks."""
        from cems.llm.observation_reflection import _parse_reflected

        response = """```json
[{"content": "User prefers Tailwind CSS for styling all frontend components", "priority": "medium", "category": "observation"}]
```"""

        result = _parse_reflected(response, max_count=5)
        assert len(result) == 1

    def test_handles_malformed_json(self):
        """Should return empty list for unparseable response."""
        from cems.llm.observation_reflection import _parse_reflected

        result = _parse_reflected("this is not json at all", max_count=5)
        assert result == []

    def test_warns_when_no_compression(self):
        """Should still return results even when output >= input count."""
        from cems.llm.observation_reflection import _parse_reflected

        response = """[
            {"content": "Observation one with enough characters to be valid content", "priority": "high"},
            {"content": "Observation two with enough characters to be valid content", "priority": "medium"},
            {"content": "Observation three with enough characters to be valid content", "priority": "low"}
        ]"""

        result = _parse_reflected(response, max_count=3)
        assert len(result) == 3  # Still returns them, just warns


class TestObservationReflector:
    """Tests for the ObservationReflector job class."""

    @pytest.mark.asyncio
    async def test_skips_when_no_observations(self):
        """Should return zero stats when no observations exist."""
        from cems.maintenance.observation_reflector import ObservationReflector

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()
        mock_doc_store.get_documents_by_category.return_value = []
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        assert result["projects_processed"] == 0
        assert result["observations_before"] == 0

    @pytest.mark.asyncio
    async def test_skips_projects_below_threshold(self):
        """Should skip projects with fewer than 10 observations."""
        from cems.maintenance.observation_reflector import ObservationReflector

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()

        # 5 observations — below threshold
        obs = [
            {"id": f"id-{i}", "content": f"Observation {i}", "source_ref": "project:test/repo",
             "priority": "medium", "created_at": f"2026-02-0{i+1}"}
            for i in range(5)
        ]
        mock_doc_store.get_documents_by_category.return_value = obs
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        assert result["projects_processed"] == 0
        assert result["observations_before"] == 5
        assert result["observations_after"] == 5

    @pytest.mark.asyncio
    @patch("cems.maintenance.observation_reflector.reflect_observations")
    async def test_consolidates_observations_above_threshold(self, mock_reflect):
        """Should consolidate when project has >= 10 observations."""
        from cems.maintenance.observation_reflector import ObservationReflector

        # 12 observations → should trigger reflection
        obs = [
            {"id": f"id-{i}", "content": f"Observation {i} with enough content to pass",
             "source_ref": "project:test/repo", "priority": "medium",
             "tags": ["observation", "medium"],
             "created_at": f"2026-01-{i+10:02d}"}
            for i in range(12)
        ]

        # LLM returns 5 consolidated observations
        mock_reflect.return_value = [
            {"content": f"Consolidated observation {i} with enough content", "priority": "high", "category": "observation"}
            for i in range(5)
        ]

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()
        mock_doc_store.get_documents_by_category.return_value = obs
        mock_doc_store.delete_document.return_value = True
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"
        mock_memory.add_async = AsyncMock(return_value={"results": [{"id": "new-id"}]})

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        assert result["projects_processed"] == 1
        assert result["observations_before"] == 12
        assert result["observations_after"] == 5
        assert result["observations_removed"] == 12

        # Verify soft-delete was called for all originals
        assert mock_doc_store.delete_document.call_count == 12
        # Verify soft-delete (not hard)
        for call in mock_doc_store.delete_document.call_args_list:
            assert call.kwargs.get("hard", call.args[1] if len(call.args) > 1 else False) is False

        # Verify new observations were stored with correct metadata
        assert mock_memory.add_async.call_count == 5
        for call in mock_memory.add_async.call_args_list:
            assert call.kwargs["category"] == "observation"
            assert "reflected" in call.kwargs["tags"]
            assert call.kwargs["source_ref"] == "project:test/repo"

    @pytest.mark.asyncio
    @patch("cems.maintenance.observation_reflector.reflect_observations")
    async def test_skips_when_reflection_fails(self, mock_reflect):
        """Should keep originals when LLM returns empty."""
        from cems.maintenance.observation_reflector import ObservationReflector

        obs = [
            {"id": f"id-{i}", "content": f"Observation {i} with enough content to pass",
             "source_ref": "project:test/repo", "priority": "medium",
             "created_at": f"2026-01-{i+10:02d}"}
            for i in range(12)
        ]

        mock_reflect.return_value = []

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()
        mock_doc_store.get_documents_by_category.return_value = obs
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        # Should keep all originals
        assert result["projects_processed"] == 0
        assert result["observations_after"] == 12
        assert result["observations_removed"] == 0

    @pytest.mark.asyncio
    @patch("cems.maintenance.observation_reflector.reflect_observations")
    async def test_skips_when_no_compression(self, mock_reflect):
        """Should keep originals when LLM produces >= input count."""
        from cems.maintenance.observation_reflector import ObservationReflector

        obs = [
            {"id": f"id-{i}", "content": f"Observation {i} with enough content to pass",
             "source_ref": "project:test/repo", "priority": "medium",
             "created_at": f"2026-01-{i+10:02d}"}
            for i in range(10)
        ]

        # LLM returns same number — no compression
        mock_reflect.return_value = [
            {"content": f"Reflected {i} with enough content to pass validation", "priority": "medium", "category": "observation"}
            for i in range(10)
        ]

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()
        mock_doc_store.get_documents_by_category.return_value = obs
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        assert result["projects_processed"] == 0
        assert result["observations_after"] == 10
        assert result["observations_removed"] == 0

    @pytest.mark.asyncio
    @patch("cems.maintenance.observation_reflector.reflect_observations")
    async def test_handles_multiple_projects(self, mock_reflect):
        """Should process each project independently."""
        from cems.maintenance.observation_reflector import ObservationReflector

        # Project A: 12 observations (should consolidate)
        obs_a = [
            {"id": f"a-{i}", "content": f"Project A observation {i} with content",
             "source_ref": "project:org/repo-a", "priority": "medium",
             "created_at": f"2026-01-{i+10:02d}"}
            for i in range(12)
        ]
        # Project B: 5 observations (below threshold)
        obs_b = [
            {"id": f"b-{i}", "content": f"Project B observation {i} with content",
             "source_ref": "project:org/repo-b", "priority": "medium",
             "created_at": f"2026-01-{i+10:02d}"}
            for i in range(5)
        ]

        all_obs = obs_a + obs_b

        mock_reflect.return_value = [
            {"content": "Consolidated A observation with sufficient length", "priority": "high", "category": "observation"}
            for _ in range(4)
        ]

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()
        mock_doc_store.get_documents_by_category.return_value = all_obs
        mock_doc_store.delete_document.return_value = True
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"
        mock_memory.add_async = AsyncMock(return_value={"results": [{"id": "new-id"}]})

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        # Only project A should be processed
        assert result["projects_processed"] == 1
        assert result["observations_before"] == 17
        # Project A: 4 consolidated + Project B: 5 untouched
        assert result["observations_after"] == 9
        assert result["observations_removed"] == 12

    @pytest.mark.asyncio
    @patch("cems.maintenance.observation_reflector.reflect_observations")
    async def test_handles_no_project_observations(self, mock_reflect):
        """Should handle observations without source_ref."""
        from cems.maintenance.observation_reflector import ObservationReflector

        obs = [
            {"id": f"id-{i}", "content": f"Unscoped observation {i} with enough content",
             "source_ref": None, "priority": "medium",
             "created_at": f"2026-01-{i+10:02d}"}
            for i in range(12)
        ]

        mock_reflect.return_value = [
            {"content": "Consolidated unscoped observation with content", "priority": "medium", "category": "observation"}
            for _ in range(3)
        ]

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()
        mock_doc_store.get_documents_by_category.return_value = obs
        mock_doc_store.delete_document.return_value = True
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"
        mock_memory.add_async = AsyncMock(return_value={"results": [{"id": "new-id"}]})

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        assert result["projects_processed"] == 1

        # Verify source_ref=None was passed to add_async
        for call in mock_memory.add_async.call_args_list:
            assert call.kwargs["source_ref"] is None

    @pytest.mark.asyncio
    @patch("cems.maintenance.observation_reflector.reflect_observations")
    async def test_preserves_originals_on_partial_store_failure(self, mock_reflect):
        """Should NOT delete originals if some consolidated stores fail."""
        from cems.maintenance.observation_reflector import ObservationReflector

        obs = [
            {"id": f"id-{i}", "content": f"Observation {i} with enough content to pass",
             "source_ref": "project:test/repo", "priority": "medium",
             "created_at": f"2026-01-{i+10:02d}"}
            for i in range(12)
        ]

        # LLM returns 5 consolidated observations
        mock_reflect.return_value = [
            {"content": f"Consolidated observation {i} with enough content", "priority": "high", "category": "observation"}
            for i in range(5)
        ]

        mock_memory = MagicMock()
        mock_doc_store = AsyncMock()
        mock_doc_store.get_documents_by_category.return_value = obs
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
        mock_memory.config.user_id = "test-user"

        # add_async succeeds twice, then fails for the rest
        call_count = 0

        async def flaky_add(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise Exception("Embedding API down")
            return {"results": [{"id": f"new-{call_count}"}]}

        mock_memory.add_async = flaky_add

        reflector = ObservationReflector(mock_memory)
        result = await reflector.run_async()

        # Should NOT have deleted originals since store was partial
        assert mock_doc_store.delete_document.call_count == 0
        # Originals preserved in count
        assert result["observations_removed"] == 0
        assert result["observations_after"] == 12
        assert result["projects_processed"] == 0
