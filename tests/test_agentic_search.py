"""Tests for the agentic search server module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestParseAgentResponse:
    """Tests for _parse_agent_response in the server module."""

    def test_valid_json_with_hex_ids(self):
        from cems.agentic.search import _parse_agent_response

        valid_ids = {"abc12345", "def67890"}
        result = _parse_agent_response('["abc12345", "def67890"]', valid_ids)
        assert result == ["abc12345", "def67890"]

    def test_filters_invalid_ids(self):
        from cems.agentic.search import _parse_agent_response

        valid_ids = {"abc12345"}
        result = _parse_agent_response('["abc12345", "ffffffff"]', valid_ids)
        assert result == ["abc12345"]

    def test_regex_fallback_for_hex_ids(self):
        from cems.agentic.search import _parse_agent_response

        valid_ids = {"abc12345"}
        result = _parse_agent_response("The most relevant memory is abc12345.", valid_ids)
        assert result == ["abc12345"]

    def test_empty_response(self):
        from cems.agentic.search import _parse_agent_response

        result = _parse_agent_response("", {"abc12345"})
        assert result == []


class TestFormatMemoriesForAgents:
    """Tests for _format_memories_for_agents."""

    def test_formats_with_metadata(self):
        from cems.agentic.search import _format_memories_for_agents

        memories = [
            {
                "id": "abc12345-full-uuid",
                "content": "User likes Python",
                "category": "preferences",
                "source_ref": "project:chocksy/cems",
                "created_at": "2026-03-22",
            }
        ]
        result = _format_memories_for_agents(memories)
        assert "abc12345" in result
        assert "User likes Python" in result
        assert "preferences" in result
        assert "chocksy/cems" in result

    def test_empty_memories(self):
        from cems.agentic.search import _format_memories_for_agents

        result = _format_memories_for_agents([])
        assert result == ""


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion in server module."""

    def test_basic_fusion(self):
        from cems.agentic.search import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
        assert result[0] == "a"

    def test_empty(self):
        from cems.agentic.search import reciprocal_rank_fusion

        assert reciprocal_rank_fusion([]) == []


class TestProfileCategories:
    """Tests for profile category constants."""

    def test_profile_categories_defined(self):
        from cems.agentic.search import PROFILE_CATEGORIES

        assert "preferences" in PROFILE_CATEGORIES
        assert "guidelines" in PROFILE_CATEGORIES
        assert "gate-rules" in PROFILE_CATEGORIES
        assert "category-summary" in PROFILE_CATEGORIES

    def test_recent_days_defined(self):
        from cems.agentic.search import RECENT_DAYS

        assert RECENT_DAYS == 14


class TestLoadContextMemories:
    """Tests for _load_context_memories 3-bucket loading."""

    @pytest.mark.asyncio
    async def test_deduplicates_across_buckets(self):
        from cems.agentic.search import _load_context_memories

        # Same doc appears in both project and profile buckets
        shared_doc = {
            "id": "doc-1",
            "content": "User prefers Python",
            "category": "preferences",
            "source_ref": "project:chocksy/cems",
            "created_at": "2026-03-22",
        }

        mock_store = AsyncMock()
        mock_store.get_all_documents = AsyncMock(return_value=[shared_doc])

        result = await _load_context_memories(
            mock_store, user_id="user-1", project="chocksy/cems"
        )

        # Should appear only once despite matching project + profile
        ids = [d["id"] for d in result]
        assert ids.count("doc-1") == 1

    @pytest.mark.asyncio
    async def test_loads_project_memories(self):
        from cems.agentic.search import _load_context_memories

        project_doc = {
            "id": "proj-1",
            "content": "CEMS uses pgvector",
            "category": "architecture",
            "source_ref": "project:chocksy/cems",
            "created_at": "2026-01-01",
        }
        other_doc = {
            "id": "other-1",
            "content": "Gooseherd uses Docker",
            "category": "architecture",
            "source_ref": "project:chocksy/gooseherd",
            "created_at": "2026-01-01",
        }

        mock_store = AsyncMock()
        # First call returns all docs (project bucket), subsequent calls for profile/recent
        mock_store.get_all_documents = AsyncMock(return_value=[project_doc, other_doc])

        result = await _load_context_memories(
            mock_store, user_id="user-1", project="chocksy/cems"
        )

        # Project doc should be included (matches source_ref)
        ids = [d["id"] for d in result]
        assert "proj-1" in ids

    @pytest.mark.asyncio
    async def test_works_without_project(self):
        from cems.agentic.search import _load_context_memories

        doc = {
            "id": "doc-1",
            "content": "User prefers Python",
            "category": "preferences",
            "source_ref": None,
            "created_at": "2026-03-22",
        }

        mock_store = AsyncMock()
        mock_store.get_all_documents = AsyncMock(return_value=[doc])

        result = await _load_context_memories(
            mock_store, user_id="user-1", project=None
        )

        # Should still load profile + recent buckets
        assert len(result) >= 1


class TestAgenticSearchAsync:
    """Tests for agentic_search_async with mocked LLM."""

    @pytest.mark.asyncio
    @patch("cems.agentic.search.get_client")
    async def test_returns_results(self, mock_get_client):
        from cems.agentic.search import agentic_search_async

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.complete.return_value = '["abc12345"]'

        mock_store = AsyncMock()
        mock_store.get_all_documents = AsyncMock(return_value=[
            {
                "id": "abc12345-full-uuid-here",
                "content": "User prefers Python",
                "category": "preferences",
                "source_ref": "project:test",
                "created_at": "2026-03-22",
                "scope": "personal",
                "tags": [],
            }
        ])

        result = await agentic_search_async(
            document_store=mock_store,
            user_id="user-1",
            query="What language does the user prefer?",
            project="test",
        )

        assert result["mode"] == "agentic"
        assert result["count"] >= 0

    @pytest.mark.asyncio
    async def test_empty_memories_returns_empty(self):
        from cems.agentic.search import agentic_search_async

        mock_store = AsyncMock()
        mock_store.get_all_documents = AsyncMock(return_value=[])

        result = await agentic_search_async(
            document_store=mock_store,
            user_id="user-1",
            query="anything",
        )

        assert result["count"] == 0
        assert result["mode"] == "agentic"
