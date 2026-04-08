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
    """Tests for _load_context_memories 3-bucket loading with DB-level project filtering."""

    def _make_mock_store(self, responses_by_call: list[list[dict]]):
        """Create a mock store that returns different results per call.

        The bucket loading makes multiple get_all_documents calls:
        - Call 1: project bucket (source_ref_prefix filter)
        - Calls 2-5: profile buckets (one per PROFILE_CATEGORIES)
        - Call 6: recent bucket (no source_ref filter)
        """
        mock_store = AsyncMock()
        mock_store.get_all_documents = AsyncMock(side_effect=responses_by_call)
        return mock_store

    @pytest.mark.asyncio
    async def test_bucket1_uses_source_ref_prefix_filter(self):
        """Bucket 1 should pass source_ref_prefix to DB query, not filter client-side."""
        from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories

        cems_doc = {
            "id": "cems-1", "content": "CEMS arch", "category": "architecture",
            "source_ref": "project:chocksy/cems", "created_at": "2026-01-01",
        }

        # Bucket 1 returns project docs, buckets 2-5 (profile) return empty, bucket 6 (recent) returns empty
        responses = [[cems_doc]] + [[] for _ in PROFILE_CATEGORIES] + [[]]
        mock_store = self._make_mock_store(responses)

        await _load_context_memories(mock_store, user_id="u1", project="chocksy/cems")

        # First call should have source_ref_prefix set
        first_call = mock_store.get_all_documents.call_args_list[0]
        assert first_call.kwargs.get("source_ref_prefix") == "project:chocksy/cems"

    @pytest.mark.asyncio
    async def test_bucket3_excludes_other_project_memories(self):
        """Bucket 3 (recent) should NOT include memories from other projects."""
        from datetime import UTC, datetime
        from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories

        now = datetime.now(UTC)
        gooseherd_doc = {
            "id": "goose-1", "content": "Gooseherd stuff", "category": "session-summary",
            "source_ref": "project:chocksy/gooseherd", "created_at": now,
        }
        cems_recent = {
            "id": "cems-recent", "content": "CEMS recent", "category": "session-summary",
            "source_ref": "project:chocksy/cems", "created_at": now,
        }
        no_project = {
            "id": "general-1", "content": "General memory", "category": "general",
            "source_ref": None, "created_at": now,
        }

        # Bucket 1 empty, buckets 2-5 empty, bucket 6 has mixed project docs
        responses = [[]] + [[] for _ in PROFILE_CATEGORIES] + [[gooseherd_doc, cems_recent, no_project]]
        mock_store = self._make_mock_store(responses)

        result = await _load_context_memories(mock_store, user_id="u1", project="chocksy/cems")

        ids = [d["id"] for d in result]
        assert "goose-1" not in ids, "Other-project memory should be excluded from bucket 3"
        assert "cems-recent" in ids, "Same-project recent memory should be included"
        assert "general-1" in ids, "No-project memory should be included"

    @pytest.mark.asyncio
    async def test_profile_categories_always_included(self):
        """Profile memories (preferences, guidelines, etc.) should always be loaded."""
        from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories

        pref_doc = {
            "id": "pref-1", "content": "User likes Python", "category": "preferences",
            "source_ref": None, "created_at": "2026-01-01",
        }

        # Bucket 1 empty, one profile category returns doc, rest empty, bucket 6 empty
        responses = [[]]
        for cat in PROFILE_CATEGORIES:
            responses.append([pref_doc] if cat == "preferences" else [])
        responses.append([])  # recent
        mock_store = self._make_mock_store(responses)

        result = await _load_context_memories(mock_store, user_id="u1", project="chocksy/cems")

        ids = [d["id"] for d in result]
        assert "pref-1" in ids

    @pytest.mark.asyncio
    async def test_deduplicates_across_buckets(self):
        """Same doc appearing in project + profile buckets should appear only once."""
        from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories

        shared_doc = {
            "id": "shared-1", "content": "CEMS pref", "category": "preferences",
            "source_ref": "project:chocksy/cems", "created_at": "2026-01-01",
        }

        # Doc appears in both bucket 1 (project) and bucket 2 (preferences)
        responses = [[shared_doc]]
        for cat in PROFILE_CATEGORIES:
            responses.append([shared_doc] if cat == "preferences" else [])
        responses.append([])  # recent
        mock_store = self._make_mock_store(responses)

        result = await _load_context_memories(mock_store, user_id="u1", project="chocksy/cems")

        ids = [d["id"] for d in result]
        assert ids.count("shared-1") == 1, "Dedup failed — doc appears multiple times"

    @pytest.mark.asyncio
    async def test_works_without_project(self):
        """When project=None, skip bucket 1, still load profile + recent."""
        from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories

        pref_doc = {
            "id": "pref-1", "content": "User likes Python", "category": "preferences",
            "source_ref": None, "created_at": "2026-03-22",
        }

        # No bucket 1 call (project=None), profile categories, then recent
        responses = []
        for cat in PROFILE_CATEGORIES:
            responses.append([pref_doc] if cat == "preferences" else [])
        responses.append([])  # recent
        mock_store = self._make_mock_store(responses)

        result = await _load_context_memories(mock_store, user_id="u1", project=None)

        assert len(result) >= 1
        # Should NOT have called with source_ref_prefix
        for call in mock_store.get_all_documents.call_args_list:
            assert call.kwargs.get("source_ref_prefix") is None

    @pytest.mark.asyncio
    async def test_context_budget_respected(self):
        """Should stop adding memories when MAX_CONTEXT_CHARS is reached."""
        from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories
        from unittest.mock import patch

        big_doc = {
            "id": "big-1", "content": "x" * 800_000, "category": "architecture",
            "source_ref": "project:chocksy/cems", "created_at": "2026-01-01",
        }
        small_doc = {
            "id": "small-1", "content": "small", "category": "preferences",
            "source_ref": None, "created_at": "2026-01-01",
        }

        responses = [[big_doc]]
        for cat in PROFILE_CATEGORIES:
            responses.append([small_doc] if cat == "preferences" else [])
        responses.append([])
        mock_store = self._make_mock_store(responses)

        with patch("cems.agentic.search.MAX_CONTEXT_CHARS", 500_000):
            result = await _load_context_memories(mock_store, user_id="u1", project="chocksy/cems")

        # big_doc exceeds budget, should be skipped
        ids = [d["id"] for d in result]
        assert "big-1" not in ids

    @pytest.mark.asyncio
    async def test_relevance_sorting_within_buckets(self):
        """Memories with higher relevant_count should come first within a bucket."""
        from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories

        low_rel = {
            "id": "low-1", "content": "low relevance", "category": "architecture",
            "source_ref": "project:chocksy/cems", "created_at": "2026-01-01",
            "relevant_count": 0, "noise_count": 5,
        }
        high_rel = {
            "id": "high-1", "content": "high relevance", "category": "architecture",
            "source_ref": "project:chocksy/cems", "created_at": "2026-01-01",
            "relevant_count": 10, "noise_count": 0,
        }

        # Return in wrong order — low relevance first
        responses = [[low_rel, high_rel]] + [[] for _ in PROFILE_CATEGORIES] + [[]]
        mock_store = self._make_mock_store(responses)

        result = await _load_context_memories(mock_store, user_id="u1", project="chocksy/cems")

        ids = [d["id"] for d in result]
        assert ids.index("high-1") < ids.index("low-1"), "High-relevance should come first"


class TestAgenticSearchAsync:
    """Tests for agentic_search_async with mocked LLM."""

    @pytest.mark.asyncio
    @patch("cems.agentic.search._load_entity_summaries", new_callable=AsyncMock, return_value=[])
    @patch("cems.agentic.search.get_client")
    async def test_returns_results(self, mock_get_client, _mock_entities):
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
        assert "entities" in result
        assert "memories" in result

    @pytest.mark.asyncio
    @patch("cems.agentic.search._load_entity_summaries", new_callable=AsyncMock, return_value=[])
    async def test_empty_memories_returns_empty(self, _mock_entities):
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
        assert result["entities"] == []

    @pytest.mark.asyncio
    @patch("cems.agentic.search.get_client")
    async def test_entity_picker_returns_entities(self, mock_get_client):
        """Entity picker agent finds relevant entity pages alongside memories."""
        from cems.agentic.search import agentic_search_async

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # Entity picker returns entity ID, memory agents return memory ID
        mock_client.complete.side_effect = [
            '["ent12345"]',   # entity picker
            '["mem12345"]',   # direct seeker
            '["mem12345"]',   # inference engine
            '["mem12345"]',   # temporal navigator
        ]

        # Mock entity summaries via _load_entity_summaries
        entity_summaries = [
            {"id": "ent12345-full-uuid", "title": "Stripe Integration",
             "summary": "Webhook-based payment flow", "sources": "20"},
        ]

        mock_store = AsyncMock()
        mock_store.get_all_documents = AsyncMock(return_value=[
            {
                "id": "mem12345-full-uuid",
                "content": "Keep Stripe requests sampled",
                "category": "preferences",
                "source_ref": "project:test",
                "created_at": "2026-03-22",
                "scope": "personal",
                "tags": [],
            }
        ])

        with patch("cems.agentic.search._load_entity_summaries",
                    new_callable=AsyncMock, return_value=entity_summaries):
            result = await agentic_search_async(
                document_store=mock_store,
                user_id="user-1",
                query="How does Stripe integration work?",
                project="test",
            )

        assert result["mode"] == "agentic"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["title"] == "Stripe Integration"
        assert len(result["memories"]) >= 1
        assert result["entity_candidates"] == 1
