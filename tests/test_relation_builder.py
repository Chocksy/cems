"""Tests for RelationBuilderJob backfill.

Tests use AsyncMock to mock DocumentStore and memory methods.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cems.config import CEMSConfig

TEST_UUID = "a6e153f9-41c5-4cbc-9a50-74160af381dd"
DOC_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
DOC_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def _run(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_doc(doc_id: str, content: str = "test") -> dict:
    """Create a mock document dict."""
    return {
        "id": doc_id,
        "content": content,
        "category": "general",
        "scope": "personal",
    }


@pytest.fixture
def mock_memory():
    """Create a mock CEMSMemory for RelationBuilderJob."""
    mock = MagicMock()
    mock.config = CEMSConfig(
        user_id=TEST_UUID,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    mock_doc_store = AsyncMock()
    mock._ensure_document_store = AsyncMock(return_value=mock_doc_store)
    mock._doc_store = mock_doc_store

    return mock, mock_doc_store


class TestRelationBuilderJob:
    """Tests for RelationBuilderJob."""

    def test_processes_documents_and_creates_relations(self, mock_memory):
        """Backfill finds neighbors and creates relations."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[_make_doc(DOC_A)])
        doc_store.get_relation_count = AsyncMock(return_value=0)
        doc_store.add_relations = AsyncMock(return_value=1)
        doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": DOC_B, "score": 0.85},
        ])

        # Mock _get_first_chunk_embedding via pool
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"embedding": [0.1] * 1536})
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        doc_store._get_pool = AsyncMock(return_value=mock_pool)

        result = _run(RelationBuilderJob(mock).run_async(limit=10))
        assert result["docs_processed"] == 1
        assert result["relations_created"] == 1

    def test_skips_docs_with_existing_relations(self, mock_memory):
        """Documents with existing relations are skipped unless force=True."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[_make_doc(DOC_A)])
        doc_store.get_relation_count = AsyncMock(return_value=5)  # Already has relations

        result = _run(RelationBuilderJob(mock).run_async(limit=10))
        assert result["docs_processed"] == 0
        assert result["docs_skipped"] == 1
        doc_store.search_chunks.assert_not_called()

    def test_force_reprocesses_all(self, mock_memory):
        """With force=True, even docs with relations are reprocessed."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[_make_doc(DOC_A)])
        doc_store.add_relations = AsyncMock(return_value=0)
        doc_store.search_chunks = AsyncMock(return_value=[])

        # Mock _get_first_chunk_embedding
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"embedding": [0.1] * 1536})
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        doc_store._get_pool = AsyncMock(return_value=mock_pool)

        result = _run(RelationBuilderJob(mock).run_async(limit=10, force=True))
        assert result["docs_processed"] == 1
        # get_relation_count should NOT be called in force mode
        doc_store.get_relation_count.assert_not_called()

    def test_empty_corpus_returns_zeros(self, mock_memory):
        """Empty document list returns all zeros."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[])

        result = _run(RelationBuilderJob(mock).run_async())
        assert result["docs_processed"] == 0
        assert result["relations_created"] == 0
        assert result["docs_skipped"] == 0

    def test_skips_doc_without_embedding(self, mock_memory):
        """Documents without chunk embeddings are skipped."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory
        doc_store.get_all_documents = AsyncMock(return_value=[_make_doc(DOC_A)])
        doc_store.get_relation_count = AsyncMock(return_value=0)

        # No embedding found
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        doc_store._get_pool = AsyncMock(return_value=mock_pool)

        result = _run(RelationBuilderJob(mock).run_async(limit=10))
        assert result["docs_processed"] == 0
        assert result["docs_skipped"] == 1
