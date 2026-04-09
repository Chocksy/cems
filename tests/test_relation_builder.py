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
        "source_ref": None,
        "tags": [],
        "created_at": None,
    }


def _mock_pool_with_rows(rows):
    """Create a mock pool that returns given rows from conn.fetch."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=rows)
    mock_conn.fetchrow = AsyncMock(return_value={"embedding": [0.1] * 1536})
    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_pool


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

    def test_processes_unlinked_docs_and_creates_relations(self, mock_memory):
        """Backfill finds neighbors and creates relations for unlinked docs."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory

        # SQL query returns unlinked docs
        mock_rows = [MagicMock(**{
            "__getitem__": lambda self, k: {
                "id": DOC_A, "content": "test", "category": "general",
                "scope": "personal", "source_ref": None, "tags": [], "created_at": None,
            }[k],
        })]
        doc_store._get_pool = AsyncMock(return_value=_mock_pool_with_rows(mock_rows))
        doc_store.add_relations = AsyncMock(return_value=1)
        doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": DOC_B, "score": 0.85},
        ])

        result = _run(RelationBuilderJob(mock).run_async(limit=10))
        assert result["docs_processed"] == 1
        assert result["relations_created"] == 1

    def test_no_unlinked_docs_returns_zeros(self, mock_memory):
        """When all docs have relations, returns zeros."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory
        doc_store._get_pool = AsyncMock(return_value=_mock_pool_with_rows([]))

        result = _run(RelationBuilderJob(mock).run_async(limit=10))
        assert result["docs_processed"] == 0
        assert result["relations_created"] == 0
        assert result["docs_skipped"] == 0

    def test_force_uses_get_all_documents(self, mock_memory):
        """With force=True, uses get_all_documents instead of unlinked query."""
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
        doc_store.get_all_documents.assert_called_once()

    def test_skips_doc_without_embedding(self, mock_memory):
        """Documents without chunk embeddings are skipped."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory

        # SQL query returns one unlinked doc
        mock_rows = [MagicMock(**{
            "__getitem__": lambda self, k: {
                "id": DOC_A, "content": "test", "category": "general",
                "scope": "personal", "source_ref": None, "tags": [], "created_at": None,
            }[k],
        })]
        # Return rows from _get_unlinked_documents, but no embedding
        pool_for_unlinked = _mock_pool_with_rows(mock_rows)
        # Override fetchrow to return None (no embedding)
        mock_conn_no_embed = AsyncMock()
        mock_conn_no_embed.fetchrow = AsyncMock(return_value=None)
        mock_conn_no_embed.fetch = AsyncMock(return_value=mock_rows)
        pool_for_unlinked.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn_no_embed)
        doc_store._get_pool = AsyncMock(return_value=pool_for_unlinked)

        result = _run(RelationBuilderJob(mock).run_async(limit=10))
        assert result["docs_processed"] == 0
        assert result["docs_skipped"] == 1

    def test_empty_corpus_returns_zeros(self, mock_memory):
        """Empty document list returns all zeros."""
        from cems.maintenance.relation_builder import RelationBuilderJob

        mock, doc_store = mock_memory
        doc_store._get_pool = AsyncMock(return_value=_mock_pool_with_rows([]))

        result = _run(RelationBuilderJob(mock).run_async())
        assert result["docs_processed"] == 0
        assert result["relations_created"] == 0
        assert result["docs_skipped"] == 0
