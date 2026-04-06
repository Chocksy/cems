"""Tests for auto-relations: DocumentStore.add_relations(), auto-link, and add_async integration.

Tests use AsyncMock to mock DB pool and verify SQL calls + behavior.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from cems.db.document_store import DocumentStore
from cems.memory.write import _auto_link_relations, AUTO_RELATION_THRESHOLD


# Valid UUIDs for testing
DOC_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
DOC_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"
USER_ID = "11111111-1111-1111-1111-111111111111"


def _run(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def doc_store():
    """Create a DocumentStore with mocked pool."""
    ds = DocumentStore.__new__(DocumentStore)
    ds.database_url = "postgresql://test:test@localhost/test"
    ds._pool = None
    ds._pool_size = 10

    # Mock the connection pool
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=0)

    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    ds._pool = mock_pool
    ds._mock_conn = mock_conn  # Expose for assertions
    return ds


class TestAddRelations:
    """Tests for DocumentStore.add_relations()."""

    def test_add_single_relation(self, doc_store):
        """Adding a single relation inserts one row."""
        relations = [
            {"target_id": DOC_B, "relation_type": "similar", "similarity": 0.85},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 1
        doc_store._mock_conn.execute.assert_called_once()

        call_args = doc_store._mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO memory_relations" in sql
        assert "ON CONFLICT" in sql

    def test_add_multiple_relations(self, doc_store):
        """Adding multiple relations inserts multiple rows."""
        relations = [
            {"target_id": DOC_B, "relation_type": "similar", "similarity": 0.85},
            {"target_id": DOC_C, "relation_type": "similar", "similarity": 0.78},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 2
        assert doc_store._mock_conn.execute.call_count == 2

    def test_skip_self_relation(self, doc_store):
        """Self-relations (source == target) are skipped."""
        relations = [
            {"target_id": DOC_A, "relation_type": "similar", "similarity": 1.0},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 0
        doc_store._mock_conn.execute.assert_not_called()

    def test_empty_relations_list(self, doc_store):
        """Empty relations list returns 0 without DB call."""
        result = _run(doc_store.add_relations(DOC_A, []))
        assert result == 0
        doc_store._mock_conn.execute.assert_not_called()

    def test_default_relation_type(self, doc_store):
        """Missing relation_type defaults to 'similar'."""
        relations = [
            {"target_id": DOC_B, "similarity": 0.9},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 1

        call_args = doc_store._mock_conn.execute.call_args
        assert call_args[0][3] == "similar"

    def test_none_relation_type_defaults_to_similar(self, doc_store):
        """Explicit None relation_type defaults to 'similar'."""
        relations = [
            {"target_id": DOC_B, "relation_type": None, "similarity": 0.9},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 1

        call_args = doc_store._mock_conn.execute.call_args
        assert call_args[0][3] == "similar"

    def test_handles_db_error_gracefully(self, doc_store):
        """DB errors on individual inserts are caught, don't fail the batch."""
        doc_store._mock_conn.execute.side_effect = [
            None,  # First succeeds
            Exception("unique violation"),  # Second fails
        ]
        relations = [
            {"target_id": DOC_B, "relation_type": "similar", "similarity": 0.85},
            {"target_id": DOC_C, "relation_type": "similar", "similarity": 0.78},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 1

    def test_upsert_passes_correct_values(self, doc_store):
        """Verify the UUID conversion and parameter ordering."""
        relations = [
            {"target_id": DOC_B, "relation_type": "extends", "similarity": 0.92},
        ]
        _run(doc_store.add_relations(DOC_A, relations))

        call_args = doc_store._mock_conn.execute.call_args
        args = call_args[0]
        assert args[1] == UUID(DOC_A)  # source_uuid
        assert args[2] == UUID(DOC_B)  # target_uuid
        assert args[3] == "extends"  # relation_type
        assert args[4] == 0.92  # similarity

    def test_invalid_source_id_returns_zero(self, doc_store):
        """Invalid source_id returns 0 without crashing."""
        relations = [{"target_id": DOC_B, "relation_type": "similar", "similarity": 0.8}]
        result = _run(doc_store.add_relations("not-a-uuid", relations))
        assert result == 0
        doc_store._mock_conn.execute.assert_not_called()

    def test_invalid_target_id_skipped(self, doc_store):
        """Invalid target_id in a relation is skipped, rest continue."""
        relations = [
            {"target_id": "not-a-uuid", "relation_type": "similar", "similarity": 0.8},
            {"target_id": DOC_B, "relation_type": "similar", "similarity": 0.9},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        # First skipped (invalid UUID), second succeeds
        assert result == 1

    def test_missing_target_id_skipped(self, doc_store):
        """Relation dict without target_id is skipped."""
        relations = [
            {"relation_type": "similar", "similarity": 0.8},  # No target_id
            {"target_id": DOC_B, "relation_type": "similar", "similarity": 0.9},
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 1

    def test_accepts_uuid_objects(self, doc_store):
        """Accepts UUID objects (not just strings) for source and target."""
        relations = [
            {"target_id": UUID(DOC_B), "relation_type": "similar", "similarity": 0.85},
        ]
        result = _run(doc_store.add_relations(UUID(DOC_A), relations))
        assert result == 1

    def test_mixed_batch_with_errors(self, doc_store):
        """Mixed batch: valid, invalid UUID, self-ref, valid — handles all gracefully."""
        relations = [
            {"target_id": DOC_B, "relation_type": "similar", "similarity": 0.9},  # OK
            {"target_id": "bad-uuid", "relation_type": "similar", "similarity": 0.8},  # Bad UUID
            {"target_id": DOC_A, "relation_type": "similar", "similarity": 1.0},  # Self
            {"target_id": DOC_C, "relation_type": "similar", "similarity": 0.8},  # OK
        ]
        result = _run(doc_store.add_relations(DOC_A, relations))
        assert result == 2  # Only DOC_B and DOC_C


class TestGetRelationCount:
    """Tests for DocumentStore.get_relation_count()."""

    def test_returns_count(self, doc_store):
        """Returns the count from the DB."""
        doc_store._mock_conn.fetchval = AsyncMock(return_value=5)
        result = _run(doc_store.get_relation_count(DOC_A))
        assert result == 5

    def test_returns_zero_for_null(self, doc_store):
        """Returns 0 when fetchval returns None."""
        doc_store._mock_conn.fetchval = AsyncMock(return_value=None)
        result = _run(doc_store.get_relation_count(DOC_A))
        assert result == 0

    def test_query_filters_soft_deleted(self, doc_store):
        """SQL query joins memory_documents and filters deleted_at IS NULL."""
        doc_store._mock_conn.fetchval = AsyncMock(return_value=3)
        _run(doc_store.get_relation_count(DOC_A))

        call_args = doc_store._mock_conn.fetchval.call_args
        sql = call_args[0][0]
        assert "memory_documents" in sql
        assert "deleted_at IS NULL" in sql


class TestAutoLinkRelations:
    """Tests for _auto_link_relations() function."""

    def test_creates_relations_for_similar_neighbors(self):
        """Creates relations for neighbors above threshold."""
        mock_doc_store = AsyncMock()
        mock_doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": DOC_B, "score": 0.85},
            {"document_id": DOC_C, "score": 0.80},
        ])
        mock_doc_store.add_relations = AsyncMock(return_value=2)

        result = _run(_auto_link_relations(
            mock_doc_store, DOC_A, [0.1] * 1536, USER_ID, None, "personal",
        ))
        assert result == 2

        # Forward relations call
        forward_call = mock_doc_store.add_relations.call_args_list[0]
        assert forward_call[0][0] == DOC_A
        assert len(forward_call[0][1]) == 2

        # Reverse relations calls (one per neighbor)
        assert mock_doc_store.add_relations.call_count == 3  # 1 forward + 2 reverse

    def test_skips_neighbors_below_threshold(self):
        """Neighbors below AUTO_RELATION_THRESHOLD are not linked."""
        mock_doc_store = AsyncMock()
        mock_doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": DOC_B, "score": 0.85},  # Above
            {"document_id": DOC_C, "score": 0.50},  # Below
        ])
        mock_doc_store.add_relations = AsyncMock(return_value=1)

        result = _run(_auto_link_relations(
            mock_doc_store, DOC_A, [0.1] * 1536, USER_ID, None, "personal",
        ))
        assert result == 1

        forward_call = mock_doc_store.add_relations.call_args_list[0]
        assert len(forward_call[0][1]) == 1  # Only DOC_B

    def test_skips_self_in_search_results(self):
        """If the document itself appears in search results, it's skipped."""
        mock_doc_store = AsyncMock()
        mock_doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": DOC_A, "score": 1.0},  # Self
            {"document_id": DOC_B, "score": 0.85},
        ])
        mock_doc_store.add_relations = AsyncMock(return_value=1)

        result = _run(_auto_link_relations(
            mock_doc_store, DOC_A, [0.1] * 1536, USER_ID, None, "personal",
        ))
        assert result == 1

    def test_deduplicates_document_ids(self):
        """Multiple chunks from the same document don't create duplicate relations."""
        mock_doc_store = AsyncMock()
        mock_doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": DOC_B, "score": 0.90},  # Chunk 1 of DOC_B
            {"document_id": DOC_B, "score": 0.85},  # Chunk 2 of DOC_B
            {"document_id": DOC_C, "score": 0.80},
        ])
        mock_doc_store.add_relations = AsyncMock(return_value=2)

        result = _run(_auto_link_relations(
            mock_doc_store, DOC_A, [0.1] * 1536, USER_ID, None, "personal",
        ))
        assert result == 2

        forward_call = mock_doc_store.add_relations.call_args_list[0]
        target_ids = [r["target_id"] for r in forward_call[0][1]]
        assert target_ids == [DOC_B, DOC_C]  # DOC_B only once

    def test_no_neighbors_returns_zero(self):
        """No neighbors above threshold returns 0."""
        mock_doc_store = AsyncMock()
        mock_doc_store.search_chunks = AsyncMock(return_value=[])

        result = _run(_auto_link_relations(
            mock_doc_store, DOC_A, [0.1] * 1536, USER_ID, None, "personal",
        ))
        assert result == 0
        mock_doc_store.add_relations.assert_not_called()

    def test_bidirectional_relations(self):
        """Creates both forward (A→B) and reverse (B→A) relations."""
        mock_doc_store = AsyncMock()
        mock_doc_store.search_chunks = AsyncMock(return_value=[
            {"document_id": DOC_B, "score": 0.85},
        ])
        mock_doc_store.add_relations = AsyncMock(return_value=1)

        _run(_auto_link_relations(
            mock_doc_store, DOC_A, [0.1] * 1536, USER_ID, None, "personal",
        ))

        # First call: forward (DOC_A → DOC_B)
        forward_call = mock_doc_store.add_relations.call_args_list[0]
        assert forward_call[0][0] == DOC_A
        assert forward_call[0][1][0]["target_id"] == DOC_B

        # Second call: reverse (DOC_B → DOC_A)
        reverse_call = mock_doc_store.add_relations.call_args_list[1]
        assert reverse_call[0][0] == DOC_B
        assert reverse_call[0][1][0]["target_id"] == DOC_A
