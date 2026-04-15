"""Tests for memory visibility scoping in DocumentStore.

Verifies ownership filter rules:
  - scope="personal": user_id = X AND scope = 'personal'
  - scope="shared": scope = 'shared' (visible to all users)
  - scope="both": user_id = X OR scope = 'shared'
"""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from cems.db.document_store import DocumentStore
from cems.db.filter_builder import FilterBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
EMBEDDING = [0.1] * 1536


def _ownership_in_where(sql: str, pattern: str) -> bool:
    """Check if a pattern appears in any WHERE clause (not just SELECT columns).

    Handles CTEs with multiple WHERE clauses by checking all of them.
    Looks for 'pattern =' to distinguish from SELECT column references.
    """
    # Find all WHERE clause segments
    parts = sql.split("WHERE")
    for part in parts[1:]:  # Skip everything before first WHERE
        # Take text up to the next major clause boundary
        for boundary in ("ORDER BY", "GROUP BY", "LIMIT", "HAVING", ")", "WITH"):
            if boundary in part:
                part = part[:part.index(boundary)]
                break
        if pattern in part:
            return True
    return False


# ---------------------------------------------------------------------------
# FilterBuilder unit tests
# ---------------------------------------------------------------------------


class TestFilterBuilderOwnershipFilter:
    """Tests for FilterBuilder.add_ownership_filter scoping logic."""

    def test_personal_scope_filters_by_user_and_scope(self):
        """Personal scope should filter by user_id AND scope='personal'."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, scope="personal")
        clause = fb.build()

        assert "user_id" in clause
        assert "scope" in clause
        assert "'personal'" not in clause  # scope is parameterized, not inline

    def test_shared_scope_filters_by_scope_only(self):
        """Shared scope should filter by scope='shared' only — visible to all users."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, scope="shared")
        clause = fb.build()

        assert "scope" in clause
        assert "user_id" not in clause  # No user filter for shared

    def test_both_scope_uses_or(self):
        """scope='both' should produce: (user_id = X OR scope = 'shared')."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, scope="both")
        clause = fb.build()

        assert "user_id" in clause
        assert "OR" in clause
        assert "scope = 'shared'" in clause

    def test_personal_scope_no_user_still_filters_scope(self):
        """Personal scope without user_id should still filter by scope."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=None, scope="personal")
        clause = fb.build()

        assert "user_id" not in clause
        assert "scope" in clause

    def test_shared_scope_no_user_filters_scope_only(self):
        """Shared scope without user_id should just filter by scope."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=None, scope="shared")
        clause = fb.build()

        assert "user_id" not in clause
        assert "scope" in clause

    def test_both_scope_no_user_no_filter(self):
        """scope='both' without user_id should add no ownership filter."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=None, scope="both")
        clause = fb.build()

        # No conditions added — build returns default
        assert clause == "TRUE"

    def test_parameter_indices_are_sequential(self):
        """Parameter indices should be sequential and correct."""
        fb = FilterBuilder(start_idx=3)
        fb.add_ownership_filter(user_id=USER_A, scope="shared")
        clause = fb.build()

        indices = [int(m) for m in re.findall(r"\$(\d+)", clause)]
        assert indices == sorted(indices), f"Indices not sequential: {indices}"
        assert indices[0] == 3, f"Expected first index 3, got {indices[0]}"

    def test_col_prefix_applied(self):
        """Column prefix should be applied to all column references."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, scope="both", col_prefix="d.")
        clause = fb.build()

        assert "d.user_id" in clause
        assert "d.scope" in clause

    def test_no_team_id_anywhere(self):
        """team_id should never appear in any ownership filter."""
        for scope in ("personal", "shared", "both"):
            fb = FilterBuilder(start_idx=1)
            fb.add_ownership_filter(user_id=USER_A, scope=scope)
            clause = fb.build()
            assert "team_id" not in clause, f"team_id found in scope={scope}: {clause}"


# ---------------------------------------------------------------------------
# DocumentStore search SQL verification (mocked DB)
# ---------------------------------------------------------------------------


class _FakeAcquire:
    """Async context manager that returns a mock connection."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def doc_store():
    """Create a DocumentStore with mocked pool."""
    store = DocumentStore("postgresql://test/test")
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    store._pool = mock_pool
    return store, mock_conn


class TestSearchChunksVisibility:
    """Verify search_chunks generates correct ownership SQL for each scope."""

    @pytest.mark.asyncio
    async def test_shared_scope_no_user_filter(self, doc_store):
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _ownership_in_where(sql, "d.scope")
        assert not _ownership_in_where(sql, "d.user_id =")  # Shared is visible to all

    @pytest.mark.asyncio
    async def test_personal_scope_filters_user(self, doc_store):
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            scope="personal",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _ownership_in_where(sql, "d.user_id =")

    @pytest.mark.asyncio
    async def test_both_scope_uses_or(self, doc_store):
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            scope="both",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _ownership_in_where(sql, "d.user_id =")
        assert _ownership_in_where(sql, "OR")
        assert _ownership_in_where(sql, "d.scope = 'shared'")


class TestHybridSearchVisibility:
    """Verify hybrid_search_chunks generates correct ownership SQL."""

    @pytest.mark.asyncio
    async def test_shared_scope_no_user_filter(self, doc_store):
        store, mock_conn = doc_store

        await store.hybrid_search_chunks(
            query="test query",
            query_embedding=EMBEDDING,
            user_id=USER_A,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _ownership_in_where(sql, "d.user_id =")

    @pytest.mark.asyncio
    async def test_personal_scope_filters_user(self, doc_store):
        store, mock_conn = doc_store

        await store.hybrid_search_chunks(
            query="test query",
            query_embedding=EMBEDDING,
            user_id=USER_A,
            scope="personal",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _ownership_in_where(sql, "d.user_id =")


class TestFullTextSearchVisibility:
    """Verify full_text_search_chunks generates correct ownership SQL."""

    @pytest.mark.asyncio
    async def test_shared_scope_no_user_filter(self, doc_store):
        store, mock_conn = doc_store

        await store.full_text_search_chunks(
            query="test query",
            user_id=USER_A,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _ownership_in_where(sql, "d.user_id =")

    @pytest.mark.asyncio
    async def test_personal_scope_filters_user(self, doc_store):
        store, mock_conn = doc_store

        await store.full_text_search_chunks(
            query="test query",
            user_id=USER_A,
            scope="personal",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _ownership_in_where(sql, "d.user_id =")


class TestNoTeamIdRegression:
    """Regression tests: verify team_id never appears in generated SQL."""

    @pytest.mark.asyncio
    async def test_search_chunks_no_team_id(self, doc_store):
        store, mock_conn = doc_store

        for scope in ("personal", "shared", "both"):
            await store.search_chunks(
                query_embedding=EMBEDDING,
                user_id=USER_A,
                scope=scope,
                limit=5,
            )

            sql = mock_conn.fetch.call_args[0][0]
            assert "team_id" not in sql, f"team_id found in scope={scope}"

    @pytest.mark.asyncio
    async def test_hybrid_search_no_team_id(self, doc_store):
        store, mock_conn = doc_store

        for scope in ("personal", "shared", "both"):
            await store.hybrid_search_chunks(
                query="test",
                query_embedding=EMBEDDING,
                user_id=USER_A,
                scope=scope,
                limit=5,
            )

            sql = mock_conn.fetch.call_args[0][0]
            assert "team_id" not in sql, f"team_id found in scope={scope}"
