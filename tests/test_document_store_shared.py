"""Tests for shared memory cross-user visibility in DocumentStore.

Verifies that shared-scope searches use OR logic for ownership:
  (d.user_id = $X OR d.team_id = $Y)
so that all team members can see shared memories, not just the author.
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from cems.db.document_store import DocumentStore
from cems.db.filter_builder import FilterBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
TEAM_1 = "11111111-1111-1111-1111-111111111111"
EMBEDDING = [0.1] * 1536


def _has_or_ownership(sql: str) -> bool:
    """Check whether the SQL contains OR-based ownership: (d.user_id = ... OR d.team_id = ...)."""
    # Normalise whitespace for easier matching
    flat = " ".join(sql.split())
    # Matches both shared pattern: (d.user_id = $X OR d.team_id = $Y)
    # and both pattern: (d.user_id = $X OR (d.team_id = $Y AND d.scope = 'shared'))
    return bool(re.search(r"\(\s*d\.user_id\s*=\s*\$\d+\s+OR\s+.*d\.team_id\s*=\s*\$\d+", flat))


def _has_and_ownership(sql: str) -> bool:
    """Check whether user_id and team_id appear as separate AND conditions (the old bug)."""
    flat = " ".join(sql.split())
    # They'd be separate conditions joined by AND — no surrounding parens with OR
    has_user = bool(re.search(r"d\.user_id\s*=\s*\$\d+", flat))
    has_team = bool(re.search(r"d\.team_id\s*=\s*\$\d+", flat))
    has_or = _has_or_ownership(flat)
    # Bug pattern: both present but NOT wrapped in an OR group
    return has_user and has_team and not has_or


# ---------------------------------------------------------------------------
# FilterBuilder unit tests
# ---------------------------------------------------------------------------


class TestFilterBuilderOwnershipFilter:
    """Tests for FilterBuilder.add_ownership_filter OR logic."""

    def test_personal_scope_uses_user_id_only(self):
        """Personal scope should filter by user_id only, no team_id."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, team_id=TEAM_1, scope="personal")
        clause = fb.build()

        assert "user_id" in clause
        assert "team_id" not in clause
        assert "scope" in clause
        # Only user_id UUID in values, plus scope string
        uuid_values = [v for v in fb.values if isinstance(v, UUID)]
        assert len(uuid_values) == 1
        assert str(uuid_values[0]) == USER_A

    def test_shared_scope_with_team_uses_or(self):
        """Shared scope with team_id should use OR logic between user_id and team_id."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, team_id=TEAM_1, scope="shared")
        clause = fb.build()

        # Must contain OR-based ownership group
        assert re.search(
            r"\(\s*user_id\s*=\s*\$\d+\s+OR\s+team_id\s*=\s*\$\d+\s*\)", clause
        ), f"Expected OR ownership clause, got: {clause}"

        # Scope filter should still be present
        assert "scope" in clause

    def test_both_scope_with_team_uses_or(self):
        """scope='both' with team_id should use OR logic with nested scope condition."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, team_id=TEAM_1, scope="both")
        clause = fb.build()

        # OR ownership with nested scope condition for team
        assert "user_id" in clause
        assert "team_id" in clause
        assert "OR" in clause

        # scope='both' should NOT add a standalone scope = $N filter
        # but may contain scope = 'shared' inside the OR clause
        assert not re.search(r"^scope\s*=\s*\$\d+$", clause)

    def test_shared_scope_without_team_falls_back_to_user_only(self):
        """Shared scope without team_id should fall back to user_id filter only."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, team_id=None, scope="shared")
        clause = fb.build()

        assert "user_id" in clause
        assert "team_id" not in clause

    def test_no_user_no_team_produces_scope_only(self):
        """No user_id or team_id should just filter by scope (or nothing for 'both')."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=None, team_id=None, scope="shared")
        clause = fb.build()

        assert "user_id" not in clause
        assert "team_id" not in clause
        assert "scope" in clause

    def test_parameter_indices_are_sequential(self):
        """Parameter indices should be sequential and correct."""
        fb = FilterBuilder(start_idx=3)
        fb.add_ownership_filter(user_id=USER_A, team_id=TEAM_1, scope="shared")
        clause = fb.build()

        # Extract all $N placeholders
        indices = [int(m) for m in re.findall(r"\$(\d+)", clause)]
        # They should start at 3 and be sequential
        assert indices == sorted(indices), f"Indices not sequential: {indices}"
        assert indices[0] == 3, f"Expected first index 3, got {indices[0]}"

    def test_shared_no_user_with_team_uses_team_only(self):
        """Shared scope with team_id but no user_id should filter by team_id only."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=None, team_id=TEAM_1, scope="shared")
        clause = fb.build()

        assert "team_id" in clause
        assert "user_id" not in clause
        assert "scope" in clause

    def test_both_no_user_with_team_uses_team_only(self):
        """scope='both' with team_id but no user_id should filter by team_id only."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=None, team_id=TEAM_1, scope="both")
        clause = fb.build()

        assert "team_id" in clause
        assert "user_id" not in clause

    def test_col_prefix_applied(self):
        """Column prefix should be applied to all column references."""
        fb = FilterBuilder(start_idx=1)
        fb.add_ownership_filter(user_id=USER_A, team_id=TEAM_1, scope="shared", col_prefix="d.")
        clause = fb.build()

        assert "d.user_id" in clause
        assert "d.team_id" in clause
        assert "d.scope" in clause


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
    store = DocumentStore("postgresql://test/test", embedding_dim=1536)
    # Create mock connection
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)

    # Create mock pool with proper async context manager for acquire()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAcquire(mock_conn)

    store._pool = mock_pool
    return store, mock_conn


class TestSearchChunksSharedVisibility:
    """Verify search_chunks generates OR-based ownership SQL for shared scope."""

    @pytest.mark.asyncio
    async def test_shared_scope_uses_or_ownership(self, doc_store):
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _has_or_ownership(sql), (
            f"search_chunks with scope='shared' should use OR ownership.\n"
            f"SQL: {sql}"
        )

    @pytest.mark.asyncio
    async def test_personal_scope_no_or(self, doc_store):
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="personal",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _has_or_ownership(sql), (
            f"search_chunks with scope='personal' should NOT use OR ownership.\n"
            f"SQL: {sql}"
        )
        assert "d.user_id" in sql

    @pytest.mark.asyncio
    async def test_both_scope_uses_or_ownership(self, doc_store):
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="both",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _has_or_ownership(sql), (
            f"search_chunks with scope='both' should use OR ownership.\n"
            f"SQL: {sql}"
        )

    @pytest.mark.asyncio
    async def test_shared_no_team_uses_user_only(self, doc_store):
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=None,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        # Extract WHERE clause only (d.team_id appears in SELECT columns too)
        where_part = sql.split("WHERE", 1)[1] if "WHERE" in sql else sql
        assert "d.user_id" in where_part
        assert "d.team_id" not in where_part


class TestHybridSearchSharedVisibility:
    """Verify hybrid_search_chunks generates OR-based ownership SQL for shared scope."""

    @pytest.mark.asyncio
    async def test_shared_scope_uses_or_ownership(self, doc_store):
        store, mock_conn = doc_store

        await store.hybrid_search_chunks(
            query="test query",
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _has_or_ownership(sql), (
            f"hybrid_search_chunks with scope='shared' should use OR ownership.\n"
            f"SQL: {sql}"
        )

    @pytest.mark.asyncio
    async def test_personal_scope_no_or(self, doc_store):
        store, mock_conn = doc_store

        await store.hybrid_search_chunks(
            query="test query",
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="personal",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _has_or_ownership(sql), (
            f"hybrid_search_chunks with scope='personal' should NOT use OR ownership.\n"
            f"SQL: {sql}"
        )


class TestFullTextSearchSharedVisibility:
    """Verify full_text_search_chunks generates OR-based ownership SQL for shared scope."""

    @pytest.mark.asyncio
    async def test_shared_scope_uses_or_ownership(self, doc_store):
        store, mock_conn = doc_store

        await store.full_text_search_chunks(
            query="test query",
            user_id=USER_A,
            team_id=TEAM_1,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert _has_or_ownership(sql), (
            f"full_text_search_chunks with scope='shared' should use OR ownership.\n"
            f"SQL: {sql}"
        )

    @pytest.mark.asyncio
    async def test_personal_scope_no_or(self, doc_store):
        store, mock_conn = doc_store

        await store.full_text_search_chunks(
            query="test query",
            user_id=USER_A,
            team_id=TEAM_1,
            scope="personal",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _has_or_ownership(sql), (
            f"full_text_search_chunks with scope='personal' should NOT use OR ownership.\n"
            f"SQL: {sql}"
        )


class TestOldBugNotPresent:
    """Regression tests: verify the old AND-based ownership bug is gone."""

    @pytest.mark.asyncio
    async def test_search_chunks_no_and_ownership(self, doc_store):
        """search_chunks should NOT produce separate AND'd user_id + team_id filters."""
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _has_and_ownership(sql), (
            f"BUG REGRESSION: search_chunks still uses AND ownership for shared scope.\n"
            f"SQL: {sql}"
        )

    @pytest.mark.asyncio
    async def test_hybrid_search_no_and_ownership(self, doc_store):
        """hybrid_search_chunks should NOT produce separate AND'd user_id + team_id filters."""
        store, mock_conn = doc_store

        await store.hybrid_search_chunks(
            query="test query",
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _has_and_ownership(sql), (
            f"BUG REGRESSION: hybrid_search_chunks still uses AND ownership.\n"
            f"SQL: {sql}"
        )

    @pytest.mark.asyncio
    async def test_fts_no_and_ownership(self, doc_store):
        """full_text_search_chunks should NOT produce separate AND'd user_id + team_id filters."""
        store, mock_conn = doc_store

        await store.full_text_search_chunks(
            query="test query",
            user_id=USER_A,
            team_id=TEAM_1,
            scope="shared",
            limit=5,
        )

        sql = mock_conn.fetch.call_args[0][0]
        assert not _has_and_ownership(sql), (
            f"BUG REGRESSION: full_text_search_chunks still uses AND ownership.\n"
            f"SQL: {sql}"
        )


class TestParameterBindingCorrectness:
    """Verify that SQL parameter values are bound in the correct positions."""

    @pytest.mark.asyncio
    async def test_search_chunks_shared_params(self, doc_store):
        """Parameters for shared scope OR clause should include both user and team UUIDs."""
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="shared",
            limit=5,
        )

        call_args = mock_conn.fetch.call_args[0]
        # call_args[0] = SQL, call_args[1] = embedding, call_args[2] = limit, rest = filter values
        param_values = call_args[1:]  # everything after SQL

        # Both UUIDs should be in the parameter values
        str_params = [str(v) for v in param_values if isinstance(v, UUID)]
        assert USER_A in str_params, f"user_id not in params: {str_params}"
        assert TEAM_1 in str_params, f"team_id not in params: {str_params}"

    @pytest.mark.asyncio
    async def test_search_chunks_personal_params(self, doc_store):
        """Personal scope should only bind user_id, not team_id."""
        store, mock_conn = doc_store

        await store.search_chunks(
            query_embedding=EMBEDDING,
            user_id=USER_A,
            team_id=TEAM_1,
            scope="personal",
            limit=5,
        )

        call_args = mock_conn.fetch.call_args[0]
        param_values = call_args[1:]

        str_params = [str(v) for v in param_values if isinstance(v, UUID)]
        assert USER_A in str_params, f"user_id not in params: {str_params}"
        # team_id should NOT be in the bound parameters for personal scope
        assert TEAM_1 not in str_params, f"team_id should not be in personal params: {str_params}"
