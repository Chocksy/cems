"""Tests for PgVectorStore methods.

Tests the vectorstore operations with mocked asyncpg connections.
"""

import pytest
from datetime import datetime, UTC
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import UUID

from cems.vectorstore import PgVectorStore


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg connection pool."""
    pool = MagicMock()
    pool.acquire = MagicMock()
    return pool


@pytest.fixture
def vectorstore():
    """Create a PgVectorStore instance with mocked pool."""
    with patch.dict("os.environ", {"CEMS_DATABASE_URL": "postgresql://test:test@localhost/test"}):
        store = PgVectorStore()
        store._pool = MagicMock()
        return store


class TestSearchByCategory:
    """Tests for search_by_category method."""

    @pytest.mark.asyncio
    async def test_search_by_category_returns_memories(self, vectorstore):
        """Test search_by_category returns matching memories."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        # Mock the database response
        mock_row = {
            "id": UUID("660e8400-e29b-41d4-a716-446655440001"),
            "content": "User prefers Python",
            "user_id": UUID(user_id),
            "team_id": None,
            "scope": "personal",
            "category": "preferences",
            "tags": ["python", "backend"],
            "source": "manual",
            "source_ref": None,
            "priority": 1.0,
            "pinned": False,
            "pin_reason": None,
            "pin_category": None,
            "archived": False,
            "access_count": 5,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_accessed": datetime.now(UTC),
            "expires_at": None,
        }

        # Setup mock connection
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        results = await vectorstore.search_by_category(
            user_id=user_id,
            category="preferences",
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["content"] == "User prefers Python"
        assert results[0]["category"] == "preferences"
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_category_with_archived(self, vectorstore):
        """Test search_by_category can include archived memories."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        await vectorstore.search_by_category(
            user_id=user_id,
            category="preferences",
            limit=10,
            include_archived=True,
        )

        # Verify the query was called - archived=False should NOT be in WHERE
        call_args = mock_conn.fetch.call_args[0][0]
        assert "archived = FALSE" not in call_args

    @pytest.mark.asyncio
    async def test_search_by_category_excludes_archived_by_default(self, vectorstore):
        """Test search_by_category excludes archived by default."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        await vectorstore.search_by_category(
            user_id=user_id,
            category="preferences",
            limit=10,
        )

        # Verify archived=FALSE is in the query
        call_args = mock_conn.fetch.call_args[0][0]
        assert "archived = FALSE" in call_args

    @pytest.mark.asyncio
    async def test_search_by_category_empty_results(self, vectorstore):
        """Test search_by_category returns empty list when no matches."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        results = await vectorstore.search_by_category(
            user_id=user_id,
            category="nonexistent",
            limit=10,
        )

        assert results == []


class TestGetRecent:
    """Tests for get_recent method."""

    @pytest.mark.asyncio
    async def test_get_recent_returns_memories(self, vectorstore):
        """Test get_recent returns recent memories."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_row = {
            "id": UUID("660e8400-e29b-41d4-a716-446655440001"),
            "content": "Discussed API design",
            "user_id": UUID(user_id),
            "team_id": None,
            "scope": "personal",
            "category": "decisions",
            "tags": ["api"],
            "source": "session",
            "source_ref": None,
            "priority": 1.0,
            "pinned": False,
            "pin_reason": None,
            "pin_category": None,
            "archived": False,
            "access_count": 1,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_accessed": datetime.now(UTC),
            "expires_at": None,
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        results = await vectorstore.get_recent(
            user_id=user_id,
            hours=24,
            limit=15,
        )

        assert len(results) == 1
        assert results[0]["content"] == "Discussed API design"
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_recent_uses_hours_parameter(self, vectorstore):
        """Test get_recent respects hours parameter."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        await vectorstore.get_recent(
            user_id=user_id,
            hours=48,
            limit=15,
        )

        # Verify hours parameter was passed (second positional arg after user_id)
        call_args = mock_conn.fetch.call_args[0]
        # The query should use INTERVAL with the hours value
        assert "INTERVAL '1 hour' * $2" in call_args[0]
        # hours=48 should be the second value
        assert call_args[2] == 48

    @pytest.mark.asyncio
    async def test_get_recent_orders_by_created_at(self, vectorstore):
        """Test get_recent orders results by created_at DESC."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        await vectorstore.get_recent(
            user_id=user_id,
            hours=24,
            limit=15,
        )

        call_args = mock_conn.fetch.call_args[0][0]
        assert "ORDER BY created_at DESC" in call_args

    @pytest.mark.asyncio
    async def test_get_recent_with_archived(self, vectorstore):
        """Test get_recent can include archived memories."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        await vectorstore.get_recent(
            user_id=user_id,
            hours=24,
            limit=15,
            include_archived=True,
        )

        call_args = mock_conn.fetch.call_args[0][0]
        assert "archived = FALSE" not in call_args

    @pytest.mark.asyncio
    async def test_get_recent_empty_results(self, vectorstore):
        """Test get_recent returns empty list when no recent memories."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        vectorstore._pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        results = await vectorstore.get_recent(
            user_id=user_id,
            hours=24,
            limit=15,
        )

        assert results == []


class TestRowToDict:
    """Tests for _row_to_dict helper method."""

    def test_row_to_dict_converts_uuids(self, vectorstore):
        """Test _row_to_dict converts UUIDs to strings."""
        row = {
            "id": UUID("660e8400-e29b-41d4-a716-446655440001"),
            "content": "Test content",
            "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
            "team_id": UUID("770e8400-e29b-41d4-a716-446655440002"),
            "scope": "personal",
            "category": "test",
            "tags": [],
            "source": None,
            "source_ref": None,
            "priority": 1.0,
            "pinned": False,
            "pin_reason": None,
            "pin_category": None,
            "archived": False,
            "access_count": 0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_accessed": None,
            "expires_at": None,
        }

        # Create a mock record that behaves like asyncpg.Record
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: row[key]
        mock_row.get = lambda key, default=None: row.get(key, default)
        mock_row.keys = lambda: row.keys()

        result = vectorstore._row_to_dict(mock_row)

        assert result["id"] == "660e8400-e29b-41d4-a716-446655440001"
        assert result["user_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert result["team_id"] == "770e8400-e29b-41d4-a716-446655440002"

    def test_row_to_dict_handles_null_ids(self, vectorstore):
        """Test _row_to_dict handles null user_id and team_id."""
        row = {
            "id": UUID("660e8400-e29b-41d4-a716-446655440001"),
            "content": "Test content",
            "user_id": None,
            "team_id": None,
            "scope": "personal",
            "category": "test",
            "tags": [],
            "source": None,
            "source_ref": None,
            "priority": 1.0,
            "pinned": False,
            "pin_reason": None,
            "pin_category": None,
            "archived": False,
            "access_count": 0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_accessed": None,
            "expires_at": None,
        }

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: row[key]
        mock_row.get = lambda key, default=None: row.get(key, default)
        mock_row.keys = lambda: row.keys()

        result = vectorstore._row_to_dict(mock_row)

        assert result["user_id"] is None
        assert result["team_id"] is None

    def test_row_to_dict_includes_score(self, vectorstore):
        """Test _row_to_dict includes score when requested."""
        row = {
            "id": UUID("660e8400-e29b-41d4-a716-446655440001"),
            "content": "Test content",
            "user_id": None,
            "team_id": None,
            "scope": "personal",
            "category": "test",
            "tags": [],
            "source": None,
            "source_ref": None,
            "priority": 1.0,
            "pinned": False,
            "pin_reason": None,
            "pin_category": None,
            "archived": False,
            "access_count": 0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_accessed": None,
            "expires_at": None,
            "score": 0.95,
        }

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: row[key]
        mock_row.get = lambda key, default=None: row.get(key, default)
        mock_row.keys = lambda: row.keys()

        result = vectorstore._row_to_dict(mock_row, include_score=True)

        assert result["score"] == 0.95

    def test_row_to_dict_excludes_score_by_default(self, vectorstore):
        """Test _row_to_dict excludes score by default."""
        row = {
            "id": UUID("660e8400-e29b-41d4-a716-446655440001"),
            "content": "Test content",
            "user_id": None,
            "team_id": None,
            "scope": "personal",
            "category": "test",
            "tags": [],
            "source": None,
            "source_ref": None,
            "priority": 1.0,
            "pinned": False,
            "pin_reason": None,
            "pin_category": None,
            "archived": False,
            "access_count": 0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_accessed": None,
            "expires_at": None,
            "score": 0.95,
        }

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: row[key]
        mock_row.get = lambda key, default=None: row.get(key, default)
        mock_row.keys = lambda: row.keys()

        result = vectorstore._row_to_dict(mock_row, include_score=False)

        assert "score" not in result
