"""Tests for /api/memory/list filters and /api/memory/facets.

Two layers, both mocked (no real Postgres):
- Handler layer: Starlette TestClient + mocked memory/doc_store, checks that
  query params reach the store and that search mode post-filters.
- Store layer: DocumentStore with a fake pool, checks the generated SQL
  (exact-tag AND via `tags @>`, source_ref prefix, facet grouping, visibility).
"""

import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import cems.api.deps as deps_module
import cems.api.handlers.memory as memory_handlers
from cems.db.document_store import DocumentStore
from cems.models import MemoryMetadata, MemoryScope, SearchResult

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
AUTH = {"Authorization": "Bearer test-api-key"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_server_state():
    deps_module._memory_cache.clear()
    deps_module._scheduler_cache.clear()
    yield
    deps_module._memory_cache.clear()
    deps_module._scheduler_cache.clear()


@pytest.fixture
def doc_store_mock():
    store = MagicMock()
    store.get_all_documents = AsyncMock(return_value=[])
    store.count_documents = AsyncMock(return_value=0)
    store.get_facets = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_memory(doc_store_mock):
    mem = MagicMock()
    mem.config = MagicMock()
    mem.config.user_id = USER_A
    mem._ensure_initialized_async = AsyncMock()
    mem._ensure_document_store = AsyncMock(return_value=doc_store_mock)
    mem.search_async = AsyncMock(return_value=[])
    return mem


@pytest.fixture
def client(mock_memory):
    """Authenticated TestClient with get_memory patched to mock_memory."""
    user = MagicMock()
    user.id = USER_A
    user.is_active = True
    user_service = MagicMock()
    user_service.get_user_by_api_key.return_value = user

    with (
        patch("cems.db.database.is_database_initialized", return_value=True),
        patch("cems.db.database.get_database") as mock_db,
        patch.object(memory_handlers, "get_memory", return_value=mock_memory),
        patch("cems.admin.services.UserService", return_value=user_service),
    ):
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)
        from cems.server import create_http_app

        yield TestClient(create_http_app())


def _result(doc_id, *, tags=(), category="general", source_ref=None, scope="shared", score=0.5):
    meta = MemoryMetadata(
        memory_id=doc_id,
        user_id=USER_A,
        scope=MemoryScope(scope),
        category=category,
        source_ref=source_ref,
        tags=list(tags),
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    return SearchResult(
        memory_id=doc_id,
        content=f"content {doc_id}",
        score=score,
        scope=MemoryScope(scope),
        metadata=meta,
        shown_count=3,
    )


# ---------------------------------------------------------------------------
# /api/memory/list — browse mode
# ---------------------------------------------------------------------------


class TestListBrowseFilters:
    def test_repeated_tag_passed_as_list(self, client, doc_store_mock):
        resp = client.get(
            "/api/memory/list?tag=slack-user:U1&tag=closeout&tag=closeout&tag=",
            headers=AUTH,
        )
        assert resp.status_code == 200
        kwargs = doc_store_mock.get_all_documents.call_args.kwargs
        assert kwargs["tags"] == ["slack-user:U1", "closeout"]
        assert doc_store_mock.count_documents.call_args.kwargs["tags"] == ["slack-user:U1", "closeout"]

    def test_source_ref_prefix_and_other_filters(self, client, doc_store_mock):
        doc_store_mock.count_documents.return_value = 7
        doc_store_mock.get_all_documents.return_value = [{
            "id": "d1", "content": "x", "category": "decisions", "tags": ["a"],
            "scope": "shared", "source_ref": "project:org/repo",
            "created_at": datetime(2026, 10, 1, tzinfo=UTC), "shown_count": 2,
        }]
        resp = client.get(
            "/api/memory/list?source_ref_prefix=project:org/&category=decisions"
            "&scope=shared&tag_prefix=session:",
            headers=AUTH,
        )
        data = resp.json()
        assert data["total"] == 7
        assert data["mode"] == "browse"
        assert data["results"][0]["source_ref"] == "project:org/repo"

        for call in (doc_store_mock.get_all_documents, doc_store_mock.count_documents):
            kw = call.call_args.kwargs
            assert kw["source_ref_prefix"] == "project:org/"
            assert kw["category"] == "decisions"
            assert kw["scope"] == "shared"
            assert kw["tag_prefix"] == "session:"
            assert kw["user_id"] == USER_A

    def test_unknown_scope_falls_back_to_visibility_default(self, client, doc_store_mock):
        client.get("/api/memory/list?scope=everything", headers=AUTH)
        assert doc_store_mock.get_all_documents.call_args.kwargs["scope"] is None


# ---------------------------------------------------------------------------
# /api/memory/list — search mode (q + filters)
# ---------------------------------------------------------------------------


class TestListSearchPostFilter:
    def test_q_with_tags_post_filters_and_overfetches(self, client, mock_memory):
        mock_memory.search_async.return_value = [
            _result("d1", tags=["slack-user:U1", "closeout"], source_ref="project:org/repo"),
            _result("d2", tags=["slack-user:U1"], source_ref="project:org/repo"),
            _result("d3", tags=["slack-user:U1", "closeout", "x"], source_ref="project:other/repo"),
            _result("d4", tags=["closeout", "slack-user:U1"], source_ref="project:org/repo2"),
        ]
        resp = client.get(
            "/api/memory/list?q=deploy&limit=5&tag=slack-user:U1&tag=closeout"
            "&source_ref_prefix=project:org/",
            headers=AUTH,
        )
        data = resp.json()
        assert data["mode"] == "search"
        assert [r["id"] for r in data["results"]] == ["d1", "d4"]
        assert data["total"] == 2
        assert mock_memory.search_async.call_args.kwargs["limit"] == 20  # limit * 4

    def test_q_overfetch_is_capped(self, client, mock_memory):
        client.get("/api/memory/list?q=x&limit=200&tag=a", headers=AUTH)
        assert mock_memory.search_async.call_args.kwargs["limit"] == 200

    def test_q_without_filters_does_not_overfetch(self, client, mock_memory):
        client.get("/api/memory/list?q=x&limit=10", headers=AUTH)
        assert mock_memory.search_async.call_args.kwargs["limit"] == 10

    def test_q_with_category_scope_and_tag_prefix(self, client, mock_memory):
        mock_memory.search_async.return_value = [
            _result("d1", category="decisions", tags=["session:abc"], scope="personal"),
            _result("d2", category="general", tags=["session:abc"], scope="personal"),
            _result("d3", category="decisions", tags=["other"], scope="personal"),
        ]
        resp = client.get(
            "/api/memory/list?q=x&category=decisions&scope=personal&tag_prefix=session:",
            headers=AUTH,
        )
        assert [r["id"] for r in resp.json()["results"]] == ["d1"]
        kw = mock_memory.search_async.call_args.kwargs
        assert kw["category"] == "decisions"
        assert kw["scope"] == "personal"

    def test_q_trims_to_limit_after_filtering(self, client, mock_memory):
        mock_memory.search_async.return_value = [
            _result(f"d{i}", tags=["keep"]) for i in range(6)
        ]
        resp = client.get("/api/memory/list?q=x&limit=3&tag=keep", headers=AUTH)
        assert [r["id"] for r in resp.json()["results"]] == ["d0", "d1", "d2"]

    def test_search_result_shape(self, client, mock_memory):
        mock_memory.search_async.return_value = [
            _result("d1", tags=["t"], category="decisions", source_ref="project:a/b"),
        ]
        item = client.get("/api/memory/list?q=x", headers=AUTH).json()["results"][0]
        assert item["id"] == "d1"
        assert item["category"] == "decisions"
        assert item["tags"] == ["t"]
        assert item["scope"] == "shared"
        assert item["source_ref"] == "project:a/b"
        assert item["created_at"].startswith("2026-10-01")
        assert item["shown_count"] == 3
        assert item["content"] == "content d1"


# ---------------------------------------------------------------------------
# /api/memory/facets — handler
# ---------------------------------------------------------------------------


class TestFacetsEndpoint:
    def test_defaults_to_tag_field(self, client, doc_store_mock):
        doc_store_mock.get_facets.return_value = [
            {"value": "slack-user:U123", "count": 88},
            {"value": "slack-user:U9", "count": 4},
        ]
        resp = client.get("/api/memory/facets?prefix=slack-user:", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json() == {
            "success": True,
            "field": "tag",
            "facets": [
                {"value": "slack-user:U123", "count": 88},
                {"value": "slack-user:U9", "count": 4},
            ],
        }
        kw = doc_store_mock.get_facets.call_args.kwargs
        assert kw["field"] == "tag"
        assert kw["prefix"] == "slack-user:"
        assert kw["limit"] == 50
        assert kw["user_id"] == USER_A

    def test_passes_narrowing_filters_and_clamps_limit(self, client, doc_store_mock):
        client.get(
            "/api/memory/facets?field=source_ref&prefix=project:&limit=9999"
            "&tag=slack-user:U1&tag=closeout&scope=shared&category=decisions"
            "&source_ref_prefix=project:org/",
            headers=AUTH,
        )
        kw = doc_store_mock.get_facets.call_args.kwargs
        assert kw["field"] == "source_ref"
        assert kw["limit"] == 500
        assert kw["tags"] == ["slack-user:U1", "closeout"]
        assert kw["scope"] == "shared"
        assert kw["category"] == "decisions"
        assert kw["source_ref_prefix"] == "project:org/"

    def test_category_field(self, client, doc_store_mock):
        resp = client.get("/api/memory/facets?field=category", headers=AUTH)
        assert resp.json()["field"] == "category"
        assert doc_store_mock.get_facets.call_args.kwargs["field"] == "category"

    def test_invalid_field_rejected(self, client, doc_store_mock):
        resp = client.get("/api/memory/facets?field=user_id", headers=AUTH)
        assert resp.status_code == 400
        doc_store_mock.get_facets.assert_not_called()

    def test_requires_auth(self, client):
        assert client.get("/api/memory/facets").status_code == 401


# ---------------------------------------------------------------------------
# DocumentStore SQL (fake pool)
# ---------------------------------------------------------------------------


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def store():
    s = DocumentStore("postgresql://test/test")
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=0)
    pool = MagicMock()
    pool.acquire.return_value = _FakeAcquire(conn)
    s._pool = pool
    return s, conn


def _param(sql: str, values: list, pattern: str):
    """Return the bound value for the first `$n` that follows `pattern` in sql."""
    m = re.search(re.escape(pattern) + r"\s*\$(\d+)", sql)
    assert m, f"{pattern!r} not found in SQL:\n{sql}"
    return values[int(m.group(1)) - 1]


class TestStoreListSQL:
    async def test_get_all_documents_tags_and(self, store):
        s, conn = store
        await s.get_all_documents(
            user_id=USER_A, tags=["slack-user:U1", "closeout"], source_ref_prefix="project:org/",
        )
        sql, *values = conn.fetch.call_args.args
        assert "tags @> $" in sql and "::text[]" in sql
        assert _param(sql, values, "tags @>") == ["slack-user:U1", "closeout"]
        assert _param(sql, values, "source_ref LIKE") == "project:org/"
        assert "deleted_at IS NULL" in sql

    async def test_count_documents_honors_new_filters(self, store):
        s, conn = store
        conn.fetchval.return_value = 12
        total = await s.count_documents(
            user_id=USER_A, tags=["a"], source_ref_prefix="project:x", category="c",
        )
        assert total == 12
        sql, *values = conn.fetchval.call_args.args
        assert _param(sql, values, "tags @>") == ["a"]
        assert _param(sql, values, "source_ref LIKE") == "project:x"
        assert _param(sql, values, "category =") == "c"

    async def test_no_tag_clause_without_tags(self, store):
        s, conn = store
        await s.count_documents(user_id=USER_A)
        assert "tags @>" not in conn.fetchval.call_args.args[0]


class TestStoreFacetsSQL:
    async def test_tag_facets_group_prefix_and_visibility(self, store):
        s, conn = store
        conn.fetch.return_value = [
            {"value": "slack-user:U1", "count": 5},
            {"value": "slack-user:U2", "count": 1},
        ]
        facets = await s.get_facets(user_id=USER_A, field="tag", prefix="slack-user:", limit=10)
        assert facets == [
            {"value": "slack-user:U1", "count": 5},
            {"value": "slack-user:U2", "count": 1},
        ]
        sql, *values = conn.fetch.call_args.args
        assert "unnest(tags)" in sql
        assert "GROUP BY t" in sql
        assert "ORDER BY count DESC" in sql
        assert _param(sql, values, "t LIKE") == "slack-user:"
        # Visibility: own docs OR shared, never soft-deleted.
        assert re.search(r"\(user_id = \$\d+ OR scope = 'shared'\)", sql)
        assert str(_param(sql, values, "(user_id =")) == USER_A
        assert "deleted_at IS NULL" in sql
        assert values[-1] == 10  # LIMIT is the last bound value

    async def test_personal_scope_facets_only_own(self, store):
        s, conn = store
        await s.get_facets(user_id=USER_A, scope="personal")
        sql, *values = conn.fetch.call_args.args
        assert str(_param(sql, values, "user_id =")) == USER_A
        assert _param(sql, values, "scope =") == "personal"
        assert "OR scope = 'shared'" not in sql

    async def test_source_ref_facets(self, store):
        s, conn = store
        await s.get_facets(user_id=USER_A, field="source_ref", prefix="project:", tags=["a"])
        sql, *values = conn.fetch.call_args.args
        assert "unnest" not in sql
        assert "GROUP BY source_ref" in sql
        assert "source_ref IS NOT NULL" in sql
        assert _param(sql, values, "source_ref LIKE") == "project:"
        assert _param(sql, values, "tags @>") == ["a"]

    async def test_category_facets(self, store):
        s, conn = store
        await s.get_facets(user_id=USER_A, field="category")
        assert "GROUP BY category" in conn.fetch.call_args.args[0]

    async def test_bad_field_raises(self, store):
        s, _ = store
        with pytest.raises(ValueError):
            await s.get_facets(user_id=USER_A, field="content")
