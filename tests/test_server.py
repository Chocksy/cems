"""Tests for CEMS REST API server.

Tests the REST API endpoints with mocked dependencies.
Uses Starlette TestClient for HTTP testing.
"""

import pytest
from starlette.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock

import cems.server as server_module


@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.user_id = "test-user-uuid"
    mock.config.team_id = "test-team"
    mock.config.storage_dir = "/tmp/cems"
    mock.config.memory_backend = "pgvector"
    mock.config.enable_graph = True
    mock.config.enable_scheduler = True
    mock.config.enable_query_synthesis = True
    mock.config.relevance_threshold = 0.01
    mock.config.default_max_tokens = 2000
    mock.config.llm_model = "anthropic/claude-3-haiku"
    mock.graph_store = None

    # Set up async methods
    mock.add_async = AsyncMock(return_value={"results": [{"id": "mem-123", "event": "ADD"}]})
    mock.search_async = AsyncMock(return_value=[])
    mock.delete_async = AsyncMock()
    mock.update_async = AsyncMock(return_value={"success": True})
    mock.retrieve_for_inference_async = AsyncMock(return_value={
        "results": [],
        "tokens_used": 0,
        "formatted_context": "",
        "queries_used": [],
        "total_candidates": 0,
        "filtered_count": 0,
        "mode": "vector",
        "intent": None,
    })
    mock.get_category_counts_async = AsyncMock(return_value={"general": 5})

    return mock


@pytest.fixture
def mock_user():
    """Create a mock user for authentication."""
    user = MagicMock()
    user.id = "test-user-uuid"
    user.username = "test-user"
    user.is_active = True
    return user


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset server global state before each test."""
    server_module._memory_cache.clear()
    server_module._scheduler_cache.clear()
    yield
    server_module._memory_cache.clear()
    server_module._scheduler_cache.clear()


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @patch("cems.db.database.get_database")
    @patch("cems.db.database.is_database_initialized", return_value=True)
    def test_health_endpoint(self, mock_is_db, mock_get_db):
        """Test /health returns healthy status."""
        mock_get_db.return_value.check_connection.return_value = True

        from cems.server import create_http_app
        app = create_http_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch("cems.db.database.is_database_initialized", return_value=True)
    def test_ping_requires_auth(self, mock_is_db):
        """Test /ping requires authentication."""
        from cems.server import create_http_app
        app = create_http_app()
        client = TestClient(app)

        response = client.get("/ping")
        assert response.status_code == 401


class TestMemoryAPI:
    """Tests for memory REST API endpoints."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(server_module, "get_memory")
    def test_memory_add(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/add endpoint."""
        mock_get_memory.return_value = mock_memory

        # Mock authentication
        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.post(
                "/api/memory/add",
                json={"content": "Test memory", "category": "test"},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(server_module, "get_memory")
    def test_memory_add_requires_content(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/add requires content field."""
        mock_get_memory.return_value = mock_memory

        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.post(
                "/api/memory/add",
                json={"category": "test"},  # Missing content
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 400
            assert "content is required" in response.json()["error"]

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(server_module, "get_memory")
    def test_memory_search(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/search endpoint."""
        mock_memory.retrieve_for_inference_async.return_value = {
            "results": [
                {"memory_id": "mem-123", "content": "Found", "score": 0.9, "scope": "personal", "category": "test"}
            ],
            "tokens_used": 50,
            "formatted_context": "=== MEMORIES ===",
            "queries_used": ["test"],
            "total_candidates": 5,
            "filtered_count": 1,
            "mode": "vector",
            "intent": None,
        }
        mock_get_memory.return_value = mock_memory

        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.post(
                "/api/memory/search",
                json={"query": "test search"},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) == 1

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(server_module, "get_memory")
    def test_memory_forget(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/forget endpoint."""
        mock_get_memory.return_value = mock_memory

        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.post(
                "/api/memory/forget",
                json={"memory_id": "mem-123"},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "archived" in data["message"]

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(server_module, "get_memory")
    def test_memory_update(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/update endpoint."""
        mock_get_memory.return_value = mock_memory

        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.post(
                "/api/memory/update",
                json={"memory_id": "mem-123", "content": "Updated content"},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestSummaryEndpoints:
    """Tests for summary endpoints."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(server_module, "get_memory")
    def test_personal_summary(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test GET /api/memory/summary/personal endpoint."""
        mock_memory.get_category_counts_async.return_value = {"general": 5, "decisions": 3}
        mock_get_memory.return_value = mock_memory

        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.get(
                "/api/memory/summary/personal",
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total"] == 8
            assert "general" in data["categories"]


class TestMaintenanceEndpoint:
    """Tests for maintenance endpoint."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(server_module, "get_scheduler")
    def test_maintenance_consolidation(self, mock_get_scheduler, mock_db, mock_is_db, mock_user):
        """Test POST /api/memory/maintenance endpoint."""
        mock_scheduler = MagicMock()
        mock_scheduler.run_now.return_value = {"duplicates_merged": 2}
        mock_get_scheduler.return_value = mock_scheduler

        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.post(
                "/api/memory/maintenance",
                json={"job_type": "consolidation"},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["job_type"] == "consolidation"


class TestAuthentication:
    """Tests for authentication."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    def test_missing_auth_header(self, mock_is_db):
        """Test that missing auth header returns 401."""
        from cems.server import create_http_app
        app = create_http_app()
        client = TestClient(app)

        response = client.post(
            "/api/memory/add",
            json={"content": "Test"}
        )

        assert response.status_code == 401
        assert "Authorization" in response.json()["error"]

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    def test_invalid_api_key(self, mock_db, mock_is_db):
        """Test that invalid API key returns 401."""
        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = None  # User not found
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app
            app = create_http_app()
            client = TestClient(app)

            response = client.post(
                "/api/memory/add",
                json={"content": "Test"},
                headers={"Authorization": "Bearer invalid-key"}
            )

            assert response.status_code == 401
            assert "Invalid API key" in response.json()["error"]
