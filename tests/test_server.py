"""Tests for CEMS REST API server.

Tests the REST API endpoints with mocked dependencies.
Uses Starlette TestClient for HTTP testing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import cems.api.deps as deps_module
import cems.api.handlers.memory as memory_handlers
import cems.api.handlers.tool as tool_handlers
from cems.config import CEMSConfig


@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.user_id = "test-user-uuid"
    mock.config.team_id = "test-team"
    mock.config.storage_dir = "/tmp/cems"
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
    deps_module._memory_cache.clear()
    deps_module._scheduler_cache.clear()
    yield
    deps_module._memory_cache.clear()
    deps_module._scheduler_cache.clear()


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
    @patch.object(memory_handlers, "get_memory")
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
    @patch.object(memory_handlers, "get_memory")
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
    @patch.object(memory_handlers, "get_memory")
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
    @patch.object(memory_handlers, "get_memory")
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
    @patch.object(memory_handlers, "get_memory")
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
    @patch.object(memory_handlers, "get_memory")
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
    @patch.object(memory_handlers, "get_memory")
    def test_maintenance_consolidation(self, mock_get_memory, mock_db, mock_is_db, mock_user):
        """Test POST /api/memory/maintenance endpoint."""
        # Mock memory instance with all methods needed by ConsolidationJob
        mock_memory = MagicMock()
        mock_memory.config = CEMSConfig(user_id="a6e153f9-41c5-4cbc-9a50-74160af381dd")
        mock_memory.get_recent_memories.return_value = []
        mock_memory.get_hot_memories.return_value = []
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


class TestProfileEndpoint:
    """Tests for /api/memory/profile endpoint."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_profile_returns_context(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test GET /api/memory/profile returns formatted context."""
        # Setup mock document store
        mock_doc_store = MagicMock()
        mock_doc_store.get_documents_by_category = AsyncMock(side_effect=[
            # preferences
            [{"id": "pref-1", "content": "Use Python for backend", "category": "preferences"}],
            # guidelines
            [{"id": "guide-1", "content": "Always write tests", "category": "guidelines"}],
            # gate-rules
            [{"id": "gate-1", "content": "Block after 10pm", "category": "gate-rules"}],
        ])
        mock_doc_store.get_recent_documents = AsyncMock(return_value=[
            {"id": "recent-1", "content": "Discussed API design", "category": "decisions"},
        ])

        mock_memory._ensure_initialized_async = AsyncMock()
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
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
                "/api/memory/profile",
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "context" in data
            assert "components" in data
            assert data["components"]["preferences"] == 1
            assert data["components"]["guidelines"] == 1
            assert data["components"]["gate_rules_count"] == 1
            assert "token_estimate" in data

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_profile_with_token_budget(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test GET /api/memory/profile respects token_budget parameter."""
        mock_doc_store = MagicMock()
        mock_doc_store.get_documents_by_category = AsyncMock(return_value=[])
        mock_doc_store.get_recent_documents = AsyncMock(return_value=[])

        mock_memory._ensure_initialized_async = AsyncMock()
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
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
                "/api/memory/profile?token_budget=1000",
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_profile_prioritizes_foundation_guidelines(
        self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user
    ):
        """Test GET /api/memory/profile puts foundation guidelines first."""
        mock_doc_store = MagicMock()
        mock_doc_store.get_documents_by_category = AsyncMock(side_effect=[
            [],  # preferences
            [  # guidelines
                {
                    "id": "guide-regular",
                    "content": "Keep functions small and focused",
                    "category": "guidelines",
                    "tags": ["style"],
                },
                {
                    "id": "guide-foundation",
                    "content": "If risk is unknown or unmitigated, stop execution.",
                    "category": "guidelines",
                    "tags": ["constitution", "foundation", "principle:15"],
                    "source_ref": "foundation:constitution:v1",
                },
            ],
            [],  # gate-rules
        ])
        mock_doc_store.get_recent_documents = AsyncMock(return_value=[])

        mock_memory._ensure_initialized_async = AsyncMock()
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
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
                "/api/memory/profile",
                headers={"Authorization": "Bearer test-api-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["components"]["foundation_guidelines"] == 1
            context = data["context"]
            assert "## Foundational Principles" in context
            assert "## Guidelines" in context
            assert context.index("## Foundational Principles") < context.index("## Guidelines")
            assert context.index("If risk is unknown or unmitigated") < context.index(
                "Keep functions small and focused"
            )

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_profile_with_project_filter(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test GET /api/memory/profile filters by project."""
        mock_doc_store = MagicMock()
        mock_doc_store.get_documents_by_category = AsyncMock(side_effect=[
            [],  # preferences
            [],  # guidelines
            [],  # gate-rules
            [  # project context
                {"id": "proj-1", "content": "CEMS uses pgvector", "source_ref": "project:org/cems"},
            ],
        ])
        mock_doc_store.get_recent_documents = AsyncMock(return_value=[])

        mock_memory._ensure_initialized_async = AsyncMock()
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
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
                "/api/memory/profile?project=org/cems",
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # get_documents_by_category called 4 times (preferences, guidelines, gate-rules, project)
            assert mock_doc_store.get_documents_by_category.call_count == 4

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_profile_empty_context(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test GET /api/memory/profile with no memories returns empty context."""
        mock_doc_store = MagicMock()
        mock_doc_store.get_documents_by_category = AsyncMock(return_value=[])
        mock_doc_store.get_recent_documents = AsyncMock(return_value=[])

        mock_memory._ensure_initialized_async = AsyncMock()
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
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
                "/api/memory/profile",
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["context"] == ""
            assert data["components"]["preferences"] == 0
            assert data["components"]["guidelines"] == 0
            assert data["components"]["recent_memories"] == 0
            assert data["token_estimate"] == 0

    @patch("cems.db.database.is_database_initialized", return_value=True)
    def test_profile_requires_auth(self, mock_is_db):
        """Test GET /api/memory/profile requires authentication."""
        from cems.server import create_http_app
        app = create_http_app()
        client = TestClient(app)

        response = client.get("/api/memory/profile")

        assert response.status_code == 401

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_profile_filters_recent_duplicates(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test GET /api/memory/profile excludes preferences/guidelines from recent."""
        mock_doc_store = MagicMock()
        mock_doc_store.get_documents_by_category = AsyncMock(side_effect=[
            [{"id": "pref-1", "content": "Prefer Python", "category": "preferences"}],
            [{"id": "guide-1", "content": "Write tests", "category": "guidelines"}],
            [],  # gate-rules
        ])
        # get_recent_documents already excludes categories, so only decisions returned
        mock_doc_store.get_recent_documents = AsyncMock(return_value=[
            {"id": "decision-1", "content": "Use REST API", "category": "decisions"},
        ])

        mock_memory._ensure_initialized_async = AsyncMock()
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
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
                "/api/memory/profile",
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # Only decision should remain in recent_memories count
            assert data["components"]["recent_memories"] == 1


class TestToolLearningEndpoint:
    """Tests for /api/tool/learning endpoint."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(tool_handlers, "get_memory")
    @patch("cems.llm.extract_tool_learning")
    def test_tool_learning_stores_learning(
        self, mock_extract, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user
    ):
        """Test POST /api/tool/learning stores extracted learning."""
        # Setup mock learning extraction
        mock_extract.return_value = {
            "type": "WORKING_SOLUTION",
            "content": "Use pgvector for hybrid search",
            "confidence": 0.8,
            "category": "database",
        }

        mock_memory.add_async = AsyncMock(return_value={
            "results": [{"id": "mem-123", "event": "ADD"}]
        })
        mock_memory._ensure_initialized_async = AsyncMock()
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
                "/api/tool/learning",
                json={
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/src/db.py"},
                    "tool_output": "Success",
                    "session_id": "test-session",
                    "context_snippet": "Working on database setup",
                },
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["stored"] is True
            assert data["memory_id"] == "mem-123"
            assert data["learning"]["type"] == "WORKING_SOLUTION"

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch("cems.llm.extract_tool_learning")
    def test_tool_learning_skips_non_learnable_tools(
        self, mock_extract, mock_db, mock_is_db, mock_user
    ):
        """Test POST /api/tool/learning skips Read/Glob/Grep tools."""
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
                "/api/tool/learning",
                json={
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/src/db.py"},
                },
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["stored"] is False
            assert data["reason"] == "skipped_non_learnable_tool"
            # extract_tool_learning should not be called
            mock_extract.assert_not_called()

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(tool_handlers, "get_memory")
    @patch("cems.llm.extract_tool_learning")
    def test_tool_learning_no_learning_extracted(
        self, mock_extract, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user
    ):
        """Test POST /api/tool/learning handles no learning extracted."""
        mock_extract.return_value = None  # No learning extracted

        mock_memory._ensure_initialized_async = AsyncMock()
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
                "/api/tool/learning",
                json={
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/src/db.py"},
                    "tool_output": "Success",
                },
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["stored"] is False
            assert data["reason"] == "skipped_no_learning_extracted"

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    def test_tool_learning_requires_tool_name(self, mock_db, mock_is_db, mock_user):
        """Test POST /api/tool/learning requires tool_name."""
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
                "/api/tool/learning",
                json={
                    "tool_input": {"file_path": "/src/db.py"},
                },
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 400
            assert "tool_name is required" in response.json()["error"]

    @patch("cems.db.database.is_database_initialized", return_value=True)
    def test_tool_learning_requires_auth(self, mock_is_db):
        """Test POST /api/tool/learning requires authentication."""
        from cems.server import create_http_app
        app = create_http_app()
        client = TestClient(app)

        response = client.post(
            "/api/tool/learning",
            json={"tool_name": "Edit"}
        )

        assert response.status_code == 401


class TestLogShownEndpoint:
    """Tests for /api/memory/log-shown endpoint."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_log_shown_increments_count(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/log-shown increments shown_count."""
        mock_doc_store = MagicMock()
        mock_doc_store.increment_shown_count = AsyncMock(return_value=3)
        mock_memory._ensure_document_store = AsyncMock(return_value=mock_doc_store)
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
                "/api/memory/log-shown",
                json={"memory_ids": ["id-1", "id-2", "id-3"]},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["updated"] == 3
            mock_doc_store.increment_shown_count.assert_called_once_with(["id-1", "id-2", "id-3"])

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_log_shown_empty_ids(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/log-shown with empty array returns 0."""
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
                "/api/memory/log-shown",
                json={"memory_ids": []},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["updated"] == 0

    @patch("cems.db.database.is_database_initialized", return_value=True)
    def test_log_shown_requires_auth(self, mock_is_db):
        """Test POST /api/memory/log-shown requires authentication."""
        from cems.server import create_http_app
        app = create_http_app()
        client = TestClient(app)

        response = client.post(
            "/api/memory/log-shown",
            json={"memory_ids": ["id-1"]}
        )

        assert response.status_code == 401


class TestSoftDelete:
    """Tests for soft-delete behavior."""

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_forget_soft_delete_default(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/forget defaults to soft-delete (archived)."""
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
            # Verify delete_async was called with hard=False
            mock_memory.delete_async.assert_called_once_with("mem-123", hard=False)

    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch.object(memory_handlers, "get_memory")
    def test_forget_hard_delete(self, mock_get_memory, mock_db, mock_is_db, mock_memory, mock_user):
        """Test POST /api/memory/forget with hard_delete=true."""
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
                json={"memory_id": "mem-123", "hard_delete": True},
                headers={"Authorization": "Bearer test-api-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "deleted" in data["message"]
            # Verify delete_async was called with hard=True
            mock_memory.delete_async.assert_called_once_with("mem-123", hard=True)
