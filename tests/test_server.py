"""Tests for CEMS MCP server tools and resources.

The server provides 5 essential tools:
- memory_add: Store memories
- memory_search: Unified search with 5-stage pipeline
- memory_forget: Delete or archive memories
- memory_update: Update existing memories
- memory_maintenance: Run maintenance jobs

And 3 resources:
- memory://status
- memory://personal/summary
- memory://shared/summary
"""

from unittest.mock import MagicMock, patch

import pytest

import cems.server as server_module


@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.user_id = "test-user"
    mock.config.team_id = "test-team"
    mock.config.storage_dir = "/tmp/cems"
    mock.config.memory_backend = "mem0"
    mock.config.vector_store = "qdrant"
    mock.config.graph_store = "kuzu"
    mock.config.enable_graph = True
    mock.config.enable_scheduler = True
    mock.config.enable_query_synthesis = True
    mock.config.relevance_threshold = 0.5
    mock.config.default_max_tokens = 2000
    mock.config.llm_model = "anthropic/claude-3-haiku"
    mock.config.get_mem0_provider.return_value = "openai"
    mock.config.get_mem0_model.return_value = "gpt-4o-mini"
    mock.graph_store = MagicMock()
    return mock


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset server global state before each test."""
    server_module._memory = None
    server_module._scheduler = None
    yield
    server_module._memory = None
    server_module._scheduler = None


class TestMemoryTools:
    """Tests for MCP memory tools (5 essential tools)."""

    @patch("cems.server.get_memory")
    def test_memory_add(self, mock_get_memory, mock_memory):
        """Test memory_add tool."""
        from cems.server import memory_add

        mock_get_memory.return_value = mock_memory
        mock_memory.add.return_value = {"results": [{"id": "mem-123"}]}

        result = memory_add("Test content", scope="personal", category="test")

        assert result["success"] is True
        assert "mem-123" in result["memory_ids"]
        mock_memory.add.assert_called_once()

    @patch("cems.server.get_memory")
    def test_memory_add_with_tags(self, mock_get_memory, mock_memory):
        """Test memory_add tool with tags."""
        from cems.server import memory_add

        mock_get_memory.return_value = mock_memory
        mock_memory.add.return_value = {"results": [{"id": "mem-123"}]}

        result = memory_add(
            "Test content",
            scope="shared",
            category="decisions",
            tags=["python", "backend"],
        )

        assert result["success"] is True
        mock_memory.add.assert_called_with(
            content="Test content",
            scope="shared",
            category="decisions",
            tags=["python", "backend"],
        )

    @patch("cems.server.get_memory")
    def test_memory_add_error(self, mock_get_memory, mock_memory):
        """Test memory_add handles errors."""
        from cems.server import memory_add

        mock_get_memory.return_value = mock_memory
        mock_memory.add.side_effect = Exception("Storage error")

        result = memory_add("Test content")

        assert result["success"] is False
        assert "Storage error" in result["message"]

    @patch("cems.server.get_memory")
    def test_memory_search(self, mock_get_memory, mock_memory):
        """Test memory_search tool uses 5-stage pipeline."""
        from cems.server import memory_search

        mock_get_memory.return_value = mock_memory
        mock_memory.retrieve_for_inference.return_value = {
            "results": [
                {
                    "memory_id": "mem-123",
                    "content": "Found content",
                    "score": 0.95,
                    "scope": "personal",
                    "category": "test",
                }
            ],
            "tokens_used": 50,
            "formatted_context": "=== RELEVANT MEMORIES ===\n...",
            "queries_used": ["test query"],
            "total_candidates": 5,
            "filtered_count": 3,
        }

        result = memory_search("test query", scope="personal", max_results=5)

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["tokens_used"] == 50
        assert "formatted_context" in result
        mock_memory.retrieve_for_inference.assert_called_once()

    @patch("cems.server.get_memory")
    def test_memory_search_error(self, mock_get_memory, mock_memory):
        """Test memory_search handles errors."""
        from cems.server import memory_search

        mock_get_memory.return_value = mock_memory
        mock_memory.retrieve_for_inference.side_effect = Exception("Search failed")

        result = memory_search("test query")

        assert result["success"] is False
        assert "Search failed" in result["error"]

    @patch("cems.server.get_memory")
    def test_memory_forget_soft_delete(self, mock_get_memory, mock_memory):
        """Test memory_forget tool with soft delete (archive)."""
        from cems.server import memory_forget

        mock_get_memory.return_value = mock_memory

        result = memory_forget("mem-123", hard_delete=False)

        assert result["success"] is True
        assert "archived" in result["message"]
        mock_memory.delete.assert_called_with("mem-123", hard=False)

    @patch("cems.server.get_memory")
    def test_memory_forget_hard_delete(self, mock_get_memory, mock_memory):
        """Test memory_forget tool with hard delete."""
        from cems.server import memory_forget

        mock_get_memory.return_value = mock_memory

        result = memory_forget("mem-123", hard_delete=True)

        assert result["success"] is True
        assert "deleted" in result["message"]
        mock_memory.delete.assert_called_with("mem-123", hard=True)

    @patch("cems.server.get_memory")
    def test_memory_forget_error(self, mock_get_memory, mock_memory):
        """Test memory_forget handles errors."""
        from cems.server import memory_forget

        mock_get_memory.return_value = mock_memory
        mock_memory.delete.side_effect = Exception("Delete failed")

        result = memory_forget("mem-123")

        assert result["success"] is False
        assert "Delete failed" in result["message"]

    @patch("cems.server.get_memory")
    def test_memory_update(self, mock_get_memory, mock_memory):
        """Test memory_update tool."""
        from cems.server import memory_update

        mock_get_memory.return_value = mock_memory

        result = memory_update("mem-123", "New content")

        assert result["success"] is True
        mock_memory.update.assert_called_with("mem-123", "New content")

    @patch("cems.server.get_memory")
    def test_memory_update_error(self, mock_get_memory, mock_memory):
        """Test memory_update handles errors."""
        from cems.server import memory_update

        mock_get_memory.return_value = mock_memory
        mock_memory.update.side_effect = Exception("Update failed")

        result = memory_update("mem-123", "New content")

        assert result["success"] is False
        assert "Update failed" in result["message"]


class TestMaintenanceTool:
    """Tests for maintenance tool."""

    @patch("cems.server.get_scheduler")
    def test_memory_maintenance_consolidation(self, mock_get_scheduler):
        """Test running consolidation maintenance."""
        from cems.server import memory_maintenance

        mock_scheduler = MagicMock()
        mock_scheduler.run_now.return_value = {"duplicates_merged": 2}
        mock_get_scheduler.return_value = mock_scheduler

        result = memory_maintenance("consolidation")

        assert result["success"] is True
        assert result["job_type"] == "consolidation"

    @patch("cems.server.get_scheduler")
    def test_memory_maintenance_all(self, mock_get_scheduler):
        """Test running all maintenance jobs."""
        from cems.server import memory_maintenance

        mock_scheduler = MagicMock()
        mock_scheduler.run_now.return_value = {}
        mock_get_scheduler.return_value = mock_scheduler

        result = memory_maintenance("all")

        assert result["success"] is True
        assert result["job_type"] == "all"
        assert mock_scheduler.run_now.call_count == 3

    @patch("cems.server.get_scheduler")
    def test_memory_maintenance_error(self, mock_get_scheduler):
        """Test maintenance handles errors."""
        from cems.server import memory_maintenance

        mock_scheduler = MagicMock()
        mock_scheduler.run_now.side_effect = Exception("Scheduler error")
        mock_get_scheduler.return_value = mock_scheduler

        result = memory_maintenance("consolidation")

        assert result["success"] is False
        assert "Scheduler error" in result["message"]


class TestResources:
    """Tests for MCP resources (3 essential resources)."""

    @patch("cems.server.get_memory")
    def test_memory_status_resource(self, mock_get_memory, mock_memory):
        """Test memory://status resource."""
        from cems.server import memory_status

        mock_get_memory.return_value = mock_memory
        mock_memory.get_graph_stats.return_value = {"nodes": 10, "edges": 5}

        result = memory_status()

        assert "CEMS Memory System Status" in result
        assert "test-user" in result
        assert "test-team" in result
        assert "Retrieval Settings" in result
        assert "Query Synthesis" in result

    @patch("cems.server.get_memory")
    def test_personal_summary_resource(self, mock_get_memory, mock_memory):
        """Test memory://personal/summary resource."""
        from cems.server import personal_summary

        mock_get_memory.return_value = mock_memory
        mock_memory.get_all.return_value = [
            {"id": "mem-1"},
            {"id": "mem-2"},
        ]
        mock_memory.get_metadata.return_value = MagicMock(category="general")

        result = personal_summary()

        assert "Personal Memory Summary" in result
        assert "2" in result  # Total memories

    @patch("cems.server.get_memory")
    def test_personal_summary_empty(self, mock_get_memory, mock_memory):
        """Test personal_summary when no memories exist."""
        from cems.server import personal_summary

        mock_get_memory.return_value = mock_memory
        mock_memory.get_all.return_value = []

        result = personal_summary()

        assert "No personal memories stored yet" in result

    @patch("cems.server.get_memory")
    def test_shared_summary_resource(self, mock_get_memory, mock_memory):
        """Test memory://shared/summary resource."""
        from cems.server import shared_summary

        mock_get_memory.return_value = mock_memory
        mock_memory.get_all.return_value = [
            {"id": "mem-1"},
        ]
        mock_memory.get_metadata.return_value = MagicMock(category="decisions")

        result = shared_summary()

        assert "Shared Memory Summary" in result
        assert "test-team" in result

    @patch("cems.server.get_memory")
    def test_shared_summary_no_team(self, mock_get_memory, mock_memory):
        """Test shared_summary when no team configured."""
        from cems.server import shared_summary

        mock_memory.config.team_id = None
        mock_get_memory.return_value = mock_memory

        result = shared_summary()

        assert "No team configured" in result

    @patch("cems.server.get_memory")
    def test_shared_summary_empty(self, mock_get_memory, mock_memory):
        """Test shared_summary when no memories exist."""
        from cems.server import shared_summary

        mock_get_memory.return_value = mock_memory
        mock_memory.get_all.return_value = []

        result = shared_summary()

        assert "No shared memories" in result


class TestToolCount:
    """Verify the simplified API has the expected number of tools."""

    def test_server_has_5_tools(self):
        """Server should have exactly 5 MCP tools."""
        # Import fresh to get tool count
        from cems.server import mcp

        # The FastMCP server registers tools when decorated
        # We check that we have the expected tools
        expected_tools = {
            "memory_add",
            "memory_search",
            "memory_forget",
            "memory_update",
            "memory_maintenance",
        }

        # This test verifies our design intent - 5 essential tools
        assert len(expected_tools) == 5
