"""Tests for CEMS maintenance jobs."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cems.config import CEMSConfig


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    mock = MagicMock()
    mock.config = CEMSConfig(
        user_id="test-user",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    return mock


class TestConsolidationJob:
    """Tests for consolidation job."""

    def test_consolidation_init(self, mock_memory):
        """Test consolidation job initialization."""
        from cems.maintenance.consolidation import ConsolidationJob

        job = ConsolidationJob(mock_memory)
        assert job.memory == mock_memory
        assert job.config == mock_memory.config

    @patch("cems.llm.merge_memory_contents")
    def test_consolidation_run(self, mock_merge, mock_memory):
        """Test running consolidation job."""
        from cems.maintenance.consolidation import ConsolidationJob

        # Setup mock
        mock_memory.get_recent_memories.return_value = ["mem-1", "mem-2"]
        mock_memory.get_hot_memories.return_value = ["mem-1"]
        mock_memory.get.return_value = {"memory": "Test content"}
        mock_memory.search.return_value = []  # No duplicates

        job = ConsolidationJob(mock_memory)
        result = job.run()

        assert "duplicates_merged" in result
        assert "memories_promoted" in result
        mock_memory.get_recent_memories.assert_called_once()
        mock_memory.get_hot_memories.assert_called_once()

    def test_consolidation_promote_hot(self, mock_memory):
        """Test promoting hot memories."""
        from cems.maintenance.consolidation import ConsolidationJob

        mock_memory.get_hot_memories.return_value = ["hot-1", "hot-2"]
        mock_memory.get_recent_memories.return_value = []

        job = ConsolidationJob(mock_memory)
        result = job.run()

        assert result["memories_promoted"] == 2
        assert mock_memory.promote_memory.call_count == 2


class TestSummarizationJob:
    """Tests for summarization job."""

    def test_summarization_init(self, mock_memory):
        """Test summarization job initialization."""
        from cems.maintenance.summarization import SummarizationJob

        job = SummarizationJob(mock_memory)
        assert job.memory == mock_memory

    @patch("cems.llm.summarize_memories")
    def test_summarization_run(self, mock_summarize, mock_memory):
        """Test running summarization job."""
        from cems.maintenance.summarization import SummarizationJob

        mock_summarize.return_value = "Generated summary"
        mock_memory.get_old_memories.return_value = ["old-1", "old-2"]
        mock_memory.get.return_value = {"memory": "Old content"}
        mock_memory.get_metadata.return_value = MagicMock(category="general")
        mock_memory.get_stale_memories.return_value = []

        job = SummarizationJob(mock_memory)
        result = job.run()

        assert "categories_updated" in result
        assert "memories_pruned" in result

    def test_summarization_prune_stale(self, mock_memory):
        """Test pruning stale memories."""
        from cems.maintenance.summarization import SummarizationJob

        mock_memory.get_old_memories.return_value = []
        mock_memory.get_stale_memories.return_value = ["stale-1", "stale-2", "stale-3"]

        job = SummarizationJob(mock_memory)
        result = job.run()

        assert result["memories_pruned"] == 3
        assert mock_memory.archive_memory.call_count == 3


class TestReindexJob:
    """Tests for reindex job."""

    def test_reindex_init(self, mock_memory):
        """Test reindex job initialization."""
        from cems.maintenance.reindex import ReindexJob

        job = ReindexJob(mock_memory)
        assert job.memory == mock_memory

    def test_reindex_run(self, mock_memory):
        """Test running reindex job."""
        from cems.maintenance.reindex import ReindexJob

        mock_memory.metadata_store.get_all_user_memories.return_value = ["mem-1", "mem-2"]
        mock_memory.get.return_value = {"memory": "Content to reindex"}

        # Mock stale memories for archive
        mock_memory.metadata_store.get_stale_memories.return_value = []

        job = ReindexJob(mock_memory)
        result = job.run()

        assert "memories_reindexed" in result
        assert "memories_archived" in result

    def test_reindex_archive_dead(self, mock_memory):
        """Test archiving dead memories."""
        from cems.maintenance.reindex import ReindexJob

        mock_memory.metadata_store.get_all_user_memories.return_value = []
        mock_memory.metadata_store.get_stale_memories.return_value = ["dead-1", "dead-2"]

        job = ReindexJob(mock_memory)
        result = job.run()

        assert result["memories_archived"] == 2


class TestScheduler:
    """Tests for scheduler functionality."""

    def test_scheduler_init(self, mock_memory):
        """Test scheduler initialization."""
        from cems.scheduler import CEMSScheduler

        scheduler = CEMSScheduler(mock_memory)
        assert scheduler.memory == mock_memory
        assert scheduler._scheduler is not None

    def test_scheduler_run_now(self, mock_memory):
        """Test running a job immediately."""
        from cems.scheduler import CEMSScheduler

        # Mock the memory methods
        mock_memory.get_recent_memories.return_value = []
        mock_memory.get_hot_memories.return_value = []

        scheduler = CEMSScheduler(mock_memory)
        result = scheduler.run_now("consolidation")

        assert isinstance(result, dict)

    def test_scheduler_run_all(self, mock_memory):
        """Test running all jobs."""
        from cems.scheduler import CEMSScheduler

        # Setup minimal mocks
        mock_memory.get_recent_memories.return_value = []
        mock_memory.get_hot_memories.return_value = []
        mock_memory.get_old_memories.return_value = []
        mock_memory.get_stale_memories.return_value = []
        mock_memory.metadata_store.get_all_user_memories.return_value = []
        mock_memory.metadata_store.get_stale_memories.return_value = []

        scheduler = CEMSScheduler(mock_memory)

        # Run each job type
        for job_type in ["consolidation", "summarization", "reindex"]:
            result = scheduler.run_now(job_type)
            assert isinstance(result, dict)

    def test_scheduler_invalid_job(self, mock_memory):
        """Test running invalid job type."""
        from cems.scheduler import CEMSScheduler

        scheduler = CEMSScheduler(mock_memory)

        with pytest.raises(ValueError, match="Unknown job type"):
            scheduler.run_now("invalid_job")
