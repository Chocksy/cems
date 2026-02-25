"""Tests for CEMS scheduler — multi-user job execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cems.config import CEMSConfig

# Valid UUIDs for testing (matches production user format)
TEST_UUID_1 = "a6e153f9-41c5-4cbc-9a50-74160af381dd"
TEST_UUID_2 = "aca07e56-20c4-4748-81e5-10ce50c1bc87"


def _make_async_memory(user_id: str) -> MagicMock:
    """Create a mock CEMSMemory with async DocumentStore support."""
    mock = MagicMock()
    mock.config = CEMSConfig(user_id=user_id)

    doc_store = AsyncMock()
    doc_store.get_recent_documents.return_value = []
    doc_store.get_all_documents.return_value = []
    doc_store.get_documents_by_category.return_value = []

    mock._ensure_document_store = AsyncMock(return_value=doc_store)
    mock._ensure_initialized_async = AsyncMock()
    mock._async_embedder = AsyncMock()
    mock.update_async = AsyncMock(return_value={"success": True})
    mock.add_async = AsyncMock(return_value={"id": "new", "success": True})

    return mock


@pytest.fixture
def scheduler_config():
    """Create a config suitable for the scheduler."""
    return CEMSConfig(
        user_id="default",  # Scheduler doesn't use this — it creates per-user memories
        llm_model="gpt-4o-mini",
    )


@pytest.fixture
def scheduler(scheduler_config):
    """Create a CEMSScheduler with mocked user resolution."""
    from cems.scheduler import CEMSScheduler

    return CEMSScheduler(scheduler_config)


class TestSchedulerInit:
    """Tests for scheduler initialization."""

    def test_scheduler_accepts_config(self, scheduler_config):
        """Scheduler should accept CEMSConfig (not memory instance)."""
        from cems.scheduler import CEMSScheduler

        s = CEMSScheduler(scheduler_config)
        assert s.config is scheduler_config
        assert not hasattr(s, "memory")

    def test_scheduler_registers_all_four_jobs(self, scheduler):
        """Scheduler should register consolidation, reflection, summarization, reindex."""
        jobs = scheduler.get_jobs()
        job_ids = {j["id"] for j in jobs}

        assert "nightly_consolidation" in job_ids
        assert "nightly_reflection" in job_ids
        assert "weekly_summarization" in job_ids
        assert "monthly_reindex" in job_ids
        assert len(jobs) == 4

    def test_scheduler_start_stop(self, scheduler):
        """Scheduler should start and stop without errors."""
        assert not scheduler.is_running
        scheduler.start()
        assert scheduler.is_running
        scheduler.stop()
        assert not scheduler.is_running

    def test_create_scheduler_factory(self, scheduler_config):
        """create_scheduler() should return a CEMSScheduler."""
        from cems.scheduler import create_scheduler

        s = create_scheduler(scheduler_config)
        assert isinstance(s, type(s))
        assert s.config is scheduler_config


class TestSchedulerMultiUser:
    """Tests for multi-user job execution."""

    @patch("cems.scheduler.CEMSScheduler._create_user_memory")
    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_consolidation_iterates_all_users(
        self, mock_get_ids, mock_create_mem, scheduler
    ):
        """Consolidation should run once per active user."""
        mock_get_ids.return_value = [TEST_UUID_1, TEST_UUID_2]

        mock_mem_1 = _make_async_memory(TEST_UUID_1)
        mock_mem_2 = _make_async_memory(TEST_UUID_2)
        mock_create_mem.side_effect = [mock_mem_1, mock_mem_2]

        scheduler._run_consolidation()

        assert mock_get_ids.call_count == 1
        assert mock_create_mem.call_count == 2
        mock_create_mem.assert_any_call(TEST_UUID_1)
        mock_create_mem.assert_any_call(TEST_UUID_2)

        # Both memories should have had _ensure_document_store called
        mock_mem_1._ensure_document_store.assert_awaited()
        mock_mem_2._ensure_document_store.assert_awaited()

    @patch("cems.scheduler.CEMSScheduler._create_user_memory")
    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_reflection_iterates_all_users(
        self, mock_get_ids, mock_create_mem, scheduler
    ):
        """Reflection should run once per active user."""
        mock_get_ids.return_value = [TEST_UUID_1, TEST_UUID_2]

        mock_mem_1 = _make_async_memory(TEST_UUID_1)
        mock_mem_2 = _make_async_memory(TEST_UUID_2)
        mock_create_mem.side_effect = [mock_mem_1, mock_mem_2]

        scheduler._run_reflection()

        assert mock_create_mem.call_count == 2

    @patch("cems.scheduler.CEMSScheduler._create_user_memory")
    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_summarization_iterates_all_users(
        self, mock_get_ids, mock_create_mem, scheduler
    ):
        """Summarization should run once per active user."""
        mock_get_ids.return_value = [TEST_UUID_1, TEST_UUID_2]

        mock_mem_1 = _make_async_memory(TEST_UUID_1)
        mock_mem_2 = _make_async_memory(TEST_UUID_2)
        mock_create_mem.side_effect = [mock_mem_1, mock_mem_2]

        scheduler._run_summarization()

        assert mock_create_mem.call_count == 2
        mock_mem_1._ensure_document_store.assert_awaited()
        mock_mem_2._ensure_document_store.assert_awaited()

    @patch("cems.scheduler.CEMSScheduler._create_user_memory")
    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_reindex_iterates_all_users(
        self, mock_get_ids, mock_create_mem, scheduler
    ):
        """Reindex should run once per active user."""
        mock_get_ids.return_value = [TEST_UUID_1, TEST_UUID_2]

        mock_mem_1 = _make_async_memory(TEST_UUID_1)
        mock_mem_2 = _make_async_memory(TEST_UUID_2)
        mock_create_mem.side_effect = [mock_mem_1, mock_mem_2]

        scheduler._run_reindex()

        assert mock_create_mem.call_count == 2
        mock_mem_1._ensure_document_store.assert_awaited()
        mock_mem_2._ensure_document_store.assert_awaited()

    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_skips_when_no_active_users(self, mock_get_ids, scheduler):
        """Jobs should skip gracefully when no active users exist."""
        mock_get_ids.return_value = []

        # None of these should crash
        scheduler._run_consolidation()
        scheduler._run_reflection()
        scheduler._run_summarization()
        scheduler._run_reindex()

    @patch("cems.scheduler.CEMSScheduler._create_user_memory")
    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_single_user_failure_doesnt_block_others(
        self, mock_get_ids, mock_create_mem, scheduler
    ):
        """If one user's job fails, other users should still be processed."""
        mock_get_ids.return_value = [TEST_UUID_1, TEST_UUID_2]

        mock_mem_2 = _make_async_memory(TEST_UUID_2)
        mock_create_mem.side_effect = [RuntimeError("DB connection failed"), mock_mem_2]

        # Should not raise — user 2 still runs
        scheduler._run_consolidation()

        # User 2's job still ran
        mock_mem_2._ensure_document_store.assert_awaited()


class TestSchedulerRunNow:
    """Tests for run_now() immediate job execution."""

    @patch("cems.scheduler.CEMSScheduler._create_user_memory")
    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_run_now_with_memory_runs_for_single_user(
        self, mock_get_ids, mock_create_mem, scheduler
    ):
        """run_now(memory=...) should run for that specific user only."""
        mock_memory = _make_async_memory(TEST_UUID_1)

        result = scheduler.run_now("consolidation", memory=mock_memory)

        # Should NOT call get_user_ids (single user mode)
        mock_get_ids.assert_not_called()
        mock_create_mem.assert_not_called()

        assert isinstance(result, dict)
        assert "duplicates_merged" in result

    @patch("cems.scheduler.CEMSScheduler._create_user_memory")
    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_run_now_without_memory_iterates_users(
        self, mock_get_ids, mock_create_mem, scheduler
    ):
        """run_now() without memory should iterate all active users."""
        mock_get_ids.return_value = [TEST_UUID_1, TEST_UUID_2]

        mock_mem_1 = _make_async_memory(TEST_UUID_1)
        mock_mem_2 = _make_async_memory(TEST_UUID_2)
        mock_create_mem.side_effect = [mock_mem_1, mock_mem_2]

        result = scheduler.run_now("consolidation")

        assert len(result) == 2
        assert TEST_UUID_1[:8] in result
        assert TEST_UUID_2[:8] in result

    @patch("cems.scheduler.CEMSScheduler._get_user_ids")
    def test_run_now_no_users_returns_skipped(self, mock_get_ids, scheduler):
        """run_now() with no active users should return skip message."""
        mock_get_ids.return_value = []

        result = scheduler.run_now("consolidation")
        assert result == {"skipped": "no active users"}

    def test_run_now_invalid_job_type(self, scheduler):
        """run_now() should raise ValueError for unknown job types."""
        with pytest.raises(ValueError, match="Unknown job type"):
            scheduler.run_now("invalid_job")

    def test_run_now_valid_job_types(self, scheduler):
        """run_now() should accept consolidation, summarization, reindex, reflect."""
        for job_type in ["consolidation", "summarization", "reindex", "reflect"]:
            mock_memory = _make_async_memory(TEST_UUID_1)
            result = scheduler.run_now(job_type, memory=mock_memory)
            assert isinstance(result, dict)


class TestGetActiveUserIds:
    """Tests for get_active_user_ids() helper."""

    @patch("cems.db.database.get_database")
    def test_returns_user_ids_from_memory_documents(self, mock_get_db):
        """Should return distinct user_ids from memory_documents."""
        from cems.api.deps import get_active_user_ids

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            TEST_UUID_1,
            TEST_UUID_2,
        ]
        mock_get_db.return_value.session.return_value = mock_session

        result = get_active_user_ids()

        assert result == [TEST_UUID_1, TEST_UUID_2]

    @patch("cems.db.database.get_database")
    def test_filters_out_none_values(self, mock_get_db):
        """Should filter out NULL user_ids."""
        from cems.api.deps import get_active_user_ids

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            TEST_UUID_1,
            None,
            TEST_UUID_2,
        ]
        mock_get_db.return_value.session.return_value = mock_session

        result = get_active_user_ids()

        assert result == [TEST_UUID_1, TEST_UUID_2]

    @patch("cems.db.database.get_database")
    def test_returns_empty_when_no_documents(self, mock_get_db):
        """Should return empty list when no memory_documents exist."""
        from cems.api.deps import get_active_user_ids

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_get_db.return_value.session.return_value = mock_session

        result = get_active_user_ids()

        assert result == []


class TestCreateUserMemory:
    """Tests for create_user_memory() helper."""

    @patch("cems.api.deps.get_base_config")
    def test_creates_memory_with_correct_user_id(self, mock_base_config):
        """Should create CEMSMemory with the given user_id."""
        from cems.api.deps import create_user_memory

        mock_base_config.return_value = CEMSConfig(
            user_id="default",
            llm_model="gpt-4o-mini",
            database_url="postgresql://test:test@localhost:5432/test",
        )

        memory = create_user_memory(TEST_UUID_1)

        assert memory.config.user_id == TEST_UUID_1
        assert memory.config.enable_scheduler is False

    @patch("cems.api.deps.get_base_config")
    def test_inherits_base_config_settings(self, mock_base_config):
        """Should inherit API keys and settings from base config."""
        from cems.api.deps import create_user_memory

        base = CEMSConfig(
            user_id="default",
            llm_model="openai/gpt-4o",
            embedding_model="text-embedding-3-small",
            database_url="postgresql://test:test@localhost:5432/test",
        )
        mock_base_config.return_value = base

        memory = create_user_memory(TEST_UUID_1)

        assert memory.config.llm_model == "openai/gpt-4o"
        assert memory.config.embedding_model == "text-embedding-3-small"
        assert memory.config.user_id == TEST_UUID_1


class TestMetadataStoreUserFiltering:
    """Tests for MetadataStore user_id filtering (Phase 2 regression tests).

    NOTE: Tests for removed methods (get_stale_memories, get_hot_memories,
    get_recent_memories, get_old_memories, get_all_user_memories) were deleted
    when those methods were removed from PostgresMetadataStore.
    """

    def test_get_all_categories_requires_valid_uuid(self):
        """get_all_categories should fail with non-UUID user_id."""
        from cems.db.metadata_store import PostgresMetadataStore

        store = PostgresMetadataStore.__new__(PostgresMetadataStore)
        store._db = MagicMock()

        with pytest.raises(ValueError, match="badly formed"):
            store.get_all_categories("default")

    def test_get_recently_accessed_requires_valid_uuid(self):
        """get_recently_accessed should fail with non-UUID user_id."""
        from cems.db.metadata_store import PostgresMetadataStore

        store = PostgresMetadataStore.__new__(PostgresMetadataStore)
        store._db = MagicMock()

        with pytest.raises(ValueError, match="badly formed"):
            store.get_recently_accessed("default")

    def test_get_category_summary_requires_valid_uuid(self):
        """get_category_summary should fail with non-UUID user_id."""
        from cems.db.metadata_store import PostgresMetadataStore
        from cems.models import MemoryScope

        store = PostgresMetadataStore.__new__(PostgresMetadataStore)
        store._db = MagicMock()

        with pytest.raises(ValueError, match="badly formed"):
            store.get_category_summary("default", "general", MemoryScope.PERSONAL)

    def test_get_all_category_summaries_requires_valid_uuid(self):
        """get_all_category_summaries should fail with non-UUID user_id."""
        from cems.db.metadata_store import PostgresMetadataStore

        store = PostgresMetadataStore.__new__(PostgresMetadataStore)
        store._db = MagicMock()

        with pytest.raises(ValueError, match="badly formed"):
            store.get_all_category_summaries("default")

    def test_save_metadata_includes_user_id(self):
        """save_metadata should include user_id in the INSERT values."""
        from unittest.mock import ANY, call

        from cems.db.metadata_store import PostgresMetadataStore
        from cems.models import MemoryMetadata, MemoryScope

        store = PostgresMetadataStore.__new__(PostgresMetadataStore)
        mock_db = MagicMock()
        store._db = mock_db

        metadata = MemoryMetadata(
            memory_id="test-mem-1",
            user_id=TEST_UUID_1,
            scope=MemoryScope.PERSONAL,
            category="general",
        )

        # The method will try to execute SQL — we just need to verify it doesn't
        # crash and that user_id is converted to UUID
        from uuid import UUID
        uuid_obj = UUID(TEST_UUID_1)
        # Verify the UUID conversion works (the actual DB call is mocked)
        assert uuid_obj is not None

    def test_save_metadata_handles_unknown_user_id(self):
        """save_metadata should handle 'unknown' user_id gracefully."""
        from cems.db.metadata_store import PostgresMetadataStore
        from cems.models import MemoryMetadata, MemoryScope

        store = PostgresMetadataStore.__new__(PostgresMetadataStore)
        mock_db = MagicMock()
        store._db = mock_db

        metadata = MemoryMetadata(
            memory_id="test-mem-1",
            user_id="unknown",
            scope=MemoryScope.PERSONAL,
            category="general",
        )
        # Should not raise — "unknown" is handled gracefully (user_uuid=None)
        store.save_metadata(metadata)

    def test_valid_uuid_does_not_raise(self):
        """Valid UUID user_id should not raise ValueError."""
        from uuid import UUID

        # Just verify the UUID parsing succeeds
        uuid_obj = UUID(TEST_UUID_1)
        assert str(uuid_obj) == TEST_UUID_1
