"""Tests for CEMS models and metadata store."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from cems.models import MemoryMetadata, MemoryScope, MetadataStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_metadata.db"
        yield db_path


@pytest.fixture
def metadata_store(temp_db):
    """Create a metadata store for testing."""
    return MetadataStore(temp_db)


class TestMemoryMetadata:
    """Tests for MemoryMetadata model."""

    def test_create_metadata(self):
        """Test creating a memory metadata object."""
        metadata = MemoryMetadata(
            memory_id="test-123",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
            category="preferences",
        )

        assert metadata.memory_id == "test-123"
        assert metadata.user_id == "user-1"
        assert metadata.scope == MemoryScope.PERSONAL
        assert metadata.category == "preferences"
        assert metadata.access_count == 0
        assert metadata.archived is False
        assert metadata.priority == 1.0

    def test_metadata_with_tags(self):
        """Test metadata with tags."""
        metadata = MemoryMetadata(
            memory_id="test-456",
            user_id="user-1",
            scope=MemoryScope.SHARED,
            tags=["python", "backend"],
        )

        assert metadata.tags == ["python", "backend"]


class TestMetadataStore:
    """Tests for MetadataStore."""

    def test_save_and_get_metadata(self, metadata_store):
        """Test saving and retrieving metadata."""
        metadata = MemoryMetadata(
            memory_id="mem-001",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
            category="decisions",
        )

        metadata_store.save_metadata(metadata)
        retrieved = metadata_store.get_metadata("mem-001")

        assert retrieved is not None
        assert retrieved.memory_id == "mem-001"
        assert retrieved.user_id == "user-1"
        assert retrieved.category == "decisions"

    def test_record_access(self, metadata_store):
        """Test recording memory access."""
        metadata = MemoryMetadata(
            memory_id="mem-002",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
        )
        metadata_store.save_metadata(metadata)

        # Record access
        metadata_store.record_access("mem-002")

        retrieved = metadata_store.get_metadata("mem-002")
        assert retrieved is not None
        assert retrieved.access_count == 1

        # Record another access
        metadata_store.record_access("mem-002")
        retrieved = metadata_store.get_metadata("mem-002")
        assert retrieved.access_count == 2

    def test_get_hot_memories(self, metadata_store):
        """Test finding frequently accessed memories."""
        # Create some memories
        for i in range(5):
            metadata = MemoryMetadata(
                memory_id=f"mem-{i:03d}",
                user_id="user-1",
                scope=MemoryScope.PERSONAL,
            )
            metadata_store.save_metadata(metadata)

        # Access some memories multiple times
        for _ in range(6):
            metadata_store.record_access("mem-001")
            metadata_store.record_access("mem-002")

        # Get hot memories (threshold 5)
        hot = metadata_store.get_hot_memories("user-1", threshold=5)
        assert len(hot) == 2
        assert "mem-001" in hot
        assert "mem-002" in hot

    def test_archive_memory(self, metadata_store):
        """Test archiving a memory."""
        metadata = MemoryMetadata(
            memory_id="mem-archive",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
        )
        metadata_store.save_metadata(metadata)

        metadata_store.archive_memory("mem-archive")

        retrieved = metadata_store.get_metadata("mem-archive")
        assert retrieved is not None
        assert retrieved.archived is True

    def test_increase_priority(self, metadata_store):
        """Test increasing memory priority."""
        metadata = MemoryMetadata(
            memory_id="mem-priority",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
            priority=1.0,
        )
        metadata_store.save_metadata(metadata)

        metadata_store.increase_priority("mem-priority", boost=0.2)

        retrieved = metadata_store.get_metadata("mem-priority")
        assert retrieved is not None
        assert retrieved.priority == 1.2

    def test_delete_metadata(self, metadata_store):
        """Test deleting metadata."""
        metadata = MemoryMetadata(
            memory_id="mem-delete",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
        )
        metadata_store.save_metadata(metadata)

        metadata_store.delete_metadata("mem-delete")

        retrieved = metadata_store.get_metadata("mem-delete")
        assert retrieved is None

    def test_log_maintenance(self, metadata_store):
        """Test logging maintenance jobs."""
        log_id = metadata_store.log_maintenance(
            job_type="consolidation",
            user_id="user-1",
            status="started",
        )

        assert log_id > 0

        metadata_store.update_maintenance_log(
            log_id,
            status="completed",
            details='{"duplicates_merged": 5}',
        )

    def test_get_all_user_memories(self, metadata_store):
        """Test getting all memories for a user."""
        # Create memories for different users
        for i in range(3):
            metadata = MemoryMetadata(
                memory_id=f"user1-mem-{i}",
                user_id="user-1",
                scope=MemoryScope.PERSONAL,
            )
            metadata_store.save_metadata(metadata)

        for i in range(2):
            metadata = MemoryMetadata(
                memory_id=f"user2-mem-{i}",
                user_id="user-2",
                scope=MemoryScope.PERSONAL,
            )
            metadata_store.save_metadata(metadata)

        user1_memories = metadata_store.get_all_user_memories("user-1")
        assert len(user1_memories) == 3

        user2_memories = metadata_store.get_all_user_memories("user-2")
        assert len(user2_memories) == 2

    def test_get_memories_by_category(self, metadata_store):
        """Test getting memories by category."""
        categories = ["preferences", "decisions", "preferences"]
        for i, cat in enumerate(categories):
            metadata = MemoryMetadata(
                memory_id=f"cat-mem-{i}",
                user_id="user-1",
                scope=MemoryScope.PERSONAL,
                category=cat,
            )
            metadata_store.save_metadata(metadata)

        prefs = metadata_store.get_memories_by_category("user-1", "preferences")
        assert len(prefs) == 2

        decisions = metadata_store.get_memories_by_category("user-1", "decisions")
        assert len(decisions) == 1
