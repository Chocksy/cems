"""Extended tests for CEMS models - categories, summaries, recently accessed."""

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


class TestCategoriesAndSummaries:
    """Tests for category and summary functionality."""

    def test_get_all_categories(self, metadata_store):
        """Test getting all categories with counts."""
        # Create memories with different categories
        categories = ["preferences", "decisions", "preferences", "patterns", "decisions"]
        for i, cat in enumerate(categories):
            metadata = MemoryMetadata(
                memory_id=f"cat-{i}",
                user_id="user-1",
                scope=MemoryScope.PERSONAL,
                category=cat,
            )
            metadata_store.save_metadata(metadata)

        result = metadata_store.get_all_categories("user-1", MemoryScope.PERSONAL)

        # Should have 3 unique categories
        assert len(result) == 3

        # Check counts
        categories_dict = {r["category"]: r["count"] for r in result}
        assert categories_dict["preferences"] == 2
        assert categories_dict["decisions"] == 2
        assert categories_dict["patterns"] == 1

    def test_get_all_categories_both_scopes(self, metadata_store):
        """Test getting categories from both scopes."""
        # Personal memories
        for i in range(2):
            metadata = MemoryMetadata(
                memory_id=f"personal-{i}",
                user_id="user-1",
                scope=MemoryScope.PERSONAL,
                category="preferences",
            )
            metadata_store.save_metadata(metadata)

        # Shared memories
        metadata = MemoryMetadata(
            memory_id="shared-1",
            user_id="user-1",
            scope=MemoryScope.SHARED,
            category="conventions",
        )
        metadata_store.save_metadata(metadata)

        result = metadata_store.get_all_categories("user-1")

        # Should have 2 categories with scope info
        assert len(result) == 2
        scopes = [r["scope"] for r in result]
        assert "personal" in scopes
        assert "shared" in scopes

    def test_get_recently_accessed(self, metadata_store):
        """Test getting recently accessed memories."""
        # Create memories
        for i in range(5):
            metadata = MemoryMetadata(
                memory_id=f"recent-{i}",
                user_id="user-1",
                scope=MemoryScope.PERSONAL,
                category="general",
            )
            metadata_store.save_metadata(metadata)

        # Access specific memories multiple times to make them "hot"
        for _ in range(3):
            metadata_store.record_access("recent-2")
        for _ in range(2):
            metadata_store.record_access("recent-4")
        metadata_store.record_access("recent-1")

        recent = metadata_store.get_recently_accessed("user-1", limit=5)

        assert len(recent) == 5
        # All memories should have access info
        for r in recent:
            assert "memory_id" in r
            assert "access_count" in r

    def test_category_summary_crud(self, metadata_store):
        """Test category summary create/read."""
        import sqlite3

        # Manually insert a summary (normally done by maintenance)
        conn = sqlite3.connect(metadata_store.db_path)
        conn.execute(
            """
            INSERT INTO category_summaries
            (user_id, category, scope, summary, item_count, last_updated, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "user-1",
                "preferences",
                "personal",
                "This is a summary of user preferences.",
                10,
                datetime.now().isoformat(),
                1,
            ),
        )
        conn.commit()
        conn.close()

        # Retrieve it
        summary = metadata_store.get_category_summary("user-1", "preferences", "personal")

        assert summary is not None
        assert summary["summary"] == "This is a summary of user preferences."
        assert summary["item_count"] == 10
        assert summary["version"] == 1

    def test_get_all_category_summaries(self, metadata_store):
        """Test getting all summaries."""
        import sqlite3

        # Insert multiple summaries
        conn = sqlite3.connect(metadata_store.db_path)
        for cat in ["preferences", "decisions", "patterns"]:
            conn.execute(
                """
                INSERT INTO category_summaries
                (user_id, category, scope, summary, item_count, last_updated, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "user-1",
                    cat,
                    "personal",
                    f"Summary for {cat}",
                    5,
                    datetime.now().isoformat(),
                    1,
                ),
            )
        conn.commit()
        conn.close()

        summaries = metadata_store.get_all_category_summaries("user-1", "personal")

        assert len(summaries) == 3
        categories = [s["category"] for s in summaries]
        assert "preferences" in categories
        assert "decisions" in categories
        assert "patterns" in categories


class TestPinnedMemories:
    """Tests for pinned memory functionality."""

    def test_pin_memory(self, metadata_store):
        """Test pinning a memory."""
        metadata = MemoryMetadata(
            memory_id="pin-test",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
        )
        metadata_store.save_metadata(metadata)

        metadata_store.pin_memory("pin-test", reason="Core guideline", pin_category="convention")

        retrieved = metadata_store.get_metadata("pin-test")
        assert retrieved is not None
        assert retrieved.pinned is True
        assert retrieved.pin_reason == "Core guideline"
        assert retrieved.pin_category == "convention"

    def test_unpin_memory(self, metadata_store):
        """Test unpinning a memory."""
        metadata = MemoryMetadata(
            memory_id="unpin-test",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
            pinned=True,
            pin_reason="Test",
        )
        metadata_store.save_metadata(metadata)

        metadata_store.unpin_memory("unpin-test")

        retrieved = metadata_store.get_metadata("unpin-test")
        assert retrieved is not None
        assert retrieved.pinned is False

    def test_get_pinned_memories(self, metadata_store):
        """Test getting all pinned memories."""
        # Create mix of pinned and non-pinned
        for i in range(5):
            metadata = MemoryMetadata(
                memory_id=f"mixed-{i}",
                user_id="user-1",
                scope=MemoryScope.PERSONAL,
                pinned=(i % 2 == 0),  # Pin every other one
            )
            metadata_store.save_metadata(metadata)

        pinned = metadata_store.get_pinned_memories("user-1")
        assert len(pinned) == 3  # 0, 2, 4 are pinned

    def test_stale_excludes_pinned(self, metadata_store):
        """Test that stale query excludes pinned memories."""
        import sqlite3

        # Create an old pinned memory
        metadata = MemoryMetadata(
            memory_id="old-pinned",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
            pinned=True,
        )
        metadata_store.save_metadata(metadata)

        # Manually backdate it
        conn = sqlite3.connect(metadata_store.db_path)
        conn.execute(
            """
            UPDATE memory_metadata
            SET last_accessed = datetime('now', '-100 days')
            WHERE memory_id = ?
            """,
            ("old-pinned",),
        )
        conn.commit()
        conn.close()

        stale = metadata_store.get_stale_memories("user-1", days=90)

        # Pinned memory should NOT be in stale list
        assert "old-pinned" not in stale
