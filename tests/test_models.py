"""Tests for CEMS Pydantic models."""

from datetime import datetime, UTC

import pytest

from cems.models import MemoryMetadata, MemoryScope, CategorySummary, PinCategory


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

    def test_metadata_defaults(self):
        """Test default values are set correctly."""
        metadata = MemoryMetadata(
            memory_id="test-defaults",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
        )

        assert metadata.category == "general"
        assert metadata.access_count == 0
        assert metadata.archived is False
        assert metadata.priority == 1.0
        assert metadata.pinned is False
        assert metadata.pin_reason is None
        assert metadata.expires_at is None
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)
        assert isinstance(metadata.last_accessed, datetime)

    def test_pinned_metadata(self):
        """Test pinned memory metadata."""
        metadata = MemoryMetadata(
            memory_id="pinned-123",
            user_id="user-1",
            scope=MemoryScope.PERSONAL,
            pinned=True,
            pin_reason="Core guideline",
            pin_category="guideline",
        )

        assert metadata.pinned is True
        assert metadata.pin_reason == "Core guideline"
        assert metadata.pin_category == "guideline"


class TestCategorySummary:
    """Tests for CategorySummary model."""

    def test_create_summary(self):
        """Test creating a category summary."""
        summary = CategorySummary(
            category="preferences",
            scope=MemoryScope.PERSONAL,
            summary="User prefers Python with type hints.",
            item_count=5,
            last_updated=datetime.now(UTC),
        )

        assert summary.category == "preferences"
        assert summary.scope == MemoryScope.PERSONAL
        assert summary.summary == "User prefers Python with type hints."
        assert summary.item_count == 5
        assert summary.version == 1

    def test_summary_version(self):
        """Test summary versioning."""
        summary = CategorySummary(
            category="decisions",
            scope=MemoryScope.SHARED,
            summary="Team uses PostgreSQL.",
            item_count=3,
            last_updated=datetime.now(UTC),
            version=5,
        )

        assert summary.version == 5


class TestEnums:
    """Tests for enum types."""

    def test_memory_scope_values(self):
        """Test MemoryScope enum values."""
        assert MemoryScope.PERSONAL.value == "personal"
        assert MemoryScope.SHARED.value == "shared"

    def test_pin_category_values(self):
        """Test PinCategory enum values."""
        assert PinCategory.GUIDELINE.value == "guideline"
        assert PinCategory.CONVENTION.value == "convention"
        assert PinCategory.ARCHITECTURE.value == "architecture"
