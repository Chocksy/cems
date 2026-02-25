"""Tests for CEMS configuration."""

import os
import tempfile
from pathlib import Path

import pytest

from cems.config import CEMSConfig


class TestCEMSConfig:
    """Tests for CEMSConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CEMSConfig()

        assert config.user_id == "default"
        assert config.team_id is None
        assert config.enable_scheduler is True
        assert config.stale_days == 90
        assert config.archive_days == 180

    def test_custom_user_id(self):
        """Test setting custom user ID."""
        config = CEMSConfig(user_id="test-user")
        assert config.user_id == "test-user"

    def test_team_id(self):
        """Test team ID configuration."""
        config = CEMSConfig(user_id="user", team_id="my-team")
        assert config.team_id == "my-team"

    def test_storage_paths(self):
        """Test storage path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CEMSConfig(storage_dir=Path(tmpdir))

            assert config.storage_dir == Path(tmpdir)

    def test_scheduler_config(self):
        """Test scheduler configuration."""
        config = CEMSConfig(
            nightly_hour=2,
            weekly_day="mon",
            weekly_hour=5,
            monthly_day=15,
            monthly_hour=6,
        )

        assert config.nightly_hour == 2
        assert config.weekly_day == "mon"
        assert config.weekly_hour == 5
        assert config.monthly_day == 15
        assert config.monthly_hour == 6

    def test_decay_settings(self):
        """Test decay settings."""
        config = CEMSConfig(
            stale_days=60,
            archive_days=120,
            hot_access_threshold=10,
            duplicate_similarity_threshold=0.95,
        )

        assert config.stale_days == 60
        assert config.archive_days == 120
        assert config.hot_access_threshold == 10
        assert config.duplicate_similarity_threshold == 0.95

    def test_llm_settings(self):
        """Test LLM configuration."""
        config = CEMSConfig(
            llm_model="anthropic/claude-3-haiku",
            embedding_model="openai/text-embedding-3-large",
        )

        assert config.llm_model == "anthropic/claude-3-haiku"
        assert config.embedding_model == "openai/text-embedding-3-large"

    def test_relevance_threshold(self):
        """Test relevance threshold default and custom values."""
        # Default: 0.4 (raised from 0.3 in Phase 2 pipeline quality)
        config = CEMSConfig()
        assert config.relevance_threshold == 0.4

        # Test custom value
        config = CEMSConfig(relevance_threshold=0.7)
        assert config.relevance_threshold == 0.7
