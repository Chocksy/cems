"""Tests for CEMS configuration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from cems.config import CEMSConfig, is_openrouter_host


class TestCEMSConfig:
    """Tests for CEMSConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CEMSConfig()

        assert config.user_id == "default"
        assert config.default_scope == "shared"
        assert config.enable_scheduler is True
        assert config.stale_days == 30
        assert config.archive_days == 60

    def test_custom_user_id(self):
        """Test setting custom user ID."""
        config = CEMSConfig(user_id="test-user")
        assert config.user_id == "test-user"

    def test_default_scope(self):
        """Test default scope configuration."""
        config = CEMSConfig(default_scope="personal")
        assert config.default_scope == "personal"

        config = CEMSConfig(default_scope="shared")
        assert config.default_scope == "shared"

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
        # Default: 0.45 (raised from 0.4 to reduce noise)
        config = CEMSConfig()
        assert config.relevance_threshold == 0.45

        # Test custom value
        config = CEMSConfig(relevance_threshold=0.7)
        assert config.relevance_threshold == 0.7


class TestProviderConfig:
    """Tests for the generic OpenAI-compatible provider settings."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-x"}, clear=False)
    def test_defaults_point_at_openrouter(self):
        cfg = CEMSConfig()
        assert cfg.llm_base_url == "https://openrouter.ai/api/v1"
        assert cfg.resolved_llm_api_key() == "sk-or-x"
        assert cfg.resolved_embedding_base_url() == "https://openrouter.ai/api/v1"
        assert cfg.resolved_embedding_api_key() == "sk-or-x"
        assert cfg.embedding_dimension == 1536
        assert cfg.enable_agentic_search is True

    @patch.dict(
        os.environ,
        {
            "CEMS_LLM_BASE_URL": "http://ollama:11434/v1",
            "CEMS_LLM_API_KEY": "ollama",
            "CEMS_EMBEDDING_DIMENSION": "768",
        },
        clear=False,
    )
    def test_custom_endpoint_falls_through_to_embeddings(self):
        cfg = CEMSConfig()
        assert cfg.llm_base_url == "http://ollama:11434/v1"
        assert cfg.resolved_llm_api_key() == "ollama"
        assert cfg.resolved_embedding_base_url() == "http://ollama:11434/v1"
        assert cfg.resolved_embedding_api_key() == "ollama"
        assert cfg.embedding_dimension == 768

    @patch.dict(
        os.environ,
        {
            "CEMS_LLM_BASE_URL": "http://ollama:11434/v1",
            "CEMS_LLM_API_KEY": "ollama",
            "CEMS_EMBEDDING_BASE_URL": "https://api.openai.com/v1",
            "CEMS_EMBEDDING_API_KEY": "sk-openai",
        },
        clear=False,
    )
    def test_embedding_endpoint_overrides_independently(self):
        cfg = CEMSConfig()
        assert cfg.resolved_embedding_base_url() == "https://api.openai.com/v1"
        assert cfg.resolved_embedding_api_key() == "sk-openai"

    @patch.dict(os.environ, {"CEMS_LLM_API_KEY": "", "OPENROUTER_API_KEY": "sk-or-x"}, clear=False)
    def test_empty_llm_api_key_falls_back_to_openrouter_key(self):
        cfg = CEMSConfig()
        assert cfg.resolved_llm_api_key() == "sk-or-x"

    @patch.dict(
        os.environ,
        {"CEMS_EMBEDDING_BASE_URL": "", "CEMS_LLM_API_KEY": "", "OPENROUTER_API_KEY": "k"},
        clear=False,
    )
    def test_empty_env_values_mean_unset(self):
        cfg = CEMSConfig()
        assert cfg.resolved_embedding_base_url() == "https://openrouter.ai/api/v1"
        assert cfg.resolved_llm_api_key() == "k"

    def test_llamacpp_fields_are_gone(self):
        assert not hasattr(CEMSConfig(), "embedding_backend")
        assert not hasattr(CEMSConfig(), "llamacpp_base_url")

    def test_is_openrouter_host(self):
        assert is_openrouter_host("https://openrouter.ai/api/v1")
        assert not is_openrouter_host("http://ollama:11434/v1")
        assert not is_openrouter_host("https://api.openai.com/v1")


class TestModelFor:
    """model_for() keeps OpenRouter installs on their tuned models."""

    def test_openrouter_host_returns_the_openrouter_default(self):
        cfg = CEMSConfig(llm_base_url="https://openrouter.ai/api/v1", llm_model="qwen/qwen3-32b")
        assert cfg.model_for("google/gemini-2.5-flash") == "google/gemini-2.5-flash"
        assert cfg.model_for("google/gemini-2.5-flash-lite") == "google/gemini-2.5-flash-lite"

    def test_local_host_returns_the_configured_llm_model(self):
        cfg = CEMSConfig(llm_base_url="http://ollama:11434/v1", llm_model="qwen3:8b")
        assert cfg.model_for("google/gemini-2.5-flash") == "qwen3:8b"
        assert cfg.model_for("google/gemini-2.5-flash-lite") == "qwen3:8b"

    def test_default_config_is_openrouter(self):
        assert CEMSConfig().model_for("google/gemini-2.5-flash") == "google/gemini-2.5-flash"
