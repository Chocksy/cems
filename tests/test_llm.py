"""Tests for CEMS LLM utilities (OpenRouter-only)."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestOpenRouterClient:
    """Tests for OpenRouterClient class."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_client_initialization(self, mock_openai_class):
        """Test client initializes with OpenRouter configuration."""
        from cems.llm import OpenRouterClient, OPENROUTER_BASE_URL

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        client = OpenRouterClient()

        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["base_url"] == OPENROUTER_BASE_URL
        assert call_kwargs["api_key"] == "test-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_client_requires_api_key(self):
        """Test that missing OPENROUTER_API_KEY raises error."""
        from cems.llm import OpenRouterClient

        os.environ.pop("OPENROUTER_API_KEY", None)

        with pytest.raises(ValueError, match="OpenRouter API key required"):
            OpenRouterClient()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_client_custom_model(self, mock_openai_class):
        """Test client with custom model."""
        from cems.llm import OpenRouterClient

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        client = OpenRouterClient(model="openai/gpt-4o")

        assert client.model == "openai/gpt-4o"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_client_attribution_headers(self, mock_openai_class):
        """Test client includes attribution headers."""
        from cems.llm import OpenRouterClient

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        OpenRouterClient(
            site_url="https://test.com",
            site_name="Test App",
        )

        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://test.com"
        assert call_kwargs["default_headers"]["X-Title"] == "Test App"


class TestModelResolution:
    """Tests for model name resolution."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_resolve_none_model(self, mock_openai_class):
        """Test resolving None returns default."""
        from cems.llm import OpenRouterClient, OPENROUTER_MODELS

        mock_openai_class.return_value = MagicMock()

        client = OpenRouterClient()
        assert client.model == OPENROUTER_MODELS["default"]

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_resolve_already_openrouter_format(self, mock_openai_class):
        """Test models already in OpenRouter format pass through."""
        from cems.llm import OpenRouterClient

        mock_openai_class.return_value = MagicMock()

        client = OpenRouterClient(model="anthropic/claude-3-haiku")
        assert client.model == "anthropic/claude-3-haiku"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_resolve_known_model(self, mock_openai_class):
        """Test known model names are mapped."""
        from cems.llm import OpenRouterClient

        mock_openai_class.return_value = MagicMock()

        client = OpenRouterClient(model="gpt-4o-mini")
        assert client.model == "openai/gpt-4o-mini"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_resolve_unknown_model(self, mock_openai_class):
        """Test unknown models pass through."""
        from cems.llm import OpenRouterClient

        mock_openai_class.return_value = MagicMock()

        client = OpenRouterClient(model="some-custom-model")
        assert client.model == "some-custom-model"


class TestClientComplete:
    """Tests for the complete method."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_complete_basic(self, mock_openai_class):
        """Test basic completion."""
        from cems.llm import OpenRouterClient

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OpenRouterClient()
        result = client.complete("Test prompt")

        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_complete_with_system(self, mock_openai_class):
        """Test completion with system prompt."""
        from cems.llm import OpenRouterClient

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OpenRouterClient()
        client.complete("User prompt", system="System prompt")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestSummarizeMemories:
    """Tests for summarize_memories function."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_summarize_memories(self, mock_openai_class):
        """Test summarization."""
        from cems.llm import summarize_memories

        # Reset global client
        import cems.llm
        cems.llm.client._client = None

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a summary."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        memories = [
            "User prefers Python for backend",
            "User likes TypeScript for frontend",
        ]

        result = summarize_memories(memories, "preferences")

        assert result == "This is a summary."

        # Cleanup
        cems.llm.client._client = None

    def test_summarize_empty_memories(self):
        """Test summarization with empty memories."""
        from cems.llm import summarize_memories

        result = summarize_memories([], "empty")
        assert "No memories" in result

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_summarize_fallback_on_error(self, mock_openai_class):
        """Test fallback when LLM fails."""
        from cems.llm import summarize_memories

        # Reset global client
        import cems.llm
        cems.llm.client._client = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_class.return_value = mock_client

        memories = ["Memory 1", "Memory 2"]

        result = summarize_memories(memories, "test")

        # Should return fallback summary
        assert "test" in result.lower() or "Test" in result
        assert "2 memories" in result

        # Cleanup
        cems.llm.client._client = None


class TestMergeMemoryContents:
    """Tests for merge_memory_contents function."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_merge_memory_contents(self, mock_openai_class):
        """Test merging duplicate memories."""
        from cems.llm import merge_memory_contents

        # Reset global client
        import cems.llm
        cems.llm.client._client = None

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Merged content."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        memories = [
            {"memory": "User likes Python"},
            {"memory": "User prefers Python for backend"},
        ]

        result = merge_memory_contents(memories)

        assert result == "Merged content."

        # Cleanup
        cems.llm.client._client = None

    def test_merge_single_memory(self):
        """Test merging single memory returns as-is."""
        from cems.llm import merge_memory_contents

        memories = [{"memory": "Single memory content"}]

        result = merge_memory_contents(memories)

        assert result == "Single memory content"

    def test_merge_empty_memories(self):
        """Test merging empty list."""
        from cems.llm import merge_memory_contents

        result = merge_memory_contents([])
        assert result == ""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_merge_fallback_on_error(self, mock_openai_class):
        """Test fallback when LLM fails."""
        from cems.llm import merge_memory_contents

        # Reset global client
        import cems.llm
        cems.llm.client._client = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_class.return_value = mock_client

        memories = [
            {"memory": "First memory"},
            {"memory": "Second memory"},
        ]

        result = merge_memory_contents(memories)

        # Should return first memory as fallback
        assert result == "First memory"

        # Cleanup
        cems.llm.client._client = None


class TestFallbackSummary:
    """Tests for fallback summary generation."""

    def test_fallback_summary(self):
        """Test fallback summary generation."""
        from cems.llm.summarization import _fallback_summary

        memories = [
            "Memory one content",
            "Memory two content",
            "Memory three content",
            "Memory four content",
            "Memory five content",
            "Memory six content",
        ]

        result = _fallback_summary(memories, "test")

        assert "Test" in result
        assert "6 memories" in result
        assert "and 1 more" in result

    def test_fallback_summary_short_list(self):
        """Test fallback summary with short list."""
        from cems.llm.summarization import _fallback_summary

        memories = ["Memory one", "Memory two"]

        result = _fallback_summary(memories, "short")

        assert "Short" in result
        assert "2 memories" in result
        assert "more" not in result


class TestGetClient:
    """Tests for get_client function."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_get_client_singleton(self, mock_openai_class):
        """Test get_client returns singleton."""
        from cems.llm import get_client

        # Reset global client
        import cems.llm
        cems.llm.client._client = None

        mock_openai_class.return_value = MagicMock()

        client1 = get_client()
        client2 = get_client()

        assert client1 is client2

        # Cleanup
        cems.llm.client._client = None


class TestBackwardsCompatibility:
    """Tests for deprecated backwards compatibility functions."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_get_llm_client_deprecated(self, mock_openai_class):
        """Test deprecated get_llm_client still works."""
        from cems.llm import get_llm_client

        # Reset global client
        import cems.llm
        cems.llm.client._client = None

        mock_openai_class.return_value = MagicMock()

        # Should work but issue deprecation warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_llm_client("openai")  # Provider ignored

        # Cleanup
        cems.llm.client._client = None

    def test_resolve_openrouter_model_deprecated(self):
        """Test deprecated model resolution."""
        from cems.llm.client import _resolve_openrouter_model

        result = _resolve_openrouter_model("gpt-4o-mini")
        assert result == "openai/gpt-4o-mini"
