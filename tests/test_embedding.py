"""Tests for embedding clients against configurable endpoints."""

import os
from unittest.mock import MagicMock, patch

import pytest


def _fake_response(n: int, dim: int = 3):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"index": i, "embedding": [0.1] * dim} for i in range(n)]}
    return resp


class TestEmbeddingClient:
    @patch.dict(os.environ, {"CEMS_LLM_BASE_URL": "http://ollama:11434/v1", "CEMS_LLM_API_KEY": "ollama"})
    @patch("cems.embedding.httpx.Client")
    def test_custom_endpoint_omits_dimensions_and_attribution(self, mock_client_class):
        from cems.embedding import EmbeddingClient

        http = MagicMock()
        http.post.return_value = _fake_response(1)
        mock_client_class.return_value = http

        client = EmbeddingClient(model="embeddinggemma", dimensions=768)
        client.embed("hello")

        assert client.embeddings_url == "http://ollama:11434/v1/embeddings"
        headers = mock_client_class.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer ollama"
        assert "HTTP-Referer" not in headers
        url, kwargs = http.post.call_args[0][0], http.post.call_args[1]
        assert url == "http://ollama:11434/v1/embeddings"
        assert "dimensions" not in kwargs["json"]
        assert kwargs["json"]["model"] == "embeddinggemma"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or"})
    @patch("cems.embedding.httpx.Client")
    def test_openrouter_sends_dimensions(self, mock_client_class):
        from cems.embedding import EmbeddingClient

        http = MagicMock()
        http.post.return_value = _fake_response(1)
        mock_client_class.return_value = http

        client = EmbeddingClient(dimensions=1536)
        client.embed("hello")

        assert client.embeddings_url == "https://openrouter.ai/api/v1/embeddings"
        assert http.post.call_args[1]["json"]["dimensions"] == 1536
        assert mock_client_class.call_args[1]["headers"]["HTTP-Referer"] == "https://github.com/cems"

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_key(self):
        from cems.embedding import EmbeddingClient

        with pytest.raises(ValueError, match="API key required"):
            EmbeddingClient()


class TestAsyncEmbeddingClient:
    @patch.dict(os.environ, {"CEMS_LLM_BASE_URL": "http://ollama:11434/v1", "CEMS_LLM_API_KEY": "ollama"})
    async def test_custom_endpoint_url(self):
        from cems.embedding import AsyncEmbeddingClient

        client = AsyncEmbeddingClient(model="embeddinggemma")
        assert client.embeddings_url == "http://ollama:11434/v1/embeddings"
        assert client.is_openrouter is False
