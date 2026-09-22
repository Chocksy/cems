"""Embedding clients for CEMS.

Uses any OpenAI-compatible embeddings endpoint (OpenRouter by default, or
Ollama, vLLM, LiteLLM). OpenRouter-only extras (attribution headers, the
``dimensions`` parameter) are sent only when the endpoint is openrouter.ai.

Environment Variables:
    CEMS_EMBEDDING_BASE_URL: Embeddings base URL. Defaults to CEMS_LLM_BASE_URL.
    CEMS_EMBEDDING_API_KEY: API key. Falls back to CEMS_LLM_API_KEY, then OPENROUTER_API_KEY.
    CEMS_EMBEDDING_MODEL: Override the default embedding model (optional).
    CEMS_EMBEDDING_DIMENSION: Vector dimension (optional).
    OPENROUTER_API_KEY: Fallback API key when no CEMS key is set.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from cems.config import CEMSConfig, is_openrouter_host

logger = logging.getLogger(__name__)

# Kept for backwards-compatible imports; the endpoint now comes from config.
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

# Default embedding model (1536 dimensions)
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536


def _resolve_endpoint(api_key: str | None, base_url: str | None) -> tuple[str, str, bool]:
    """Return (embeddings_url, api_key, is_openrouter) from args and config."""
    cfg = CEMSConfig()
    resolved_base = (base_url or cfg.resolved_embedding_base_url()).rstrip("/")
    key = api_key or cfg.resolved_embedding_api_key()
    if not key:
        raise ValueError(
            "Embedding API key required. Set CEMS_EMBEDDING_API_KEY, CEMS_LLM_API_KEY "
            "or OPENROUTER_API_KEY, or pass api_key."
        )
    return f"{resolved_base}/embeddings", key, is_openrouter_host(resolved_base)


def _headers(api_key: str, is_openrouter: bool) -> dict[str, str]:
    """Build request headers, adding OpenRouter attribution only for OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/cems"
        headers["X-Title"] = "CEMS Memory Server"
    return headers


class EmbeddingClient:
    """Client for generating embeddings via an OpenAI-compatible API.

    Supports single and batch embedding generation with automatic
    rate limiting and retries.

    Example:
        client = EmbeddingClient()
        embedding = client.embed("Hello, world!")

        # Batch embedding
        embeddings = client.embed_batch([
            "First text",
            "Second text",
            "Third text",
        ])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ):
        """Initialize the embedding client.

        Args:
            api_key: API key. Defaults to CEMS_EMBEDDING_API_KEY, then the LLM key.
            model: Embedding model name.
                   Defaults to CEMS_EMBEDDING_MODEL or openai/text-embedding-3-small.
            dimensions: Output dimensions (OpenRouter only, model-dependent).
            base_url: OpenAI-compatible base URL. Defaults to the configured endpoint.
        """
        self.embeddings_url, self.api_key, self.is_openrouter = _resolve_endpoint(api_key, base_url)
        self.model = model or os.getenv("CEMS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.dimensions = dimensions
        self._client = httpx.Client(timeout=30.0, headers=_headers(self.api_key, self.is_openrouter))

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        result = self._call_api([text])
        return result[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Automatically batches large requests to avoid API limits.

        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"Embedded batch {i // batch_size + 1}, {len(batch)} texts")

        return all_embeddings

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Make API call to the configured embeddings endpoint.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If API call fails
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        if self.dimensions and self.is_openrouter:
            payload["dimensions"] = self.dimensions

        try:
            response = self._client.post(
                self.embeddings_url,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            # OpenAI-compatible format: {"data": [{"embedding": [...], "index": 0}, ...]}
            embeddings = [None] * len(texts)
            for item in data["data"]:
                embeddings[item["index"]] = item["embedding"]

            # Verify all embeddings were returned
            if None in embeddings:
                raise ValueError("Missing embeddings in API response")

            return embeddings  # type: ignore

        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Embedding API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Embedding request failed: {type(e).__name__}: {e!r}")
            raise ValueError(f"Embedding request failed: {type(e).__name__}: {e!r}") from e
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Invalid API response: {e}")
            raise ValueError(f"Invalid embedding response: {e}") from e

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncEmbeddingClient:
    """Async client for generating embeddings via an OpenAI-compatible API.

    Example:
        async with AsyncEmbeddingClient() as client:
            embedding = await client.embed("Hello, world!")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ):
        """Initialize the async embedding client.

        Args:
            api_key: API key. Defaults to CEMS_EMBEDDING_API_KEY, then the LLM key.
            model: Embedding model name.
            dimensions: Output dimensions (OpenRouter only, model-dependent).
            base_url: OpenAI-compatible base URL. Defaults to the configured endpoint.
        """
        self.embeddings_url, self.api_key, self.is_openrouter = _resolve_endpoint(api_key, base_url)
        self.model = model or os.getenv("CEMS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.dimensions = dimensions
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=_headers(self.api_key, self.is_openrouter),
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        For multiple texts, prefer embed_batch() which is more efficient.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        import time

        start_time = time.time()
        result = await self._call_api([text])
        elapsed = time.time() - start_time
        logger.debug(f"[EMBEDDING] Single embed in {elapsed:.2f}s")
        return result[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts, batched into API calls.

        Automatically splits into batches of batch_size to avoid API limits.
        Faster than sequential embed() calls (batch ~500ms vs sequential ~2500ms).

        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call

        Returns:
            List of embedding vectors
        """
        import time

        if not texts:
            return []

        start_time = time.time()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self._call_api(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"Embedded batch {i // batch_size + 1}, {len(batch)} texts")

        elapsed = time.time() - start_time
        logger.info(f"[EMBEDDING] Batch embedded {len(texts)} texts in {elapsed:.2f}s ({elapsed/len(texts)*1000:.0f}ms/text)")

        return all_embeddings

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Make async API call to the configured embeddings endpoint.

        Note: Mirrors EmbeddingClient._call_api — kept separate to avoid
        mixing sync/async client lifecycle (httpx.Client vs httpx.AsyncClient).

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If API call fails
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        if self.dimensions and self.is_openrouter:
            payload["dimensions"] = self.dimensions

        try:
            response = await client.post(
                self.embeddings_url,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            embeddings = [None] * len(texts)
            for item in data["data"]:
                embeddings[item["index"]] = item["embedding"]

            if None in embeddings:
                raise ValueError("Missing embeddings in API response")

            return embeddings  # type: ignore

        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Embedding API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Embedding request failed: {type(e).__name__}: {e!r}")
            raise ValueError(f"Embedding request failed: {type(e).__name__}: {e!r}") from e
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Invalid API response: {e}")
            raise ValueError(f"Invalid embedding response: {e}") from e

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncEmbeddingClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
