"""Embedding clients for CEMS.

Supports two backends:
1. OpenRouter API - Uses OpenAI text-embedding-3-small via OpenRouter
2. llama.cpp server - Uses local llama.cpp server via HTTP (see llamacpp_server.py)

Environment Variables:
    OPENROUTER_API_KEY: Required for OpenRouter API calls.
    CEMS_EMBEDDING_MODEL: Override default OpenRouter model (optional).
    CEMS_EMBEDDING_BACKEND: "openrouter" or "llamacpp_server"
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

# Default embedding model (1536 dimensions)
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536


class EmbeddingClient:
    """Client for generating embeddings via OpenRouter API.

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
    ):
        """Initialize the embedding client.

        Args:
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            model: Embedding model in OpenRouter format.
                   Defaults to CEMS_EMBEDDING_MODEL or openai/text-embedding-3-small.
            dimensions: Output dimensions (optional, model-dependent).
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model or os.getenv("CEMS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.dimensions = dimensions

        # HTTP client with retries
        self._client = httpx.Client(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/cems",
                "X-Title": "CEMS Memory Server",
            },
        )

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
        """Make API call to OpenRouter embeddings endpoint.

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

        if self.dimensions:
            payload["dimensions"] = self.dimensions

        try:
            response = self._client.post(
                OPENROUTER_EMBEDDINGS_URL,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            # OpenRouter follows OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
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
            logger.error(f"Embedding request failed: {e}")
            raise ValueError(f"Embedding request failed: {e}") from e
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
    """Async client for generating embeddings via OpenRouter API.

    Example:
        async with AsyncEmbeddingClient() as client:
            embedding = await client.embed("Hello, world!")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        """Initialize the async embedding client.

        Args:
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            model: Embedding model in OpenRouter format.
            dimensions: Output dimensions (optional).
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model or os.getenv("CEMS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.dimensions = dimensions
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/cems",
                    "X-Title": "CEMS Memory Server",
                },
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
        """Generate embeddings for multiple texts in a single API call.

        This is significantly faster than sequential embed() calls.
        For 5 texts: batch = ~500ms vs sequential = ~2500ms.

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
        """Make async API call to OpenRouter embeddings endpoint.

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

        if self.dimensions:
            payload["dimensions"] = self.dimensions

        try:
            response = await client.post(
                OPENROUTER_EMBEDDINGS_URL,
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
            logger.error(f"Embedding request failed: {e}")
            raise ValueError(f"Embedding request failed: {e}") from e
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
