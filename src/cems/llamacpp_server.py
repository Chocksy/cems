"""llama.cpp server HTTP client for embeddings and reranking.

This module provides HTTP clients for llama.cpp server endpoints:
- Embeddings via /v1/embeddings (OpenAI-compatible)
- Reranking via /rerank endpoint

References:
- https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
- https://llamaedge.com/docs/user-guide/llm/api-reference
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from cems.config import CEMSConfig

logger = logging.getLogger(__name__)


class LlamaCppServerClient:
    """HTTP client for llama.cpp server embeddings and reranking.

    Supports:
    - /v1/embeddings (OpenAI-compatible) for embeddings
    - /rerank for cross-encoder reranking

    Usage:
        client = LlamaCppServerClient(
            base_url="http://localhost:8081",
            embed_path="/v1/embeddings",
        )
        embeddings = await client.embed_batch("model", ["text1", "text2"])
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        embed_path: str = "/v1/embeddings",
        rerank_path: str = "/rerank",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.embed_path = embed_path
        self.rerank_path = rerank_path
        self.timeout = timeout
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def embed(self, model: str, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_batch(model, [text])
        return embeddings[0]

    async def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            model: Model name/path (used by server for identification)
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        payload = {
            "model": model,
            "input": texts,
            "encoding_format": "float",
        }
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            resp = await client.post(f"{self.base_url}{self.embed_path}", json=payload)
            resp.raise_for_status()
            data = resp.json()

        # OpenAI-compatible response: data["data"][i]["embedding"]
        return [item["embedding"] for item in data["data"]]

    async def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
    ) -> list[dict]:
        """Rerank documents against a query.

        Args:
            model: Model name/path
            query: Query text
            documents: List of document texts to rerank

        Returns:
            List of dicts with 'index' and 'relevance_score' keys,
            sorted by relevance (highest first)
        """
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
        }
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            resp = await client.post(f"{self.base_url}{self.rerank_path}", json=payload)
            resp.raise_for_status()
            return resp.json()


class AsyncLlamaCppEmbeddingClient:
    """Async embedding client wrapping LlamaCppServerClient.

    Provides the same interface as AsyncEmbeddingClient for drop-in replacement.
    """

    # Model context limits (in characters, conservative estimate ~4 chars/token)
    # embeddinggemma: 2048 tokens => ~6000 chars
    # nomic-embed: 8192 tokens => ~24000 chars
    MAX_CHARS = 6000  # Safe for embeddinggemma-300M (2048 token limit)

    def __init__(self, config: "CEMSConfig"):
        self.config = config
        self._client = LlamaCppServerClient(
            base_url=config.llamacpp_base_url,
            api_key=config.llamacpp_api_key,
            embed_path=config.llamacpp_embed_path,
            timeout=config.llamacpp_timeout_seconds,
        )
        self.model = config.llamacpp_embed_model

    def _truncate(self, text: str) -> str:
        """Truncate text to fit within model's context limit."""
        if len(text) <= self.MAX_CHARS:
            return text
        # Truncate and log warning
        logger.warning(
            f"[EMBEDDING] Truncating text from {len(text)} to {self.MAX_CHARS} chars "
            f"(model limit ~2048 tokens)"
        )
        return text[: self.MAX_CHARS]

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return await self._client.embed(self.model, self._truncate(text))

    async def embed_batch(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Uses smaller batches (8) to avoid timeout issues with llama.cpp server.
        Each batch of 8 texts typically completes in ~3-4s vs 30s+ for 32 texts.
        """
        # Truncate all texts to fit model context
        truncated = [self._truncate(t) for t in texts]
        # Process in batches to avoid overwhelming the server
        all_embeddings = []
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i : i + batch_size]
            logger.debug(f"[EMBEDDING] Processing batch {i // batch_size + 1}, {len(batch)} texts")
            embeddings = await self._client.embed_batch(self.model, batch)
            all_embeddings.extend(embeddings)
        return all_embeddings


class AsyncLlamaCppRerankerClient:
    """Async reranker client wrapping LlamaCppServerClient.

    Provides reranking via llama.cpp server's /rerank endpoint.
    """

    def __init__(self, config: "CEMSConfig"):
        self.config = config
        self._client = LlamaCppServerClient(
            base_url=config.llamacpp_rerank_url,
            api_key=config.llamacpp_api_key,
            rerank_path=config.llamacpp_rerank_path,
            timeout=config.llamacpp_timeout_seconds,
        )
        self.model = config.llamacpp_rerank_model

    async def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """Rerank documents against a query.

        Returns list of dicts with 'index' and 'relevance_score'.
        """
        return await self._client.rerank(self.model, query, documents)
