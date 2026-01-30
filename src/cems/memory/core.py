"""CEMS Memory with native pgvector storage.

This module provides unified memory management using PostgreSQL with pgvector
for both vector embeddings and metadata storage. It replaces the previous
Mem0 + Qdrant architecture with a simpler, ACID-compliant solution.

Key features:
- Vector similarity search (HNSW index)
- Full-text search (GIN index on tsvector)
- Hybrid search using RRF (Reciprocal Rank Fusion)
- ACID transactions for data consistency
- Namespace isolation (personal vs shared memories)
- Extended metadata tracking (access counts, priorities)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from cems.config import CEMSConfig
from cems.memory.analytics import AnalyticsMixin
from cems.memory.crud import CRUDMixin
from cems.memory.metadata import MetadataMixin
from cems.memory.relations import RelationsMixin
from cems.memory.retrieval import RetrievalMixin
from cems.memory.search import SearchMixin
from cems.memory.write import WriteMixin
from cems.models import (
    MemoryMetadata,
    MemoryScope,
    SearchResult,
)

if TYPE_CHECKING:
    from cems.db.metadata_store import PostgresMetadataStore
    from cems.embedding import AsyncEmbeddingClient, EmbeddingClient
    from cems.fact_extraction import FactExtractor
    from cems.llamacpp_server import AsyncLlamaCppEmbeddingClient
    from cems.vectorstore import PgVectorStore

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine in a sync context.

    NOTE: This is for sync contexts only (CLI, MCP stdio).
    For async contexts (HTTP server), use the async methods directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context - caller should use async methods
        raise RuntimeError(
            "Cannot use sync method from async context. "
            "Use the async version (e.g., add_async instead of add)."
        )
    else:
        # No running loop - create one
        return asyncio.run(coro)


class CEMSMemory(WriteMixin, SearchMixin, CRUDMixin, AnalyticsMixin, MetadataMixin, RelationsMixin, RetrievalMixin):
    """Memory system with personal/shared namespace isolation.

    Built on PostgreSQL + pgvector, this class provides:
    - Namespace isolation (personal vs shared memories)
    - Extended metadata tracking (access counts, priorities)
    - Unified search across namespaces
    - Hybrid search (vector + full-text)
    - ACID transactions for consistency
    """

    def __init__(self, config: CEMSConfig | None = None):
        """Initialize CEMS memory.

        Args:
            config: CEMS configuration. If None, loads from environment.
        """
        self.config = config or CEMSConfig()

        # Validate database URL
        if not self.config.database_url:
            raise ValueError(
                "CEMS_DATABASE_URL is required. "
                "CEMS runs in Docker/server mode only (no local SQLite mode)."
            )

        # Initialize components lazily
        self._vectorstore: PgVectorStore | None = None
        self._embedder: EmbeddingClient | None = None
        self._async_embedder: AsyncEmbeddingClient | AsyncLlamaCppEmbeddingClient | None = None
        self._fact_extractor: FactExtractor | None = None
        self._metadata: PostgresMetadataStore | None = None
        self._initialized = False
        self._async_initialized = False  # Track async initialization separately

        # Initialize legacy metadata store for backwards compatibility
        # This will be deprecated once migration is complete
        from cems.db.database import init_database, is_database_initialized
        from cems.db.metadata_store import PostgresMetadataStore

        if not is_database_initialized():
            init_database(self.config.database_url)
        self._metadata = PostgresMetadataStore()

        # Graph store (using PostgreSQL relations table instead of Kuzu)
        self._use_pg_relations = True

    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized (sync version for CLI/MCP).

        Note: llamacpp_server backend requires async - use _ensure_initialized_async().
        """
        if self._initialized:
            return

        from cems.embedding import EmbeddingClient
        from cems.fact_extraction import FactExtractor
        from cems.vectorstore import PgVectorStore

        # Initialize vectorstore with configured dimension
        if self._vectorstore is None:
            self._vectorstore = PgVectorStore(
                database_url=self.config.database_url,
                embedding_dim=self.config.embedding_dimension,
            )
            # Connect synchronously - only works in sync contexts
            _run_async(self._vectorstore.connect())

        # Initialize embedder based on config
        if self._embedder is None:
            if self.config.embedding_backend == "llamacpp_server":
                # llamacpp_server requires async - fall back to OpenRouter for sync
                logger.warning(
                    "[MEMORY] llamacpp_server requires async. "
                    "Use async methods or switch to openrouter backend."
                )
                self._embedder = EmbeddingClient(model=self.config.embedding_model)
            else:
                # OpenRouter embedder
                self._embedder = EmbeddingClient(model=self.config.embedding_model)
                logger.info(f"[MEMORY] Using OpenRouter embeddings ({self.config.embedding_dimension}-dim)")

        # Initialize fact extractor
        if self._fact_extractor is None:
            self._fact_extractor = FactExtractor()

        self._initialized = True

    async def _ensure_initialized_async(self) -> None:
        """Ensure all components are initialized (async version for HTTP server)."""
        if self._async_initialized:
            return

        from cems.embedding import AsyncEmbeddingClient, EmbeddingClient
        from cems.fact_extraction import FactExtractor
        from cems.llamacpp_server import AsyncLlamaCppEmbeddingClient
        from cems.vectorstore import PgVectorStore

        # Initialize vectorstore with configured dimension
        if self._vectorstore is None:
            self._vectorstore = PgVectorStore(
                database_url=self.config.database_url,
                embedding_dim=self.config.embedding_dimension,
            )
            await self._vectorstore.connect()

        # Initialize embedders based on config
        if self.config.embedding_backend == "llamacpp_server":
            # llama.cpp server embeddings (768-dim via HTTP)
            if self._embedder is None:
                # Sync embedder falls back to OpenRouter for fact extraction
                self._embedder = EmbeddingClient(model=self.config.embedding_model)

            if self._async_embedder is None:
                self._async_embedder = AsyncLlamaCppEmbeddingClient(self.config)

            logger.info(
                f"[MEMORY] Using llama.cpp server embeddings "
                f"({self.config.embedding_dimension}-dim at {self.config.llamacpp_base_url})"
            )
        else:
            # OpenRouter embedders
            if self._embedder is None:
                self._embedder = EmbeddingClient(model=self.config.embedding_model)

            if self._async_embedder is None:
                self._async_embedder = AsyncEmbeddingClient(model=self.config.embedding_model)

            logger.info(f"[MEMORY] Using OpenRouter embeddings ({self.config.embedding_dimension}-dim)")

        # Initialize fact extractor
        if self._fact_extractor is None:
            self._fact_extractor = FactExtractor()

        self._initialized = True
        self._async_initialized = True

    def _infer_category_from_query(self, query: str) -> str | None:
        """Infer the likely category from a search query.

        Uses keyword matching to determine if a query is about a specific domain.
        Returns None if no category can be inferred.

        Args:
            query: Search query text

        Returns:
            Inferred category name or None
        """
        query_lower = query.lower()

        # Category keyword mappings (keywords -> category)
        category_keywords = {
            "memory": ["memory", "recall", "remember", "retrieval", "search", "embedding"],
            "deployment": ["deploy", "coolify", "server", "production", "hosting", "docker"],
            "development": ["code", "coding", "programming", "debug", "git", "refactor"],
            "ai": ["llm", "claude", "openai", "gpt", "ai", "model", "prompt"],
            "project": ["project", "repo", "repository", "codebase"],
            "preferences": ["prefer", "preference", "like", "favorite", "style"],
            "workflow": ["workflow", "process", "habit", "routine", "automate"],
        }

        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return category

        return None

    # add() and add_async() are provided by WriteMixin
    # search(), search_async(), _search_raw(), _search_raw_async() are provided by SearchMixin
    # get(), get_all(), update(), update_async(), delete(), delete_async(), forget(), history() are provided by CRUDMixin
    # get_stale_memories(), get_hot_memories(), get_recent_memories(), get_old_memories() are provided by AnalyticsMixin
    # promote_memory(), archive_memory() are provided by AnalyticsMixin
    # get_metadata(), get_category_counts(), get_category_counts_async(), get_all_categories() are provided by MetadataMixin
    # get_recently_accessed(), get_category_summary(), get_all_category_summaries() are provided by MetadataMixin
    # metadata_store and vectorstore properties are provided by MetadataMixin
    # graph_store, get_related_memories(), get_related_memories_async(), get_memories_by_entity(), get_graph_stats() are provided by RelationsMixin
    # retrieve_for_inference(), retrieve_for_inference_async() are provided by RetrievalMixin
