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

import logging
from typing import TYPE_CHECKING

from cems.config import CEMSConfig
from cems.memory.crud import CRUDMixin
from cems.memory.metadata import MetadataMixin
from cems.memory.relations import RelationsMixin
from cems.memory.retrieval import RetrievalMixin
from cems.memory.search import SearchMixin
from cems.memory.write import WriteMixin

if TYPE_CHECKING:
    from cems.embedding import AsyncEmbeddingClient, EmbeddingClient

logger = logging.getLogger(__name__)


class CEMSMemory(WriteMixin, SearchMixin, CRUDMixin, MetadataMixin, RelationsMixin, RetrievalMixin):
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
        self._embedder: EmbeddingClient | None = None
        self._async_embedder: AsyncEmbeddingClient | None = None
        self._initialized = False
        self._async_initialized = False  # Track async initialization separately

        # Initialize database connection
        from cems.db.database import init_database, is_database_initialized

        if not is_database_initialized():
            init_database(self.config.database_url)


    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized (sync version for CLI/MCP)."""
        if self._initialized:
            return

        from cems.embedding import EmbeddingClient

        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.config.embedding_model)
            logger.info(
                f"[MEMORY] Embeddings via {self._embedder.embeddings_url} "
                f"({self.config.embedding_dimension}-dim)"
            )

        self._initialized = True

    async def _ensure_initialized_async(self) -> None:
        """Ensure all components are initialized (async version for HTTP server)."""
        if self._async_initialized:
            return

        from cems.embedding import AsyncEmbeddingClient, EmbeddingClient

        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.config.embedding_model)
        if self._async_embedder is None:
            self._async_embedder = AsyncEmbeddingClient(model=self.config.embedding_model)
        logger.info(
            f"[MEMORY] Embeddings via {self._async_embedder.embeddings_url} "
            f"({self.config.embedding_dimension}-dim)"
        )

        self._initialized = True
        self._async_initialized = True

    # add() and add_async() are provided by WriteMixin
    # search(), search_async(), _search_raw(), _search_raw_async() are provided by SearchMixin
    # get(), get_all(), update(), update_async(), delete(), delete_async(), forget() are provided by CRUDMixin
    # get_metadata(), get_metadata_async(), get_category_counts_async() are provided by MetadataMixin
    # get_related_memories(), get_related_memories_async() are provided by RelationsMixin
    # retrieve_for_inference(), retrieve_for_inference_async() are provided by RetrievalMixin
