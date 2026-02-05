"""Search operations for CEMSMemory.

Chunk-based search: Searches memory_chunks table, returns best chunk per document.
This avoids truncation issues and provides better recall with snippet-level matching.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from cems.models import MemoryMetadata, MemoryScope, SearchResult

if TYPE_CHECKING:
    from cems.db.document_store import DocumentStore
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine in a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        raise RuntimeError(
            "Cannot use sync method from async context. "
            "Use the async version (e.g., search_async instead of search)."
        )
    else:
        return asyncio.run(coro)


def _make_search_result_from_chunk(chunk: dict, user_id: str) -> SearchResult:
    """Create SearchResult from a chunk search result dict.

    Args:
        chunk: Dict from DocumentStore.search_chunks with chunk + doc metadata
        user_id: Fallback user ID

    Returns:
        SearchResult with chunk content and document metadata
    """
    memory_scope = MemoryScope(chunk.get("scope", "personal"))

    # Use document_id as the memory_id for compatibility
    # chunk_id is available in metadata if needed
    memory_id = chunk.get("document_id", chunk.get("chunk_id", ""))

    metadata = MemoryMetadata(
        memory_id=memory_id,
        user_id=chunk.get("user_id", user_id),
        scope=memory_scope,
        category=chunk.get("category", "document"),
        source=chunk.get("source"),
        source_ref=chunk.get("source_ref"),
        tags=chunk.get("tags", []),
        priority=1.0,  # Documents don't have priority yet
        pinned=False,  # Documents don't have pinned yet
        pin_reason=None,
        archived=False,
        access_count=0,
        created_at=chunk.get("created_at", datetime.now(UTC)),
        updated_at=chunk.get("created_at", datetime.now(UTC)),
        last_accessed=chunk.get("created_at", datetime.now(UTC)),
        expires_at=None,
    )

    return SearchResult(
        memory_id=memory_id,
        content=chunk.get("content", chunk.get("chunk_content", "")),
        score=chunk.get("score", 0.0),
        scope=memory_scope,
        metadata=metadata,
    )


def _make_search_result_from_memory(mem: dict, user_id: str) -> SearchResult:
    """Create SearchResult from a raw memory dict (legacy memories table)."""
    memory_scope = MemoryScope(mem.get("scope", "personal"))
    metadata = MemoryMetadata(
        memory_id=mem["id"],
        user_id=mem.get("user_id", user_id),
        scope=memory_scope,
        category=mem.get("category", "general"),
        source=mem.get("source"),
        source_ref=mem.get("source_ref"),
        tags=mem.get("tags", []),
        priority=mem.get("priority", 1.0),
        pinned=mem.get("pinned", False),
        pin_reason=mem.get("pin_reason"),
        archived=mem.get("archived", False),
        access_count=mem.get("access_count", 0),
        created_at=mem.get("created_at", datetime.now(UTC)),
        updated_at=mem.get("updated_at", datetime.now(UTC)),
        last_accessed=mem.get("last_accessed", datetime.now(UTC)),
        expires_at=mem.get("expires_at"),
    )
    return SearchResult(
        memory_id=mem["id"],
        content=mem.get("content", ""),
        score=mem.get("score", 0.0),
        scope=memory_scope,
        metadata=metadata,
    )


def _dedupe_by_document(results: list[SearchResult]) -> list[SearchResult]:
    """Keep only the best-scoring chunk per document.

    Args:
        results: List of SearchResults (may have multiple chunks from same doc)

    Returns:
        Deduplicated list with best chunk per document
    """
    seen_docs: dict[str, SearchResult] = {}
    for result in results:
        doc_id = result.memory_id
        if doc_id not in seen_docs or result.score > seen_docs[doc_id].score:
            seen_docs[doc_id] = result
    return list(seen_docs.values())


def _apply_score_adjustments(results: list[SearchResult], inferred_category: str | None) -> None:
    """Apply priority boost and time decay to scores in-place."""
    now = datetime.now(UTC)
    for result in results:
        if result.metadata:
            # Priority boost
            result.score *= result.metadata.priority
            # Time decay
            days_since_access = (now - result.metadata.last_accessed).days
            time_decay = 1.0 / (1.0 + (days_since_access / 60))
            result.score *= time_decay
            # Pinned boost
            if result.metadata.pinned:
                result.score *= 1.1


class SearchMixin:
    """Mixin class providing search operations for CEMSMemory.

    Uses document+chunk model for search:
    - Searches memory_chunks table for semantic matching
    - Returns best chunk per document (deduplication)
    - Chunk content is returned as the searchable snippet
    """

    # Document store instance (lazy initialized)
    _document_store: "DocumentStore | None" = None

    async def _ensure_document_store_search(self: "CEMSMemory") -> "DocumentStore":
        """Ensure document store is initialized for search."""
        if self._document_store is None:
            from cems.db.document_store import DocumentStore

            self._document_store = DocumentStore(
                database_url=self.config.database_url,
                embedding_dim=self.config.embedding_dimension,
            )
            await self._document_store.connect()
        return self._document_store

    def search(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories across namespaces (sync version)."""
        return _run_async(
            self.search_async(query, scope, category, limit)
        )

    async def search_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Async search using document+chunk model, with fallback to memories table."""
        await self._ensure_initialized_async()
        assert self._async_embedder is not None

        query_embedding = await self._async_embedder.embed(query)
        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Try document+chunk model first, fall back to memories table if tables don't exist
        try:
            doc_store = await self._ensure_document_store_search()
            # Search chunks with hybrid (vector + full-text)
            raw_results = await doc_store.hybrid_search_chunks(
                query=query,
                query_embedding=query_embedding,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                category=category,
                limit=limit * 3,  # Fetch more for deduplication
                vector_weight=getattr(self.config, "hybrid_vector_weight", 0.7),
            )

            # Convert to SearchResult objects
            results = [_make_search_result_from_chunk(chunk, user_id) for chunk in raw_results]

            # Dedupe by document (keep best chunk per doc)
            results = _dedupe_by_document(results)

        except Exception as e:
            # Fallback to legacy memories table search if document/chunks tables don't exist
            if "memory_chunks" in str(e) or "memory_documents" in str(e) or "UndefinedTable" in str(e):
                logger.info("Document store not available, falling back to memories table search")
                raw_results = await self.vectorstore.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    user_id=user_id,
                    team_id=team_id,
                    scope=scope,
                    category=category,
                    limit=limit,
                    vector_weight=getattr(self.config, "hybrid_vector_weight", 0.7),
                )
                results = [_make_search_result_from_memory(mem, user_id) for mem in raw_results]
            else:
                raise

        # Apply score adjustments
        inferred_category = self._infer_category_from_query(query)
        _apply_score_adjustments(results, inferred_category)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def _search_raw(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Raw search without score adjustments - for use in retrieve_for_inference (sync)."""
        return _run_async(
            self._search_raw_async(query, scope, category, limit)
        )

    def _search_lexical_raw(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        limit: int = 5,
    ) -> list[SearchResult]:
        """BM25 (full-text) search without score adjustments (sync)."""
        return _run_async(
            self._search_lexical_raw_async(query, scope, limit)
        )

    async def _search_raw_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Async hybrid (vector+BM25) search on chunks without score adjustments.

        Used by retrieve_for_inference pipeline. Uses hybrid search to combine
        vector similarity with BM25 lexical matching for better entity recall.
        Falls back to legacy memories table if document/chunk tables don't exist.
        """
        await self._ensure_initialized_async()
        assert self._async_embedder is not None

        if query_embedding is None:
            query_embedding = await self._async_embedder.embed(query)

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Try document+chunk model first, fall back to memories table
        try:
            doc_store = await self._ensure_document_store_search()
            # Hybrid search (vector + BM25) on chunks - better for entity matching
            raw_results = await doc_store.hybrid_search_chunks(
                query=query,
                query_embedding=query_embedding,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                category=category,
                limit=limit * 2,  # Fetch extra for deduplication
                vector_weight=getattr(self.config, "hybrid_vector_weight", 0.4),
            )

            # Convert to SearchResult objects
            results = [_make_search_result_from_chunk(chunk, user_id) for chunk in raw_results]

            # Dedupe by document
            results = _dedupe_by_document(results)

        except Exception as e:
            # Fallback to legacy memories table if document/chunks tables don't exist
            if "memory_chunks" in str(e) or "memory_documents" in str(e) or "UndefinedTable" in str(e):
                logger.info("Document store not available, falling back to memories table for _search_raw")
                raw_results = await self.vectorstore.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    user_id=user_id,
                    team_id=team_id,
                    scope=scope,
                    category=category,
                    limit=limit,
                    vector_weight=getattr(self.config, "hybrid_vector_weight", 0.4),
                )
                results = [_make_search_result_from_memory(mem, user_id) for mem in raw_results]
            else:
                raise

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def _search_lexical_raw_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        limit: int = 5,
    ) -> list[SearchResult]:
        """BM25 (full-text) search on chunks without score adjustments.

        Used for:
        1. Strong-signal detection before query expansion
        2. Lexical stream in hybrid retrieval

        Falls back to hybrid search with low vector weight if document tables don't exist.
        """
        await self._ensure_initialized_async()

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Try document+chunk model first, fall back to hybrid search
        try:
            doc_store = await self._ensure_document_store_search()
            # Full-text search on chunks
            raw_results = await doc_store.full_text_search_chunks(
                query=query,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                limit=limit * 2,  # Fetch extra for deduplication
            )

            # Convert to SearchResult objects
            results = [_make_search_result_from_chunk(chunk, user_id) for chunk in raw_results]

            # Dedupe by document
            results = _dedupe_by_document(results)

        except Exception as e:
            # Fallback: use hybrid search with low vector weight to approximate lexical
            if "memory_chunks" in str(e) or "memory_documents" in str(e) or "UndefinedTable" in str(e):
                logger.info("Document store not available, falling back to hybrid search for lexical")
                assert self._async_embedder is not None
                query_embedding = await self._async_embedder.embed(query)
                raw_results = await self.vectorstore.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    user_id=user_id,
                    team_id=team_id,
                    scope=scope,
                    limit=limit,
                    vector_weight=0.1,  # Low vector weight = more lexical
                )
                results = [_make_search_result_from_memory(mem, user_id) for mem in raw_results]
            else:
                raise

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    # Legacy methods for backwards compatibility with old memories table
    async def _search_raw_legacy_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Legacy search using old memories table (for migration)."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None
        assert self._async_embedder is not None

        if query_embedding is None:
            query_embedding = await self._async_embedder.embed(query)

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        raw_results = await self._vectorstore.search(
            query_embedding=query_embedding,
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            category=category,
            limit=limit,
        )

        results = [_make_search_result_from_memory(mem, user_id) for mem in raw_results]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
