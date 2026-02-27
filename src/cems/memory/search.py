"""Search operations for CEMSMemory.

Chunk-based search: Searches memory_chunks table, returns best chunk per document.
This avoids truncation issues and provides better recall with snippet-level matching.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from cems.lib.async_utils import run_async as _run_async
from cems.models import MemoryMetadata, MemoryScope, SearchResult

if TYPE_CHECKING:
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)


def _make_search_result_from_chunk(chunk: dict, user_id: str) -> SearchResult:
    """Create SearchResult from a chunk search result dict.

    Args:
        chunk: Dict from DocumentStore.search_chunks with chunk + doc metadata
        user_id: Fallback user ID

    Returns:
        SearchResult with chunk content and document metadata
    """
    memory_scope = MemoryScope(chunk.get("scope", "personal"))
    memory_id = chunk.get("document_id", chunk.get("chunk_id", ""))
    created_at = chunk.get("created_at", datetime.now(UTC))

    metadata = MemoryMetadata(
        memory_id=memory_id,
        user_id=chunk.get("user_id", user_id),
        scope=memory_scope,
        category=chunk.get("category", "document"),
        source=chunk.get("source"),
        source_ref=chunk.get("source_ref"),
        tags=chunk.get("tags", []),
        priority=1.0,
        pinned=False,
        pin_reason=None,
        archived=False,
        access_count=0,
        created_at=created_at,
        updated_at=created_at,
        last_accessed=created_at,
        expires_at=None,
    )

    return SearchResult(
        memory_id=memory_id,
        content=chunk.get("content", chunk.get("chunk_content", "")),
        score=chunk.get("score", 0.0),
        scope=memory_scope,
        metadata=metadata,
    )


def _dedupe_by_document(results: list[SearchResult]) -> list[SearchResult]:
    """Keep only the best-scoring chunk per document."""
    seen_docs: dict[str, SearchResult] = {}
    for result in results:
        doc_id = result.memory_id
        if doc_id not in seen_docs or result.score > seen_docs[doc_id].score:
            seen_docs[doc_id] = result
    return list(seen_docs.values())


def _apply_score_adjustments(results: list[SearchResult]) -> list[SearchResult]:
    """Apply priority boost and time decay to scores in-place.

    Mutates scores on each result and returns the same list for chaining.
    """
    now = datetime.now(UTC)
    for result in results:
        if result.metadata:
            result.score *= result.metadata.priority
            days_since_access = (now - result.metadata.last_accessed).days
            time_decay = 1.0 / (1.0 + (days_since_access / 60))
            result.score *= time_decay
            if result.metadata.pinned:
                result.score *= 1.1
    return results


class SearchMixin:
    """Mixin class providing search operations for CEMSMemory.

    Uses document+chunk model for search:
    - Searches memory_chunks table for semantic matching
    - Returns best chunk per document (deduplication)
    - Chunk content is returned as the searchable snippet
    """

    def search(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories across namespaces (sync version)."""
        return _run_async(self.search_async(query, scope, category, limit))

    async def search_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Async search using document+chunk model."""
        await self._ensure_initialized_async()
        assert self._async_embedder is not None

        doc_store = await self._ensure_document_store()
        query_embedding = await self._async_embedder.embed(query)
        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        raw_results = await doc_store.hybrid_search_chunks(
            query=query,
            query_embedding=query_embedding,
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            category=category,
            limit=limit * 3,
            vector_weight=getattr(self.config, "hybrid_vector_weight", 0.7),
        )

        results = [_make_search_result_from_chunk(chunk, user_id) for chunk in raw_results]
        results = _dedupe_by_document(results)

        _apply_score_adjustments(results)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def _search_raw(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Raw search without score adjustments (sync)."""
        return _run_async(self._search_raw_async(query, scope, category, limit))

    def _search_lexical_raw(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        limit: int = 5,
    ) -> list[SearchResult]:
        """BM25 (full-text) search without score adjustments (sync)."""
        return _run_async(self._search_lexical_raw_async(query, scope, limit))

    async def _search_raw_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Async hybrid (vector+BM25) search on chunks without score adjustments.

        Used by retrieve_for_inference pipeline.
        """
        await self._ensure_initialized_async()
        assert self._async_embedder is not None

        doc_store = await self._ensure_document_store()

        if query_embedding is None:
            query_embedding = await self._async_embedder.embed(query)

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        raw_results = await doc_store.hybrid_search_chunks(
            query=query,
            query_embedding=query_embedding,
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            category=category,
            limit=limit * 2,
            vector_weight=getattr(self.config, "hybrid_vector_weight", 0.4),
        )

        results = [_make_search_result_from_chunk(chunk, user_id) for chunk in raw_results]
        results = _dedupe_by_document(results)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def _search_lexical_raw_async(
        self: "CEMSMemory",
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        limit: int = 5,
    ) -> list[SearchResult]:
        """BM25 (full-text) search on chunks without score adjustments.

        Used for strong-signal detection and lexical stream in hybrid retrieval.
        """
        await self._ensure_initialized_async()

        doc_store = await self._ensure_document_store()
        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        raw_results = await doc_store.full_text_search_chunks(
            query=query,
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            limit=limit * 2,
        )

        results = [_make_search_result_from_chunk(chunk, user_id) for chunk in raw_results]
        results = _dedupe_by_document(results)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
