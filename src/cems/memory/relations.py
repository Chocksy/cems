"""Relations operations for CEMSMemory (graph-like queries via DocumentStore)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
            "Use the async version if available."
        )
    else:
        return asyncio.run(coro)


class RelationsMixin:
    """Mixin class providing relations/graph operations for CEMSMemory."""

    @property
    def graph_store(self: "CEMSMemory"):
        """Access the graph store directly for graph queries.

        Returns:
            None (Kuzu replaced by PostgreSQL relations)
        """
        return None

    def get_related_memories(
        self: "CEMSMemory",
        memory_id: str,
        max_depth: int = 2,
        limit: int = 10,
    ) -> list[dict]:
        """Find memories related to a given memory via relations.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum path length (not used with PostgreSQL relations)
            limit: Maximum results

        Returns:
            List of related memories
        """
        return _run_async(self.get_related_memories_async(memory_id, limit=limit))

    async def get_related_memories_async(
        self: "CEMSMemory",
        memory_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Async version of get_related_memories().

        Queries memory_relations joined with memory_documents.
        """
        doc_store = await self._ensure_document_store()
        results = await doc_store.get_related_documents(memory_id, limit=limit)

        # Convert to legacy format expected by retrieval pipeline
        return [
            {
                "id": doc["id"],
                "content": doc.get("content", ""),
                "relation_type": doc.get("relation_type"),
                "relation_similarity": doc.get("relation_similarity"),
            }
            for doc in results
        ]

    def get_memories_by_entity(
        self: "CEMSMemory",
        entity_name: str,
        entity_type: str = "tool",
        limit: int = 20,
    ) -> list[dict]:
        """Find memories that mention a specific entity via full-text search.

        Args:
            entity_name: Entity name (e.g., "Python", "Docker")
            entity_type: Entity type (ignored)
            limit: Maximum results

        Returns:
            List of memories mentioning the entity
        """
        return _run_async(
            self._get_memories_by_entity_async(entity_name, limit=limit)
        )

    async def _get_memories_by_entity_async(
        self: "CEMSMemory",
        entity_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """Async implementation of get_memories_by_entity using DocumentStore."""
        doc_store = await self._ensure_document_store()
        results = await doc_store.full_text_search_chunks(
            query=entity_name,
            user_id=self.config.user_id,
            limit=limit,
        )

        # Convert chunk results to memory-style dicts
        return [
            {
                "id": r["document_id"],
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]

    def get_graph_stats(self: "CEMSMemory") -> dict[str, int]:
        """Get statistics about the memory relations.

        Returns:
            Dict with counts (empty since Kuzu is removed)
        """
        return {"nodes": 0, "edges": 0}
