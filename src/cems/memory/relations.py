"""Relations operations for CEMSMemory (graph-like queries via PostgreSQL relations)."""

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
        self._ensure_initialized()
        assert self._vectorstore is not None

        return _run_async(
            self._vectorstore.get_related_memories(memory_id, limit=limit)
        )

    async def get_related_memories_async(
        self: "CEMSMemory",
        memory_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Async version of get_related_memories()."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None

        return await self._vectorstore.get_related_memories(memory_id, limit=limit)

    def get_memories_by_entity(
        self: "CEMSMemory",
        entity_name: str,
        entity_type: str = "tool",
        limit: int = 20,
    ) -> list[dict]:
        """Find memories that mention a specific entity.

        Note: This uses full-text search instead of graph traversal.

        Args:
            entity_name: Entity name (e.g., "Python", "Docker")
            entity_type: Entity type (ignored in pgvector implementation)
            limit: Maximum results

        Returns:
            List of memories mentioning the entity
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        return _run_async(
            self._vectorstore.full_text_search(
                query=entity_name,
                user_id=self.config.user_id,
                limit=limit,
            )
        )

    def get_graph_stats(self: "CEMSMemory") -> dict[str, int]:
        """Get statistics about the memory relations.

        Returns:
            Dict with counts (empty since Kuzu is removed)
        """
        return {"nodes": 0, "edges": 0}
