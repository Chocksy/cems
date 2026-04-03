"""Relations operations for CEMSMemory (graph-like queries via DocumentStore)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cems.lib.async_utils import run_async as _run_async

if TYPE_CHECKING:
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)


class RelationsMixin:
    """Mixin class providing relations/graph operations for CEMSMemory."""

    def get_related_memories(
        self: "CEMSMemory",
        memory_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """Find memories related to a given memory via relations.

        Args:
            memory_id: Starting memory ID
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

