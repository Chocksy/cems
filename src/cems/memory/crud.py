"""CRUD operations for CEMSMemory (get, get_all, update, delete, forget, history)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

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
            "Use the async version (e.g., delete_async instead of delete)."
        )
    else:
        return asyncio.run(coro)


class CRUDMixin:
    """Mixin class providing CRUD operations for CEMSMemory."""

    def get(self: "CEMSMemory", memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Memory dict or None if not found
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        result = _run_async(self._vectorstore.get(memory_id))
        if result:
            _run_async(self._vectorstore.record_access(memory_id))
            # Return in Mem0-compatible format
            return {
                "id": result["id"],
                "memory": result["content"],
                "metadata": {
                    "category": result.get("category"),
                    "source": result.get("source"),
                    "tags": result.get("tags", []),
                },
            }
        return None

    def get_all(
        self: "CEMSMemory",
        scope: Literal["personal", "shared", "both"] = "both",
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get all memories in a scope.

        Args:
            scope: Which namespace to get
            include_archived: Whether to include archived memories

        Returns:
            List of memory dicts
        """
        self._ensure_initialized()
        assert self._vectorstore is not None
        assert self._embedder is not None

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Use a dummy embedding to get all (or use a dedicated method)
        # For now, we'll do a broad search with high limit
        dummy_embedding = self._embedder.embed("memory")

        results = _run_async(
            self._vectorstore.search(
                query_embedding=dummy_embedding,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                limit=1000,
                include_archived=include_archived,
            )
        )

        # Convert to Mem0-compatible format
        memories = []
        for mem in results:
            memories.append({
                "id": mem["id"],
                "memory": mem["content"],
                "scope": mem.get("scope", "personal"),
                "metadata": {
                    "category": mem.get("category"),
                    "source": mem.get("source"),
                    "tags": mem.get("tags", []),
                },
            })

        return memories

    def update(self: "CEMSMemory", memory_id: str, content: str) -> dict[str, Any]:
        """Update a memory's content.

        Args:
            memory_id: The memory ID to update
            content: New content

        Returns:
            Update result dict
        """
        self._ensure_initialized()
        assert self._vectorstore is not None
        assert self._embedder is not None

        # Generate new embedding
        embedding = self._embedder.embed(content)

        success = _run_async(
            self._vectorstore.update(
                memory_id=memory_id,
                content=content,
                embedding=embedding,
            )
        )

        if success:
            return {"status": "updated", "id": memory_id}
        return {"status": "not_found", "id": memory_id}

    async def update_async(
        self: "CEMSMemory", memory_id: str, content: str
    ) -> dict[str, Any]:
        """Async version of update(). Use this from async contexts (HTTP server)."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None
        assert self._async_embedder is not None

        # Generate new embedding using async embedder
        embedding = await self._async_embedder.embed(content)

        await self._vectorstore.update(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
        )

        return {"success": True, "memory_id": memory_id}

    def delete(
        self: "CEMSMemory", memory_id: str, hard: bool = False
    ) -> dict[str, Any]:
        """Delete or archive a memory.

        Args:
            memory_id: The memory ID to delete
            hard: If True, permanently delete. If False, archive.

        Returns:
            Delete result dict
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        success = _run_async(self._vectorstore.delete(memory_id, hard=hard))

        if success:
            status = "deleted" if hard else "archived"
            return {"status": status, "memory_id": memory_id}
        return {"status": "not_found", "memory_id": memory_id}

    async def delete_async(
        self: "CEMSMemory", memory_id: str, hard: bool = False
    ) -> None:
        """Async version of delete(). Use this from async contexts (HTTP server)."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None

        if hard:
            await self._vectorstore.delete(memory_id)
        else:
            await self._vectorstore.update(memory_id, archived=True)

    def forget(self: "CEMSMemory", memory_id: str) -> dict[str, Any]:
        """Forget (soft delete) a memory.

        Args:
            memory_id: The memory ID to forget

        Returns:
            Result dict
        """
        return self.delete(memory_id, hard=False)

    def history(self: "CEMSMemory", memory_id: str) -> list[dict[str, Any]]:
        """Get the history of a memory.

        Note: pgvector doesn't track history by default.
        This returns an empty list for compatibility.

        Args:
            memory_id: The memory ID

        Returns:
            List of history entries (empty for pgvector)
        """
        # History tracking not implemented in pgvector
        # Would need a separate audit table
        return []
