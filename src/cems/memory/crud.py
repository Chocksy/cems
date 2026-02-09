"""CRUD operations for CEMSMemory (get, get_all, update, delete, forget, history).

Uses DocumentStore (memory_documents/memory_chunks) exclusively.
"""

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


def _doc_to_mem0_format(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert a DocumentStore document dict to Mem0-compatible format."""
    return {
        "id": doc["id"],
        "memory": doc["content"],
        "scope": doc.get("scope", "personal"),
        "metadata": {
            "category": doc.get("category"),
            "source": doc.get("source"),
            "source_ref": doc.get("source_ref"),
            "tags": doc.get("tags", []),
        },
    }


class CRUDMixin:
    """Mixin class providing CRUD operations for CEMSMemory.

    All operations use DocumentStore (memory_documents/memory_chunks tables).
    """

    def get(self: "CEMSMemory", memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Memory dict or None if not found
        """
        return _run_async(self._get_async(memory_id))

    async def _get_async(self: "CEMSMemory", memory_id: str) -> dict[str, Any] | None:
        """Async get from DocumentStore."""
        doc_store = await self._ensure_document_store()
        result = await doc_store.get_document(memory_id)
        if result:
            return _doc_to_mem0_format(result)
        return None

    def get_all(
        self: "CEMSMemory",
        scope: Literal["personal", "shared", "both"] = "both",
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get all memories in a scope.

        Args:
            scope: Which namespace to get
            include_archived: Ignored (document model has no archive concept)

        Returns:
            List of memory dicts
        """
        return _run_async(self._get_all_async(scope))

    async def _get_all_async(
        self: "CEMSMemory",
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> list[dict[str, Any]]:
        """Async get_all from DocumentStore."""
        doc_store = await self._ensure_document_store()
        user_id = self.config.user_id

        scope_filter = scope if scope != "both" else None

        docs = await doc_store.get_all_documents(
            user_id=user_id,
            scope=scope_filter,
            limit=1000,
        )

        return [_doc_to_mem0_format(doc) for doc in docs]

    def update(self: "CEMSMemory", memory_id: str, content: str) -> dict[str, Any]:
        """Update a memory's content.

        Args:
            memory_id: The memory ID to update
            content: New content

        Returns:
            Update result dict
        """
        return _run_async(self.update_async(memory_id, content))

    async def update_async(
        self: "CEMSMemory", memory_id: str, content: str
    ) -> dict[str, Any]:
        """Async update via DocumentStore.

        Re-chunks content, re-embeds, replaces document+chunks in a transaction.
        """
        await self._ensure_initialized_async()
        assert self._async_embedder is not None

        doc_store = await self._ensure_document_store()

        from cems.chunking import chunk_document

        chunks = chunk_document(content)
        if not chunks:
            return {"success": False, "memory_id": memory_id, "error": "Chunking produced no output"}

        chunk_texts = [c.content for c in chunks]
        embeddings = await self._async_embedder.embed_batch(chunk_texts)

        success = await doc_store.update_document(
            document_id=memory_id,
            content=content,
            chunks=chunks,
            embeddings=embeddings,
        )

        if success:
            return {"success": True, "memory_id": memory_id}
        return {"success": False, "memory_id": memory_id, "error": "Document not found"}

    def delete(
        self: "CEMSMemory", memory_id: str, hard: bool = False
    ) -> dict[str, Any]:
        """Delete a memory.

        Args:
            memory_id: The memory ID to delete
            hard: If True, permanently removes. If False, soft-deletes (sets deleted_at).

        Returns:
            Delete result dict
        """
        return _run_async(self._delete_async_internal(memory_id, hard=hard))

    async def _delete_async_internal(
        self: "CEMSMemory", memory_id: str, hard: bool = False
    ) -> dict[str, Any]:
        """Internal async delete with result dict."""
        doc_store = await self._ensure_document_store()
        success = await doc_store.delete_document(memory_id, hard=hard)

        action = "deleted" if hard else "archived"
        if success:
            return {"status": action, "memory_id": memory_id}
        return {"status": "not_found", "memory_id": memory_id}

    async def delete_async(
        self: "CEMSMemory", memory_id: str, hard: bool = False
    ) -> None:
        """Async delete via DocumentStore.

        Args:
            memory_id: The memory ID to delete
            hard: If True, permanently removes. If False, soft-deletes.
        """
        doc_store = await self._ensure_document_store()
        await doc_store.delete_document(memory_id, hard=hard)

    def forget(self: "CEMSMemory", memory_id: str) -> dict[str, Any]:
        """Forget (delete) a memory.

        Args:
            memory_id: The memory ID to forget

        Returns:
            Result dict
        """
        return self.delete(memory_id, hard=True)

    def history(self: "CEMSMemory", memory_id: str) -> list[dict[str, Any]]:
        """Get the history of a memory.

        Note: History tracking not implemented.
        Returns an empty list for compatibility.

        Args:
            memory_id: The memory ID

        Returns:
            List of history entries (always empty)
        """
        return []
