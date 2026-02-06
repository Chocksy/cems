"""Metadata operations for CEMSMemory (category counts, summaries, etc.)."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from cems.models import MemoryMetadata, MemoryScope

if TYPE_CHECKING:
    from cems.db.metadata_store import PostgresMetadataStore
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


class MetadataMixin:
    """Mixin class providing metadata operations for CEMSMemory."""

    async def get_category_counts_async(
        self: "CEMSMemory",
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> dict[str, int]:
        """Async version of get_category_counts(). Reads from DocumentStore."""
        await self._ensure_initialized_async()

        user_id = self.config.user_id
        doc_store = await self._ensure_document_store()

        return await doc_store.get_document_category_counts(
            user_id=user_id,
            scope=scope if scope != "both" else None,
        )

    def get_metadata(self: "CEMSMemory", memory_id: str) -> MemoryMetadata | None:
        """Get extended metadata for a memory from DocumentStore.

        Args:
            memory_id: The memory ID

        Returns:
            MemoryMetadata or None
        """
        return _run_async(self.get_metadata_async(memory_id))

    async def get_metadata_async(self: "CEMSMemory", memory_id: str) -> MemoryMetadata | None:
        """Async get_metadata from DocumentStore."""
        doc_store = await self._ensure_document_store()
        doc = await doc_store.get_document(memory_id)
        if not doc:
            return None

        return MemoryMetadata(
            memory_id=doc["id"],
            user_id=doc.get("user_id", self.config.user_id),
            scope=MemoryScope(doc.get("scope", "personal")),
            category=doc.get("category", "general"),
            source=doc.get("source"),
            source_ref=doc.get("source_ref"),
            tags=doc.get("tags", []),
            priority=1.0,
            pinned=False,
            pin_reason=None,
            archived=False,
            access_count=0,
            created_at=doc.get("created_at", datetime.now(UTC)),
            updated_at=doc.get("updated_at", datetime.now(UTC)),
            last_accessed=doc.get("updated_at", datetime.now(UTC)),
            expires_at=None,
        )

    def get_category_counts(
        self: "CEMSMemory", scope: Literal["personal", "shared"] | None = None
    ) -> dict[str, int]:
        """Get document counts grouped by category.

        Args:
            scope: Optional filter by scope

        Returns:
            Dict mapping category name to count
        """
        async def _get():
            doc_store = await self._ensure_document_store()
            return await doc_store.get_document_category_counts(
                self.config.user_id, scope
            )

        return _run_async(_get())

    @property
    def metadata_store(self: "CEMSMemory") -> "PostgresMetadataStore":
        """Access the metadata store directly for maintenance operations."""
        return self._metadata

    def get_all_categories(
        self: "CEMSMemory",
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> list[dict]:
        """Get all categories with their memory counts.

        Args:
            scope: Which namespace to get categories for

        Returns:
            List of dicts with category name, scope, and count
        """
        if self._metadata:
            if scope == "personal":
                return self._metadata.get_all_categories(
                    self.config.user_id, MemoryScope.PERSONAL
                )
            elif scope == "shared" and self.config.team_id:
                return self._metadata.get_all_categories(
                    self.config.user_id, MemoryScope.SHARED
                )
            else:
                return self._metadata.get_all_categories(self.config.user_id)
        return []

    def get_recently_accessed(self: "CEMSMemory", limit: int = 10) -> list[dict]:
        """Get recently accessed memories.

        Args:
            limit: Maximum number of results

        Returns:
            List of dicts with memory info and access timestamps
        """
        if self._metadata:
            return self._metadata.get_recently_accessed(self.config.user_id, limit)
        return []

    def get_category_summary(
        self: "CEMSMemory",
        category: str,
        scope: Literal["personal", "shared"] = "personal",
    ) -> dict | None:
        """Get the LLM-generated summary for a category.

        Args:
            category: Category name
            scope: "personal" or "shared"

        Returns:
            Summary dict with content, item_count, last_updated, or None
        """
        if self._metadata:
            return self._metadata.get_category_summary(
                self.config.user_id, category, scope
            )
        return None

    def get_all_category_summaries(
        self: "CEMSMemory",
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> list[dict]:
        """Get all category summaries.

        Args:
            scope: Which namespace to get summaries for

        Returns:
            List of summary dicts
        """
        if self._metadata:
            if scope == "personal":
                return self._metadata.get_all_category_summaries(
                    self.config.user_id, "personal"
                )
            elif scope == "shared":
                return self._metadata.get_all_category_summaries(
                    self.config.user_id, "shared"
                )
            else:
                return self._metadata.get_all_category_summaries(self.config.user_id)
        return []
