"""Metadata operations for CEMSMemory (category counts, summaries, etc.)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from cems.lib.async_utils import run_async as _run_async
from cems.models import MemoryMetadata, MemoryScope

if TYPE_CHECKING:
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)


class MetadataMixin:
    """Mixin class providing metadata operations for CEMSMemory."""

    async def get_category_counts_async(
        self: "CEMSMemory",
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> dict[str, int]:
        """Async version of get_category_counts(). Reads from DocumentStore."""
        await self._ensure_initialized_async()

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None
        doc_store = await self._ensure_document_store()

        return await doc_store.get_document_category_counts(
            user_id=user_id,
            team_id=team_id,
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
            created_at=doc.get("created_at", datetime.now(UTC)),
            updated_at=doc.get("updated_at", datetime.now(UTC)),
            last_accessed=doc.get("updated_at", datetime.now(UTC)),
        )

