"""Analytics operations for CEMSMemory (stale, hot, recent, old memories)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

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


class AnalyticsMixin:
    """Mixin class providing analytics operations for CEMSMemory."""

    def get_stale_memories(self: "CEMSMemory", days: int | None = None) -> list[str]:
        """Get memories that haven't been accessed in N days.

        Args:
            days: Days threshold. Uses config default if not specified.

        Returns:
            List of stale memory IDs
        """
        days = days or self.config.stale_days
        if self._metadata:
            return self._metadata.get_stale_memories(self.config.user_id, days)
        return []

    def get_hot_memories(self: "CEMSMemory", threshold: int | None = None) -> list[str]:
        """Get frequently accessed memories.

        Args:
            threshold: Access count threshold. Uses config default if not specified.

        Returns:
            List of hot memory IDs
        """
        if self._metadata:
            threshold = threshold or self.config.hot_access_threshold
            return self._metadata.get_hot_memories(self.config.user_id, threshold)
        return []

    def get_recent_memories(self: "CEMSMemory", hours: int = 24) -> list[str]:
        """Get memories created in the last N hours.

        Args:
            hours: Hours to look back

        Returns:
            List of memory IDs
        """
        if self._metadata:
            return self._metadata.get_recent_memories(self.config.user_id, hours)
        return []

    def get_old_memories(self: "CEMSMemory", days: int = 30) -> list[str]:
        """Get memories older than N days.

        Args:
            days: Days threshold

        Returns:
            List of memory IDs
        """
        if self._metadata:
            return self._metadata.get_old_memories(self.config.user_id, days)
        return []

    def promote_memory(self: "CEMSMemory", memory_id: str, boost: float = 0.1) -> None:
        """Increase a memory's priority via metadata store.

        Args:
            memory_id: The memory ID
            boost: Priority boost amount
        """
        if not self._metadata:
            return

        # Get current metadata and boost priority
        batch = self._metadata.get_metadata_batch([memory_id])
        meta = batch.get(memory_id)
        if meta:
            meta.priority = min(meta.priority + boost, 2.0)
            self._metadata.save_metadata(meta)

    def archive_memory(self: "CEMSMemory", memory_id: str) -> None:
        """Archive a memory by deleting its document.

        In the document model, archiving is equivalent to deletion since
        memory_documents has no archived column.

        Args:
            memory_id: The memory ID
        """
        self.delete(memory_id, hard=True)
