"""Monthly re-indexing job — rebuild embeddings, archive dead memories.

Uses DocumentStore (memory_documents) exclusively via async pattern.
Follows the ObservationReflector pattern for async + DocumentStore access.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)


class ReindexJob:
    """Monthly maintenance job for memory re-indexing.

    Rebuilds embeddings for all memories (using latest embedding model)
    and soft-deletes dead nodes not updated in 180+ days.
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(self) -> dict[str, int]:
        """Run the re-indexing job.

        Returns:
            Dict with counts of operations performed
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        all_docs = await doc_store.get_all_documents(user_id, limit=5000)
        logger.info(f"Found {len(all_docs)} documents to check for re-indexing")

        reindexed = await self._refresh_embeddings(all_docs)
        archived = await self._archive_dead(doc_store, all_docs)

        result = {
            "memories_reindexed": reindexed,
            "memories_archived": archived,
            "total_memories": len(all_docs),
        }
        logger.info(f"Re-indexing completed: {result}")
        return result

    async def _refresh_embeddings(self, docs: list[dict]) -> int:
        """Refresh embeddings for documents.

        Triggers re-chunking and re-embedding via update_async which:
        1. Re-generates embeddings with the current embedding model
        2. Replaces chunks in the database

        Args:
            docs: List of document dicts

        Returns:
            Number of documents re-indexed
        """
        refreshed = 0
        failed = 0
        log_interval = 10
        total = len(docs)

        for doc in docs:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            if not doc_id or not content:
                continue

            try:
                result = await self.memory.update_async(doc_id, content)
                if result.get("success"):
                    refreshed += 1
                else:
                    failed += 1

                if refreshed % log_interval == 0:
                    logger.info(f"Re-indexing progress: {refreshed}/{total} documents")
            except Exception as e:
                logger.warning(f"Failed to re-index document {doc_id}: {e}")
                failed += 1

        logger.info(
            f"Re-indexing complete: {refreshed} succeeded, {failed} failed out of {total}"
        )
        return refreshed

    async def _archive_dead(self, doc_store, docs: list[dict]) -> int:
        """Soft-delete documents not updated in archive_days.

        Args:
            doc_store: DocumentStore instance
            docs: All user documents

        Returns:
            Number of documents archived
        """
        archive_days = self.config.archive_days
        cutoff = datetime.now(UTC) - timedelta(days=archive_days)
        archived = 0

        for doc in docs:
            updated_at = doc.get("updated_at") or doc.get("created_at")
            if not updated_at:
                continue
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)

            if updated_at < cutoff:
                doc_id = doc.get("id")
                if doc_id:
                    try:
                        await doc_store.delete_document(doc_id, hard=False)
                        archived += 1
                    except Exception as e:
                        logger.error(f"Failed to archive document {doc_id}: {e}")

        if archived:
            logger.info(f"Soft-deleted {archived} dead documents (>{archive_days} days unused)")

        return archived

    def run(self) -> dict[str, int]:
        """Synchronous wrapper for run_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()
