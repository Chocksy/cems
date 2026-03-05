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

# Skip docs updated within this many days (embeddings are still fresh)
SKIP_FRESH_DAYS = 7


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

        # Oldest first so we re-embed the stalest docs first
        all_docs = await doc_store.get_all_documents(user_id, limit=5000, order="asc")
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

        Skips docs updated within SKIP_FRESH_DAYS (embeddings are still current).

        Args:
            docs: List of document dicts

        Returns:
            Number of documents re-indexed
        """
        refreshed = 0
        failed = 0
        skipped = 0
        log_interval = 50
        fresh_cutoff = datetime.now(UTC) - timedelta(days=SKIP_FRESH_DAYS)

        # Filter to stale docs only
        stale_docs = []
        for doc in docs:
            updated_at = doc.get("updated_at") or doc.get("created_at")
            if updated_at:
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at)
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=UTC)
                if updated_at >= fresh_cutoff:
                    skipped += 1
                    continue
            stale_docs.append(doc)

        total = len(stale_docs)
        logger.info(
            f"Re-indexing {total} stale docs (skipped {skipped} fresh, "
            f"<{SKIP_FRESH_DAYS} days old)"
        )

        for doc in stale_docs:
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

                if refreshed > 0 and refreshed % log_interval == 0:
                    logger.info(f"Re-indexing progress: {refreshed}/{total} documents")
            except Exception as e:
                logger.warning(f"Failed to re-index document {doc_id}: {repr(e)}")
                failed += 1

        logger.info(
            f"Re-indexing complete: {refreshed} succeeded, {failed} failed "
            f"out of {total} stale ({skipped} skipped fresh)"
        )
        return refreshed

    # Categories that should never be archived (long-lived config docs)
    PROTECTED_CATEGORIES = {
        "gate-rules", "guidelines", "preferences",
        "category-summary", "session-summary",
    }

    async def _archive_dead(self, doc_store, docs: list[dict]) -> int:
        """Soft-delete documents not updated in archive_days.

        Skips protected categories (gate-rules, guidelines, etc.) which are
        long-lived config docs that should never be auto-archived.

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
            # Never archive protected categories (long-lived config docs)
            if doc.get("category", "general") in self.PROTECTED_CATEGORIES:
                continue

            updated_at = doc.get("updated_at") or doc.get("created_at")
            if not updated_at:
                continue
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)

            if updated_at < cutoff:
                # Skip if recently shown — actively-surfaced memories shouldn't be archived
                last_shown = doc.get("last_shown_at")
                if last_shown:
                    if isinstance(last_shown, str):
                        last_shown = datetime.fromisoformat(last_shown)
                    if last_shown.tzinfo is None:
                        last_shown = last_shown.replace(tzinfo=UTC)
                    if last_shown > cutoff:
                        continue

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
