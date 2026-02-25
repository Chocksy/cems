"""Weekly summarization job — compress old memories, prune stale ones.

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


def _is_older_than(doc: dict, days: int) -> bool:
    """Check if a document is older than N days."""
    created_at = doc.get("created_at")
    if not created_at:
        return False
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at < datetime.now(UTC) - timedelta(days=days)


class SummarizationJob:
    """Weekly maintenance job for memory summarization.

    Compresses old memories into category summaries and prunes
    stale memories that haven't been accessed in 90+ days.
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(self) -> dict[str, int]:
        """Run the summarization job.

        Returns:
            Dict with counts of operations performed
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        # Get all docs, filter to 30+ days old
        all_docs = await doc_store.get_all_documents(user_id, limit=500)
        old_docs = [d for d in all_docs if _is_older_than(d, days=30)]
        logger.info(f"Found {len(old_docs)} old documents to potentially summarize")

        categories_updated = await self._compress_by_category(old_docs)
        pruned = await self._prune_stale(doc_store, user_id, all_docs)

        result = {
            "categories_updated": categories_updated,
            "memories_pruned": pruned,
            "old_memories_checked": len(old_docs),
        }
        logger.info(f"Summarization completed: {result}")
        return result

    async def _compress_by_category(self, old_docs: list[dict]) -> int:
        """Compress old memories into category summaries.

        Groups documents by category and creates summary documents
        for categories with 3+ old memories.

        Args:
            old_docs: List of old document dicts

        Returns:
            Number of categories updated
        """
        if not old_docs:
            return 0

        # Group by category
        categories: dict[str, list[dict]] = {}
        for doc in old_docs:
            cat = doc.get("category", "general")
            categories.setdefault(cat, []).append(doc)

        updated = 0
        for category, docs in categories.items():
            if len(docs) >= 3:
                try:
                    await self._create_category_summary(category, docs)
                    updated += 1
                except Exception as e:
                    logger.warning(f"Failed to summarize category {category}: {e}")

        return updated

    async def _create_category_summary(
        self, category: str, docs: list[dict]
    ) -> None:
        """Create an LLM-generated summary for a category.

        Args:
            category: The category name
            docs: Documents in this category
        """
        from cems.llm import summarize_memories

        contents = [d.get("content", "") for d in docs if d.get("content")]
        if not contents:
            logger.debug(f"No content found for category {category}")
            return

        logger.info(
            f"Generating LLM summary for category '{category}' with {len(contents)} memories"
        )
        summary_text = summarize_memories(
            memories=contents,
            category=category,
            model=self.config.llm_model,
        )

        # Store as a new document with category-summary tag
        await self.memory.add_async(
            content=summary_text,
            scope="personal",
            category="category-summary",
            tags=["category-summary", f"category:{category}"],
            infer=False,
            source_ref=f"summary:{category}",
        )

        logger.info(f"Created LLM summary for category {category} with {len(contents)} items")

    async def _prune_stale(
        self, doc_store, user_id: str, all_docs: list[dict]
    ) -> int:
        """Soft-delete documents not updated in stale_days.

        Args:
            doc_store: DocumentStore instance
            user_id: User ID
            all_docs: All user documents

        Returns:
            Number of documents pruned
        """
        stale_days = self.config.stale_days
        stale_docs = [d for d in all_docs if self._is_stale(d, stale_days)]
        pruned = 0

        for doc in stale_docs:
            doc_id = doc.get("id")
            if doc_id:
                try:
                    await doc_store.delete_document(doc_id, hard=False)
                    pruned += 1
                except Exception as e:
                    logger.error(f"Failed to prune stale document {doc_id}: {e}")

        if pruned:
            logger.info(f"Soft-deleted {pruned} stale documents (>{stale_days} days)")

        return pruned

    @staticmethod
    def _is_stale(doc: dict, days: int) -> bool:
        """Check if a document is stale (not updated in N days)."""
        updated_at = doc.get("updated_at") or doc.get("created_at")
        if not updated_at:
            return False
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        return updated_at < datetime.now(UTC) - timedelta(days=days)

    def run(self) -> dict[str, int]:
        """Synchronous wrapper for run_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()
