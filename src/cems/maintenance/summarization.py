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

# Categories that should never be summarized or pruned
PROTECTED_CATEGORIES = {
    "gate-rules", "guidelines", "preferences",
    "category-summary", "session-summary",
}


def _doc_age_exceeds(doc: dict, days: int, field: str = "created_at") -> bool:
    """Check if a document's timestamp field is older than N days.

    Args:
        doc: Document dict
        days: Age threshold in days
        field: Timestamp field to check (default "created_at")
    """
    ts = doc.get(field) or doc.get("created_at")
    if not ts:
        return False
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts < datetime.now(UTC) - timedelta(days=days)


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

        # Get oldest docs first (ASC) so old docs aren't cut off by the limit
        all_docs = await doc_store.get_all_documents(user_id, limit=2000, order="asc")

        # Filter to 30+ days old, excluding protected categories
        old_docs = [
            d for d in all_docs
            if _doc_age_exceeds(d, days=30)
            and d.get("category", "general") not in PROTECTED_CATEGORIES
        ]
        logger.info(f"Found {len(old_docs)} old documents to potentially summarize")

        categories_updated, originals_deleted = await self._compress_by_category(
            doc_store, old_docs
        )
        pruned = await self._prune_stale(doc_store, user_id, all_docs)

        result = {
            "categories_updated": categories_updated,
            "originals_deleted": originals_deleted,
            "memories_pruned": pruned,
            "old_memories_checked": len(old_docs),
        }
        logger.info(f"Summarization completed: {result}")
        return result

    async def _compress_by_category(
        self, doc_store, old_docs: list[dict]
    ) -> tuple[int, int]:
        """Compress old memories into category summaries and soft-delete originals.

        Groups documents by category and creates summary documents
        for categories with 3+ old memories.

        Args:
            doc_store: DocumentStore instance
            old_docs: List of old document dicts

        Returns:
            Tuple of (categories_updated, originals_deleted)
        """
        if not old_docs:
            return 0, 0

        # Group by category
        categories: dict[str, list[dict]] = {}
        for doc in old_docs:
            cat = doc.get("category", "general")
            categories.setdefault(cat, []).append(doc)

        updated = 0
        total_deleted = 0
        for category, docs in categories.items():
            if len(docs) >= 3:
                try:
                    await self._create_category_summary(category, docs)
                    # Soft-delete originals now that summary exists
                    deleted = 0
                    for doc in docs:
                        doc_id = doc.get("id")
                        if doc_id:
                            try:
                                await doc_store.delete_document(doc_id, hard=False)
                                deleted += 1
                            except Exception as e:
                                logger.error(
                                    f"Failed to soft-delete summarized doc {doc_id}: {e}"
                                )
                    logger.info(
                        f"Summarized category '{category}': {len(docs)} docs → 1 summary, "
                        f"{deleted} originals soft-deleted"
                    )
                    updated += 1
                    total_deleted += deleted
                except Exception as e:
                    logger.warning(f"Failed to summarize category {category}: {e}")

        return updated, total_deleted

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
        stale_docs = [
            d for d in all_docs
            if _doc_age_exceeds(d, stale_days, field="updated_at")
            and d.get("category", "general") not in PROTECTED_CATEGORIES
        ]
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

    def run(self) -> dict[str, int]:
        """Synchronous wrapper for run_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()
