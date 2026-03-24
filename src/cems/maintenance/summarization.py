"""Weekly summarization job — compress old memories, consolidate never-shown.

Philosophy: consolidate, never delete (except proven noise).
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

    Compresses old memories into category summaries and consolidates
    never-shown memories. Philosophy: consolidate, never delete
    (except proven noise via _prune_chronically_noisy).
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

        # Filter to 14+ days old, excluding protected categories
        old_docs = [
            d for d in all_docs
            if _doc_age_exceeds(d, days=14)
            and d.get("category", "general") not in PROTECTED_CATEGORIES
        ]
        logger.info(f"Found {len(old_docs)} old documents to potentially summarize")

        categories_updated, originals_deleted = await self._compress_by_category(
            doc_store, old_docs
        )
        noise_pruned = await self._prune_chronically_noisy(doc_store, all_docs)
        consolidated = await self._consolidate_never_shown(doc_store, all_docs)

        result = {
            "categories_updated": categories_updated,
            "originals_deleted": originals_deleted,
            "memories_pruned": 0,  # Kept for backwards compat (pruning removed)
            "never_shown_pruned": 0,  # Kept for backwards compat (pruning removed)
            "noise_pruned": noise_pruned,
            "never_shown_consolidated": consolidated,
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
                                await doc_store.delete_document(doc_id, hard=False, user_id=self.config.user_id)
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

        # Prefer content_detailed (full original) over content (may be condensed)
        contents = [d.get("content_detailed") or d.get("content", "") for d in docs if d.get("content")]
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

    # Progressive consolidation: never-shown memories per project/category
    # that exceed this count get consolidated into summary docs.
    CONSOLIDATION_THRESHOLD = 20  # If >20 never-shown memories in one project+category, consolidate

    # Noise pruning thresholds
    NOISE_MIN_SIGNALS = 5      # Minimum total feedback signals to act
    NOISE_MAX_RATIO = 0.50     # Noise rate above which to prune (50%+)

    async def _consolidate_never_shown(
        self, doc_store, all_docs: list[dict]
    ) -> int:
        """Progressively consolidate never-shown memories within the same project+category.

        When a project+category has more than CONSOLIDATION_THRESHOLD never-shown
        memories, condense them into fewer summary documents via LLM. This prevents
        memory count from growing unboundedly for active projects.

        The consolidation is progressive — each run halves the count until it
        reaches the threshold.

        Returns:
            Number of documents consolidated (originals soft-deleted)
        """
        from collections import defaultdict

        # Group never-shown memories by project + category
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for d in all_docs:
            if d.get("shown_count", 0) == 0 and d.get("category", "general") not in PROTECTED_CATEGORIES:
                project = d.get("source_ref") or "(none)"
                category = d.get("category", "general")
                groups[(project, category)].append(d)

        total_consolidated = 0
        for (project, category), docs in groups.items():
            if len(docs) <= self.CONSOLIDATION_THRESHOLD:
                continue

            # Take half of the oldest docs and consolidate them into one summary
            docs.sort(key=lambda d: d.get("created_at", ""))
            batch_size = len(docs) // 2
            batch = docs[:batch_size]

            if batch_size < 5:
                continue

            # Combine content for LLM summarization
            # Prefer content_detailed (full original) over content (may be condensed)
            combined = "\n\n".join(
                (d.get("content_detailed") or d.get("content", ""))[:500]
                for d in batch  # Cap per-doc to avoid huge prompts
            )

            try:
                from cems.llm.client import get_client
                client = get_client()
                summary = client.complete(
                    prompt=(
                        f"Consolidate these {len(batch)} memory entries into a single, dense summary.\n"
                        f"Keep all unique facts, decisions, and important details.\n"
                        f"Remove redundancy. Preserve exact values (names, numbers, dates).\n"
                        f"Category: {category}\n"
                        f"Project: {project}\n\n"
                        f"{combined}"
                    ),
                    system="You are a memory consolidation agent. Output a concise summary preserving all unique information.",
                    model="google/gemini-2.5-flash-lite",
                    temperature=0.1,
                    max_tokens=2000,
                    fast_route=False,
                )

                if not summary or len(summary) < 50:
                    continue

                # Store the consolidated summary via memory.add_async (handles chunking+embedding)
                await self.memory.add_async(
                    content=summary,
                    category=category,
                    source_ref=project if project != "(none)" else None,
                    tags=[f"consolidated:{len(batch)}"],
                )

                # Soft-delete the originals
                for d in batch:
                    doc_id = d.get("id")
                    if doc_id:
                        await doc_store.delete_document(doc_id, hard=False, user_id=self.config.user_id)
                        total_consolidated += 1

                logger.info(
                    f"Consolidated {len(batch)} never-shown '{category}' memories "
                    f"in {project} into 1 summary"
                )
            except Exception as e:
                logger.error(f"Failed to consolidate {project}/{category}: {e}")

        if total_consolidated:
            logger.info(f"Progressive consolidation: {total_consolidated} memories merged")

        return total_consolidated

    async def _prune_chronically_noisy(
        self, doc_store, all_docs: list[dict]
    ) -> int:
        """Soft-delete memories with consistently high noise feedback.

        If a memory has accumulated enough feedback signals (>=5) and more than
        50% are noise, it's chronically irrelevant and should be pruned.

        This closes the feedback loop: users mark memories as noise via the
        "Memory relevance: #N was noise" line, and this pruner acts on it.

        Args:
            doc_store: DocumentStore instance
            all_docs: All user documents

        Returns:
            Number of documents pruned
        """
        candidates = []
        for d in all_docs:
            relevant = d.get("relevant_count", 0)
            noise = d.get("noise_count", 0)
            total = relevant + noise
            if total < self.NOISE_MIN_SIGNALS:
                continue
            noise_ratio = noise / total
            if noise_ratio <= self.NOISE_MAX_RATIO:
                continue
            # Skip protected categories
            if d.get("category", "general") in PROTECTED_CATEGORIES:
                continue
            candidates.append(d)

        pruned = 0
        for doc in candidates:
            doc_id = doc.get("id")
            if doc_id:
                noise = doc.get("noise_count", 0)
                shown = doc.get("shown_count", 0)
                cat = doc.get("category", "?")
                try:
                    await doc_store.delete_document(doc_id, hard=False, user_id=self.config.user_id)
                    pruned += 1
                    logger.info(
                        f"Pruned noisy memory {doc_id[:8]} "
                        f"(noise={noise}, shown={shown}, cat={cat})"
                    )
                except Exception as e:
                    logger.error(f"Failed to prune noisy document {doc_id}: {e}")

        if pruned:
            logger.info(
                f"Soft-deleted {pruned} chronically noisy documents "
                f"(>{self.NOISE_MAX_RATIO:.0%} noise rate, "
                f">={self.NOISE_MIN_SIGNALS} signals)"
            )

        return pruned

    def run(self) -> dict[str, int]:
        """Synchronous wrapper for run_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()
