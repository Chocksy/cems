"""Weekly summarization job — archive compiled memories, prune noise.

Philosophy: entity pages are the permanent knowledge store. Individual memories
are raw material that flows through: link → compile → archive.
Only archives memories that have been compiled into entity pages (knowledge preserved).
Uses DocumentStore (memory_documents) exclusively via async pattern.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

from cems.maintenance import PROTECTED_CATEGORIES


class SummarizationJob:
    """Weekly maintenance job for memory archival.

    Archives compiled source memories and prunes chronically noisy ones.
    Entity pages replace category summaries as the permanent knowledge store.
    """

    # Noise pruning thresholds
    NOISE_MIN_SIGNALS = 5      # Minimum total feedback signals to act
    NOISE_MAX_RATIO = 0.50     # Noise rate above which to prune (50%+)

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(self) -> dict[str, int]:
        """Simplified maintenance: archive compiled memories, prune noise.

        Returns:
            Dict with counts of operations performed
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        # Get all docs for noise pruning
        all_docs = await doc_store.get_all_documents(user_id, limit=2000, order="asc")

        # 1. Archive compiled memories (knowledge preserved in entity pages)
        archived_compiled = await self._archive_compiled_sources(doc_store, user_id)

        # 2. Prune chronically noisy memories
        noise_pruned = await self._prune_chronically_noisy(doc_store, all_docs)

        result = {
            "archived_compiled": archived_compiled,
            "noise_pruned": noise_pruned,
        }
        logger.info(f"Summarization completed: {result}")
        return result

    async def _archive_compiled_sources(self, doc_store, user_id: str) -> int:
        """Archive memories that have been compiled into entity pages.

        Only archives memories that:
        - Have a compiled_from relation (knowledge preserved in entity page)
        - Are 14+ days old
        - Are not hot (shown_count < 20)
        """
        compiled_sources = await doc_store.get_compiled_sources(
            user_id=user_id,
            min_age_days=14,
            max_shown_count=20,
            limit=100,
        )

        archived = 0
        for doc in compiled_sources:
            doc_id = doc.get("id")
            if doc_id:
                try:
                    await doc_store.delete_document(doc_id, hard=False, user_id=user_id)
                    archived += 1
                except Exception as e:
                    logger.error(f"Failed to archive compiled source {doc_id[:8]}: {e}")

        if archived:
            logger.info(f"Archived {archived} compiled source memories")

        return archived

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
            # Skip protected categories and pinned memories
            if d.get("category", "general") in PROTECTED_CATEGORIES:
                continue
            if "pinned" in (d.get("tags") or []):
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
