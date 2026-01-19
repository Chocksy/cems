"""Weekly summarization job - compress old memories, prune stale ones."""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)


class SummarizationJob:
    """Weekly maintenance job for memory summarization.

    Runs every week to:
    1. Compress old memories into category summaries
    2. Prune memories not accessed in 90+ days
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    def run(self) -> dict[str, int]:
        """Run the summarization job.

        Returns:
            Dict with counts of operations performed
        """
        user_id = self.config.user_id
        log_id = self.memory.metadata_store.log_maintenance(
            "summarization", user_id, "started"
        )

        try:
            # Step 1: Get old memories (30+ days)
            old_ids = self.memory.get_old_memories(days=30)
            logger.info(f"Found {len(old_ids)} old memories to potentially summarize")

            # Step 2: Compress old memories by category
            categories_updated = self._compress_by_category(old_ids)

            # Step 3: Prune stale memories
            pruned = self._prune_stale()

            result = {
                "categories_updated": categories_updated,
                "memories_pruned": pruned,
                "old_memories_checked": len(old_ids),
            }

            self.memory.metadata_store.update_maintenance_log(
                log_id, "completed", str(result)
            )
            logger.info(f"Summarization completed: {result}")
            return result

        except Exception as e:
            self.memory.metadata_store.update_maintenance_log(
                log_id, "failed", str(e)
            )
            logger.error(f"Summarization failed: {e}")
            raise

    def _compress_by_category(self, memory_ids: list[str]) -> int:
        """Compress old memories into category summaries.

        Groups memories by category and creates/updates summary documents.

        Args:
            memory_ids: List of old memory IDs

        Returns:
            Number of categories updated
        """
        if not memory_ids:
            return 0

        # Group by category
        categories: dict[str, list[str]] = {}
        for mem_id in memory_ids:
            metadata = self.memory.get_metadata(mem_id)
            if metadata:
                cat = metadata.category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(mem_id)

        updated = 0
        for category, ids in categories.items():
            if len(ids) >= 3:  # Only summarize if we have enough memories
                try:
                    self._create_category_summary(category, ids)
                    updated += 1
                except Exception as e:
                    logger.warning(f"Failed to summarize category {category}: {e}")

        return updated

    def _create_category_summary(self, category: str, memory_ids: list[str]) -> None:
        """Create an LLM-generated summary for a category.

        Uses the configured LLM to generate a coherent summary of all memories
        in the category, then stores it in the database.

        Args:
            category: The category name
            memory_ids: Memory IDs in this category
        """
        import sqlite3

        from cems.llm import summarize_memories

        # Gather memory contents
        contents = []
        for mem_id in memory_ids:
            mem = self.memory.get(mem_id)
            if mem:
                content = mem.get("memory", "")
                if content:
                    contents.append(content)

        if not contents:
            logger.debug(f"No content found for category {category}")
            return

        # Generate LLM summary (uses OpenRouter via llm.py)
        logger.info(f"Generating LLM summary for category '{category}' with {len(contents)} memories")
        summary = summarize_memories(
            memories=contents,
            category=category,
            model=self.config.llm_model,  # OpenRouter format: provider/model
        )

        # Store in database
        conn = sqlite3.connect(self.config.metadata_db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO category_summaries
                (user_id, category, scope, summary, item_count, last_updated, version)
                VALUES (?, ?, 'personal', ?, ?, ?, COALESCE(
                    (SELECT version + 1 FROM category_summaries
                     WHERE user_id = ? AND category = ? AND scope = 'personal'), 1
                ))
                """,
                (
                    self.config.user_id,
                    category,
                    summary,
                    len(contents),
                    datetime.now(UTC).isoformat(),
                    self.config.user_id,
                    category,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Created LLM summary for category {category} with {len(contents)} items")

    def _prune_stale(self) -> int:
        """Prune memories not accessed in configured days.

        Archives stale memories rather than deleting them.

        Returns:
            Number of memories pruned
        """
        stale_ids = self.memory.get_stale_memories()
        pruned = 0

        for mem_id in stale_ids:
            self.memory.archive_memory(mem_id)
            pruned += 1

        if pruned:
            logger.info(f"Archived {pruned} stale memories")

        return pruned


def compress_old_memories(memory: "CEMSMemory", older_than_days: int = 30) -> int:
    """Standalone function to compress old memories.

    Args:
        memory: CEMSMemory instance
        older_than_days: Age threshold for memories

    Returns:
        Number of categories updated
    """
    job = SummarizationJob(memory)
    old_ids = memory.get_old_memories(days=older_than_days)
    return job._compress_by_category(old_ids)


def prune_stale(memory: "CEMSMemory", not_accessed_days: int = 90) -> int:
    """Standalone function to prune stale memories.

    Args:
        memory: CEMSMemory instance
        not_accessed_days: Days since last access

    Returns:
        Number of memories pruned
    """
    # Override threshold temporarily
    original_days = memory.config.stale_days
    memory.config.stale_days = not_accessed_days

    try:
        job = SummarizationJob(memory)
        return job._prune_stale()
    finally:
        memory.config.stale_days = original_days
