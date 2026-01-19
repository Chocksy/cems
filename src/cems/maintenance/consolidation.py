"""Nightly consolidation job - merge duplicates, promote hot memories."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)


class ConsolidationJob:
    """Nightly maintenance job for memory consolidation.

    Runs every night to:
    1. Find and merge semantically duplicate memories
    2. Promote frequently accessed (hot) memories
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    def run(self) -> dict[str, int]:
        """Run the consolidation job.

        Returns:
            Dict with counts of operations performed
        """
        user_id = self.config.user_id
        log_id = self.memory.metadata_store.log_maintenance(
            "consolidation", user_id, "started"
        )

        try:
            # Step 1: Get recent memories (last 24 hours)
            recent_ids = self.memory.get_recent_memories(hours=24)
            logger.info(f"Found {len(recent_ids)} recent memories to check")

            # Step 2: Find and merge duplicates
            duplicates_merged = self._merge_duplicates(recent_ids)

            # Step 3: Promote hot memories
            promoted = self._promote_hot_memories()

            result = {
                "duplicates_merged": duplicates_merged,
                "memories_promoted": promoted,
                "memories_checked": len(recent_ids),
            }

            self.memory.metadata_store.update_maintenance_log(
                log_id, "completed", str(result)
            )
            logger.info(f"Consolidation completed: {result}")
            return result

        except Exception as e:
            self.memory.metadata_store.update_maintenance_log(
                log_id, "failed", str(e)
            )
            logger.error(f"Consolidation failed: {e}")
            raise

    def _merge_duplicates(self, memory_ids: list[str]) -> int:
        """Find and merge semantically duplicate memories using LLM.

        Uses vector similarity to find duplicates above the configured threshold,
        then uses LLM to merge their content into a single comprehensive memory.

        Args:
            memory_ids: List of memory IDs to check

        Returns:
            Number of duplicates merged
        """
        from cems.llm import merge_memory_contents

        if len(memory_ids) < 2:
            return 0

        merged_count = 0
        threshold = self.config.duplicate_similarity_threshold
        processed = set()

        for memory_id in memory_ids:
            if memory_id in processed:
                continue

            # Get the memory content
            mem = self.memory.get(memory_id)
            if not mem:
                continue

            content = mem.get("memory", "")
            if not content:
                continue

            # Search for similar memories
            similar = self.memory.search(content, scope="personal", limit=5)

            # Find duplicates above threshold
            duplicates = []
            for result in similar:
                if (
                    result.memory_id != memory_id
                    and result.memory_id not in processed
                    and result.score >= threshold
                ):
                    duplicates.append(result)

            if duplicates:
                # Gather all memories to merge (original + duplicates)
                memories_to_merge = [mem]
                duplicate_ids = []
                for dup in duplicates:
                    dup_mem = self.memory.get(dup.memory_id)
                    if dup_mem:
                        memories_to_merge.append(dup_mem)
                        duplicate_ids.append(dup.memory_id)
                        processed.add(dup.memory_id)

                if len(memories_to_merge) > 1:
                    # Use LLM to merge the content (uses OpenRouter via llm.py)
                    logger.info(f"Merging {len(memories_to_merge)} similar memories using LLM")
                    merged_content = merge_memory_contents(
                        memories=memories_to_merge,
                        model=self.config.llm_model,  # OpenRouter format: provider/model
                    )

                    # Update the original memory with merged content
                    if merged_content and merged_content != content:
                        self.memory.update(memory_id, merged_content)
                        logger.debug(f"Updated memory {memory_id} with merged content")

                    # Delete the duplicates (hard delete since content is merged)
                    for dup_id in duplicate_ids:
                        self.memory.delete(dup_id, hard=True)
                        merged_count += 1

                    logger.info(
                        f"Merged {len(duplicate_ids)} duplicates into memory {memory_id}"
                    )

            processed.add(memory_id)

        return merged_count

    def _promote_hot_memories(self) -> int:
        """Promote frequently accessed memories.

        Increases priority of memories accessed more than the threshold.

        Returns:
            Number of memories promoted
        """
        hot_ids = self.memory.get_hot_memories()
        promoted = 0

        for memory_id in hot_ids:
            self.memory.promote_memory(memory_id, boost=0.1)
            promoted += 1

        if promoted:
            logger.info(f"Promoted {promoted} hot memories")

        return promoted


def merge_duplicates(memory: "CEMSMemory", similarity_threshold: float = 0.92) -> int:
    """Standalone function to merge duplicates.

    Args:
        memory: CEMSMemory instance
        similarity_threshold: Cosine similarity threshold

    Returns:
        Number of duplicates merged
    """
    # Override threshold temporarily
    original_threshold = memory.config.duplicate_similarity_threshold
    memory.config.duplicate_similarity_threshold = similarity_threshold

    try:
        job = ConsolidationJob(memory)
        recent_ids = memory.get_recent_memories(hours=24)
        return job._merge_duplicates(recent_ids)
    finally:
        memory.config.duplicate_similarity_threshold = original_threshold


def promote_hot_memories(memory: "CEMSMemory", access_threshold: int = 5) -> int:
    """Standalone function to promote hot memories.

    Args:
        memory: CEMSMemory instance
        access_threshold: Access count threshold

    Returns:
        Number of memories promoted
    """
    # Override threshold temporarily
    original_threshold = memory.config.hot_access_threshold
    memory.config.hot_access_threshold = access_threshold

    try:
        job = ConsolidationJob(memory)
        return job._promote_hot_memories()
    finally:
        memory.config.hot_access_threshold = original_threshold
