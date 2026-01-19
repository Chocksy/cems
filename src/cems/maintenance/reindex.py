"""Monthly re-indexing job - rebuild embeddings, archive dead memories."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)


class ReindexJob:
    """Monthly maintenance job for memory re-indexing.

    Runs monthly to:
    1. Rebuild embeddings for all memories (using latest model)
    2. Archive dead nodes (not accessed in 180+ days)
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    def run(self) -> dict[str, int]:
        """Run the re-indexing job.

        Returns:
            Dict with counts of operations performed
        """
        user_id = self.config.user_id
        log_id = self.memory.metadata_store.log_maintenance(
            "reindex", user_id, "started"
        )

        try:
            # Step 1: Get all non-archived memories
            all_ids = self.memory.metadata_store.get_all_user_memories(
                user_id, include_archived=False
            )
            logger.info(f"Found {len(all_ids)} memories to check for re-indexing")

            # Step 2: Rebuild embeddings
            # Note: In Mem0, embeddings are managed internally
            # We trigger a refresh by doing a search which updates the index
            reindexed = self._refresh_embeddings(all_ids)

            # Step 3: Archive dead memories (180+ days unused)
            archived = self._archive_dead()

            result = {
                "memories_reindexed": reindexed,
                "memories_archived": archived,
                "total_memories": len(all_ids),
            }

            self.memory.metadata_store.update_maintenance_log(
                log_id, "completed", str(result)
            )
            logger.info(f"Re-indexing completed: {result}")
            return result

        except Exception as e:
            self.memory.metadata_store.update_maintenance_log(
                log_id, "failed", str(e)
            )
            logger.error(f"Re-indexing failed: {e}")
            raise

    def _refresh_embeddings(self, memory_ids: list[str]) -> int:
        """Refresh embeddings for memories.

        With Mem0, we force re-embedding by updating the memory content,
        which triggers the embedding pipeline. This is the supported approach
        when using Mem0 as a dependency since it doesn't expose direct
        embedding APIs.

        The update triggers Mem0's internal pipeline which:
        1. Re-generates embeddings with the current embedding model
        2. Updates the vector store with new embeddings
        3. Maintains version history

        Args:
            memory_ids: List of memory IDs to refresh

        Returns:
            Number of memories re-indexed
        """
        refreshed = 0
        failed = 0
        batch_size = 10  # Process in batches for better progress tracking

        total = len(memory_ids)
        for i, mem_id in enumerate(memory_ids):
            try:
                mem = self.memory.get(mem_id)
                if mem and "memory" in mem:
                    content = mem["memory"]
                    # Update triggers Mem0's embedding regeneration
                    self.memory.update(mem_id, content)
                    refreshed += 1

                    # Log progress every batch_size
                    if refreshed % batch_size == 0:
                        logger.info(f"Re-indexing progress: {refreshed}/{total} memories")

            except Exception as e:
                logger.warning(f"Failed to re-index memory {mem_id}: {e}")
                failed += 1

        logger.info(
            f"Re-indexing complete: {refreshed} succeeded, {failed} failed out of {total}"
        )

        return refreshed

    def _archive_dead(self) -> int:
        """Archive memories not accessed in archive_days.

        Returns:
            Number of memories archived
        """
        archive_days = self.config.archive_days
        dead_ids = self.memory.metadata_store.get_stale_memories(
            self.config.user_id, archive_days
        )
        archived = 0

        for mem_id in dead_ids:
            self.memory.archive_memory(mem_id)
            archived += 1

        if archived:
            logger.info(f"Archived {archived} dead memories (>{archive_days} days unused)")

        return archived


def rebuild_embeddings(memory: "CEMSMemory") -> int:
    """Standalone function to rebuild all embeddings.

    Args:
        memory: CEMSMemory instance

    Returns:
        Number of memories re-indexed
    """
    job = ReindexJob(memory)
    all_ids = memory.metadata_store.get_all_user_memories(
        memory.config.user_id, include_archived=False
    )
    return job._refresh_embeddings(all_ids)


def archive_dead(memory: "CEMSMemory", not_accessed_days: int = 180) -> int:
    """Standalone function to archive dead memories.

    Args:
        memory: CEMSMemory instance
        not_accessed_days: Days since last access

    Returns:
        Number of memories archived
    """
    # Override threshold temporarily
    original_days = memory.config.archive_days
    memory.config.archive_days = not_accessed_days

    try:
        job = ReindexJob(memory)
        return job._archive_dead()
    finally:
        memory.config.archive_days = original_days
