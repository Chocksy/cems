"""Nightly consolidation job — three-tier smart deduplication.

Uses DocumentStore (memory_documents) exclusively via async pattern.
Follows the ObservationReflector pattern for async + DocumentStore access.

Three-tier approach:
  - Tier 1 (>= 0.98): Auto-merge near-identical memories
  - Tier 2 (0.80-0.98): LLM classifies as duplicate/related/conflicting/distinct
  - Tier 3 (< 0.80): Skip — too different
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from cems.llm import classify_memory_pair, merge_memory_contents

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Minimum confidence from LLM to act on a classification
MIN_CONFIDENCE = 0.7


class ConsolidationJob:
    """Nightly maintenance job for memory consolidation.

    Finds and merges semantically duplicate memories using vector
    similarity search and LLM-powered classification + merging.
    Also detects conflicting memories and records them for resolution.
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(
        self, *, full_sweep: bool = False, limit: int = 0, offset: int = 0
    ) -> dict[str, int]:
        """Run the consolidation job.

        Args:
            full_sweep: If True, process ALL documents (not just last 7 days).
            limit: Override document limit (0 = use defaults: 5000 nightly, 500 sweep).
            offset: Skip first N documents (for paginated sweeps).

        Returns:
            Dict with counts of operations performed
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        if full_sweep:
            effective_limit = limit or 500
            docs = await doc_store.get_all_documents(
                user_id, limit=effective_limit, offset=offset
            )
            logger.info(
                f"Full sweep: {len(docs)} documents (offset={offset}, limit={effective_limit})"
            )
        else:
            effective_limit = limit or 5000
            docs = await doc_store.get_recent_documents(
                user_id, hours=168, limit=effective_limit
            )
            logger.info(f"Nightly consolidation: {len(docs)} documents from last 7 days")

        result = await self._merge_duplicates(docs)
        result["memories_checked"] = len(docs)
        if full_sweep:
            result["offset"] = offset

        logger.info(f"Consolidation completed: {result}")
        return result

    async def _merge_duplicates(self, docs: list[dict]) -> dict[str, int]:
        """Three-tier deduplication with conflict detection.

        Tier 1 (>= automerge_threshold): Auto-merge without LLM
        Tier 2 (llm_threshold - automerge_threshold): LLM classification
        Tier 3 (< llm_threshold): Skip

        Args:
            docs: List of document dicts from DocumentStore

        Returns:
            Dict with duplicates_merged, conflicts_found, llm_classifications
        """
        if len(docs) < 2:
            return {"duplicates_merged": 0, "conflicts_found": 0, "llm_classifications": 0}

        await self.memory._ensure_initialized_async()
        assert self.memory._async_embedder is not None
        doc_store = await self.memory._ensure_document_store()

        automerge_threshold = self.config.dedup_automerge_threshold
        llm_threshold = self.config.dedup_llm_threshold

        merged_count = 0
        conflicts_found = 0
        llm_classifications = 0
        processed: set[str] = set()

        # Pre-embed all document contents in batches to avoid N API round-trips.
        # Each doc gets one embedding for its full content (used for similarity search).
        contents = []
        content_doc_ids = []
        for doc in docs:
            c = doc.get("content", "")
            if c:
                contents.append(c)
                content_doc_ids.append(doc["id"])

        embedding_cache: dict[str, list[float]] = {}
        if contents:
            BATCH_SIZE = 100
            for batch_start in range(0, len(contents), BATCH_SIZE):
                batch_texts = contents[batch_start : batch_start + BATCH_SIZE]
                batch_ids = content_doc_ids[batch_start : batch_start + BATCH_SIZE]
                batch_embeddings = await self.memory._async_embedder.embed_batch(batch_texts)
                for did, emb in zip(batch_ids, batch_embeddings):
                    if emb:
                        embedding_cache[did] = emb
            logger.info(f"Pre-embedded {len(embedding_cache)} documents for dedup")

        for doc in docs:
            doc_id = doc["id"]
            if doc_id in processed:
                continue

            content = doc.get("content", "")
            if not content:
                continue

            embedding = embedding_cache.get(doc_id)
            if not embedding:
                continue

            # Search for similar chunks
            similar_chunks = await doc_store.search_chunks(
                query_embedding=embedding,
                user_id=self.config.user_id,
                scope="personal",
                limit=10,
            )

            # Group by document_id, find candidates above LLM threshold
            seen_doc_ids: set[str] = set()
            for chunk in similar_chunks:
                chunk_doc_id = chunk.get("document_id", "")
                score = chunk.get("score", 0)

                if (
                    not chunk_doc_id
                    or chunk_doc_id == doc_id
                    or chunk_doc_id in processed
                    or chunk_doc_id in seen_doc_ids
                    or score < llm_threshold
                ):
                    continue

                seen_doc_ids.add(chunk_doc_id)

                # Fetch the candidate document
                dup_doc = await doc_store.get_document(chunk_doc_id)
                if not dup_doc:
                    continue

                # Tier 1: Auto-merge near-identical
                if score >= automerge_threshold:
                    logger.info(
                        f"Auto-merging {doc_id[:8]}+{chunk_doc_id[:8]} (score={score:.3f})"
                    )
                    merged_content = merge_memory_contents(
                        memories=[{"memory": content}, {"memory": dup_doc.get("content", "")}],
                        model=self.config.llm_model,
                    )
                    if merged_content and merged_content != content:
                        await self.memory.update_async(doc_id, merged_content)
                        content = merged_content  # Update for subsequent merges in this loop
                    await doc_store.delete_document(chunk_doc_id, hard=False)
                    processed.add(chunk_doc_id)
                    merged_count += 1
                    continue

                # Tier 2: LLM classification (llm_threshold <= score < automerge_threshold)
                # Metadata guards: skip if different category or project
                if self._metadata_distinct(doc, dup_doc):
                    continue

                classification = classify_memory_pair(
                    content, dup_doc.get("content", ""),
                    model=self.config.llm_model,
                )
                llm_classifications += 1

                cls = classification["classification"]
                confidence = classification["confidence"]

                if cls == "duplicate" and confidence >= MIN_CONFIDENCE:
                    logger.info(
                        f"LLM merge {doc_id[:8]}+{chunk_doc_id[:8]} "
                        f"(score={score:.3f}, confidence={confidence:.2f})"
                    )
                    merged_content = merge_memory_contents(
                        memories=[{"memory": content}, {"memory": dup_doc.get("content", "")}],
                        model=self.config.llm_model,
                    )
                    if merged_content:
                        if merged_content != content:
                            await self.memory.update_async(doc_id, merged_content)
                            content = merged_content  # Update for subsequent merges
                        await doc_store.delete_document(chunk_doc_id, hard=False)
                        processed.add(chunk_doc_id)
                        merged_count += 1

                elif cls == "conflicting" and confidence >= MIN_CONFIDENCE:
                    logger.info(
                        f"Conflict detected: {doc_id[:8]} vs {chunk_doc_id[:8]} "
                        f"(confidence={confidence:.2f})"
                    )
                    try:
                        conflict_id = await doc_store.add_conflict(
                            user_id=self.config.user_id,
                            doc_a_id=doc_id,
                            doc_b_id=chunk_doc_id,
                            explanation=classification["explanation"],
                        )
                        if conflict_id:
                            conflicts_found += 1
                    except Exception:
                        # memory_conflicts table may not exist yet (migration pending)
                        logger.warning(
                            f"Could not store conflict {doc_id[:8]} vs {chunk_doc_id[:8]} "
                            f"— run migrate_conflicts.sql"
                        )

                # "related" and "distinct" → skip (both memories stay)

            processed.add(doc_id)

        return {
            "duplicates_merged": merged_count,
            "conflicts_found": conflicts_found,
            "llm_classifications": llm_classifications,
        }

    @staticmethod
    def _metadata_distinct(doc_a: dict, doc_b: dict) -> bool:
        """Check if metadata signals the documents are clearly distinct.

        Guards against LLM hallucination by checking structural differences.

        Returns:
            True if documents are clearly distinct (skip LLM classification)
        """
        # Different categories = different kinds of knowledge
        cat_a = doc_a.get("category", "")
        cat_b = doc_b.get("category", "")
        if cat_a and cat_b and cat_a != cat_b:
            return True

        # Different source_ref = different projects
        ref_a = doc_a.get("source_ref", "")
        ref_b = doc_b.get("source_ref", "")
        if ref_a and ref_b and ref_a != ref_b:
            return True

        return False

    def run(self) -> dict[str, int]:
        """Synchronous wrapper for run_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()
