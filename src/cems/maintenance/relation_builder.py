"""Backfill job — populate memory_relations for existing documents.

Finds and links related memories by reusing existing chunk embeddings.
Safe to re-run: uses upsert semantics (ON CONFLICT DO UPDATE).
Follows the standard async maintenance job pattern.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Match the threshold used in write.py auto-link
SIMILARITY_THRESHOLD = 0.75
NEIGHBORS_PER_DOC = 10


class RelationBuilderJob:
    """Backfill job to populate memory_relations for existing documents.

    Processes documents in batches, finding and linking neighbors
    using existing chunk embeddings. Skips documents that already
    have relations unless force=True.
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        force: bool = False,
    ) -> dict:
        """Run the backfill.

        Args:
            limit: Batch size (keep small for Coolify proxy timeout ~60s)
            offset: Starting offset for pagination
            force: If True, re-process documents that already have relations

        Returns:
            Dict with stats: docs_processed, relations_created, docs_skipped
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        # Get documents ordered by created_at ASC (oldest first)
        docs = await doc_store.get_all_documents(
            user_id, limit=limit, offset=offset, order="asc",
        )

        if not docs:
            logger.info("RelationBuilder: no documents to process")
            return {
                "docs_processed": 0,
                "relations_created": 0,
                "docs_skipped": 0,
            }

        logger.info(
            f"RelationBuilder: processing {len(docs)} documents "
            f"(offset={offset}, limit={limit}, force={force})"
        )

        total_relations = 0
        docs_processed = 0
        docs_skipped = 0

        for doc in docs:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            # Skip docs that already have relations (unless force)
            if not force:
                existing = await doc_store.get_relation_count(doc_id)
                if existing > 0:
                    docs_skipped += 1
                    continue

            # Get the first chunk's embedding for this document
            embedding = await self._get_first_chunk_embedding(doc_store, doc_id)
            if not embedding:
                docs_skipped += 1
                continue

            # Search for neighbors
            neighbors = await doc_store.search_chunks(
                query_embedding=embedding,
                user_id=user_id,
                limit=NEIGHBORS_PER_DOC,
            )

            # Build relations for neighbors above threshold
            relations = []
            seen_docs = {doc_id}

            for neighbor in neighbors:
                neighbor_doc_id = neighbor["document_id"]
                score = neighbor.get("score", 0)

                if neighbor_doc_id in seen_docs:
                    continue
                if score < SIMILARITY_THRESHOLD:
                    continue

                seen_docs.add(neighbor_doc_id)
                relations.append({
                    "target_id": neighbor_doc_id,
                    "relation_type": "similar",
                    "similarity": score,
                })

            if relations:
                # Forward relations
                created = await doc_store.add_relations(doc_id, relations)
                # Reverse relations (bidirectional)
                for rel in relations:
                    await doc_store.add_relations(
                        rel["target_id"],
                        [{"target_id": doc_id, "relation_type": "similar", "similarity": rel["similarity"]}],
                    )
                total_relations += created

            docs_processed += 1

        result = {
            "docs_processed": docs_processed,
            "relations_created": total_relations,
            "docs_skipped": docs_skipped,
        }
        logger.info(f"RelationBuilder completed: {result}")
        return result

    async def _get_first_chunk_embedding(
        self, doc_store, document_id: str
    ) -> list[float] | None:
        """Get the embedding of the first chunk for a document."""
        from uuid import UUID

        pool = await doc_store._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT embedding FROM memory_chunks
                WHERE document_id = $1
                ORDER BY seq ASC
                LIMIT 1
                """,
                UUID(document_id),
            )

        if row is not None and row["embedding"] is not None:
            return list(row["embedding"])
        return None
