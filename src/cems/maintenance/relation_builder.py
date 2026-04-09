"""Backfill job — populate memory_relations for existing documents.

Finds and links related memories by reusing existing chunk embeddings.
Safe to re-run: uses upsert semantics (ON CONFLICT DO UPDATE).
Follows the standard async maintenance job pattern.
"""

import logging
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Lower than write.py auto-link (0.75) to catch more semantic connections
SIMILARITY_THRESHOLD = 0.65
NEIGHBORS_PER_DOC = 10
# Days before rechecking a doc that previously found no neighbors
RECHECK_DAYS = 3


class RelationBuilderJob:
    """Backfill job to populate memory_relations for existing documents.

    Processes documents in batches, finding and linking neighbors
    using existing chunk embeddings. Uses SQL-level filtering to
    skip already-processed docs (those with 'similar' relations).
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(
        self,
        *,
        limit: int = 50,
        force: bool = False,
    ) -> dict:
        """Run the backfill.

        Args:
            limit: Batch size (keep small for Coolify proxy timeout ~60s)
            force: If True, re-process documents that already have relations

        Returns:
            Dict with stats: docs_processed, relations_created, docs_skipped
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        if force:
            docs = await doc_store.get_all_documents(
                user_id, limit=limit, order="asc",
            )
        else:
            docs = await self._get_unlinked_documents(doc_store, user_id, limit)

        if not docs:
            logger.info("RelationBuilder: no unlinked documents remaining")
            return {
                "docs_processed": 0,
                "relations_created": 0,
                "docs_skipped": 0,
            }

        logger.info(
            f"RelationBuilder: processing {len(docs)} documents "
            f"(limit={limit}, force={force})"
        )

        total_relations = 0
        docs_processed = 0
        docs_skipped = 0

        for doc in docs:
            doc_id = doc.get("id")
            if not doc_id:
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
            else:
                # No neighbors found — write a self-referencing 'checked'
                # marker directly (bypasses add_relations self-ref guard).
                # The _get_unlinked_documents query rechecks these after
                # RECHECK_DAYS in case new similar memories arrive later.
                try:
                    pool = await doc_store._get_pool()
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO memory_relations
                                (source_id, target_id, relation_type, similarity)
                            VALUES ($1, $1, 'checked', 0.0)
                            ON CONFLICT (source_id, target_id, relation_type)
                            DO UPDATE SET similarity = 0.0, created_at = NOW()
                            """,
                            UUID(doc_id),
                        )
                except Exception as e:
                    logger.debug(f"Failed to write checked marker for {doc_id[:8]}: {e}")

            docs_processed += 1

        result = {
            "docs_processed": docs_processed,
            "relations_created": total_relations,
            "docs_skipped": docs_skipped,
        }
        logger.info(f"RelationBuilder completed: {result}")
        return result

    async def _get_unlinked_documents(
        self, doc_store, user_id: str, limit: int
    ) -> list[dict]:
        """Get documents eligible for relation building.

        Returns docs that either:
        1. Have no 'similar' relations AND no 'checked' marker (never processed), OR
        2. Have a 'checked' marker (no matches found) older than RECHECK_DAYS —
           so new memories added later can still be matched.

        Uses SQL-level filtering to avoid the offset starvation bug
        where the same first N docs were re-read every run.
        """
        pool = await doc_store._get_pool()
        user_uuid = UUID(user_id)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, category, scope, source_ref, tags, created_at
                FROM memory_documents d
                WHERE d.user_id = $1
                  AND d.deleted_at IS NULL
                  AND d.category != 'entity-page'
                  AND (
                      -- Never processed: no similar relations and no checked marker
                      (
                          NOT EXISTS (
                              SELECT 1 FROM memory_relations r
                              WHERE r.source_id = d.id
                                AND r.relation_type = 'similar'
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM memory_relations r
                              WHERE r.source_id = d.id
                                AND r.relation_type = 'checked'
                          )
                      )
                      OR
                      -- Previously checked but found no matches, and the
                      -- check is stale — recheck for newly added memories
                      EXISTS (
                          SELECT 1 FROM memory_relations r
                          WHERE r.source_id = d.id
                            AND r.relation_type = 'checked'
                            AND r.created_at < NOW() - INTERVAL '%s days'
                      )
                  )
                ORDER BY d.created_at ASC
                LIMIT $2
                """
                % RECHECK_DAYS,
                user_uuid,
                limit,
            )

        return [
            {
                "id": str(row["id"]),
                "content": row["content"],
                "category": row["category"],
                "scope": row["scope"],
                "source_ref": row["source_ref"],
                "tags": row["tags"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

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
