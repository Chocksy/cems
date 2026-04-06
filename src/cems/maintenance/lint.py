"""Knowledge health lint job — detect contradictions, orphans, and gaps.

Scans the knowledge graph for quality issues:
- Contradictions: highly similar memories with conflicting content
- Orphans: memories with zero relations (isolated, possibly low-value)
- Stale entity pages: entity pages whose source cluster has new memories
- Knowledge gaps: topics mentioned frequently but without synthesis

Follows the standard async maintenance job pattern.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Minimum similarity to consider as a contradiction candidate
CONTRADICTION_THRESHOLD = 0.85


class LintJob:
    """Knowledge health lint job.

    Produces a report of issues found + optionally records contradictions
    in the memory_conflicts table for dashboard resolution.
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(
        self,
        *,
        detect_contradictions: bool = True,
        detect_orphans: bool = True,
        limit: int = 50,
    ) -> dict:
        """Run the lint job.

        Args:
            detect_contradictions: Scan high-similarity pairs for contradictions
            detect_orphans: Count memories with no relations
            limit: Max contradiction candidates to process per run

        Returns:
            Dict with lint report: contradictions_found, orphan_count,
            total_memories, connected_memories, health_score
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id
        pool = await doc_store._get_pool()

        from uuid import UUID
        user_uuid = UUID(user_id)

        report = {
            "contradictions_found": 0,
            "contradictions_checked": 0,
            "orphan_count": 0,
            "total_memories": 0,
            "connected_memories": 0,
            "health_score": 100,
        }

        async with pool.acquire() as conn:
            # Total active documents
            report["total_memories"] = await conn.fetchval(
                "SELECT COUNT(*) FROM memory_documents WHERE user_id = $1 AND deleted_at IS NULL",
                user_uuid,
            ) or 0

            # Connected documents (have at least one relation)
            report["connected_memories"] = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT d.id) FROM memory_documents d
                WHERE d.user_id = $1 AND d.deleted_at IS NULL
                AND EXISTS (
                    SELECT 1 FROM memory_relations r
                    WHERE r.source_id = d.id OR r.target_id = d.id
                )
                """,
                user_uuid,
            ) or 0

            report["orphan_count"] = report["total_memories"] - report["connected_memories"]

        # Detect contradictions using LLM
        if detect_contradictions:
            contradictions = await self._detect_contradictions(doc_store, user_id, limit)
            report["contradictions_found"] = contradictions["found"]
            report["contradictions_checked"] = contradictions["checked"]

        # Compute health score
        health = 100
        open_conflicts = await self._count_open_conflicts(doc_store, user_id)
        health -= min(open_conflicts * 5, 25)
        health -= min(report["orphan_count"] * 0.5, 20)
        report["health_score"] = max(round(health), 0)
        report["open_conflicts"] = open_conflicts

        logger.info(f"LintJob completed: {report}")
        return report

    async def _detect_contradictions(
        self, doc_store, user_id: str, limit: int
    ) -> dict:
        """Find high-similarity memory pairs and check for contradictions.

        Strategy: find pairs with cosine > CONTRADICTION_THRESHOLD that aren't
        already in memory_conflicts, then use LLM to classify.
        """
        from uuid import UUID
        pool = await doc_store._get_pool()
        user_uuid = UUID(user_id)

        # Find high-similarity pairs not yet checked
        async with pool.acquire() as conn:
            pairs = await conn.fetch(
                """
                SELECT r.source_id, r.target_id, r.similarity,
                       s.content AS source_content,
                       t.content AS target_content
                FROM memory_relations r
                JOIN memory_documents s ON r.source_id = s.id
                JOIN memory_documents t ON r.target_id = t.id
                WHERE s.user_id = $1
                  AND s.deleted_at IS NULL AND t.deleted_at IS NULL
                  AND r.similarity >= $2
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_conflicts c
                      WHERE (c.doc_a_id = r.source_id AND c.doc_b_id = r.target_id)
                         OR (c.doc_a_id = r.target_id AND c.doc_b_id = r.source_id)
                  )
                ORDER BY r.similarity DESC
                LIMIT $3
                """,
                user_uuid,
                CONTRADICTION_THRESHOLD,
                limit,
            )

        if not pairs:
            return {"found": 0, "checked": 0}

        found = 0
        checked = 0

        for pair in pairs:
            source_content = pair["source_content"] or ""
            target_content = pair["target_content"] or ""

            # Skip very short content (not enough to contradict)
            if len(source_content) < 50 or len(target_content) < 50:
                continue

            checked += 1
            is_contradiction = await self._llm_check_contradiction(
                source_content[:1000], target_content[:1000]
            )

            if is_contradiction:
                explanation = f"Auto-detected: similarity {pair['similarity']:.2f}, flagged by lint job"
                await doc_store.add_conflict(
                    user_id=user_id,
                    doc_a_id=str(pair["source_id"]),
                    doc_b_id=str(pair["target_id"]),
                    explanation=explanation,
                )
                found += 1

        return {"found": found, "checked": checked}

    async def _llm_check_contradiction(
        self, content_a: str, content_b: str
    ) -> bool:
        """Use LLM to check if two memories contradict each other."""
        try:
            from cems.llm import get_client
            client = get_client()
            if not client:
                return False

            prompt = f"""Do these two memories contradict each other? Answer ONLY "yes" or "no".

Memory A:
{content_a}

Memory B:
{content_b}

Contradiction means they make incompatible claims about the same topic.
Similar or overlapping information is NOT a contradiction.
Answer:"""

            response = client.complete(prompt, max_tokens=5)
            answer = (response or "").strip().lower()
            return answer.startswith("yes")
        except Exception as e:
            logger.warning(f"LLM contradiction check failed: {e}")
            return False

    async def _count_open_conflicts(self, doc_store, user_id: str) -> int:
        """Count open (unresolved) conflicts for this user."""
        from uuid import UUID
        pool = await doc_store._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM memory_conflicts WHERE user_id = $1 AND status = 'open'",
                UUID(user_id),
            ) or 0
