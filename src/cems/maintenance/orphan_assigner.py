"""Orphan assigner job — link uncompiled memories to existing entity pages via LLM.

Finds memories with no `compiled_from` relation (orphans) and asks an LLM
which existing entity page they belong to. This reduces orphan count and
ensures knowledge is captured in entity pages.

Follows the standard async maintenance job pattern.
"""

import logging
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Categories that should never be assigned as orphans
EXCLUDED_CATEGORIES = frozenset({
    "entity-page", "gate-rules", "guidelines",
    "preferences", "category-summary", "session-summary",
})


class OrphanAssignerJob:
    """Assign orphan memories to existing entity pages via LLM."""

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(self, *, limit: int = 50) -> dict:
        """Find orphan memories and assign to entity pages.

        Args:
            limit: Max orphans to process per run

        Returns:
            Dict with: orphans_found, assigned, skipped
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        # 1. Get entity page titles + IDs
        entity_pages = await self._get_entity_index(doc_store, user_id)
        if not entity_pages:
            return {"orphans_found": 0, "assigned": 0, "skipped": 0,
                    "message": "no entity pages exist yet"}

        # 2. Get orphan memories (no compiled_from relation)
        orphans = await self._get_orphan_memories(doc_store, user_id, limit)
        if not orphans:
            return {"orphans_found": 0, "assigned": 0, "skipped": 0}

        logger.info(
            f"OrphanAssigner: processing {len(orphans)} orphans "
            f"against {len(entity_pages)} entity pages"
        )

        # 3. For each orphan, ask LLM which entity it belongs to
        assigned = 0
        skipped = 0
        for orphan in orphans:
            try:
                entity_id = await self._assign_orphan(orphan, entity_pages)
                if entity_id:
                    # Use 'assigned_to' (not 'compiled_from') so archival
                    # doesn't treat LLM-assigned orphans as truly compiled
                    await doc_store.add_relations(
                        entity_id,
                        [{"target_id": orphan["id"],
                          "relation_type": "assigned_to",
                          "similarity": 0.8}],
                    )
                    # Reverse relation (source → entity)
                    await doc_store.add_relations(
                        orphan["id"],
                        [{"target_id": entity_id,
                          "relation_type": "assigned_to",
                          "similarity": 0.8}],
                    )
                    assigned += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"Failed to assign orphan {orphan['id'][:8]}: {e}")
                skipped += 1

        result = {"orphans_found": len(orphans), "assigned": assigned, "skipped": skipped}
        logger.info(f"OrphanAssigner completed: {result}")
        return result

    async def _get_entity_index(
        self, doc_store, user_id: str
    ) -> list[dict]:
        """Get all entity pages with their titles and first ~200 chars of content."""
        pool = await doc_store._get_pool()
        user_uuid = UUID(user_id)

        rows = await pool.fetch(
            """
            SELECT id, title, LEFT(content, 200) AS summary
            FROM memory_documents
            WHERE user_id = $1
              AND deleted_at IS NULL
              AND category = 'entity-page'
            ORDER BY created_at DESC
            """,
            user_uuid,
        )

        return [
            {"id": str(row["id"]), "title": row["title"] or "", "summary": row["summary"] or ""}
            for row in rows
        ]

    async def _get_orphan_memories(
        self, doc_store, user_id: str, limit: int
    ) -> list[dict]:
        """Get memories with no compiled_from relation as target."""
        pool = await doc_store._get_pool()
        user_uuid = UUID(user_id)

        excluded = list(EXCLUDED_CATEGORIES)
        rows = await pool.fetch(
            """
            SELECT d.id, d.content, d.category, d.source_ref
            FROM memory_documents d
            WHERE d.user_id = $1
              AND d.deleted_at IS NULL
              AND d.category != ALL($2::text[])
              AND NOT EXISTS (
                  SELECT 1 FROM memory_relations r
                  WHERE r.target_id = d.id
                    AND r.relation_type IN ('compiled_from', 'assigned_to')
              )
            ORDER BY d.created_at ASC
            LIMIT $3
            """,
            user_uuid, excluded, limit,
        )

        return [
            {
                "id": str(row["id"]),
                "content": row["content"] or "",
                "category": row["category"] or "general",
                "source_ref": row["source_ref"],
            }
            for row in rows
        ]

    async def _assign_orphan(
        self, orphan: dict, entity_pages: list[dict]
    ) -> str | None:
        """Ask LLM which entity page an orphan memory belongs to.

        Returns entity page ID if assigned, None if no match.
        """
        from cems.llm.client import get_client

        client = get_client()
        if not client:
            return None

        # Build entity index for the prompt (cap at 100 to stay within context limits)
        capped_pages = entity_pages[:100]
        entity_index = "\n".join(
            f"- ID: {ep['id'][:8]} | Title: {ep['title']}"
            for ep in capped_pages
        )

        memory_content = (orphan["content"] or "")[:500]
        if not memory_content.strip():
            return None

        prompt = (
            f"Given this memory:\n"
            f'"{memory_content}"\n\n'
            f"Which of these knowledge topics does it belong to?\n"
            f"{entity_index}\n\n"
            f"Return ONLY the 8-character ID if it clearly belongs to one topic. "
            f'Return "none" if it doesn\'t fit any topic. '
            f"Return at most 1 ID. No explanation."
        )

        try:
            response = client.complete(
                prompt=prompt,
                system="You are a memory classification agent. Output only an ID or 'none'.",
                model="google/gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=50,
                fast_route=False,
            )
        except Exception as e:
            logger.warning(f"LLM assignment failed: {e}")
            return None

        if not response:
            return None

        answer = response.strip().lower()
        if answer == "none" or len(answer) < 8:
            return None

        # Match the 8-char prefix to a full entity ID
        for ep in entity_pages:
            if ep["id"].startswith(answer[:8]):
                return ep["id"]

        return None
