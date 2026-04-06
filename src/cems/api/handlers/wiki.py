"""Wiki API handlers for the Knowledge Engine dashboard.

Provides endpoints for browsing the relation graph, viewing memory clusters,
and knowledge health stats.
"""

import logging
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def _safe_int(val, default: int, max_val: int = 1000) -> int:
    try:
        return min(int(val), max_val)
    except (TypeError, ValueError):
        return default


def get_memory():
    """Get the shared CEMSMemory instance."""
    from cems.api.handlers.memory import get_memory as _get_memory
    return _get_memory()


async def api_wiki_graph(request: Request):
    """Return nodes and edges for the relation graph visualization.

    GET /api/wiki/graph?limit=200

    Returns nodes (documents with metadata) and edges (relations with similarity).
    """
    try:
        limit = _safe_int(request.query_params.get("limit"), 200, max_val=500)

        memory = get_memory()
        doc_store = await memory._ensure_document_store()
        user_id = memory.config.user_id
        pool = await doc_store._get_pool()

        # Get all relations for this user's documents
        async with pool.acquire() as conn:
            edges_rows = await conn.fetch(
                """
                SELECT r.source_id, r.target_id, r.relation_type, r.similarity
                FROM memory_relations r
                JOIN memory_documents s ON r.source_id = s.id
                JOIN memory_documents t ON r.target_id = t.id
                WHERE s.user_id = $1 AND s.deleted_at IS NULL AND t.deleted_at IS NULL
                ORDER BY r.similarity DESC NULLS LAST
                LIMIT $2
                """,
                UUID(user_id),
                limit * 5,  # More edges than nodes to show connectivity
            )

        # Collect unique document IDs from edges
        doc_ids = set()
        for row in edges_rows:
            doc_ids.add(str(row["source_id"]))
            doc_ids.add(str(row["target_id"]))

        # Fetch document metadata for all nodes
        nodes = []
        if doc_ids:
            docs_list = list(doc_ids)[:limit]
            for doc_id in docs_list:
                try:
                    doc = await doc_store.get_document(doc_id)
                    if doc:
                        content = doc.get("content", "")
                        nodes.append({
                            "id": doc_id,
                            "title": doc.get("title") or content[:80],
                            "category": doc.get("category", "general"),
                            "source": doc.get("source"),
                            "source_ref": doc.get("source_ref"),
                            "shown_count": doc.get("shown_count", 0),
                            "relevant_count": doc.get("relevant_count", 0),
                            "created_at": str(doc.get("created_at", "")),
                        })
                except Exception:
                    pass

        # Format edges
        edges = []
        node_ids = {n["id"] for n in nodes}
        for row in edges_rows:
            src = str(row["source_id"])
            tgt = str(row["target_id"])
            if src in node_ids and tgt in node_ids:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "type": row["relation_type"],
                    "similarity": row["similarity"],
                })

        return JSONResponse({
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        })
    except Exception as e:
        logger.error(f"API wiki_graph error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_wiki_stats(request: Request):
    """Return knowledge health stats for the dashboard.

    GET /api/wiki/stats

    Returns: total memories, relation count, avg relations per doc,
    orphan count, category distribution, heat distribution.
    """
    try:
        memory = get_memory()
        doc_store = await memory._ensure_document_store()
        user_id = memory.config.user_id
        pool = await doc_store._get_pool()

        async with pool.acquire() as conn:
            user_uuid = UUID(user_id)

            # Total active documents
            total_docs = await conn.fetchval(
                "SELECT COUNT(*) FROM memory_documents WHERE user_id = $1 AND deleted_at IS NULL",
                user_uuid,
            )

            # Total relations
            total_relations = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_relations r
                JOIN memory_documents d ON r.source_id = d.id
                WHERE d.user_id = $1 AND d.deleted_at IS NULL
                """,
                user_uuid,
            )

            # Documents with at least one relation (connected)
            connected_docs = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT d.id) FROM memory_documents d
                JOIN memory_relations r ON (r.source_id = d.id OR r.target_id = d.id)
                WHERE d.user_id = $1 AND d.deleted_at IS NULL
                """,
                user_uuid,
            )

            # Category distribution
            cat_rows = await conn.fetch(
                """
                SELECT category, COUNT(*) as cnt
                FROM memory_documents
                WHERE user_id = $1 AND deleted_at IS NULL
                GROUP BY category
                ORDER BY cnt DESC
                """,
                user_uuid,
            )

            # Heat distribution (by shown_count tiers)
            heat_rows = await conn.fetch(
                """
                SELECT
                    CASE
                        WHEN COALESCE(shown_count, 0) >= 20 THEN 'hot'
                        WHEN COALESCE(shown_count, 0) >= 5 THEN 'warm'
                        WHEN COALESCE(shown_count, 0) >= 1 THEN 'cool'
                        ELSE 'cold'
                    END AS tier,
                    COUNT(*) as cnt
                FROM memory_documents
                WHERE user_id = $1 AND deleted_at IS NULL
                GROUP BY tier
                ORDER BY cnt DESC
                """,
                user_uuid,
            )

            # Open conflicts count
            conflicts_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_conflicts
                WHERE user_id = $1 AND status = 'open'
                """,
                user_uuid,
            )

        orphan_count = (total_docs or 0) - (connected_docs or 0)
        avg_relations = round((total_relations or 0) / max(total_docs or 1, 1), 1)

        # Health score: 100 - penalties
        health = 100
        health -= min((conflicts_count or 0) * 5, 25)  # Contradictions: -5 each, max -25
        health -= min(orphan_count * 0.5, 20)  # Orphans: -0.5 each, max -20
        health = max(health, 0)

        return JSONResponse({
            "success": True,
            "stats": {
                "total_memories": total_docs or 0,
                "total_relations": total_relations or 0,
                "connected_memories": connected_docs or 0,
                "orphan_memories": orphan_count,
                "avg_relations_per_doc": avg_relations,
                "open_conflicts": conflicts_count or 0,
                "health_score": round(health),
                "categories": {row["category"]: row["cnt"] for row in cat_rows},
                "heat_tiers": {row["tier"]: row["cnt"] for row in heat_rows},
            },
        })
    except Exception as e:
        logger.error(f"API wiki_stats error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_wiki_memory_relations(request: Request):
    """Get relations for a specific memory.

    GET /api/wiki/relations?id=<memory_id>

    Returns the memory's content and all related memories.
    """
    try:
        memory_id = request.query_params.get("id")
        if not memory_id:
            return JSONResponse({"error": "id parameter required"}, status_code=400)

        memory = get_memory()
        doc_store = await memory._ensure_document_store()

        # Get the document
        doc = await doc_store.get_document(memory_id)
        if not doc:
            return JSONResponse({"error": "Memory not found"}, status_code=404)

        # Get related documents
        related = await doc_store.get_related_documents(
            memory_id, limit=20, user_id=memory.config.user_id,
        )

        return JSONResponse({
            "success": True,
            "memory": {
                "id": memory_id,
                "content": doc.get("content", ""),
                "category": doc.get("category"),
                "source_ref": doc.get("source_ref"),
                "shown_count": doc.get("shown_count", 0),
                "created_at": str(doc.get("created_at", "")),
            },
            "relations": [
                {
                    "id": str(r["id"]),
                    "content": (r.get("content", "") or "")[:200],
                    "category": r.get("category"),
                    "relation_type": r.get("relation_type"),
                    "similarity": r.get("relation_similarity"),
                }
                for r in related
            ],
        })
    except Exception as e:
        logger.error(f"API wiki_memory_relations error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)
