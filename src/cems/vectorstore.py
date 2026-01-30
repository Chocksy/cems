"""pgvector-based vector store for CEMS.

This module provides the PgVectorStore class which handles all vector
operations using PostgreSQL with pgvector extension. It replaces Qdrant
with a unified PostgreSQL-based solution that combines:
- Vector similarity search (HNSW index)
- Full-text search (GIN index on tsvector)
- Hybrid search using RRF (Reciprocal Rank Fusion)
- ACID transactions for data consistency

Strategy A: Single embedding column at configured dimension (768-dim for llama.cpp).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

import asyncpg
from pgvector.asyncpg import register_vector

from cems.db import MEMORY_COLUMNS, MEMORY_COLUMNS_PREFIXED, FilterBuilder, row_to_dict
from cems.db.constants import DEFAULT_EMBEDDING_DIM

if TYPE_CHECKING:
    from cems.models import MemoryScope

logger = logging.getLogger(__name__)


class PgVectorStore:
    """PostgreSQL + pgvector based vector store.

    Provides unified storage for memories with:
    - Vector embeddings for semantic search
    - Full-text search for keyword matching
    - Hybrid search combining both approaches
    - ACID transactions for consistency

    Uses a single 'embedding' column. Dimension is configurable (default: 768 for llama.cpp).
    """

    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int = 10,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        """Initialize the vector store.

        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            embedding_dim: Embedding dimension (768 for llama.cpp, 1536 for OpenRouter)
        """
        self.database_url = database_url or os.getenv("CEMS_DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "Database URL required. Set CEMS_DATABASE_URL environment variable "
                "or pass database_url parameter."
            )

        self.pool_size = pool_size
        self.embedding_dim = embedding_dim
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=self.pool_size,
            setup=self._setup_connection,
        )
        logger.info("PgVectorStore connected to database")

    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Setup each connection with pgvector extension."""
        await register_vector(conn)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PgVectorStore disconnected")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get the connection pool, connecting if needed."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def add(
        self,
        content: str,
        embedding: list[float],
        user_id: str | UUID | None = None,
        team_id: str | UUID | None = None,
        scope: str = "personal",
        category: str = "general",
        tags: list[str] | None = None,
        source: str | None = None,
        source_ref: str | None = None,
        priority: float = 1.0,
        pinned: bool = False,
        pin_reason: str | None = None,
        pin_category: str | None = None,
        expires_at=None,
        created_at=None,
    ) -> str:
        """Add a memory to the store.

        Args:
            content: Memory content
            embedding: Embedding vector (dimension must match configured embedding_dim)
            user_id: User ID
            team_id: Team ID (for shared memories)
            scope: "personal" or "shared"
            category: Memory category
            tags: Optional tags
            source: Source identifier
            source_ref: Source reference (e.g., project:org/repo)
            priority: Priority score
            pinned: Whether memory is pinned
            pin_reason: Reason for pinning
            pin_category: Category for pinned memory
            expires_at: Optional expiration datetime
            created_at: Optional creation datetime (for imports)

        Returns:
            Memory ID as string
        """
        pool = await self._get_pool()

        user_uuid = UUID(user_id) if isinstance(user_id, str) and user_id else None
        team_uuid = UUID(team_id) if isinstance(team_id, str) and team_id else None
        memory_id = uuid4()

        async with pool.acquire() as conn:
            if created_at:
                await conn.execute(
                    """
                    INSERT INTO memories (
                        id, content, embedding, user_id, team_id, scope, category,
                        tags, source, source_ref, priority, pinned, pin_reason,
                        pin_category, expires_at, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $16)
                    """,
                    memory_id, content, embedding, user_uuid, team_uuid, scope, category,
                    tags or [], source, source_ref, priority, pinned, pin_reason,
                    pin_category, expires_at, created_at,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO memories (
                        id, content, embedding, user_id, team_id, scope, category,
                        tags, source, source_ref, priority, pinned, pin_reason,
                        pin_category, expires_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    memory_id, content, embedding, user_uuid, team_uuid, scope, category,
                    tags or [], source, source_ref, priority, pinned, pin_reason,
                    pin_category, expires_at,
                )

        logger.debug(f"Added memory {memory_id}")
        return str(memory_id)

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {MEMORY_COLUMNS} FROM memories WHERE id = $1",
                UUID(memory_id),
            )

        return row_to_dict(row) if row else None

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        embedding: list[float] | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        priority: float | None = None,
        pinned: bool | None = None,
        pin_reason: str | None = None,
        archived: bool | None = None,
    ) -> bool:
        """Update a memory."""
        pool = await self._get_pool()

        updates = []
        values = []
        param_idx = 1

        for field, value in [
            ("content", content),
            ("embedding", embedding),
            ("category", category),
            ("tags", tags),
            ("priority", priority),
            ("pinned", pinned),
            ("pin_reason", pin_reason),
            ("archived", archived),
        ]:
            if value is not None:
                updates.append(f"{field} = ${param_idx}")
                values.append(value)
                param_idx += 1

        if not updates:
            return True

        values.append(UUID(memory_id))
        query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ${param_idx}"

        async with pool.acquire() as conn:
            result = await conn.execute(query, *values)

        return result == "UPDATE 1"

    async def delete(self, memory_id: str, hard: bool = False) -> bool:
        """Delete or archive a memory."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if hard:
                result = await conn.execute(
                    "DELETE FROM memories WHERE id = $1",
                    UUID(memory_id),
                )
                return result == "DELETE 1"
            else:
                result = await conn.execute(
                    "UPDATE memories SET archived = TRUE WHERE id = $1",
                    UUID(memory_id),
                )
                return result == "UPDATE 1"

    async def record_access(self, memory_id: str) -> None:
        """Record an access to a memory."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memories
                SET last_accessed = NOW(), access_count = access_count + 1
                WHERE id = $1
                """,
                UUID(memory_id),
            )

    async def record_access_batch(self, memory_ids: list[str]) -> None:
        """Record access to multiple memories in a single query."""
        if not memory_ids:
            return

        pool = await self._get_pool()
        uuids = [UUID(mid) for mid in memory_ids]

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memories
                SET last_accessed = NOW(), access_count = access_count + 1
                WHERE id = ANY($1)
                """,
                uuids,
            )

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query_embedding: list[float],
        user_id: str | None = None,
        team_id: str | None = None,
        scope: str | Literal["both"] = "both",
        category: str | None = None,
        limit: int = 10,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for similar memories using vector similarity."""
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=3)
        if not include_archived:
            fb.add_not_archived()
        fb.add_scope_filter(scope, user_id, team_id)
        fb.add_if(category, "category = ${}", category)
        # Only search memories that have embeddings
        fb.add("embedding IS NOT NULL")

        query = f"""
            SELECT {MEMORY_COLUMNS}, 1 - (embedding <=> $1) AS score
            FROM memories
            WHERE {fb.build()}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, query_embedding, limit, *fb.values)

        return [row_to_dict(row, include_score=True) for row in rows]

    async def full_text_search(
        self,
        query: str,
        user_id: str | None = None,
        team_id: str | None = None,
        scope: str | Literal["both"] = "both",
        category: str | None = None,
        limit: int = 10,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Search memories using full-text search (BM25-style ranking)."""
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=3)
        fb.add("content_tsv @@ plainto_tsquery('english', $1)")
        if not include_archived:
            fb.add_not_archived()
        fb.add_scope_filter(scope, user_id, team_id)
        fb.add_if(category, "category = ${}", category)

        query_sql = f"""
            SELECT {MEMORY_COLUMNS},
                   ts_rank(content_tsv, plainto_tsquery('english', $1)) AS score
            FROM memories
            WHERE {fb.build()}
            ORDER BY score DESC
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, query, limit, *fb.values)

        return [row_to_dict(row, include_score=True) for row in rows]

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        team_id: str | None = None,
        scope: str | Literal["both"] = "both",
        category: str | None = None,
        limit: int = 10,
        vector_weight: float = 0.7,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and full-text search using RRF."""
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=4)
        if not include_archived:
            fb.add_not_archived()
        fb.add_scope_filter(scope, user_id, team_id)
        fb.add_if(category, "category = ${}", category)

        where_clause = fb.build()
        # Add embedding null check for vector search CTE
        vector_where = f"{where_clause} AND embedding IS NOT NULL"
        text_weight = 1 - vector_weight

        query_sql = f"""
            WITH vector_search AS (
                SELECT id, 1 - (embedding <=> $1) AS vector_score,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS vector_rank
                FROM memories
                WHERE {vector_where}
                ORDER BY embedding <=> $1
                LIMIT $3
            ),
            text_search AS (
                SELECT id, ts_rank(content_tsv, plainto_tsquery('english', $2)) AS text_score,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $2)) DESC) AS text_rank
                FROM memories
                WHERE {where_clause}
                  AND content_tsv @@ plainto_tsquery('english', $2)
                ORDER BY text_score DESC
                LIMIT $3
            )
            SELECT {MEMORY_COLUMNS_PREFIXED},
                   COALESCE(1.0 / (60 + v.vector_rank), 0) * {vector_weight} +
                   COALESCE(1.0 / (60 + t.text_rank), 0) * {text_weight} AS score
            FROM memories m
            LEFT JOIN vector_search v ON m.id = v.id
            LEFT JOIN text_search t ON m.id = t.id
            WHERE v.id IS NOT NULL OR t.id IS NOT NULL
            ORDER BY score DESC
            LIMIT {limit}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, query_embedding, query, limit * 2, *fb.values)

        return [row_to_dict(row, include_score=True) for row in rows]

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def add_batch(self, memories: list[dict[str, Any]]) -> list[str]:
        """Add multiple memories in a single transaction."""
        pool = await self._get_pool()
        memory_ids = []

        async with pool.acquire() as conn:
            async with conn.transaction():
                for mem in memories:
                    memory_id = uuid4()
                    memory_ids.append(str(memory_id))

                    user_uuid = UUID(mem.get("user_id")) if mem.get("user_id") else None
                    team_uuid = UUID(mem.get("team_id")) if mem.get("team_id") else None

                    await conn.execute(
                        """
                        INSERT INTO memories (
                            id, content, embedding, user_id, team_id, scope, category,
                            tags, source, source_ref, priority, pinned, pin_reason,
                            pin_category, expires_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        """,
                        memory_id, mem["content"], mem["embedding"], user_uuid, team_uuid,
                        mem.get("scope", "personal"), mem.get("category", "general"),
                        mem.get("tags", []), mem.get("source"), mem.get("source_ref"),
                        mem.get("priority", 1.0), mem.get("pinned", False),
                        mem.get("pin_reason"), mem.get("pin_category"), mem.get("expires_at"),
                    )

        logger.info(f"Added {len(memory_ids)} memories in batch")
        return memory_ids

    async def get_batch(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get multiple memories by ID."""
        if not memory_ids:
            return {}

        pool = await self._get_pool()
        uuids = [UUID(mid) for mid in memory_ids]

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {MEMORY_COLUMNS} FROM memories WHERE id = ANY($1)",
                uuids,
            )

        return {str(row["id"]): row_to_dict(row) for row in rows}

    # =========================================================================
    # Memory Relations
    # =========================================================================

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "similar",
        similarity: float | None = None,
    ) -> None:
        """Add a relation between two memories."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_relations (source_id, target_id, relation_type, similarity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (source_id, target_id, relation_type) DO UPDATE
                SET similarity = EXCLUDED.similarity
                """,
                UUID(source_id), UUID(target_id), relation_type, similarity,
            )

    async def get_related_memories(
        self,
        memory_id: str,
        relation_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get memories related to a given memory."""
        pool = await self._get_pool()

        if relation_type:
            query = f"""
                SELECT {MEMORY_COLUMNS_PREFIXED}, r.relation_type, r.similarity AS relation_similarity
                FROM memory_relations r
                JOIN memories m ON r.target_id = m.id
                WHERE r.source_id = $1 AND r.relation_type = $2
                ORDER BY r.similarity DESC NULLS LAST
                LIMIT $3
            """
            values = [UUID(memory_id), relation_type, limit]
        else:
            query = f"""
                SELECT {MEMORY_COLUMNS_PREFIXED}, r.relation_type, r.similarity AS relation_similarity
                FROM memory_relations r
                JOIN memories m ON r.target_id = m.id
                WHERE r.source_id = $1
                ORDER BY r.similarity DESC NULLS LAST
                LIMIT $2
            """
            values = [UUID(memory_id), limit]

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *values)

        results = []
        for row in rows:
            mem = row_to_dict(row)
            mem["relation_type"] = row["relation_type"]
            mem["relation_similarity"] = row["relation_similarity"]
            results.append(mem)

        return results

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    async def get_stale_memories(self, user_id: str, days: int = 90) -> list[str]:
        """Get memory IDs that haven't been accessed in N days."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM memories
                WHERE user_id = $1
                  AND archived = FALSE
                  AND pinned = FALSE
                  AND last_accessed < NOW() - INTERVAL '1 day' * $2
                """,
                UUID(user_id), days,
            )

        return [str(row["id"]) for row in rows]

    async def get_expired_memories(self) -> list[str]:
        """Get memory IDs that have expired."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < NOW()"
            )

        return [str(row["id"]) for row in rows]

    async def delete_expired(self) -> int:
        """Delete all expired memories."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < NOW()"
            )

        count = int(result.split()[1]) if result else 0
        logger.info(f"Deleted {count} expired memories")
        return count

    async def get_category_counts(
        self,
        user_id: str,
        scope: str | None = None,
    ) -> dict[str, int]:
        """Get memory counts by category."""
        pool = await self._get_pool()

        if scope:
            query = """
                SELECT category, COUNT(*) as count
                FROM memories
                WHERE user_id = $1 AND scope = $2 AND archived = FALSE
                GROUP BY category
            """
            values = [UUID(user_id), scope]
        else:
            query = """
                SELECT category, COUNT(*) as count
                FROM memories
                WHERE user_id = $1 AND archived = FALSE
                GROUP BY category
            """
            values = [UUID(user_id)]

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *values)

        return {row["category"]: row["count"] for row in rows}

    async def search_by_category(
        self,
        user_id: str,
        category: str,
        limit: int = 10,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get memories by category without semantic search."""
        pool = await self._get_pool()

        fb = FilterBuilder()
        fb.add_param("user_id = ${}", UUID(user_id))
        fb.add_param("category = ${}", category)
        if not include_archived:
            fb.add_not_archived()

        query = f"""
            SELECT {MEMORY_COLUMNS}
            FROM memories
            WHERE {fb.build()}
            ORDER BY priority DESC, created_at DESC
            LIMIT ${fb.next_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *fb.values, limit)

        return [row_to_dict(row) for row in rows]

    async def get_recent(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 15,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get memories created within the last N hours."""
        pool = await self._get_pool()

        fb = FilterBuilder()
        fb.add_param("user_id = ${}", UUID(user_id))
        fb.add_param("created_at > NOW() - INTERVAL '1 hour' * ${}", hours)
        if not include_archived:
            fb.add_not_archived()

        query = f"""
            SELECT {MEMORY_COLUMNS}
            FROM memories
            WHERE {fb.build()}
            ORDER BY created_at DESC
            LIMIT ${fb.next_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *fb.values, limit)

        return [row_to_dict(row) for row in rows]

    # =========================================================================
    # Legacy compatibility
    # =========================================================================

    def _row_to_dict(
        self,
        row: asyncpg.Record,
        include_score: bool = False,
    ) -> dict[str, Any]:
        """Convert a database row to a dictionary (legacy method)."""
        return row_to_dict(row, include_score)


# Module-level instance (lazy initialization)
_store: PgVectorStore | None = None


def get_vectorstore() -> PgVectorStore:
    """Get the shared PgVectorStore instance."""
    global _store
    if _store is None:
        _store = PgVectorStore()
    return _store
