"""pgvector-based vector store for CEMS.

This module provides the PgVectorStore class which handles all vector
operations using PostgreSQL with pgvector extension. It replaces Qdrant
with a unified PostgreSQL-based solution that combines:
- Vector similarity search (HNSW index)
- Full-text search (GIN index on tsvector)
- Hybrid search using RRF (Reciprocal Rank Fusion)
- ACID transactions for data consistency
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

import asyncpg
from pgvector.asyncpg import register_vector

if TYPE_CHECKING:
    from cems.models import MemoryScope

logger = logging.getLogger(__name__)

# Default embedding dimension for OpenAI text-embedding-3-small
DEFAULT_EMBEDDING_DIM = 1536


class PgVectorStore:
    """PostgreSQL + pgvector based vector store.

    Provides unified storage for memories with:
    - Vector embeddings for semantic search
    - Full-text search for keyword matching
    - Hybrid search combining both approaches
    - ACID transactions for consistency

    Example:
        store = PgVectorStore(database_url="postgresql://...")
        await store.connect()

        memory_id = await store.add(
            content="User prefers Python for backend",
            embedding=[0.1, 0.2, ...],  # 1536 dimensions
            user_id="user123",
            scope="personal",
        )

        results = await store.search(
            query_embedding=[0.1, 0.2, ...],
            user_id="user123",
            limit=10,
        )
    """

    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int = 10,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        """Initialize the vector store.

        Args:
            database_url: PostgreSQL connection URL. Defaults to CEMS_DATABASE_URL env var.
            pool_size: Connection pool size.
            embedding_dim: Dimension of embeddings (default 1536 for OpenAI).
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
        expires_at: datetime | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Add a memory to the store.

        Args:
            content: Memory text content
            embedding: Vector embedding (1536 dimensions)
            user_id: User ID (UUID or string)
            team_id: Team ID (UUID or string)
            scope: Memory scope (personal, shared, team, company)
            category: Memory category
            tags: List of tags
            source: Source identifier
            source_ref: Source reference (e.g., project:org/repo)
            priority: Priority weight (default 1.0)
            pinned: Whether memory is pinned
            pin_reason: Reason for pinning
            pin_category: Pin category
            expires_at: Expiration timestamp
            created_at: Optional historical timestamp (for imports/evals)

        Returns:
            Memory ID as string (UUID)
        """
        pool = await self._get_pool()

        # Convert string UUIDs to UUID objects
        user_uuid = UUID(user_id) if isinstance(user_id, str) and user_id else None
        team_uuid = UUID(team_id) if isinstance(team_id, str) and team_id else None

        memory_id = uuid4()

        async with pool.acquire() as conn:
            if created_at:
                # Use provided timestamp for created_at only
                # Keep last_accessed as current time to avoid time decay penalties
                # This is important for eval/import scenarios where we want
                # historical timestamps but normal recency scoring
                await conn.execute(
                    """
                    INSERT INTO memories (
                        id, content, embedding, user_id, team_id, scope, category,
                        tags, source, source_ref, priority, pinned, pin_reason,
                        pin_category, expires_at, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $16
                    )
                    """,
                    memory_id,
                    content,
                    embedding,
                    user_uuid,
                    team_uuid,
                    scope,
                    category,
                    tags or [],
                    source,
                    source_ref,
                    priority,
                    pinned,
                    pin_reason,
                    pin_category,
                    expires_at,
                    created_at,  # Only for created_at and updated_at; last_accessed uses DB default (now)
                )
            else:
                # Use database defaults for current time
                await conn.execute(
                    """
                    INSERT INTO memories (
                        id, content, embedding, user_id, team_id, scope, category,
                        tags, source, source_ref, priority, pinned, pin_reason,
                        pin_category, expires_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                    )
                    """,
                    memory_id,
                    content,
                    embedding,
                    user_uuid,
                    team_uuid,
                    scope,
                    category,
                    tags or [],
                    source,
                    source_ref,
                    priority,
                    pinned,
                    pin_reason,
                    pin_category,
                    expires_at,
                )

        logger.debug(f"Added memory {memory_id}")
        return str(memory_id)

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID.

        Args:
            memory_id: Memory UUID as string

        Returns:
            Memory dict or None if not found
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, content, user_id, team_id, scope, category, tags,
                       source, source_ref, priority, pinned, pin_reason,
                       pin_category, archived, access_count, created_at,
                       updated_at, last_accessed, expires_at
                FROM memories
                WHERE id = $1
                """,
                UUID(memory_id),
            )

        if row is None:
            return None

        return self._row_to_dict(row)

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
        """Update a memory.

        Args:
            memory_id: Memory UUID as string
            content: New content (also updates embedding if provided)
            embedding: New embedding vector
            category: New category
            tags: New tags
            priority: New priority
            pinned: New pinned status
            pin_reason: New pin reason
            archived: New archived status

        Returns:
            True if updated, False if not found
        """
        pool = await self._get_pool()

        # Build dynamic update query
        updates = []
        values = []
        param_idx = 1

        if content is not None:
            updates.append(f"content = ${param_idx}")
            values.append(content)
            param_idx += 1

        if embedding is not None:
            updates.append(f"embedding = ${param_idx}")
            values.append(embedding)
            param_idx += 1

        if category is not None:
            updates.append(f"category = ${param_idx}")
            values.append(category)
            param_idx += 1

        if tags is not None:
            updates.append(f"tags = ${param_idx}")
            values.append(tags)
            param_idx += 1

        if priority is not None:
            updates.append(f"priority = ${param_idx}")
            values.append(priority)
            param_idx += 1

        if pinned is not None:
            updates.append(f"pinned = ${param_idx}")
            values.append(pinned)
            param_idx += 1

        if pin_reason is not None:
            updates.append(f"pin_reason = ${param_idx}")
            values.append(pin_reason)
            param_idx += 1

        if archived is not None:
            updates.append(f"archived = ${param_idx}")
            values.append(archived)
            param_idx += 1

        if not updates:
            return True  # Nothing to update

        values.append(UUID(memory_id))

        query = f"""
            UPDATE memories
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
        """

        async with pool.acquire() as conn:
            result = await conn.execute(query, *values)

        return result == "UPDATE 1"

    async def delete(self, memory_id: str, hard: bool = False) -> bool:
        """Delete or archive a memory.

        Args:
            memory_id: Memory UUID as string
            hard: If True, permanently delete. If False, archive.

        Returns:
            True if deleted/archived, False if not found
        """
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
        """Record an access to a memory (updates last_accessed and access_count)."""
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
        """Search for similar memories using vector similarity.

        Uses cosine distance for similarity scoring.

        Args:
            query_embedding: Query vector (1536 dimensions)
            user_id: Filter by user ID
            team_id: Filter by team ID
            scope: Filter by scope ("personal", "shared", "both")
            category: Filter by category
            limit: Maximum results
            include_archived: Include archived memories

        Returns:
            List of memory dicts with similarity scores
        """
        pool = await self._get_pool()

        # Build WHERE clauses
        conditions = []
        values = [query_embedding, limit]
        param_idx = 3

        if not include_archived:
            conditions.append("archived = FALSE")
            conditions.append("(expires_at IS NULL OR expires_at > NOW())")

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            values.append(UUID(user_id))
            param_idx += 1

        if team_id and scope in ("shared", "both"):
            conditions.append(f"team_id = ${param_idx}")
            values.append(UUID(team_id))
            param_idx += 1

        if scope != "both":
            conditions.append(f"scope = ${param_idx}")
            values.append(scope)
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            values.append(category)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT id, content, user_id, team_id, scope, category, tags,
                   source, source_ref, priority, pinned, pin_reason,
                   pin_category, archived, access_count, created_at,
                   updated_at, last_accessed, expires_at,
                   1 - (embedding <=> $1) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *values)

        return [self._row_to_dict(row, include_score=True) for row in rows]

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
        """Search memories using full-text search (BM25-style ranking).

        Args:
            query: Text search query
            user_id: Filter by user ID
            team_id: Filter by team ID
            scope: Filter by scope
            category: Filter by category
            limit: Maximum results
            include_archived: Include archived memories

        Returns:
            List of memory dicts with text rank scores
        """
        pool = await self._get_pool()

        # Build WHERE clauses
        conditions = ["content_tsv @@ plainto_tsquery('english', $1)"]
        values = [query, limit]
        param_idx = 3

        if not include_archived:
            conditions.append("archived = FALSE")
            conditions.append("(expires_at IS NULL OR expires_at > NOW())")

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            values.append(UUID(user_id))
            param_idx += 1

        if team_id and scope in ("shared", "both"):
            conditions.append(f"team_id = ${param_idx}")
            values.append(UUID(team_id))
            param_idx += 1

        if scope != "both":
            conditions.append(f"scope = ${param_idx}")
            values.append(scope)
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            values.append(category)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query_sql = f"""
            SELECT id, content, user_id, team_id, scope, category, tags,
                   source, source_ref, priority, pinned, pin_reason,
                   pin_category, archived, access_count, created_at,
                   updated_at, last_accessed, expires_at,
                   ts_rank(content_tsv, plainto_tsquery('english', $1)) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY score DESC
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *values)

        return [self._row_to_dict(row, include_score=True) for row in rows]

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
        """Hybrid search combining vector similarity and full-text search.

        Uses RRF (Reciprocal Rank Fusion) to combine results from both
        retrieval methods.

        Args:
            query: Text search query (for full-text search)
            query_embedding: Query vector (for vector search)
            user_id: Filter by user ID
            team_id: Filter by team ID
            scope: Filter by scope
            category: Filter by category
            limit: Maximum results
            vector_weight: Weight for vector results (0-1), text gets (1-vector_weight)
            include_archived: Include archived memories

        Returns:
            List of memory dicts with combined scores
        """
        pool = await self._get_pool()

        # Build WHERE clauses
        conditions = []
        values = [query_embedding, query, limit * 2]  # Fetch more for fusion
        param_idx = 4

        if not include_archived:
            conditions.append("archived = FALSE")
            conditions.append("(expires_at IS NULL OR expires_at > NOW())")

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            values.append(UUID(user_id))
            param_idx += 1

        if team_id and scope in ("shared", "both"):
            conditions.append(f"team_id = ${param_idx}")
            values.append(UUID(team_id))
            param_idx += 1

        if scope != "both":
            conditions.append(f"scope = ${param_idx}")
            values.append(scope)
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            values.append(category)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        # Combined query using RRF
        # We compute both vector and text scores, then combine with weights
        query_sql = f"""
            WITH vector_search AS (
                SELECT id, 1 - (embedding <=> $1) AS vector_score,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS vector_rank
                FROM memories
                WHERE {where_clause}
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
            SELECT m.id, m.content, m.user_id, m.team_id, m.scope, m.category, m.tags,
                   m.source, m.source_ref, m.priority, m.pinned, m.pin_reason,
                   m.pin_category, m.archived, m.access_count, m.created_at,
                   m.updated_at, m.last_accessed, m.expires_at,
                   -- RRF fusion: 1/(k+rank) with k=60 (standard)
                   COALESCE(1.0 / (60 + v.vector_rank), 0) * {vector_weight} +
                   COALESCE(1.0 / (60 + t.text_rank), 0) * {1 - vector_weight} AS score
            FROM memories m
            LEFT JOIN vector_search v ON m.id = v.id
            LEFT JOIN text_search t ON m.id = t.id
            WHERE v.id IS NOT NULL OR t.id IS NOT NULL
            ORDER BY score DESC
            LIMIT {limit}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *values)

        return [self._row_to_dict(row, include_score=True) for row in rows]

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def add_batch(
        self,
        memories: list[dict[str, Any]],
    ) -> list[str]:
        """Add multiple memories in a single transaction.

        Args:
            memories: List of memory dicts with keys:
                - content: str (required)
                - embedding: list[float] (required)
                - user_id, team_id, scope, category, tags, etc. (optional)

        Returns:
            List of memory IDs
        """
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
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                        )
                        """,
                        memory_id,
                        mem["content"],
                        mem["embedding"],
                        user_uuid,
                        team_uuid,
                        mem.get("scope", "personal"),
                        mem.get("category", "general"),
                        mem.get("tags", []),
                        mem.get("source"),
                        mem.get("source_ref"),
                        mem.get("priority", 1.0),
                        mem.get("pinned", False),
                        mem.get("pin_reason"),
                        mem.get("pin_category"),
                        mem.get("expires_at"),
                    )

        logger.info(f"Added {len(memory_ids)} memories in batch")
        return memory_ids

    async def get_batch(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get multiple memories by ID.

        Args:
            memory_ids: List of memory UUIDs as strings

        Returns:
            Dict mapping memory_id to memory dict
        """
        if not memory_ids:
            return {}

        pool = await self._get_pool()
        uuids = [UUID(mid) for mid in memory_ids]

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, user_id, team_id, scope, category, tags,
                       source, source_ref, priority, pinned, pin_reason,
                       pin_category, archived, access_count, created_at,
                       updated_at, last_accessed, expires_at
                FROM memories
                WHERE id = ANY($1)
                """,
                uuids,
            )

        return {str(row["id"]): self._row_to_dict(row) for row in rows}

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
        """Add a relation between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relation_type: Type of relation (e.g., "similar", "related")
            similarity: Similarity score (optional)
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_relations (source_id, target_id, relation_type, similarity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (source_id, target_id, relation_type) DO UPDATE
                SET similarity = EXCLUDED.similarity
                """,
                UUID(source_id),
                UUID(target_id),
                relation_type,
                similarity,
            )

    async def get_related_memories(
        self,
        memory_id: str,
        relation_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get memories related to a given memory.

        Args:
            memory_id: Memory ID to find relations for
            relation_type: Filter by relation type
            limit: Maximum results

        Returns:
            List of related memory dicts with relation info
        """
        pool = await self._get_pool()

        # Build query
        if relation_type:
            query = """
                SELECT m.*, r.relation_type, r.similarity AS relation_similarity
                FROM memory_relations r
                JOIN memories m ON r.target_id = m.id
                WHERE r.source_id = $1 AND r.relation_type = $2
                ORDER BY r.similarity DESC NULLS LAST
                LIMIT $3
            """
            values = [UUID(memory_id), relation_type, limit]
        else:
            query = """
                SELECT m.*, r.relation_type, r.similarity AS relation_similarity
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
            mem = self._row_to_dict(row)
            mem["relation_type"] = row["relation_type"]
            mem["relation_similarity"] = row["relation_similarity"]
            results.append(mem)

        return results

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    async def get_stale_memories(
        self,
        user_id: str,
        days: int = 90,
    ) -> list[str]:
        """Get memory IDs that haven't been accessed in N days.

        Args:
            user_id: User ID to filter by
            days: Days threshold

        Returns:
            List of stale memory IDs
        """
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
                UUID(user_id),
                days,
            )

        return [str(row["id"]) for row in rows]

    async def get_expired_memories(self) -> list[str]:
        """Get memory IDs that have expired.

        Returns:
            List of expired memory IDs
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM memories
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )

        return [str(row["id"]) for row in rows]

    async def delete_expired(self) -> int:
        """Delete all expired memories.

        Returns:
            Number of deleted memories
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM memories
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )

        # Parse "DELETE N" response
        count = int(result.split()[1]) if result else 0
        logger.info(f"Deleted {count} expired memories")
        return count

    async def get_category_counts(
        self,
        user_id: str,
        scope: str | None = None,
    ) -> dict[str, int]:
        """Get memory counts by category.

        Args:
            user_id: User ID to filter by
            scope: Optional scope filter

        Returns:
            Dict mapping category to count
        """
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
        """Get memories by category without semantic search.

        Directly queries by category field for fast, exact matching.
        Used for profile retrieval and gate rules.

        Args:
            user_id: User ID to filter by
            category: Category to match exactly
            limit: Maximum results
            include_archived: Include archived memories

        Returns:
            List of memory dicts
        """
        pool = await self._get_pool()

        conditions = ["user_id = $1", "category = $2"]
        values: list = [UUID(user_id), category, limit]

        if not include_archived:
            conditions.append("archived = FALSE")
            conditions.append("(expires_at IS NULL OR expires_at > NOW())")

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT id, content, user_id, team_id, scope, category, tags,
                   source, source_ref, priority, pinned, pin_reason,
                   pin_category, archived, access_count, created_at,
                   updated_at, last_accessed, expires_at
            FROM memories
            WHERE {where_clause}
            ORDER BY priority DESC, created_at DESC
            LIMIT $3
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *values)

        return [self._row_to_dict(row) for row in rows]

    async def get_recent(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 15,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get memories created within the last N hours.

        Used for session profile context injection.

        Args:
            user_id: User ID to filter by
            hours: Hours to look back (default 24)
            limit: Maximum results
            include_archived: Include archived memories

        Returns:
            List of memory dicts ordered by recency
        """
        pool = await self._get_pool()

        conditions = [
            "user_id = $1",
            "created_at > NOW() - INTERVAL '1 hour' * $2",
        ]
        values: list = [UUID(user_id), hours, limit]

        if not include_archived:
            conditions.append("archived = FALSE")
            conditions.append("(expires_at IS NULL OR expires_at > NOW())")

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT id, content, user_id, team_id, scope, category, tags,
                   source, source_ref, priority, pinned, pin_reason,
                   pin_category, archived, access_count, created_at,
                   updated_at, last_accessed, expires_at
            FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT $3
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *values)

        return [self._row_to_dict(row) for row in rows]

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_dict(
        self,
        row: asyncpg.Record,
        include_score: bool = False,
    ) -> dict[str, Any]:
        """Convert a database row to a dictionary.

        Args:
            row: Database row
            include_score: Include score field if present

        Returns:
            Memory dict
        """
        result = {
            "id": str(row["id"]),
            "content": row["content"],
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "team_id": str(row["team_id"]) if row["team_id"] else None,
            "scope": row["scope"],
            "category": row["category"],
            "tags": row["tags"],
            "source": row["source"],
            "source_ref": row["source_ref"],
            "priority": row["priority"],
            "pinned": row["pinned"],
            "pin_reason": row["pin_reason"],
            "pin_category": row.get("pin_category"),
            "archived": row["archived"],
            "access_count": row["access_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_accessed": row["last_accessed"],
            "expires_at": row["expires_at"],
        }

        if include_score and "score" in row.keys():
            result["score"] = row["score"]

        return result


# Module-level instance (lazy initialization)
_store: PgVectorStore | None = None


def get_vectorstore() -> PgVectorStore:
    """Get the shared PgVectorStore instance.

    Returns:
        PgVectorStore instance

    Raises:
        ValueError: If CEMS_DATABASE_URL is not set
    """
    global _store
    if _store is None:
        _store = PgVectorStore()
    return _store
