"""Document store for CEMS with chunked content.

This module provides the DocumentStore class which handles:
- Document storage with deduplication by content hash
- Chunked content with embeddings for search
- Vector and hybrid search on chunks
- Document-level metadata management

The document+chunk model replaces the flat memory model to:
- Avoid truncation issues with limited-context embedding models
- Provide better search recall with chunk-level matching
- Enable deduplication by content hash
"""

from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import UUID, uuid4

import asyncpg
from pgvector.asyncpg import register_vector

from cems.chunking import Chunk, content_hash
from cems.db.filter_builder import FilterBuilder

logger = logging.getLogger(__name__)

# Column definitions for documents
DOCUMENT_COLUMNS = """
    id, user_id, team_id, scope, category, title, source, source_ref,
    tags, content, content_hash, content_bytes, created_at, updated_at,
    deleted_at, shown_count, last_shown_at
"""

# Column definitions for chunks (with document join)
CHUNK_COLUMNS = """
    c.id AS chunk_id, c.document_id, c.seq, c.pos, c.content AS chunk_content,
    c.tokens, c.bytes, c.created_at AS chunk_created_at
"""

CHUNK_WITH_DOC_COLUMNS = f"""
    {CHUNK_COLUMNS},
    d.user_id, d.team_id, d.scope, d.category, d.title, d.source, d.source_ref,
    d.tags, d.created_at AS document_created_at
"""


def chunk_row_to_result(row: asyncpg.Record, include_score: bool = False) -> dict[str, Any]:
    """Convert a chunk search row to a dictionary."""
    result = {
        "chunk_id": str(row["chunk_id"]),
        "document_id": str(row["document_id"]),
        "seq": row["seq"],
        "pos": row["pos"],
        "content": row["chunk_content"],  # Use chunk content as primary content
        "chunk_content": row["chunk_content"],
        "tokens": row["tokens"],
        "bytes": row["bytes"],
        # Document metadata
        "user_id": str(row["user_id"]) if row["user_id"] else None,
        "team_id": str(row["team_id"]) if row["team_id"] else None,
        "scope": row["scope"],
        "category": row["category"],
        "title": row["title"],
        "source": row["source"],
        "source_ref": row["source_ref"],
        "tags": row["tags"] or [],
        "created_at": row["document_created_at"],
    }
    if include_score and "score" in row.keys():
        result["score"] = row["score"]
    return result


class DocumentStore:
    """PostgreSQL store for documents and chunks.

    Provides:
    - Document storage with deduplication
    - Chunk-based vector search
    - Hybrid search combining vector and full-text
    - Document-level CRUD operations
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
    ):
        """Initialize the document store.

        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
        """
        self.database_url = database_url
        self.pool_size = pool_size
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
        logger.info("DocumentStore connected to database")

    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Setup each connection with pgvector extension."""
        await register_vector(conn)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("DocumentStore disconnected")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get the connection pool, connecting if needed."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def add_document(
        self,
        content: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        user_id: str | UUID,
        team_id: str | UUID | None = None,
        scope: str = "personal",
        category: str = "document",
        title: str | None = None,
        source: str | None = None,
        source_ref: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[str, bool]:
        """Add a document with its chunks.

        Args:
            content: Full document content
            chunks: Pre-chunked content
            embeddings: Embedding for each chunk (must match chunks length)
            user_id: User ID
            team_id: Team ID (for shared documents)
            scope: "personal" or "shared"
            category: Document category
            title: Optional document title
            source: Source identifier
            source_ref: Source reference (e.g., project:org/repo)
            tags: Optional tags

        Returns:
            Tuple of (document_id, is_new). is_new is False if document was deduplicated.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have same length: {len(chunks)} vs {len(embeddings)}"
            )

        pool = await self._get_pool()
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
        team_uuid = UUID(team_id) if isinstance(team_id, str) and team_id else None
        doc_hash = content_hash(content)
        doc_bytes = len(content.encode("utf-8"))

        async with pool.acquire() as conn:
            # All dedup checks + insert run inside a single transaction to
            # prevent TOCTOU races (two concurrent adds both passing dedup).
            doc_id = uuid4()
            try:
                async with conn.transaction():
                    # Check for existing document with same hash (with row lock)
                    existing = await conn.fetchrow(
                        """
                        SELECT id FROM memory_documents
                        WHERE content_hash = $1 AND user_id = $2 AND deleted_at IS NULL
                        FOR UPDATE
                        """,
                        doc_hash,
                        user_uuid,
                    )

                    if existing:
                        logger.debug(f"Document already exists with hash {doc_hash[:8]}...")
                        return str(existing["id"]), False

                    # Semantic dedup: check if very similar content already exists
                    # Uses first chunk's embedding to find near-duplicates (cosine > 0.92)
                    if embeddings:
                        semantic_dup = await conn.fetchrow(
                            """
                            SELECT d.id FROM memory_chunks c
                            JOIN memory_documents d ON c.document_id = d.id
                            WHERE d.user_id = $1 AND d.deleted_at IS NULL AND c.seq = 0
                              AND 1 - (c.embedding <=> $2) > 0.92
                            LIMIT 1
                            """,
                            user_uuid,
                            embeddings[0],
                        )
                        if semantic_dup:
                            logger.debug(f"Semantic duplicate found (>{0.92} similarity)")
                            return str(semantic_dup["id"]), False

                    # Insert new document
                    await conn.execute(
                        """
                        INSERT INTO memory_documents (
                            id, user_id, team_id, scope, category, title,
                            source, source_ref, tags, content, content_hash, content_bytes
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        """,
                        doc_id,
                        user_uuid,
                        team_uuid,
                        scope,
                        category,
                        title,
                        source,
                        source_ref,
                        tags or [],
                        content,
                        doc_hash,
                        doc_bytes,
                    )

                    # Insert chunks in a single batch call
                    chunk_rows = [
                        (uuid4(), doc_id, chunk.seq, chunk.pos,
                         chunk.content, embedding, chunk.tokens, chunk.bytes)
                        for chunk, embedding in zip(chunks, embeddings)
                    ]
                    await conn.executemany(
                        """
                        INSERT INTO memory_chunks (
                            id, document_id, seq, pos, content, embedding, tokens, bytes
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        chunk_rows,
                    )
            except asyncpg.UniqueViolationError:
                # Concurrent insert won the race — fetch the existing doc
                existing = await conn.fetchrow(
                    """
                    SELECT id FROM memory_documents
                    WHERE content_hash = $1 AND user_id = $2 AND deleted_at IS NULL
                    """,
                    doc_hash,
                    user_uuid,
                )
                if existing:
                    logger.debug(f"Concurrent duplicate detected for hash {doc_hash[:8]}...")
                    return str(existing["id"]), False
                raise  # Unexpected — re-raise if we can't find it

        logger.debug(f"Added document {doc_id} with {len(chunks)} chunks")
        return str(doc_id), True

    def _doc_row_to_dict(self, row: asyncpg.Record) -> dict[str, Any]:
        """Convert a document row to a dictionary."""
        return {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "team_id": str(row["team_id"]) if row["team_id"] else None,
            "scope": row["scope"],
            "category": row["category"],
            "title": row["title"],
            "source": row["source"],
            "source_ref": row["source_ref"],
            "tags": row["tags"] or [],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "content_bytes": row["content_bytes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted_at": row["deleted_at"],
            "shown_count": row["shown_count"],
            "last_shown_at": row["last_shown_at"],
        }

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get a document by ID (excludes soft-deleted)."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {DOCUMENT_COLUMNS} FROM memory_documents WHERE id = $1 AND deleted_at IS NULL",
                UUID(document_id),
            )

        if not row:
            return None

        return self._doc_row_to_dict(row)

    async def delete_document(self, document_id: str, hard: bool = False) -> bool:
        """Delete a document (soft by default, hard if specified).

        Soft delete sets deleted_at timestamp; hard delete removes permanently.
        Chunks are cascade-deleted on hard delete, hidden by JOIN filter on soft.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if hard:
                result = await conn.execute(
                    "DELETE FROM memory_documents WHERE id = $1",
                    UUID(document_id),
                )
                return result == "DELETE 1"
            else:
                result = await conn.execute(
                    "UPDATE memory_documents SET deleted_at = NOW() WHERE id = $1 AND deleted_at IS NULL",
                    UUID(document_id),
                )
                return result == "UPDATE 1"

    async def delete_by_source_ref(self, source_ref: str, user_id: str) -> int:
        """Hard-delete all documents with a given source_ref for a user.

        Intentionally uses hard delete (not soft-delete) because this is
        used for eval cleanup where we need complete removal of test data,
        including chunks (via CASCADE).
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memory_documents WHERE source_ref = $1 AND user_id = $2",
                source_ref,
                UUID(user_id),
            )

        count = int(result.split()[1]) if result else 0
        logger.debug(f"Deleted {count} documents with source_ref={source_ref}")
        return count

    async def delete_by_tag(self, tag: str, user_id: str) -> int:
        """Hard-delete all documents containing a specific tag for a user.

        Intentionally uses hard delete (not soft-delete) because this is
        used for eval cleanup where we need complete removal of test data,
        including chunks (via CASCADE).
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memory_documents WHERE $1 = ANY(tags) AND user_id = $2",
                tag,
                UUID(user_id),
            )

        count = int(result.split()[1]) if result else 0
        logger.debug(f"Deleted {count} documents with tag={tag}")
        return count

    async def find_document_by_tag(
        self,
        tag: str,
        user_id: str,
        category: str | None = None,
    ) -> dict[str, Any] | None:
        """Find a single document containing a specific tag.

        Returns the most recently updated document matching the tag.
        Used for session summary upsert (find existing by session tag).

        Args:
            tag: Tag to search for (e.g., "session:b40eb706")
            user_id: User ID
            category: Optional category filter

        Returns:
            Document dict or None if not found
        """
        pool = await self._get_pool()
        user_uuid = UUID(user_id)

        if category:
            query = f"""
                SELECT {DOCUMENT_COLUMNS} FROM memory_documents
                WHERE $1 = ANY(tags) AND user_id = $2 AND category = $3
                  AND deleted_at IS NULL
                ORDER BY updated_at DESC NULLS LAST
                LIMIT 1
            """
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, tag, user_uuid, category)
        else:
            query = f"""
                SELECT {DOCUMENT_COLUMNS} FROM memory_documents
                WHERE $1 = ANY(tags) AND user_id = $2
                  AND deleted_at IS NULL
                ORDER BY updated_at DESC NULLS LAST
                LIMIT 1
            """
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, tag, user_uuid)

        if not row:
            return None
        return self._doc_row_to_dict(row)

    async def update_document(
        self,
        document_id: str,
        content: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> bool:
        """Update a document's content and replace its chunks.

        Replaces content, hash, bytes, and all chunks in a single transaction.

        Args:
            document_id: The document ID to update
            content: New full content
            chunks: New pre-chunked content
            embeddings: New embeddings (must match chunks length)

        Returns:
            True if document was found and updated, False if not found
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have same length: {len(chunks)} vs {len(embeddings)}"
            )

        pool = await self._get_pool()
        doc_uuid = UUID(document_id)
        doc_hash = content_hash(content)
        doc_bytes = len(content.encode("utf-8"))

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Update document row
                result = await conn.execute(
                    """
                    UPDATE memory_documents
                    SET content = $1, content_hash = $2, content_bytes = $3,
                        updated_at = NOW()
                    WHERE id = $4
                    """,
                    content, doc_hash, doc_bytes, doc_uuid,
                )
                if result != "UPDATE 1":
                    return False

                # Delete old chunks
                await conn.execute(
                    "DELETE FROM memory_chunks WHERE document_id = $1",
                    doc_uuid,
                )

                # Insert new chunks
                for chunk, embedding in zip(chunks, embeddings):
                    await conn.execute(
                        """
                        INSERT INTO memory_chunks (
                            id, document_id, seq, pos, content, embedding, tokens, bytes
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        uuid4(),
                        doc_uuid,
                        chunk.seq,
                        chunk.pos,
                        chunk.content,
                        embedding,
                        chunk.tokens,
                        chunk.bytes,
                    )

        logger.debug(f"Updated document {document_id} with {len(chunks)} chunks")
        return True

    async def upsert_document_by_tag(
        self,
        tag: str,
        user_id: str,
        content: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        category: str,
        mode: Literal["append", "replace", "skip"] = "replace",
        scope: str = "personal",
        title: str | None = None,
        source_ref: str | None = None,
        tags: list[str] | None = None,
        append_separator: str = "\n\n---\n\n",
    ) -> tuple[str, str]:
        """Atomically find-or-create a document by tag.

        Uses SELECT ... FOR UPDATE to prevent TOCTOU race conditions.
        If a document with the given tag exists, updates it.
        If not, creates a new one.

        Args:
            tag: Tag to match (e.g., "session:b40eb706")
            user_id: User ID
            content: New content
            chunks: Pre-chunked content for the NEW content
            embeddings: Embeddings for the new chunks
            category: Document category
            mode: "replace" overwrites content, "append" appends with separator,
                  "finalize" replaces entire content (same as replace)
            scope: "personal" or "shared"
            title: Optional document title (used for create, or update if provided)
            source_ref: Source reference
            tags: Full tag list (used for create only)
            append_separator: Separator for append mode

        Returns:
            Tuple of (document_id, action) where action is "created", "appended",
            or "finalized"/"replaced".
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have same length: "
                f"{len(chunks)} vs {len(embeddings)}"
            )

        pool = await self._get_pool()
        user_uuid = UUID(user_id)
        doc_hash = content_hash(content)
        doc_bytes = len(content.encode("utf-8"))

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Atomic find with row lock — prevents concurrent creates
                existing = await conn.fetchrow(
                    f"""
                    SELECT {DOCUMENT_COLUMNS} FROM memory_documents
                    WHERE $1 = ANY(tags) AND user_id = $2 AND category = $3
                      AND deleted_at IS NULL
                    ORDER BY updated_at DESC NULLS LAST
                    LIMIT 1
                    FOR UPDATE
                    """,
                    tag, user_uuid, category,
                )

                if existing:
                    doc_id = existing["id"]
                    doc_id_str = str(doc_id)

                    # Determine final content based on mode
                    if mode == "append":
                        existing_content = existing["content"] or ""
                        final_content = (
                            f"{existing_content}{append_separator}{content}"
                            if existing_content else content
                        )
                        action = "appended"
                    else:
                        # "replace" or "finalize" — overwrite
                        final_content = content
                        action = "finalized" if mode == "finalize" else "replaced"

                    final_hash = content_hash(final_content)
                    final_bytes = len(final_content.encode("utf-8"))

                    # Update document
                    update_fields = """
                        SET content = $1, content_hash = $2, content_bytes = $3,
                            updated_at = NOW()
                    """
                    params: list = [final_content, final_hash, final_bytes]

                    if title:
                        update_fields += ", title = $4 WHERE id = $5"
                        params.extend([title, doc_id])
                    else:
                        update_fields += " WHERE id = $4"
                        params.append(doc_id)

                    await conn.execute(
                        f"UPDATE memory_documents {update_fields}",
                        *params,
                    )

                    # Replace chunks with new embeddings
                    await conn.execute(
                        "DELETE FROM memory_chunks WHERE document_id = $1",
                        doc_id,
                    )
                    for chunk, embedding in zip(chunks, embeddings):
                        await conn.execute(
                            """
                            INSERT INTO memory_chunks (
                                id, document_id, seq, pos, content,
                                embedding, tokens, bytes
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                            uuid4(), doc_id, chunk.seq, chunk.pos,
                            chunk.content, embedding, chunk.tokens, chunk.bytes,
                        )

                    logger.debug(
                        f"Upsert: {action} document {doc_id_str} "
                        f"(tag={tag}, {len(chunks)} chunks)"
                    )
                    return doc_id_str, action

                else:
                    # Create new document
                    doc_id = uuid4()
                    await conn.execute(
                        """
                        INSERT INTO memory_documents (
                            id, user_id, scope, category, title,
                            source_ref, tags, content, content_hash, content_bytes
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """,
                        doc_id, user_uuid, scope, category, title,
                        source_ref, tags or [], content, doc_hash, doc_bytes,
                    )

                    for chunk, embedding in zip(chunks, embeddings):
                        await conn.execute(
                            """
                            INSERT INTO memory_chunks (
                                id, document_id, seq, pos, content,
                                embedding, tokens, bytes
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                            uuid4(), doc_id, chunk.seq, chunk.pos,
                            chunk.content, embedding, chunk.tokens, chunk.bytes,
                        )

                    logger.debug(
                        f"Upsert: created document {doc_id} "
                        f"(tag={tag}, {len(chunks)} chunks)"
                    )
                    return str(doc_id), "created"

    async def get_documents_by_category(
        self,
        user_id: str,
        category: str,
        limit: int = 50,
        source_ref_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get documents by category for a user.

        Args:
            user_id: User ID
            category: Category to filter by
            limit: Maximum results
            source_ref_prefix: Optional prefix filter on source_ref (e.g. "project:org/repo")

        Returns:
            List of document dicts
        """
        pool = await self._get_pool()
        user_uuid = UUID(user_id)

        if source_ref_prefix:
            query = f"""
                SELECT {DOCUMENT_COLUMNS} FROM memory_documents
                WHERE user_id = $1 AND category = $2 AND source_ref LIKE $3
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $4
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, user_uuid, category, f"{source_ref_prefix}%", limit)
        else:
            query = f"""
                SELECT {DOCUMENT_COLUMNS} FROM memory_documents
                WHERE user_id = $1 AND category = $2
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $3
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, user_uuid, category, limit)

        return [self._doc_row_to_dict(row) for row in rows]

    async def get_recent_documents(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 15,
        exclude_categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get recently created documents.

        Args:
            user_id: User ID
            hours: Look back window in hours
            limit: Maximum results
            exclude_categories: Categories to exclude from results

        Returns:
            List of document dicts ordered by created_at DESC
        """
        pool = await self._get_pool()
        user_uuid = UUID(user_id)

        if exclude_categories:
            query = f"""
                SELECT {DOCUMENT_COLUMNS} FROM memory_documents
                WHERE user_id = $1
                  AND created_at > NOW() - INTERVAL '1 hour' * $2
                  AND NOT (category = ANY($3))
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $4
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, user_uuid, hours, exclude_categories, limit)
        else:
            query = f"""
                SELECT {DOCUMENT_COLUMNS} FROM memory_documents
                WHERE user_id = $1
                  AND created_at > NOW() - INTERVAL '1 hour' * $2
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $3
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, user_uuid, hours, limit)

        return [self._doc_row_to_dict(row) for row in rows]

    async def get_all_documents(
        self,
        user_id: str,
        team_id: str | None = None,
        scope: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all documents for a user with pagination and filtering.

        For shared/both scopes with a team_id, uses OR logic so users can see
        their own docs plus shared docs from their team.

        Args:
            user_id: User ID
            team_id: Team ID (enables cross-user visibility for shared scope)
            scope: Optional scope filter ("personal", "shared", or None for both)
            limit: Maximum results
            offset: Number of rows to skip (for pagination)
            category: Optional category filter

        Returns:
            List of document dicts
        """
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=1)
        fb.add("deleted_at IS NULL")
        fb.add_ownership_filter(
            user_id, team_id, scope or "both",
        )
        if category:
            fb.add_param("category = ${}", category)

        limit_idx, offset_idx = fb.add_raw_values(limit, offset)

        query = f"""
            SELECT {DOCUMENT_COLUMNS} FROM memory_documents
            WHERE {fb.build()}
            ORDER BY created_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *fb.values)

        return [self._doc_row_to_dict(row) for row in rows]

    async def count_documents(
        self,
        user_id: str,
        team_id: str | None = None,
        scope: str | None = None,
        category: str | None = None,
    ) -> int:
        """Count documents for a user with optional filters.

        For shared/both scopes with a team_id, uses OR logic so users can see
        their own docs plus shared docs from their team.

        Args:
            user_id: User ID
            team_id: Team ID (enables cross-user visibility for shared scope)
            scope: Optional scope filter
            category: Optional category filter

        Returns:
            Document count matching the filters
        """
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=1)
        fb.add("deleted_at IS NULL")
        fb.add_ownership_filter(
            user_id, team_id, scope or "both",
        )
        if category:
            fb.add_param("category = ${}", category)

        query = f"SELECT COUNT(*) FROM memory_documents WHERE {fb.build()}"

        async with pool.acquire() as conn:
            result = await conn.fetchval(query, *fb.values)

        return result or 0

    async def get_document_category_counts(
        self,
        user_id: str,
        team_id: str | None = None,
        scope: str | None = None,
    ) -> dict[str, int]:
        """Get document counts grouped by category.

        For shared/both scopes with a team_id, uses OR logic so users can see
        their own docs plus shared docs from their team.

        Args:
            user_id: User ID
            team_id: Team ID (enables cross-user visibility for shared scope)
            scope: Optional scope filter

        Returns:
            Dict mapping category name to document count
        """
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=1)
        fb.add("deleted_at IS NULL")
        fb.add_ownership_filter(
            user_id, team_id, scope or "both",
        )

        query = f"""
            SELECT category, COUNT(*) as count
            FROM memory_documents
            WHERE {fb.build()}
            GROUP BY category
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *fb.values)

        return {row["category"]: row["count"] for row in rows}

    # =========================================================================
    # Chunk Search Operations
    # =========================================================================

    async def search_chunks(
        self,
        query_embedding: list[float],
        user_id: str | None = None,
        team_id: str | None = None,
        scope: str | Literal["both"] = "both",
        category: str | None = None,
        source_ref: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Returns chunk content with document metadata.
        """
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=3)
        fb.add("d.deleted_at IS NULL")
        fb.add_ownership_filter(user_id, team_id, scope, col_prefix="d.")
        if category:
            fb.add_param("d.category = ${}", category)
        if source_ref:
            fb.add_param("d.source_ref = ${}", source_ref)
        if tags:
            for tag in tags:
                fb.add_param("${} = ANY(d.tags)", tag)

        query = f"""
            SELECT {CHUNK_WITH_DOC_COLUMNS},
                   1 - (c.embedding <=> $1) AS score
            FROM memory_chunks c
            JOIN memory_documents d ON c.document_id = d.id
            WHERE {fb.build()}
            ORDER BY c.embedding <=> $1
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, query_embedding, limit, *fb.values)

        return [chunk_row_to_result(row, include_score=True) for row in rows]

    async def hybrid_search_chunks(
        self,
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
        team_id: str | None = None,
        scope: str | Literal["both"] = "both",
        category: str | None = None,
        source_ref: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and full-text search using RRF."""
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=4)
        fb.add("d.deleted_at IS NULL")
        fb.add_ownership_filter(user_id, team_id, scope, col_prefix="d.")
        if category:
            fb.add_param("d.category = ${}", category)
        if source_ref:
            fb.add_param("d.source_ref = ${}", source_ref)
        if tags:
            for tag in tags:
                fb.add_param("${} = ANY(d.tags)", tag)

        where_clause = fb.build()
        text_weight = 1 - vector_weight

        # Convert query to OR-based tsquery for better temporal question matching
        # "Rack Fest Turbocharged Tuesdays" -> 'rack' | 'fest' | 'turbocharg' | 'tuesday'
        # This finds docs containing ANY term, not just ALL terms
        query_sql = f"""
            WITH or_tsquery AS (
                -- Generate OR-based tsquery from query words
                -- "Rack Fest Turbocharged" -> 'rack' | 'fest' | 'turbocharg'
                SELECT to_tsquery('english',
                    string_agg((t).lexeme::text, ' | ' ORDER BY (t).lexeme)
                ) as q
                FROM unnest(to_tsvector('english', $2)) t
            ),
            vector_search AS (
                SELECT c.id, 1 - (c.embedding <=> $1) AS vector_score,
                       ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1) AS vector_rank
                FROM memory_chunks c
                JOIN memory_documents d ON c.document_id = d.id
                WHERE {where_clause}
                ORDER BY c.embedding <=> $1
                LIMIT $3
            ),
            text_search AS (
                SELECT c.id, ts_rank(c.content_tsv, oq.q) AS text_score,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(c.content_tsv, oq.q) DESC) AS text_rank
                FROM memory_chunks c
                JOIN memory_documents d ON c.document_id = d.id
                CROSS JOIN or_tsquery oq
                WHERE {where_clause}
                  AND c.content_tsv @@ oq.q
                ORDER BY text_score DESC
                LIMIT $3
            )
            SELECT {CHUNK_WITH_DOC_COLUMNS},
                   COALESCE(1.0 / (60 + v.vector_rank), 0) * {vector_weight} +
                   COALESCE(1.0 / (60 + t.text_rank), 0) * {text_weight} AS score
            FROM memory_chunks c
            JOIN memory_documents d ON c.document_id = d.id
            LEFT JOIN vector_search v ON c.id = v.id
            LEFT JOIN text_search t ON c.id = t.id
            WHERE v.id IS NOT NULL OR t.id IS NOT NULL
            ORDER BY score DESC
            LIMIT {limit}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, query_embedding, query, limit * 2, *fb.values)

        return [chunk_row_to_result(row, include_score=True) for row in rows]

    async def full_text_search_chunks(
        self,
        query: str,
        user_id: str | None = None,
        team_id: str | None = None,
        scope: str | Literal["both"] = "both",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search chunks using full-text search (BM25-style ranking).

        Uses OR logic so documents matching ANY query term are found.
        This is critical for temporal questions that mention multiple events
        which may be in different documents.
        """
        pool = await self._get_pool()

        fb = FilterBuilder(start_idx=3)
        fb.add("d.deleted_at IS NULL")
        fb.add_ownership_filter(user_id, team_id, scope, col_prefix="d.")
        if category:
            fb.add_param("d.category = ${}", category)

        where_clause = fb.build()
        if where_clause:
            where_clause = f"AND {where_clause}"

        # Use OR-based tsquery to find docs with ANY term
        query_sql = f"""
            WITH or_tsquery AS (
                SELECT to_tsquery('english',
                    string_agg((t).lexeme::text, ' | ' ORDER BY (t).lexeme)
                ) as q
                FROM unnest(to_tsvector('english', $1)) t
            )
            SELECT {CHUNK_WITH_DOC_COLUMNS},
                   ts_rank(c.content_tsv, oq.q) AS score
            FROM memory_chunks c
            JOIN memory_documents d ON c.document_id = d.id
            CROSS JOIN or_tsquery oq
            WHERE c.content_tsv @@ oq.q
                  {where_clause}
            ORDER BY score DESC
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, query, limit, *fb.values)

        return [chunk_row_to_result(row, include_score=True) for row in rows]

    # =========================================================================
    # Relations Operations
    # =========================================================================

    async def get_related_documents(
        self,
        document_id: str,
        relation_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get documents related to a given document via memory_relations.

        Joins memory_relations with memory_documents (target_id = document ID).
        Returns empty list if no relations exist for this document.

        Args:
            document_id: Starting document ID
            relation_type: Optional relation type filter
            limit: Maximum results

        Returns:
            List of related document dicts with relation_type and relation_similarity
        """
        pool = await self._get_pool()
        doc_uuid = UUID(document_id)

        # Prefix document columns with table alias 'd.'
        doc_cols = ", ".join(
            f"d.{col.strip()}" for col in DOCUMENT_COLUMNS.split(",")
        )

        if relation_type:
            query = f"""
                SELECT {doc_cols}, r.relation_type, r.similarity AS relation_similarity
                FROM memory_relations r
                JOIN memory_documents d ON r.target_id = d.id
                WHERE r.source_id = $1 AND r.relation_type = $2
                ORDER BY r.similarity DESC NULLS LAST
                LIMIT $3
            """
            values: list = [doc_uuid, relation_type, limit]
        else:
            query = f"""
                SELECT {doc_cols}, r.relation_type, r.similarity AS relation_similarity
                FROM memory_relations r
                JOIN memory_documents d ON r.target_id = d.id
                WHERE r.source_id = $1
                ORDER BY r.similarity DESC NULLS LAST
                LIMIT $2
            """
            values = [doc_uuid, limit]

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *values)

        results = []
        for row in rows:
            doc = self._doc_row_to_dict(row)
            doc["relation_type"] = row["relation_type"]
            doc["relation_similarity"] = row["relation_similarity"]
            results.append(doc)

        return results

    # =========================================================================
    # Utility Operations
    # =========================================================================

    async def get_document_count(self, user_id: str) -> int:
        """Get total document count for a user (excludes soft-deleted)."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM memory_documents WHERE user_id = $1 AND deleted_at IS NULL",
                UUID(user_id),
            )

        return result or 0

    async def get_chunk_count(self, user_id: str) -> int:
        """Get total chunk count for a user (excludes soft-deleted documents)."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_chunks c
                JOIN memory_documents d ON c.document_id = d.id
                WHERE d.user_id = $1 AND d.deleted_at IS NULL
                """,
                UUID(user_id),
            )

        return result or 0

    # =========================================================================
    # Feedback Operations
    # =========================================================================

    async def increment_shown_count(self, document_ids: list[str]) -> int:
        """Increment shown_count and update last_shown_at for documents.

        Called when memories are surfaced in search results to track usage.

        Args:
            document_ids: List of document IDs that were shown

        Returns:
            Number of documents updated
        """
        if not document_ids:
            return 0

        pool = await self._get_pool()
        uuids = [UUID(did) for did in document_ids]

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE memory_documents
                SET shown_count = shown_count + 1,
                    last_shown_at = NOW()
                WHERE id = ANY($1) AND deleted_at IS NULL
                """,
                uuids,
            )

        count = int(result.split()[1]) if result else 0
        logger.debug(f"Incremented shown_count for {count} documents")
        return count

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def add_documents_batch(
        self,
        documents: list[dict],
        all_chunks: list[list[Chunk]],
        all_embeddings: list[list[list[float]]],
        user_id: str | UUID,
        team_id: str | UUID | None = None,
        scope: str = "personal",
    ) -> list[tuple[str, bool]]:
        """Add multiple documents with their chunks in a single transaction.

        This is optimized for bulk ingestion (e.g., eval benchmarks) where
        we want to minimize HTTP calls and database round-trips.

        Args:
            documents: List of document dicts with keys:
                - content: Full document content
                - category: Document category
                - source_ref: Source reference (e.g., project:longmemeval:session_id)
                - tags: Optional tags
                - title: Optional title
                - source: Optional source identifier
            all_chunks: List of chunk lists (one list per document)
            all_embeddings: List of embedding lists (one list per document)
            user_id: User ID
            team_id: Team ID (for shared documents)
            scope: "personal" or "shared"

        Returns:
            List of tuples (document_id, is_new). is_new is False if deduplicated.
        """
        if len(documents) != len(all_chunks) or len(documents) != len(all_embeddings):
            raise ValueError(
                f"Mismatched lengths: {len(documents)} docs, "
                f"{len(all_chunks)} chunk lists, {len(all_embeddings)} embedding lists"
            )

        pool = await self._get_pool()
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
        team_uuid = UUID(team_id) if isinstance(team_id, str) and team_id else None

        results: list[tuple[str, bool]] = []

        async with pool.acquire() as conn:
            async with conn.transaction():
                for doc, chunks, embeddings in zip(documents, all_chunks, all_embeddings):
                    if len(chunks) != len(embeddings):
                        raise ValueError(
                            f"Chunks/embeddings mismatch for document: "
                            f"{len(chunks)} chunks, {len(embeddings)} embeddings"
                        )

                    content = doc["content"]
                    doc_hash = content_hash(content)
                    doc_bytes = len(content.encode("utf-8"))

                    # Check for existing document with same hash (non-deleted only)
                    existing = await conn.fetchrow(
                        """
                        SELECT id FROM memory_documents
                        WHERE content_hash = $1 AND user_id = $2 AND deleted_at IS NULL
                        """,
                        doc_hash,
                        user_uuid,
                    )

                    if existing:
                        results.append((str(existing["id"]), False))
                        continue

                    # Insert new document
                    doc_id = uuid4()
                    await conn.execute(
                        """
                        INSERT INTO memory_documents (
                            id, user_id, team_id, scope, category, title,
                            source, source_ref, tags, content, content_hash, content_bytes
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        """,
                        doc_id,
                        user_uuid,
                        team_uuid,
                        scope,
                        doc.get("category", "document"),
                        doc.get("title"),
                        doc.get("source"),
                        doc.get("source_ref"),
                        doc.get("tags") or [],
                        content,
                        doc_hash,
                        doc_bytes,
                    )

                    # Insert chunks
                    for chunk, embedding in zip(chunks, embeddings):
                        await conn.execute(
                            """
                            INSERT INTO memory_chunks (
                                id, document_id, seq, pos, content, embedding, tokens, bytes
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                            uuid4(),
                            doc_id,
                            chunk.seq,
                            chunk.pos,
                            chunk.content,
                            embedding,
                            chunk.tokens,
                            chunk.bytes,
                        )

                    results.append((str(doc_id), True))

        new_count = sum(1 for _, is_new in results if is_new)
        logger.info(
            f"Batch added {new_count} new documents ({len(results) - new_count} deduplicated) "
            f"with {sum(len(c) for c in all_chunks)} total chunks"
        )
        return results


    # =========================================================================
    # Conflict CRUD (memory_conflicts table)
    # =========================================================================

    async def add_conflict(
        self,
        user_id: str,
        doc_a_id: str,
        doc_b_id: str,
        explanation: str,
    ) -> str | None:
        """Record a conflict between two documents.

        Uses ON CONFLICT DO NOTHING to avoid duplicates.

        Args:
            user_id: Owner user ID
            doc_a_id: First conflicting document ID
            doc_b_id: Second conflicting document ID
            explanation: LLM-generated explanation of the conflict

        Returns:
            Conflict ID if created, None if already exists
        """
        pool = await self._get_pool()
        # Normalize order to prevent duplicate (a,b) vs (b,a)
        id_a, id_b = sorted([doc_a_id, doc_b_id])
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_conflicts (user_id, doc_a_id, doc_b_id, explanation)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (doc_a_id, doc_b_id) DO NOTHING
                RETURNING id
                """,
                UUID(user_id) if isinstance(user_id, str) else user_id,
                UUID(id_a) if isinstance(id_a, str) else id_a,
                UUID(id_b) if isinstance(id_b, str) else id_b,
                explanation,
            )
            if row:
                logger.info(f"Recorded conflict between {id_a[:8]} and {id_b[:8]}")
                return str(row["id"])
            return None

    async def get_open_conflicts(
        self,
        user_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get unresolved conflicts for a user, with document content.

        Args:
            user_id: Owner user ID
            limit: Max conflicts to return

        Returns:
            List of conflict dicts with doc_a_content and doc_b_content
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id, c.doc_a_id, c.doc_b_id, c.explanation,
                    c.status, c.created_at,
                    da.content AS doc_a_content,
                    db.content AS doc_b_content
                FROM memory_conflicts c
                LEFT JOIN memory_documents da ON da.id = c.doc_a_id AND da.deleted_at IS NULL
                LEFT JOIN memory_documents db ON db.id = c.doc_b_id AND db.deleted_at IS NULL
                WHERE c.user_id = $1 AND c.status = 'open'
                ORDER BY c.created_at DESC
                LIMIT $2
                """,
                UUID(user_id) if isinstance(user_id, str) else user_id,
                limit,
            )
            return [
                {
                    "id": str(r["id"]),
                    "doc_a_id": str(r["doc_a_id"]),
                    "doc_b_id": str(r["doc_b_id"]),
                    "explanation": r["explanation"],
                    "status": r["status"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "doc_a_content": r["doc_a_content"],
                    "doc_b_content": r["doc_b_content"],
                }
                for r in rows
            ]

    async def get_conflict(
        self,
        conflict_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Get a single conflict by ID with authorization check.

        Args:
            conflict_id: Conflict UUID
            user_id: Owner user ID (for authorization)

        Returns:
            Conflict dict with doc content, or None if not found/not authorized
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    c.id, c.doc_a_id, c.doc_b_id, c.explanation,
                    c.status, c.created_at,
                    da.content AS doc_a_content,
                    db.content AS doc_b_content
                FROM memory_conflicts c
                LEFT JOIN memory_documents da ON da.id = c.doc_a_id AND da.deleted_at IS NULL
                LEFT JOIN memory_documents db ON db.id = c.doc_b_id AND db.deleted_at IS NULL
                WHERE c.id = $1 AND c.user_id = $2 AND c.status = 'open'
                """,
                UUID(conflict_id) if isinstance(conflict_id, str) else conflict_id,
                UUID(user_id) if isinstance(user_id, str) else user_id,
            )
            if not row:
                return None
            return {
                "id": str(row["id"]),
                "doc_a_id": str(row["doc_a_id"]),
                "doc_b_id": str(row["doc_b_id"]),
                "explanation": row["explanation"],
                "status": row["status"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "doc_a_content": row["doc_a_content"],
                "doc_b_content": row["doc_b_content"],
            }

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str = "resolved",
    ) -> bool:
        """Mark a conflict as resolved or dismissed.

        Args:
            conflict_id: Conflict UUID
            resolution: Status to set ("resolved" or "dismissed")

        Returns:
            True if conflict was found and updated
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE memory_conflicts
                SET status = $2, resolved_at = NOW()
                WHERE id = $1 AND status = 'open'
                """,
                UUID(conflict_id) if isinstance(conflict_id, str) else conflict_id,
                resolution,
            )
            return result == "UPDATE 1"


# Module-level instance (lazy initialization)
_store: DocumentStore | None = None


def get_document_store(database_url: str | None = None) -> DocumentStore:
    """Get the shared DocumentStore instance."""
    global _store
    if _store is None:
        import os

        url = database_url or os.getenv("CEMS_DATABASE_URL")
        if not url:
            raise ValueError("Database URL required")
        _store = DocumentStore(url)
    return _store
