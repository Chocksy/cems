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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

import asyncpg
from pgvector.asyncpg import register_vector

from cems.chunking import Chunk, content_hash
from cems.db.filter_builder import FilterBuilder

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Column definitions for documents
DOCUMENT_COLUMNS = """
    id, user_id, team_id, scope, category, title, source, source_ref,
    tags, content, content_hash, content_bytes, created_at, updated_at
"""

# Column definitions for chunks (with document join)
CHUNK_COLUMNS = """
    c.id AS chunk_id, c.document_id, c.seq, c.pos, c.content AS chunk_content,
    c.tokens, c.bytes, c.created_at AS chunk_created_at
"""

CHUNK_WITH_DOC_COLUMNS = f"""
    {CHUNK_COLUMNS},
    d.user_id, d.team_id, d.scope, d.category, d.title, d.source, d.source_ref,
    d.tags, d.content AS document_content, d.content_hash, d.content_bytes,
    d.created_at AS document_created_at
"""


@dataclass
class ChunkSearchResult:
    """Result from chunk search with document metadata."""

    chunk_id: str
    document_id: str
    seq: int
    pos: int
    chunk_content: str
    score: float
    tokens: int | None
    bytes: int
    # Document metadata
    user_id: str | None
    team_id: str | None
    scope: str
    category: str
    title: str | None
    source: str | None
    source_ref: str | None
    tags: list[str]
    document_created_at: Any


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
        embedding_dim: int = 768,
    ):
        """Initialize the document store.

        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            embedding_dim: Embedding dimension (768 for llama.cpp)
        """
        self.database_url = database_url
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
            # Check for existing document with same hash for this user
            existing = await conn.fetchrow(
                """
                SELECT id FROM memory_documents
                WHERE content_hash = $1 AND user_id = $2
                """,
                doc_hash,
                user_uuid,
            )

            if existing:
                logger.debug(f"Document already exists with hash {doc_hash[:8]}...")
                return str(existing["id"]), False

            # Insert new document
            doc_id = uuid4()
            async with conn.transaction():
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

        logger.debug(f"Added document {doc_id} with {len(chunks)} chunks")
        return str(doc_id), True

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get a document by ID."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {DOCUMENT_COLUMNS} FROM memory_documents WHERE id = $1",
                UUID(document_id),
            )

        if not row:
            return None

        return {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "team_id": str(row["team_id"]) if row["team_id"] else None,
            "scope": row["scope"],
            "category": row["category"],
            "title": row["title"],
            "source": row["source"],
            "source_ref": row["source_ref"],
            "tags": row["tags"],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "content_bytes": row["content_bytes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memory_documents WHERE id = $1",
                UUID(document_id),
            )

        return result == "DELETE 1"

    async def delete_by_source_ref(self, source_ref: str, user_id: str) -> int:
        """Delete all documents with a given source_ref for a user.

        Useful for eval cleanup.
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
        """Delete all documents containing a specific tag for a user.

        Useful for eval cleanup.
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
        if user_id:
            fb.add_param("d.user_id = ${}", UUID(user_id))
        if team_id and scope in ("shared", "both"):
            fb.add_param("d.team_id = ${}", UUID(team_id))
        if scope != "both":
            fb.add_param("d.scope = ${}", scope)
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
        if user_id:
            fb.add_param("d.user_id = ${}", UUID(user_id))
        if team_id and scope in ("shared", "both"):
            fb.add_param("d.team_id = ${}", UUID(team_id))
        if scope != "both":
            fb.add_param("d.scope = ${}", scope)
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
        # Filter will be built with OR-based tsquery from CTE
        if user_id:
            fb.add_param("d.user_id = ${}", UUID(user_id))
        if team_id and scope in ("shared", "both"):
            fb.add_param("d.team_id = ${}", UUID(team_id))
        if scope != "both":
            fb.add_param("d.scope = ${}", scope)
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
    # Utility Operations
    # =========================================================================

    async def get_document_count(self, user_id: str) -> int:
        """Get total document count for a user."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM memory_documents WHERE user_id = $1",
                UUID(user_id),
            )

        return result or 0

    async def get_chunk_count(self, user_id: str) -> int:
        """Get total chunk count for a user."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_chunks c
                JOIN memory_documents d ON c.document_id = d.id
                WHERE d.user_id = $1
                """,
                UUID(user_id),
            )

        return result or 0

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

                    # Check for existing document with same hash
                    existing = await conn.fetchrow(
                        """
                        SELECT id FROM memory_documents
                        WHERE content_hash = $1 AND user_id = $2
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
