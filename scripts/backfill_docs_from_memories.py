#!/usr/bin/env python
"""Backfill memory_documents and memory_chunks from old memories table.

This script migrates data from the flat memories table to the new
document+chunk model:
- Each memory becomes a document
- Documents are chunked using QMD-style chunking (800 tokens, 15% overlap)
- Chunks are embedded using the configured embedding backend

Usage:
    # Run with default settings (llamacpp_server embeddings)
    uv run python scripts/backfill_docs_from_memories.py

    # Dry run (no changes)
    uv run python scripts/backfill_docs_from_memories.py --dry-run

    # Limit to N memories (for testing)
    uv run python scripts/backfill_docs_from_memories.py --limit 10

    # Skip existing (resume after failure)
    uv run python scripts/backfill_docs_from_memories.py --skip-existing

Environment variables:
    CEMS_DATABASE_URL: PostgreSQL connection URL
    CEMS_EMBEDDING_BACKEND: "llamacpp_server" or "openrouter"
    CEMS_LLAMACPP_BASE_URL: llama.cpp server URL (for llamacpp_server backend)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cems.chunking import Chunk, chunk_document, content_hash
from cems.config import CEMSConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def get_all_memories(pool, limit: int | None = None) -> list[dict]:
    """Get all memories from the old table."""
    query = """
        SELECT
            id, content, user_id, team_id, scope, category,
            tags, source, source_ref, created_at
        FROM memories
        WHERE archived = FALSE
        ORDER BY created_at ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    return [
        {
            "id": str(row["id"]),
            "content": row["content"],
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "team_id": str(row["team_id"]) if row["team_id"] else None,
            "scope": row["scope"],
            "category": row["category"],
            "tags": row["tags"] or [],
            "source": row["source"],
            "source_ref": row["source_ref"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


async def check_document_exists(pool, content_hash_val: str, user_id: str) -> bool:
    """Check if a document with this hash already exists for this user."""
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            """
            SELECT 1 FROM memory_documents
            WHERE content_hash = $1 AND user_id = $2
            """,
            content_hash_val,
            user_id,
        )
    return result is not None


async def insert_document_with_chunks(
    pool,
    memory: dict,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> str:
    """Insert a document and its chunks."""
    import uuid

    doc_id = uuid.uuid4()
    doc_hash = content_hash(memory["content"])
    doc_bytes = len(memory["content"].encode("utf-8"))

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Insert document
            await conn.execute(
                """
                INSERT INTO memory_documents (
                    id, user_id, team_id, scope, category, title,
                    source, source_ref, tags, content, content_hash, content_bytes,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $13)
                """,
                doc_id,
                uuid.UUID(memory["user_id"]) if memory["user_id"] else None,
                uuid.UUID(memory["team_id"]) if memory["team_id"] else None,
                memory["scope"],
                memory["category"],
                None,  # title
                memory["source"],
                memory["source_ref"],
                memory["tags"],
                memory["content"],
                doc_hash,
                doc_bytes,
                memory["created_at"],
            )

            # Insert chunks
            for chunk, embedding in zip(chunks, embeddings):
                await conn.execute(
                    """
                    INSERT INTO memory_chunks (
                        id, document_id, seq, pos, content, embedding, tokens, bytes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    uuid.uuid4(),
                    doc_id,
                    chunk.seq,
                    chunk.pos,
                    chunk.content,
                    embedding,
                    chunk.tokens,
                    chunk.bytes,
                )

    return str(doc_id)


async def migrate(
    dry_run: bool = False,
    limit: int | None = None,
    skip_existing: bool = False,
    batch_size: int = 32,
):
    """Run the migration."""
    import asyncpg
    from pgvector.asyncpg import register_vector

    config = CEMSConfig()
    logger.info(f"Connecting to database: {config.database_url[:50]}...")

    # Create connection pool
    pool = await asyncpg.create_pool(
        config.database_url,
        min_size=2,
        max_size=10,
        setup=lambda conn: register_vector(conn),
    )

    # Initialize embedder based on config
    if config.embedding_backend == "llamacpp_server":
        from cems.llamacpp_server import AsyncLlamaCppEmbeddingClient
        embedder = AsyncLlamaCppEmbeddingClient(config)
        logger.info(f"Using llama.cpp server embeddings at {config.llamacpp_base_url}")
    else:
        from cems.embedding import AsyncEmbeddingClient
        embedder = AsyncEmbeddingClient(model=config.embedding_model)
        logger.info(f"Using OpenRouter embeddings ({config.embedding_model})")

    try:
        # Get all memories
        logger.info("Fetching memories from old table...")
        memories = await get_all_memories(pool, limit)
        logger.info(f"Found {len(memories)} memories to migrate")

        if dry_run:
            logger.info("DRY RUN - no changes will be made")

        migrated = 0
        skipped = 0
        errors = 0

        for i, memory in enumerate(memories):
            try:
                # Skip if no user_id
                if not memory["user_id"]:
                    logger.warning(f"  [{i+1}/{len(memories)}] Skipping memory with no user_id")
                    skipped += 1
                    continue

                # Skip if no content
                if not memory["content"] or not memory["content"].strip():
                    logger.warning(f"  [{i+1}/{len(memories)}] Skipping empty memory")
                    skipped += 1
                    continue

                # Check if already migrated
                doc_hash = content_hash(memory["content"])
                if skip_existing:
                    exists = await check_document_exists(pool, doc_hash, memory["user_id"])
                    if exists:
                        logger.debug(f"  [{i+1}/{len(memories)}] Already exists, skipping")
                        skipped += 1
                        continue

                # Chunk the content
                chunks = chunk_document(memory["content"])
                if not chunks:
                    logger.warning(f"  [{i+1}/{len(memories)}] Chunking produced no output")
                    skipped += 1
                    continue

                if dry_run:
                    logger.info(
                        f"  [{i+1}/{len(memories)}] Would migrate: "
                        f"{len(memory['content'])} chars -> {len(chunks)} chunks"
                    )
                    migrated += 1
                    continue

                # Embed all chunks (batched)
                chunk_texts = [c.content for c in chunks]
                embeddings = await embedder.embed_batch(chunk_texts, batch_size=batch_size)

                # Insert document + chunks
                doc_id = await insert_document_with_chunks(
                    pool, memory, chunks, embeddings
                )

                logger.info(
                    f"  [{i+1}/{len(memories)}] Migrated: {len(chunks)} chunks -> {doc_id[:8]}..."
                )
                migrated += 1

            except Exception as e:
                logger.error(f"  [{i+1}/{len(memories)}] Error: {e}")
                errors += 1

        logger.info("=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total memories:  {len(memories)}")
        logger.info(f"Migrated:        {migrated}")
        logger.info(f"Skipped:         {skipped}")
        logger.info(f"Errors:          {errors}")

        if dry_run:
            logger.info("(DRY RUN - no changes were made)")

    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate memories to documents+chunks")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to N memories (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip memories that have already been migrated (resume mode)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )

    args = parser.parse_args()

    asyncio.run(
        migrate(
            dry_run=args.dry_run,
            limit=args.limit,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
