#!/usr/bin/env python3
"""Migration script: Qdrant + memory_metadata -> pgvector unified memories table.

This script migrates existing CEMS data from the old architecture (Qdrant vectors +
PostgreSQL memory_metadata) to the new unified pgvector architecture (PostgreSQL
memories table with embedded vectors).

Usage:
    # Dry run (shows what would be migrated)
    python scripts/migrate_to_pgvector.py --dry-run

    # Full migration
    python scripts/migrate_to_pgvector.py

    # With custom database URL
    CEMS_DATABASE_URL=postgresql://... python scripts/migrate_to_pgvector.py

Prerequisites:
    1. The new memories table must exist (run deploy/init.sql first)
    2. The old memory_metadata table must have data
    3. For Qdrant migration, Qdrant must be accessible

Steps:
    1. Read all entries from memory_metadata table
    2. For each entry, fetch the vector from Qdrant
    3. Insert into the new memories table with all metadata + vector
    4. Verify counts match
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncpg
from pgvector.asyncpg import register_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def _count(conn: asyncpg.Connection, table: str) -> int:
    """Get row count for a table."""
    result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
    return result or 0


async def fetch_all_metadata(conn: asyncpg.Connection) -> list[dict[str, Any]]:
    """Fetch all entries from memory_metadata table."""
    rows = await conn.fetch("""
        SELECT
            id, memory_id, user_id, team_id, scope, category,
            created_at, updated_at, last_accessed, access_count,
            source, source_ref, tags, archived, priority,
            pinned, pin_reason, pin_category, expires_at
        FROM memory_metadata
        ORDER BY created_at
    """)
    return [dict(row) for row in rows]


async def fetch_vector_from_qdrant(
    qdrant_url: str,
    collection: str,
    memory_id: str,
) -> tuple[list[float] | None, str | None]:
    """Fetch vector and content from Qdrant.

    Returns:
        Tuple of (embedding vector, content text) or (None, None) if not found
    """
    import httpx

    # Qdrant scroll API to find by ID
    async with httpx.AsyncClient() as client:
        # First, try to get the point directly
        try:
            response = await client.post(
                f"{qdrant_url}/collections/{collection}/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {"key": "id", "match": {"value": memory_id}}
                        ]
                    },
                    "with_payload": True,
                    "with_vector": True,
                    "limit": 1,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                points = data.get("result", {}).get("points", [])
                if points:
                    point = points[0]
                    vector = point.get("vector")
                    payload = point.get("payload", {})
                    content = payload.get("data", payload.get("memory", payload.get("text", "")))
                    return vector, content
        except Exception as e:
            logger.debug(f"Qdrant fetch failed for {memory_id}: {e}")

    return None, None


async def insert_memory(
    conn: asyncpg.Connection,
    metadata: dict[str, Any],
    embedding: list[float],
    content: str,
) -> bool:
    """Insert a memory into the new memories table."""
    try:
        await conn.execute("""
            INSERT INTO memories (
                id, content, embedding, user_id, team_id, scope, category,
                tags, source, source_ref, priority, pinned, pin_reason,
                pin_category, archived, access_count, created_at, updated_at,
                last_accessed, expires_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18, $19, $20
            )
            ON CONFLICT (id) DO NOTHING
        """,
            metadata["id"],
            content,
            embedding,
            metadata.get("user_id"),
            metadata.get("team_id"),
            metadata.get("scope", "personal"),
            metadata.get("category", "general"),
            metadata.get("tags", []),
            metadata.get("source"),
            metadata.get("source_ref"),
            metadata.get("priority", 1.0),
            metadata.get("pinned", False),
            metadata.get("pin_reason"),
            metadata.get("pin_category"),
            metadata.get("archived", False),
            metadata.get("access_count", 0),
            metadata.get("created_at", datetime.now(UTC)),
            metadata.get("updated_at", datetime.now(UTC)),
            metadata.get("last_accessed", datetime.now(UTC)),
            metadata.get("expires_at"),
        )
        return True
    except Exception as e:
        logger.error(f"Failed to insert memory {metadata['id']}: {e}")
        return False


async def migrate_from_metadata_only(
    conn: asyncpg.Connection,
    metadata_list: list[dict[str, Any]],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate metadata entries, generating new embeddings.

    This is used when Qdrant is not available or vectors can't be retrieved.
    It creates placeholder embeddings that should be regenerated.

    Returns:
        Tuple of (migrated count, skipped count)
    """
    # Without Qdrant access, this path cannot retrieve original content.
    # All memories are skipped. Use migrate_from_qdrant() instead.
    logger.warning(
        f"metadata-only migration: {len(metadata_list)} memories skipped "
        "(no content available without Qdrant)"
    )
    return 0, len(metadata_list)


async def migrate_from_qdrant(
    conn: asyncpg.Connection,
    qdrant_url: str,
    metadata_list: list[dict[str, Any]],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate data from Qdrant + metadata to unified memories table.

    Returns:
        Tuple of (migrated count, skipped count)
    """
    migrated = 0
    skipped = 0

    # Group metadata by scope to determine Qdrant collection
    for i, metadata in enumerate(metadata_list):
        memory_id = metadata.get("memory_id", str(metadata["id"]))
        user_id = metadata.get("user_id")
        team_id = metadata.get("team_id")
        scope = metadata.get("scope", "personal")

        # Determine Qdrant collection name (Mem0 naming convention)
        if scope == "personal" and user_id:
            collection = f"mem0_{user_id}"
        elif scope == "shared" and team_id:
            collection = f"mem0_{team_id}"
        else:
            collection = "mem0_default"

        # Fetch vector and content from Qdrant
        embedding, content = await fetch_vector_from_qdrant(
            qdrant_url, collection, memory_id
        )

        if embedding is None or content is None:
            logger.warning(
                f"[{i+1}/{len(metadata_list)}] Memory {memory_id}: "
                f"Not found in Qdrant collection {collection}. Skipping."
            )
            skipped += 1
            continue

        if dry_run:
            logger.info(
                f"[{i+1}/{len(metadata_list)}] Would migrate {memory_id}: "
                f"{content[:50]}... ({len(embedding)} dims)"
            )
            migrated += 1
            continue

        # Insert into new memories table
        success = await insert_memory(conn, metadata, embedding, content)

        if success:
            migrated += 1
            if migrated % 100 == 0:
                logger.info(f"Migrated {migrated} memories...")
        else:
            skipped += 1

    return migrated, skipped


async def main(
    database_url: str,
    qdrant_url: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run the migration."""
    logger.info("=" * 60)
    logger.info("CEMS pgvector Migration")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    # Connect to PostgreSQL
    logger.info(f"Connecting to database...")
    conn = await asyncpg.connect(database_url)
    await register_vector(conn)

    try:
        # Check table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'memories'
            )
        """)

        if not table_exists:
            logger.error(
                "The 'memories' table does not exist. "
                "Please run deploy/init.sql first to create the new schema."
            )
            return

        # Get counts
        metadata_count = await _count(conn, "memory_metadata")
        memories_count = await _count(conn, "memories")

        logger.info(f"Source: memory_metadata table has {metadata_count} entries")
        logger.info(f"Target: memories table has {memories_count} entries")

        if metadata_count == 0:
            logger.info("No data to migrate from memory_metadata table.")
            return

        if memories_count > 0:
            logger.warning(
                f"Target memories table already has {memories_count} entries. "
                "Migration will skip existing IDs."
            )

        # Fetch all metadata
        logger.info("Fetching metadata...")
        metadata_list = await fetch_all_metadata(conn)
        logger.info(f"Found {len(metadata_list)} metadata entries to migrate")

        # Migrate
        if qdrant_url:
            logger.info(f"Migrating from Qdrant ({qdrant_url})...")
            migrated, skipped = await migrate_from_qdrant(
                conn, qdrant_url, metadata_list, dry_run
            )
        else:
            logger.warning(
                "No Qdrant URL provided. Migration will be limited. "
                "Set CEMS_QDRANT_URL or use --qdrant-url to enable full migration."
            )
            migrated, skipped = await migrate_from_metadata_only(
                conn, metadata_list, dry_run
            )

        # Summary
        logger.info("=" * 60)
        logger.info("Migration Summary")
        logger.info("=" * 60)
        logger.info(f"Total metadata entries: {len(metadata_list)}")
        logger.info(f"Migrated: {migrated}")
        logger.info(f"Skipped: {skipped}")

        if not dry_run:
            final_count = await _count(conn, "memories")
            logger.info(f"Final memories count: {final_count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate CEMS data from Qdrant + memory_metadata to pgvector"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("CEMS_DATABASE_URL"),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("CEMS_QDRANT_URL"),
        help="Qdrant server URL (e.g., http://localhost:6333)",
    )

    args = parser.parse_args()

    if not args.database_url:
        logger.error(
            "Database URL required. Set CEMS_DATABASE_URL or use --database-url"
        )
        sys.exit(1)

    asyncio.run(main(
        database_url=args.database_url,
        qdrant_url=args.qdrant_url,
        dry_run=args.dry_run,
    ))
