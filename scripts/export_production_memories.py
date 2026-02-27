#!/usr/bin/env python3
"""Export memories from production for local import.

Exports memory_metadata content from production PostgreSQL.
The local import will re-generate embeddings using OpenRouter.

Usage:
    # Export from production
    PROD_DATABASE_URL="postgresql://..." python scripts/export_production_memories.py export > memories.json
    
    # Import to local (will re-embed)
    CEMS_DATABASE_URL="postgresql://..." python scripts/export_production_memories.py import < memories.json
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import asyncpg


def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


async def export_memories():
    """Export all memory metadata from production."""
    db_url = os.environ.get("PROD_DATABASE_URL") or os.environ.get("CEMS_DATABASE_URL")
    if not db_url:
        print("Error: PROD_DATABASE_URL or CEMS_DATABASE_URL required", file=sys.stderr)
        sys.exit(1)

    conn = await asyncpg.connect(db_url)
    
    # Get all memory metadata with content from the old schema
    rows = await conn.fetch("""
        SELECT 
            memory_id,
            user_id,
            scope,
            category,
            created_at,
            updated_at,
            last_accessed,
            access_count,
            source,
            source_ref,
            tags,
            archived,
            priority,
            pinned,
            pin_reason,
            pin_category,
            expires_at
        FROM memory_metadata
        WHERE archived = FALSE
        ORDER BY created_at DESC
    """)
    
    memories = []
    for row in rows:
        memories.append({
            "memory_id": row["memory_id"],
            "user_id": row["user_id"],
            "scope": row["scope"],
            "category": row["category"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_accessed": row["last_accessed"],
            "access_count": row["access_count"],
            "source": row["source"],
            "source_ref": row["source_ref"],
            "tags": row["tags"] or [],
            "archived": row["archived"],
            "priority": row["priority"],
            "pinned": row["pinned"],
            "pin_reason": row["pin_reason"],
            "pin_category": row["pin_category"],
            "expires_at": row["expires_at"],
        })
    
    await conn.close()
    
    print(json.dumps(memories, default=json_serializer, indent=2))
    print(f"Exported {len(memories)} memories", file=sys.stderr)


async def import_memories():
    """Import memories to local pgvector database."""
    from cems.embedding import EmbeddingClient
    
    db_url = os.environ.get("CEMS_DATABASE_URL")
    if not db_url:
        print("Error: CEMS_DATABASE_URL required", file=sys.stderr)
        sys.exit(1)
    
    # Read memories from stdin
    data = json.load(sys.stdin)
    print(f"Importing {len(data)} memories...", file=sys.stderr)
    
    conn = await asyncpg.connect(db_url)
    embedder = EmbeddingClient()

    try:
        # Register pgvector
        from pgvector.asyncpg import register_vector
        await register_vector(conn)

        imported = 0
        skipped = 0

        for mem in data:
            content = mem.get("content", "")
            if not content:
                skipped += 1
                continue

            try:
                # Generate embedding
                embedding = embedder.embed(content)

                # Insert into memories table (legacy — orphaned, kept for migration scripts)
                await conn.execute("""
                    INSERT INTO memories (
                        id, content, embedding, user_id, scope, category,
                        tags, source, source_ref, priority, pinned, pin_reason,
                        archived, access_count, created_at, updated_at, last_accessed, expires_at
                    ) VALUES (
                        $1::uuid, $2, $3, $4::uuid, $5, $6,
                        $7, $8, $9, $10, $11, $12,
                        $13, $14, $15, $16, $17, $18
                    )
                    ON CONFLICT (id) DO NOTHING
                """,
                    mem["memory_id"],
                    content,
                    embedding,
                    mem["user_id"],
                    mem["scope"],
                    mem["category"],
                    mem["tags"],
                    mem["source"],
                    mem["source_ref"],
                    mem["priority"] or 1.0,
                    mem["pinned"] or False,
                    mem["pin_reason"],
                    mem["archived"] or False,
                    mem["access_count"] or 0,
                    datetime.fromisoformat(mem["created_at"]) if mem["created_at"] else datetime.utcnow(),
                    datetime.fromisoformat(mem["updated_at"]) if mem["updated_at"] else datetime.utcnow(),
                    datetime.fromisoformat(mem["last_accessed"]) if mem["last_accessed"] else datetime.utcnow(),
                    datetime.fromisoformat(mem["expires_at"]) if mem.get("expires_at") else None,
                )
                imported += 1

                if imported % 100 == 0:
                    print(f"  Imported {imported}...", file=sys.stderr)

            except Exception as e:
                print(f"  Error importing {mem['memory_id']}: {e}", file=sys.stderr)
                skipped += 1

        print(f"Done: {imported} imported, {skipped} skipped", file=sys.stderr)
    finally:
        await conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_production_memories.py [export|import]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == "export":
        asyncio.run(export_memories())
    elif cmd == "import":
        asyncio.run(import_memories())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
