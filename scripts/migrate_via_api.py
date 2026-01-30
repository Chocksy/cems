#!/usr/bin/env python3
"""Migrate memories from production CEMS to local via API.

Uses the CEMS REST API to fetch memories with content, then inserts to local pgvector.

Usage:
    # Set production and local credentials
    export PROD_API_URL="https://cems.chocksy.com"
    export PROD_API_KEY="cems_ak_..."
    export LOCAL_DB_URL="postgresql://cems:cems_secure_password@localhost:5432/cems"
    
    python scripts/migrate_via_api.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, UTC
from uuid import UUID

import asyncpg
import httpx


async def fetch_all_memories(api_url: str, api_key: str) -> list[dict]:
    """Fetch all memories from production API using broad search."""
    memories = []
    seen_ids = set()
    
    # Search queries to capture different memory types
    queries = [
        "code programming development",
        "preferences settings configuration", 
        "deployment server infrastructure",
        "testing debugging error",
        "database sql query",
        "api endpoint request",
        "design ui ux",
        "documentation readme",
        "git commit branch",
        "python ruby javascript",
        "memory recall search",
        "pattern convention",
        "learning insight",
        "project feature",
        "",  # Empty query for broad match
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for query in queries:
            try:
                # Use raw mode to get all results without heavy filtering
                response = await client.post(
                    f"{api_url}/api/memory/search",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query or "memory",
                        "limit": 500,
                        "raw": True,
                        "scope": "both",
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for mem in results:
                        mem_id = mem.get("memory_id") or mem.get("id")
                        if mem_id and mem_id not in seen_ids:
                            seen_ids.add(mem_id)
                            memories.append(mem)
                    
                    print(f"  Query '{query[:30]}...': {len(results)} results, total unique: {len(memories)}", file=sys.stderr)
                else:
                    print(f"  Query '{query[:30]}...' failed: {response.status_code}", file=sys.stderr)
                    
            except Exception as e:
                print(f"  Error with query '{query[:30]}...': {e}", file=sys.stderr)
    
    return memories


async def import_to_local(memories: list[dict], db_url: str):
    """Import memories to local pgvector database."""
    from pgvector.asyncpg import register_vector
    
    # Import embedding client
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.cems.embedding import EmbeddingClient
    
    conn = await asyncpg.connect(db_url)
    await register_vector(conn)
    
    embedder = EmbeddingClient()
    
    imported = 0
    skipped = 0
    errors = []
    
    print(f"\nImporting {len(memories)} memories to local database...", file=sys.stderr)
    
    for i, mem in enumerate(memories):
        content = mem.get("content", "")
        if not content or len(content) < 5:
            skipped += 1
            continue
        
        mem_id = mem.get("memory_id") or mem.get("id")
        
        try:
            # Generate fresh embedding
            embedding = embedder.embed(content)
            
            # Parse metadata
            metadata = mem.get("metadata", {}) or {}
            scope = mem.get("scope", "personal")
            if hasattr(scope, "value"):
                scope = scope.value
            
            # Get user_id - need to map to local user
            # For now, use a fixed local user ID
            local_user_id = "e96b0454-0d1d-4f15-8a97-fb52d97f1d6c"  # test-user created earlier
            
            category = metadata.get("category") or mem.get("category") or "general"
            tags = metadata.get("tags") or mem.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            
            # Insert into memories table
            await conn.execute("""
                INSERT INTO memories (
                    id, content, embedding, user_id, scope, category,
                    tags, source, source_ref, priority, pinned, pin_reason,
                    archived, access_count, created_at, updated_at, last_accessed
                ) VALUES (
                    $1::uuid, $2, $3, $4::uuid, $5, $6,
                    $7, $8, $9, $10, $11, $12,
                    $13, $14, NOW(), NOW(), NOW()
                )
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    category = EXCLUDED.category,
                    updated_at = NOW()
            """,
                mem_id,
                content,
                embedding,
                local_user_id,
                scope,
                category,
                tags,
                metadata.get("source"),
                metadata.get("source_ref"),
                float(metadata.get("priority", 1.0) or 1.0),
                bool(metadata.get("pinned", False)),
                metadata.get("pin_reason"),
                bool(metadata.get("archived", False)),
                int(metadata.get("access_count", 0) or 0),
            )
            imported += 1
            
            if imported % 50 == 0:
                print(f"  Progress: {imported}/{len(memories)} imported...", file=sys.stderr)
                
        except Exception as e:
            errors.append(f"{mem_id}: {e}")
            skipped += 1
    
    await conn.close()
    
    print(f"\nDone: {imported} imported, {skipped} skipped", file=sys.stderr)
    if errors[:5]:
        print(f"Sample errors: {errors[:5]}", file=sys.stderr)


async def main():
    prod_url = os.environ.get("PROD_API_URL", "https://cems.chocksy.com")
    prod_key = os.environ.get("PROD_API_KEY")
    local_db = os.environ.get("LOCAL_DB_URL", "postgresql://cems:cems_secure_password@localhost:5432/cems")
    
    if not prod_key:
        print("Error: PROD_API_KEY required", file=sys.stderr)
        print("Set it with: export PROD_API_KEY='cems_ak_...'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Fetching memories from {prod_url}...", file=sys.stderr)
    memories = await fetch_all_memories(prod_url, prod_key)
    
    print(f"\nFetched {len(memories)} unique memories", file=sys.stderr)
    
    if memories:
        await import_to_local(memories, local_db)


if __name__ == "__main__":
    asyncio.run(main())
