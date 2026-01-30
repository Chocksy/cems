#!/usr/bin/env python3
"""Re-embed all memories using the configured embedding backend.

This script is part of the Strategy A migration to 768-dim embeddings.
It reads all memories and regenerates embeddings via the configured backend
(llamacpp_server or openrouter).

Usage:
    CEMS_DATABASE_URL=postgresql://... \
    CEMS_EMBEDDING_BACKEND=llamacpp_server \
    CEMS_EMBEDDING_DIMENSION=768 \
    uv run python scripts/reembed_all_memories.py
"""

import asyncio
from cems.config import CEMSConfig
from cems.memory import CEMSMemory


async def reembed_all():
    cfg = CEMSConfig()
    memory = CEMSMemory(cfg)
    await memory._ensure_initialized_async()

    ids = memory.metadata_store.get_all_user_memories(cfg.user_id, include_archived=False)
    total = len(ids)
    print(f"[reembed] Found {total} memories to re-embed")
    print(f"[reembed] Backend: {cfg.embedding_backend}, Dimension: {cfg.embedding_dimension}")

    for i, mem_id in enumerate(ids, 1):
        mem = memory.get(mem_id)
        if not mem:
            continue
        # update() regenerates embedding via configured backend
        memory.update(mem_id, mem["memory"])
        if i % 50 == 0:
            print(f"[reembed] Progress: {i}/{total}")

    print(f"[reembed] Done. Re-embedded {total} memories.")


if __name__ == "__main__":
    asyncio.run(reembed_all())
