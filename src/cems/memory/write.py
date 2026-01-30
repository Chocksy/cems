"""Write operations for CEMSMemory (add, add_async).

Document-first ingest: Every add() goes through document + chunk storage.
- Short memories = single chunk
- Long documents = multiple chunks
- Deduplication by content hash
- No fact extraction (documents stored as-is)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from cems.chunking import Chunk, chunk_document, content_hash
from cems.models import MemoryScope

if TYPE_CHECKING:
    from cems.db.document_store import DocumentStore
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine in a sync context."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        raise RuntimeError(
            "Cannot use sync method from async context. "
            "Use the async version (e.g., add_async instead of add)."
        )
    else:
        return asyncio.run(coro)


class WriteMixin:
    """Mixin class providing write operations for CEMSMemory.

    Document-first ingest model:
    - Every add() stores content as a document with chunks
    - Chunks are embedded individually for search
    - No truncation issues (chunking handles long content)
    - Deduplication by content hash
    """

    # Document store instance (lazy initialized)
    _document_store: "DocumentStore | None" = None

    async def _ensure_document_store(self: "CEMSMemory") -> "DocumentStore":
        """Ensure document store is initialized."""
        if self._document_store is None:
            from cems.db.document_store import DocumentStore

            self._document_store = DocumentStore(
                database_url=self.config.database_url,
                embedding_dim=self.config.embedding_dimension,
            )
            await self._document_store.connect()
        return self._document_store

    def add(
        self: "CEMSMemory",
        content: str,
        scope: Literal["personal", "shared"] = "personal",
        category: str = "general",
        source: str | None = None,
        tags: list[str] | None = None,
        infer: bool = True,  # Ignored - kept for API compatibility
        source_ref: str | None = None,
        ttl_hours: int | None = None,
        pinned: bool = False,
        pin_reason: str | None = None,
    ) -> dict[str, Any]:
        """Add a memory to the specified namespace (sync version).

        Args:
            content: The content to remember
            scope: "personal" or "shared"
            category: Category for organization
            source: Optional source identifier
            tags: Optional tags for organization
            infer: Ignored (kept for API compatibility)
            source_ref: Optional project reference for scoped recall
            ttl_hours: Optional TTL in hours (not yet implemented for documents)
            pinned: If True, memory is pinned (not yet implemented for documents)
            pin_reason: Optional reason for pinning the memory

        Returns:
            Dict with memory operation results
        """
        return _run_async(
            self.add_async(
                content=content,
                scope=scope,
                category=category,
                source=source,
                tags=tags,
                infer=infer,
                source_ref=source_ref,
                ttl_hours=ttl_hours,
                pinned=pinned,
                pin_reason=pin_reason,
            )
        )

    async def add_async(
        self: "CEMSMemory",
        content: str,
        scope: Literal["personal", "shared"] = "personal",
        category: str = "general",
        source: str | None = None,
        tags: list[str] | None = None,
        infer: bool = True,  # Ignored - kept for API compatibility
        source_ref: str | None = None,
        ttl_hours: int | None = None,
        pinned: bool = False,
        pin_reason: str | None = None,
        timestamp: datetime | None = None,  # For historical imports
    ) -> dict[str, Any]:
        """Async add - document-first ingest with chunking.

        This is the primary method for adding content to CEMS:
        1. Chunks content (800 tokens, 15% overlap)
        2. Embeds each chunk
        3. Stores document + chunks in PostgreSQL

        Args:
            content: The content to remember
            scope: "personal" or "shared"
            category: Category for organization
            source: Optional source identifier
            tags: Optional tags for organization
            infer: Ignored (kept for API compatibility, fact extraction disabled)
            source_ref: Optional project reference for scoped recall
            ttl_hours: Optional TTL in hours (not yet implemented for documents)
            pinned: If True, memory is pinned (not yet implemented for documents)
            pin_reason: Optional reason for pinning
            timestamp: Optional historical timestamp for the memory (for imports/evals)

        Returns:
            Dict with memory operation results including document_id
        """
        await self._ensure_initialized_async()
        doc_store = await self._ensure_document_store()
        assert self._async_embedder is not None

        # Validate
        if not content or not content.strip():
            return {"results": [{"event": "ERROR", "error": "Empty content"}]}

        # Get user/team IDs
        user_id = self.config.user_id
        team_id = self.config.team_id if scope == "shared" else None

        if not user_id:
            return {"results": [{"event": "ERROR", "error": "No user_id configured"}]}

        try:
            # Step 1: Chunk the content
            chunks: list[Chunk] = chunk_document(content)
            if not chunks:
                return {"results": [{"event": "ERROR", "error": "Chunking produced no output"}]}

            logger.debug(f"[WRITE] Chunked content into {len(chunks)} chunks")

            # Step 2: Embed all chunks (batched for efficiency)
            chunk_texts = [c.content for c in chunks]
            embeddings = await self._async_embedder.embed_batch(chunk_texts)

            if len(embeddings) != len(chunks):
                return {
                    "results": [
                        {
                            "event": "ERROR",
                            "error": f"Embedding mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings",
                        }
                    ]
                }

            logger.debug(f"[WRITE] Generated {len(embeddings)} embeddings")

            # Step 3: Store document + chunks
            doc_id, is_new = await doc_store.add_document(
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                category=category,
                title=None,  # Could extract title from content in future
                source=source,
                source_ref=source_ref,
                tags=tags,
            )

            if is_new:
                logger.info(
                    f"[WRITE] Added document {doc_id[:8]}... with {len(chunks)} chunks, "
                    f"scope={scope}, category={category}, source_ref={source_ref}, tags={tags}"
                )
                event = "ADD"
            else:
                logger.debug(f"[WRITE] Document {doc_id[:8]}... already exists (deduplicated)")
                event = "DUPLICATE"

            return {
                "results": [
                    {
                        "id": doc_id,
                        "event": event,
                        "memory": content[:200] + "..." if len(content) > 200 else content,
                        "chunks": len(chunks),
                        "is_new": is_new,
                    }
                ]
            }

        except Exception as e:
            logger.error(f"[WRITE] Failed to add document: {e}")
            return {"results": [{"event": "ERROR", "error": str(e)}]}

    async def add_legacy_async(
        self: "CEMSMemory",
        content: str,
        scope: Literal["personal", "shared"] = "personal",
        category: str = "general",
        source: str | None = None,
        tags: list[str] | None = None,
        infer: bool = True,
        source_ref: str | None = None,
        ttl_hours: int | None = None,
        pinned: bool = False,
        pin_reason: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Legacy add using old memories table (for backwards compatibility).

        Use add_async() for new code - this is only for migration purposes.
        """
        await self._ensure_initialized_async()
        assert self._vectorstore is not None
        assert self._async_embedder is not None

        memory_scope = MemoryScope(scope)

        # Extract facts if infer is enabled
        contents_to_store = [content]
        if infer and self._fact_extractor:
            try:
                facts = self._fact_extractor.extract(content)
                if facts:
                    contents_to_store = facts
            except Exception as e:
                logger.warning(f"Fact extraction failed, storing raw content: {e}")

        # Calculate expires_at if TTL is set
        expires_at = None
        if ttl_hours:
            expires_at = datetime.now(UTC) + timedelta(hours=ttl_hours)

        # Get user/team IDs
        user_id = self.config.user_id
        team_id = self.config.team_id if scope == "shared" else None

        results = []
        for fact_content in contents_to_store:
            try:
                # Generate embedding using async embedder
                embedding = await self._async_embedder.embed(fact_content)

                # Store in pgvector (old memories table)
                memory_id = await self._vectorstore.add(
                    content=fact_content,
                    embedding=embedding,
                    user_id=user_id,
                    team_id=team_id,
                    scope=scope,
                    category=category,
                    tags=tags,
                    source=source,
                    source_ref=source_ref,
                    priority=1.0,
                    pinned=pinned,
                    pin_reason=pin_reason,
                    expires_at=expires_at,
                    created_at=timestamp,  # For historical imports
                )

                results.append(
                    {
                        "id": memory_id,
                        "event": "ADD",
                        "memory": fact_content,
                    }
                )

                # Add relations to similar memories
                if self._use_pg_relations:
                    try:
                        similar = await self._vectorstore.search(
                            query_embedding=embedding,
                            user_id=user_id,
                            team_id=team_id,
                            scope=scope,
                            limit=5,
                        )
                        for sim_mem in similar:
                            if sim_mem["id"] != memory_id and sim_mem.get("score", 0) > 0.7:
                                await self._vectorstore.add_relation(
                                    source_id=memory_id,
                                    target_id=sim_mem["id"],
                                    relation_type="similar",
                                    similarity=sim_mem.get("score"),
                                )
                    except Exception as e:
                        logger.debug(f"Failed to add relations: {e}")

            except Exception as e:
                logger.error(f"Failed to add memory: {e}")
                results.append(
                    {
                        "event": "ERROR",
                        "error": str(e),
                    }
                )

        return {"results": results}
