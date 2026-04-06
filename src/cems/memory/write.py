"""Write operations for CEMSMemory (add, add_async).

Document-first ingest: Every add() goes through document + chunk storage.
- Short memories = single chunk
- Long documents = multiple chunks
- Deduplication by content hash
- No fact extraction (documents stored as-is)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from cems.chunking import Chunk, chunk_document, content_hash
from cems.lib.async_utils import run_async as _run_async

if TYPE_CHECKING:
    from cems.db.document_store import DocumentStore
    from cems.memory.core import CEMSMemory

logger = logging.getLogger(__name__)

# Minimum cosine similarity to create an auto-relation
AUTO_RELATION_THRESHOLD = 0.75
# Maximum neighbors to consider per add
AUTO_RELATION_LIMIT = 10


async def _auto_link_relations(
    doc_store: "DocumentStore",
    doc_id: str,
    embedding: list[float],
    user_id: str,
    team_id: str | None,
    scope: str,
) -> int:
    """Find and link related memories for a newly added document.

    Reuses the already-computed first-chunk embedding to search for neighbors.
    Creates bidirectional relations (A→B and B→A) for each match above threshold.

    Returns:
        Number of forward relations created.
    """
    # Search for similar chunks (same user, all scopes for personal connectivity)
    neighbors = await doc_store.search_chunks(
        query_embedding=embedding,
        user_id=user_id,
        limit=AUTO_RELATION_LIMIT,
    )

    # Build relations for neighbors above threshold
    relations: list[dict[str, Any]] = []
    seen_docs: set[str] = {doc_id}  # Skip self

    for neighbor in neighbors:
        neighbor_doc_id = neighbor["document_id"]
        score = neighbor.get("score", 0)

        if neighbor_doc_id in seen_docs:
            continue
        if score < AUTO_RELATION_THRESHOLD:
            continue

        seen_docs.add(neighbor_doc_id)
        relations.append({
            "target_id": neighbor_doc_id,
            "relation_type": "similar",
            "similarity": score,
        })

    if not relations:
        return 0

    # Insert forward relations (new → existing)
    created = await doc_store.add_relations(doc_id, relations)

    # Insert reverse relations (existing → new) for bidirectional graph
    for rel in relations:
        await doc_store.add_relations(
            rel["target_id"],
            [{"target_id": doc_id, "relation_type": rel["relation_type"], "similarity": rel["similarity"]}],
        )

    logger.info(f"[WRITE] Auto-linked {doc_id[:8]} to {created} neighbors (bidirectional)")
    return created


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
        timestamp: datetime | None = None,  # For historical imports
        title: str | None = None,
        content_detailed: str | None = None,
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
                title=title,
                source=source,
                source_ref=source_ref,
                tags=tags,
                content_detailed=content_detailed,
            )

            if is_new:
                logger.info(
                    f"[WRITE] Added document {doc_id[:8]}... with {len(chunks)} chunks, "
                    f"scope={scope}, category={category}, source_ref={source_ref}, tags={tags}"
                )
                event = "ADD"

                # Step 4: Auto-link to related memories (best-effort)
                relations_created = 0
                try:
                    relations_created = await _auto_link_relations(
                        doc_store, doc_id, embeddings[0], user_id, team_id, scope,
                    )
                except Exception as link_err:
                    logger.warning(f"[WRITE] Auto-link failed for {doc_id[:8]}: {link_err}")
            else:
                logger.debug(f"[WRITE] Document {doc_id[:8]}... already exists (deduplicated)")
                event = "DUPLICATE"
                relations_created = 0

            result_dict: dict[str, Any] = {
                "id": doc_id,
                "event": event,
                "memory": content[:200] + "..." if len(content) > 200 else content,
                "chunks": len(chunks),
                "is_new": is_new,
            }
            if relations_created > 0:
                result_dict["relations_created"] = relations_created

            return {"results": [result_dict]}

        except Exception as e:
            logger.error(f"[WRITE] Failed to add document: {e}")
            return {"results": [{"event": "ERROR", "error": str(e)}]}

