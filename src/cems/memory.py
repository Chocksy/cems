"""CEMS Memory with native pgvector storage.

This module provides unified memory management using PostgreSQL with pgvector
for both vector embeddings and metadata storage. It replaces the previous
Mem0 + Qdrant architecture with a simpler, ACID-compliant solution.

Key features:
- Vector similarity search (HNSW index)
- Full-text search (GIN index on tsvector)
- Hybrid search using RRF (Reciprocal Rank Fusion)
- ACID transactions for data consistency
- Namespace isolation (personal vs shared memories)
- Extended metadata tracking (access counts, priorities)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from cems.config import CEMSConfig
from cems.models import (
    MemoryMetadata,
    MemoryScope,
    SearchResult,
)

if TYPE_CHECKING:
    from cems.db.metadata_store import PostgresMetadataStore
    from cems.embedding import EmbeddingClient
    from cems.fact_extraction import FactExtractor
    from cems.vectorstore import PgVectorStore

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine in a sync context.

    NOTE: This is for sync contexts only (CLI, MCP stdio).
    For async contexts (HTTP server), use the async methods directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context - caller should use async methods
        raise RuntimeError(
            "Cannot use sync method from async context. "
            "Use the async version (e.g., add_async instead of add)."
        )
    else:
        # No running loop - create one
        return asyncio.run(coro)


class CEMSMemory:
    """Memory system with personal/shared namespace isolation.

    Built on PostgreSQL + pgvector, this class provides:
    - Namespace isolation (personal vs shared memories)
    - Extended metadata tracking (access counts, priorities)
    - Unified search across namespaces
    - Hybrid search (vector + full-text)
    - ACID transactions for consistency
    """

    def __init__(self, config: CEMSConfig | None = None):
        """Initialize CEMS memory.

        Args:
            config: CEMS configuration. If None, loads from environment.
        """
        self.config = config or CEMSConfig()

        # Validate database URL
        if not self.config.database_url:
            raise ValueError(
                "CEMS_DATABASE_URL is required. "
                "CEMS runs in Docker/server mode only (no local SQLite mode)."
            )

        # Initialize components lazily
        self._vectorstore: PgVectorStore | None = None
        self._embedder: EmbeddingClient | None = None
        self._fact_extractor: FactExtractor | None = None
        self._metadata: PostgresMetadataStore | None = None
        self._initialized = False

        # Initialize legacy metadata store for backwards compatibility
        # This will be deprecated once migration is complete
        from cems.db.database import init_database, is_database_initialized
        from cems.db.metadata_store import PostgresMetadataStore

        if not is_database_initialized():
            init_database(self.config.database_url)
        self._metadata = PostgresMetadataStore()

        # Graph store (using PostgreSQL relations table instead of Kuzu)
        self._use_pg_relations = True

    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized (sync version for CLI/MCP)."""
        if self._initialized:
            return

        from cems.embedding import EmbeddingClient
        from cems.fact_extraction import FactExtractor
        from cems.vectorstore import PgVectorStore

        # Initialize vectorstore
        if self._vectorstore is None:
            self._vectorstore = PgVectorStore(self.config.database_url)
            # Connect synchronously - only works in sync contexts
            _run_async(self._vectorstore.connect())

        # Initialize embedder
        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.config.embedding_model)

        # Initialize fact extractor
        if self._fact_extractor is None:
            self._fact_extractor = FactExtractor()

        self._initialized = True

    async def _ensure_initialized_async(self) -> None:
        """Ensure all components are initialized (async version for HTTP server)."""
        if self._initialized:
            return

        from cems.embedding import EmbeddingClient
        from cems.fact_extraction import FactExtractor
        from cems.vectorstore import PgVectorStore

        # Initialize vectorstore
        if self._vectorstore is None:
            self._vectorstore = PgVectorStore(self.config.database_url)
            await self._vectorstore.connect()

        # Initialize embedder
        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.config.embedding_model)

        # Initialize fact extractor
        if self._fact_extractor is None:
            self._fact_extractor = FactExtractor()

        self._initialized = True

    def _infer_category_from_query(self, query: str) -> str | None:
        """Infer the likely category from a search query.

        Uses keyword matching to determine if a query is about a specific domain.
        Returns None if no category can be inferred.

        Args:
            query: Search query text

        Returns:
            Inferred category name or None
        """
        query_lower = query.lower()

        # Category keyword mappings (keywords -> category)
        category_keywords = {
            "memory": ["memory", "recall", "remember", "retrieval", "search", "embedding"],
            "deployment": ["deploy", "coolify", "server", "production", "hosting", "docker"],
            "development": ["code", "coding", "programming", "debug", "git", "refactor"],
            "ai": ["llm", "claude", "openai", "gpt", "ai", "model", "prompt"],
            "project": ["project", "repo", "repository", "codebase"],
            "preferences": ["prefer", "preference", "like", "favorite", "style"],
            "workflow": ["workflow", "process", "habit", "routine", "automate"],
        }

        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return category

        return None

    def add(
        self,
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
    ) -> dict[str, Any]:
        """Add a memory to the specified namespace.

        Args:
            content: The content to remember
            scope: "personal" or "shared"
            category: Category for organization
            source: Optional source identifier
            tags: Optional tags for organization
            infer: If True (default), use LLM for fact extraction.
                   If False, store raw content directly (faster).
            source_ref: Optional project reference for scoped recall
            ttl_hours: Optional TTL in hours. If set, memory expires after this time.
            pinned: If True, memory is pinned and never auto-pruned.
            pin_reason: Optional reason for pinning the memory.

        Returns:
            Dict with memory operation results
        """
        self._ensure_initialized()
        assert self._vectorstore is not None
        assert self._embedder is not None

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
                # Generate embedding
                embedding = self._embedder.embed(fact_content)

                # Store in pgvector
                memory_id = _run_async(
                    self._vectorstore.add(
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
                    )
                )

                results.append({
                    "id": memory_id,
                    "event": "ADD",
                    "memory": fact_content,
                })

                # Add relations to similar memories
                if self._use_pg_relations:
                    try:
                        similar = _run_async(
                            self._vectorstore.search(
                                query_embedding=embedding,
                                user_id=user_id,
                                team_id=team_id,
                                scope=scope,
                                limit=5,
                            )
                        )
                        for sim_mem in similar:
                            if sim_mem["id"] != memory_id and sim_mem.get("score", 0) > 0.7:
                                _run_async(
                                    self._vectorstore.add_relation(
                                        source_id=memory_id,
                                        target_id=sim_mem["id"],
                                        relation_type="similar",
                                        similarity=sim_mem.get("score"),
                                    )
                                )
                    except Exception as e:
                        logger.debug(f"Failed to add relations: {e}")

            except Exception as e:
                logger.error(f"Failed to add memory: {e}")
                results.append({
                    "event": "ERROR",
                    "error": str(e),
                })

        return {"results": results}

    async def add_async(
        self,
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
    ) -> dict[str, Any]:
        """Async version of add(). Use this from async contexts (HTTP server)."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None
        assert self._embedder is not None

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
                # Generate embedding
                embedding = self._embedder.embed(fact_content)

                # Store in pgvector
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
                )

                results.append({
                    "id": memory_id,
                    "event": "ADD",
                    "memory": fact_content,
                })

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
                results.append({
                    "event": "ERROR",
                    "error": str(e),
                })

        return {"results": results}

    def search(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories across namespaces.

        Args:
            query: Search query
            scope: Which namespace(s) to search
            category: Optional category filter
            limit: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        self._ensure_initialized()
        assert self._vectorstore is not None
        assert self._embedder is not None

        # Generate query embedding
        query_embedding = self._embedder.embed(query)

        # Determine user/team for filtering
        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Search with hybrid (vector + full-text)
        raw_results = _run_async(
            self._vectorstore.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                category=category,
                limit=limit * 2,  # Fetch extra for filtering
                vector_weight=self.config.hybrid_vector_weight if hasattr(self.config, 'hybrid_vector_weight') else 0.7,
            )
        )

        # Convert to SearchResult objects
        results: list[SearchResult] = []
        memory_ids: list[str] = []

        for mem in raw_results:
            memory_scope = MemoryScope(mem.get("scope", "personal"))

            # Create MemoryMetadata from the result
            metadata = MemoryMetadata(
                memory_id=mem["id"],
                user_id=mem.get("user_id", user_id),
                scope=memory_scope,
                category=mem.get("category", "general"),
                source=mem.get("source"),
                source_ref=mem.get("source_ref"),
                tags=mem.get("tags", []),
                priority=mem.get("priority", 1.0),
                pinned=mem.get("pinned", False),
                pin_reason=mem.get("pin_reason"),
                archived=mem.get("archived", False),
                access_count=mem.get("access_count", 0),
                created_at=mem.get("created_at", datetime.now(UTC)),
                updated_at=mem.get("updated_at", datetime.now(UTC)),
                last_accessed=mem.get("last_accessed", datetime.now(UTC)),
                expires_at=mem.get("expires_at"),
            )

            results.append(
                SearchResult(
                    memory_id=mem["id"],
                    content=mem.get("content", ""),
                    score=mem.get("score", 0.0),
                    scope=memory_scope,
                    metadata=metadata,
                )
            )
            memory_ids.append(mem["id"])

        # Record access
        if memory_ids:
            _run_async(self._vectorstore.record_access_batch(memory_ids))

        # Apply priority boost and time decay to scores
        now = datetime.now(UTC)
        inferred_category = self._infer_category_from_query(query)

        for result in results:
            if result.metadata:
                # Priority boost (1.0 default, up to 2.0 for hot memories)
                result.score *= result.metadata.priority

                # Time decay: 50% penalty per month since last access
                days_since_access = (now - result.metadata.last_accessed).days
                time_decay = 1.0 / (1.0 + (days_since_access / 30))
                result.score *= time_decay

                # Boost pinned memories slightly (they're important)
                if result.metadata.pinned:
                    result.score *= 1.1

                # Cross-category penalty
                if inferred_category and result.metadata.category:
                    if result.metadata.category.lower() != inferred_category:
                        result.score *= 0.8

        # Sort by adjusted score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def search_async(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Async version of search(). Use this from async contexts (HTTP server)."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None
        assert self._embedder is not None

        # Generate query embedding
        query_embedding = self._embedder.embed(query)

        # Determine user/team for filtering
        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Search with hybrid (vector + full-text)
        raw_results = await self._vectorstore.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            category=category,
            limit=limit * 2,  # Fetch extra for filtering
            vector_weight=self.config.hybrid_vector_weight if hasattr(self.config, 'hybrid_vector_weight') else 0.7,
        )

        # Convert to SearchResult objects
        results: list[SearchResult] = []
        memory_ids: list[str] = []

        for mem in raw_results:
            memory_scope = MemoryScope(mem.get("scope", "personal"))

            # Create MemoryMetadata from the result
            metadata = MemoryMetadata(
                memory_id=mem["id"],
                user_id=mem.get("user_id", user_id),
                scope=memory_scope,
                category=mem.get("category", "general"),
                source=mem.get("source"),
                source_ref=mem.get("source_ref"),
                tags=mem.get("tags", []),
                priority=mem.get("priority", 1.0),
                pinned=mem.get("pinned", False),
                pin_reason=mem.get("pin_reason"),
                archived=mem.get("archived", False),
                access_count=mem.get("access_count", 0),
                created_at=mem.get("created_at", datetime.now(UTC)),
                updated_at=mem.get("updated_at", datetime.now(UTC)),
                last_accessed=mem.get("last_accessed", datetime.now(UTC)),
                expires_at=mem.get("expires_at"),
            )

            results.append(
                SearchResult(
                    memory_id=mem["id"],
                    content=mem.get("content", ""),
                    score=mem.get("score", 0.0),
                    scope=memory_scope,
                    metadata=metadata,
                )
            )
            memory_ids.append(mem["id"])

        # Record access
        if memory_ids:
            await self._vectorstore.record_access_batch(memory_ids)

        # Apply priority boost and time decay to scores
        now = datetime.now(UTC)
        inferred_category = self._infer_category_from_query(query)

        for result in results:
            if result.metadata:
                result.score *= result.metadata.priority
                days_since_access = (now - result.metadata.last_accessed).days
                time_decay = 1.0 / (1.0 + (days_since_access / 30))
                result.score *= time_decay
                if result.metadata.pinned:
                    result.score *= 1.1
                if inferred_category and result.metadata.category:
                    if result.metadata.category.lower() != inferred_category:
                        result.score *= 0.8

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def delete_async(self, memory_id: str, hard: bool = False) -> None:
        """Async version of delete(). Use this from async contexts (HTTP server)."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None

        if hard:
            await self._vectorstore.delete(memory_id)
        else:
            await self._vectorstore.update(memory_id, archived=True)

    async def update_async(self, memory_id: str, content: str) -> dict[str, Any]:
        """Async version of update(). Use this from async contexts (HTTP server)."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None
        assert self._embedder is not None

        # Generate new embedding
        embedding = self._embedder.embed(content)

        await self._vectorstore.update(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
        )

        return {"success": True, "memory_id": memory_id}

    async def get_category_counts_async(
        self,
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> dict[str, int]:
        """Async version of get_category_counts()."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None

        user_id = self.config.user_id

        return await self._vectorstore.get_category_counts(
            user_id=user_id,
            scope=scope if scope != "both" else None,
        )

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Memory dict or None if not found
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        result = _run_async(self._vectorstore.get(memory_id))
        if result:
            _run_async(self._vectorstore.record_access(memory_id))
            # Return in Mem0-compatible format
            return {
                "id": result["id"],
                "memory": result["content"],
                "metadata": {
                    "category": result.get("category"),
                    "source": result.get("source"),
                    "tags": result.get("tags", []),
                },
            }
        return None

    def get_all(
        self,
        scope: Literal["personal", "shared", "both"] = "both",
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get all memories in a scope.

        Args:
            scope: Which namespace to get
            include_archived: Whether to include archived memories

        Returns:
            List of memory dicts
        """
        self._ensure_initialized()
        assert self._vectorstore is not None
        assert self._embedder is not None

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Use a dummy embedding to get all (or use a dedicated method)
        # For now, we'll do a broad search with high limit
        dummy_embedding = self._embedder.embed("memory")

        results = _run_async(
            self._vectorstore.search(
                query_embedding=dummy_embedding,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                limit=1000,
                include_archived=include_archived,
            )
        )

        # Convert to Mem0-compatible format
        memories = []
        for mem in results:
            memories.append({
                "id": mem["id"],
                "memory": mem["content"],
                "scope": mem.get("scope", "personal"),
                "metadata": {
                    "category": mem.get("category"),
                    "source": mem.get("source"),
                    "tags": mem.get("tags", []),
                },
            })

        return memories

    def update(self, memory_id: str, content: str) -> dict[str, Any]:
        """Update a memory's content.

        Args:
            memory_id: The memory ID to update
            content: New content

        Returns:
            Update result dict
        """
        self._ensure_initialized()
        assert self._vectorstore is not None
        assert self._embedder is not None

        # Generate new embedding
        embedding = self._embedder.embed(content)

        success = _run_async(
            self._vectorstore.update(
                memory_id=memory_id,
                content=content,
                embedding=embedding,
            )
        )

        if success:
            return {"status": "updated", "id": memory_id}
        return {"status": "not_found", "id": memory_id}

    def delete(self, memory_id: str, hard: bool = False) -> dict[str, Any]:
        """Delete or archive a memory.

        Args:
            memory_id: The memory ID to delete
            hard: If True, permanently delete. If False, archive.

        Returns:
            Delete result dict
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        success = _run_async(self._vectorstore.delete(memory_id, hard=hard))

        if success:
            status = "deleted" if hard else "archived"
            return {"status": status, "memory_id": memory_id}
        return {"status": "not_found", "memory_id": memory_id}

    def forget(self, memory_id: str) -> dict[str, Any]:
        """Forget (soft delete) a memory.

        Args:
            memory_id: The memory ID to forget

        Returns:
            Result dict
        """
        return self.delete(memory_id, hard=False)

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        """Get the history of a memory.

        Note: pgvector doesn't track history by default.
        This returns an empty list for compatibility.

        Args:
            memory_id: The memory ID

        Returns:
            List of history entries (empty for pgvector)
        """
        # History tracking not implemented in pgvector
        # Would need a separate audit table
        return []

    def get_stale_memories(self, days: int | None = None) -> list[str]:
        """Get memories that haven't been accessed in N days.

        Args:
            days: Days threshold. Uses config default if not specified.

        Returns:
            List of stale memory IDs
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        days = days or self.config.stale_days
        return _run_async(
            self._vectorstore.get_stale_memories(self.config.user_id, days)
        )

    def get_hot_memories(self, threshold: int | None = None) -> list[str]:
        """Get frequently accessed memories.

        Args:
            threshold: Access count threshold. Uses config default if not specified.

        Returns:
            List of hot memory IDs
        """
        # This requires a query on access_count
        # For now, delegate to metadata store if available
        if self._metadata:
            threshold = threshold or self.config.hot_access_threshold
            return self._metadata.get_hot_memories(self.config.user_id, threshold)
        return []

    def get_recent_memories(self, hours: int = 24) -> list[str]:
        """Get memories created in the last N hours.

        Args:
            hours: Hours to look back

        Returns:
            List of memory IDs
        """
        if self._metadata:
            return self._metadata.get_recent_memories(self.config.user_id, hours)
        return []

    def get_old_memories(self, days: int = 30) -> list[str]:
        """Get memories older than N days.

        Args:
            days: Days threshold

        Returns:
            List of memory IDs
        """
        if self._metadata:
            return self._metadata.get_old_memories(self.config.user_id, days)
        return []

    def promote_memory(self, memory_id: str, boost: float = 0.1) -> None:
        """Increase a memory's priority.

        Args:
            memory_id: The memory ID
            boost: Priority boost amount
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        # Get current priority
        mem = _run_async(self._vectorstore.get(memory_id))
        if mem:
            new_priority = min(mem.get("priority", 1.0) + boost, 2.0)
            _run_async(
                self._vectorstore.update(memory_id, priority=new_priority)
            )

    def archive_memory(self, memory_id: str) -> None:
        """Archive a memory (soft delete).

        Args:
            memory_id: The memory ID
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        _run_async(self._vectorstore.update(memory_id, archived=True))

    def get_metadata(self, memory_id: str) -> MemoryMetadata | None:
        """Get extended metadata for a memory.

        Args:
            memory_id: The memory ID

        Returns:
            MemoryMetadata or None
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        mem = _run_async(self._vectorstore.get(memory_id))
        if not mem:
            return None

        return MemoryMetadata(
            memory_id=mem["id"],
            user_id=mem.get("user_id", self.config.user_id),
            scope=MemoryScope(mem.get("scope", "personal")),
            category=mem.get("category", "general"),
            source=mem.get("source"),
            source_ref=mem.get("source_ref"),
            tags=mem.get("tags", []),
            priority=mem.get("priority", 1.0),
            pinned=mem.get("pinned", False),
            pin_reason=mem.get("pin_reason"),
            archived=mem.get("archived", False),
            access_count=mem.get("access_count", 0),
            created_at=mem.get("created_at", datetime.now(UTC)),
            updated_at=mem.get("updated_at", datetime.now(UTC)),
            last_accessed=mem.get("last_accessed", datetime.now(UTC)),
            expires_at=mem.get("expires_at"),
        )

    def get_category_counts(
        self, scope: Literal["personal", "shared"] | None = None
    ) -> dict[str, int]:
        """Get memory counts grouped by category.

        Args:
            scope: Optional filter by scope

        Returns:
            Dict mapping category name to count
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        return _run_async(
            self._vectorstore.get_category_counts(self.config.user_id, scope)
        )

    @property
    def metadata_store(self) -> "PostgresMetadataStore":
        """Access the metadata store directly for maintenance operations."""
        return self._metadata

    @property
    def vectorstore(self) -> "PgVectorStore":
        """Access the underlying vectorstore instance."""
        self._ensure_initialized()
        return self._vectorstore

    def get_all_categories(
        self,
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> list[dict]:
        """Get all categories with their memory counts.

        Args:
            scope: Which namespace to get categories for

        Returns:
            List of dicts with category name, scope, and count
        """
        if self._metadata:
            if scope == "personal":
                return self._metadata.get_all_categories(
                    self.config.user_id, MemoryScope.PERSONAL
                )
            elif scope == "shared" and self.config.team_id:
                return self._metadata.get_all_categories(
                    self.config.user_id, MemoryScope.SHARED
                )
            else:
                return self._metadata.get_all_categories(self.config.user_id)
        return []

    def get_recently_accessed(self, limit: int = 10) -> list[dict]:
        """Get recently accessed memories.

        Args:
            limit: Maximum number of results

        Returns:
            List of dicts with memory info and access timestamps
        """
        if self._metadata:
            return self._metadata.get_recently_accessed(self.config.user_id, limit)
        return []

    def get_category_summary(
        self,
        category: str,
        scope: Literal["personal", "shared"] = "personal",
    ) -> dict | None:
        """Get the LLM-generated summary for a category.

        Args:
            category: Category name
            scope: "personal" or "shared"

        Returns:
            Summary dict with content, item_count, last_updated, or None
        """
        if self._metadata:
            return self._metadata.get_category_summary(
                self.config.user_id, category, scope
            )
        return None

    def get_all_category_summaries(
        self,
        scope: Literal["personal", "shared", "both"] = "both",
    ) -> list[dict]:
        """Get all category summaries.

        Args:
            scope: Which namespace to get summaries for

        Returns:
            List of summary dicts
        """
        if self._metadata:
            if scope == "personal":
                return self._metadata.get_all_category_summaries(
                    self.config.user_id, "personal"
                )
            elif scope == "shared":
                return self._metadata.get_all_category_summaries(
                    self.config.user_id, "shared"
                )
            else:
                return self._metadata.get_all_category_summaries(self.config.user_id)
        return []

    # =========================================================================
    # Graph Store Methods (using PostgreSQL relations)
    # =========================================================================

    @property
    def graph_store(self):
        """Access the graph store directly for graph queries.

        Returns:
            None (Kuzu replaced by PostgreSQL relations)
        """
        return None

    def get_related_memories(
        self,
        memory_id: str,
        max_depth: int = 2,
        limit: int = 10,
    ) -> list[dict]:
        """Find memories related to a given memory via relations.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum path length (not used with PostgreSQL relations)
            limit: Maximum results

        Returns:
            List of related memories
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        return _run_async(
            self._vectorstore.get_related_memories(memory_id, limit=limit)
        )

    def get_memories_by_entity(
        self,
        entity_name: str,
        entity_type: str = "tool",
        limit: int = 20,
    ) -> list[dict]:
        """Find memories that mention a specific entity.

        Note: This uses full-text search instead of graph traversal.

        Args:
            entity_name: Entity name (e.g., "Python", "Docker")
            entity_type: Entity type (ignored in pgvector implementation)
            limit: Maximum results

        Returns:
            List of memories mentioning the entity
        """
        self._ensure_initialized()
        assert self._vectorstore is not None

        return _run_async(
            self._vectorstore.full_text_search(
                query=entity_name,
                user_id=self.config.user_id,
                limit=limit,
            )
        )

    def get_graph_stats(self) -> dict[str, int]:
        """Get statistics about the memory relations.

        Returns:
            Dict with counts (empty since Kuzu is removed)
        """
        return {"nodes": 0, "edges": 0}

    # =========================================================================
    # Unified Retrieval Pipeline (5-Stage Inference Retrieval)
    # =========================================================================

    def retrieve_for_inference(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        max_tokens: int = 2000,
        enable_query_synthesis: bool = True,
        enable_graph: bool = True,
        project: str | None = None,
        mode: Literal["auto", "vector", "hybrid"] = "auto",
        enable_hyde: bool = True,
        enable_rerank: bool = True,
    ) -> dict[str, Any]:
        """The enhanced inference retrieval pipeline.

        Implements 9 stages:
        1. Query understanding (intent, domains, entities)
        2. Query synthesis (LLM expansion)
        3. HyDE (hypothetical document generation)
        4. Candidate retrieval (vector + hybrid search)
        5. RRF fusion (combine multi-query results)
        6. LLM re-ranking (smarter relevance)
        7. Relevance filtering (threshold)
        8. Scoring adjustments (unified: time decay, priority, project)
        9. Token-budgeted assembly

        This is the primary search method for LLM context injection.

        Args:
            query: User's search query
            scope: "personal", "shared", or "both"
            max_tokens: Token budget for results
            enable_query_synthesis: Use LLM to expand query
            enable_graph: Include relation traversal
            project: Optional project ID for project-scoped scoring
            mode: Retrieval mode
            enable_hyde: Use HyDE for better vector matching
            enable_rerank: Use LLM re-ranking for smarter results

        Returns:
            Dict with results, tokens_used, metadata
        """
        from cems.retrieval import (
            apply_score_adjustments,
            assemble_context,
            deduplicate_results,
            extract_query_intent,
            format_memory_context,
            generate_hypothetical_memory,
            reciprocal_rank_fusion,
            rerank_with_llm,
            route_to_strategy,
            synthesize_query,
        )

        log = logger
        log.info(f"[RETRIEVAL] Starting retrieve_for_inference: query='{query[:50]}...', mode={mode}")

        # Get LLM client for advanced features
        client = None
        try:
            from cems.llm import get_client
            client = get_client()
        except Exception as e:
            log.warning(f"[RETRIEVAL] Could not get LLM client: {e}")

        # Stage 1: Query understanding (for auto mode routing)
        intent = None
        selected_mode = mode
        if mode == "auto" and client:
            try:
                intent = extract_query_intent(query, client)
                selected_mode = route_to_strategy(intent)
                log.info(f"[RETRIEVAL] Auto mode selected: {selected_mode}")
            except Exception as e:
                log.warning(f"[RETRIEVAL] Query understanding failed: {e}")
                selected_mode = "hybrid"

        # Infer category from query for scoring
        inferred_category = self._infer_category_from_query(query)

        # Stage 2: Query synthesis
        queries_to_search = [query]
        if enable_query_synthesis and self.config.enable_query_synthesis and client:
            try:
                expanded = synthesize_query(query, client)
                queries_to_search = [query] + expanded[:3]
                log.info(f"[RETRIEVAL] Query synthesis: {len(queries_to_search)} queries")
            except Exception as e:
                log.warning(f"[RETRIEVAL] Query synthesis failed: {e}")

        # Stage 3: HyDE (if enabled and in hybrid mode)
        if enable_hyde and selected_mode == "hybrid" and client:
            try:
                hypothetical = generate_hypothetical_memory(query, client)
                if hypothetical:
                    queries_to_search.append(hypothetical)
                    log.info(f"[RETRIEVAL] HyDE generated")
            except Exception as e:
                log.warning(f"[RETRIEVAL] HyDE generation failed: {e}")

        # Stage 4: Candidate retrieval
        query_results: list[list[SearchResult]] = []

        for search_query in queries_to_search:
            vector_results = self._search_raw(
                search_query, scope, limit=self.config.max_candidates_per_query
            )
            query_results.append(vector_results)

        # Relation traversal (if enabled)
        if enable_graph and query_results and query_results[0]:
            relation_results: list[SearchResult] = []
            for top_result in query_results[0][:5]:
                try:
                    related = self.get_related_memories(top_result.memory_id, limit=8)
                    for rel in related:
                        metadata = self.get_metadata(rel["id"])
                        if metadata:
                            base_score = rel.get("relation_similarity", 0.5) or 0.5
                            relation_results.append(
                                SearchResult(
                                    memory_id=rel["id"],
                                    content=rel.get("content", ""),
                                    score=base_score,
                                    scope=metadata.scope,
                                    metadata=metadata,
                                )
                            )
                except Exception as e:
                    log.debug(f"[RETRIEVAL] Relation traversal error: {e}")

            if relation_results:
                query_results.append(relation_results)

        # Stage 5: RRF Fusion
        if len(query_results) > 1:
            candidates = reciprocal_rank_fusion(query_results)
            log.info(f"[RETRIEVAL] RRF fusion: {sum(len(r) for r in query_results)} -> {len(candidates)} results")
        else:
            candidates = query_results[0] if query_results else []

        # Deduplicate
        candidates = deduplicate_results(candidates)

        # Stage 6: LLM Re-ranking
        if enable_rerank and selected_mode == "hybrid" and client and len(candidates) > 3:
            try:
                candidates = rerank_with_llm(
                    query, candidates, client,
                    top_k=self.config.rerank_output_limit,
                    config=self.config
                )
                log.info(f"[RETRIEVAL] LLM reranking complete: {len(candidates)} results")
            except Exception as e:
                log.warning(f"[RETRIEVAL] LLM reranking failed: {e}")

        # Stage 7: Relevance filtering
        threshold = self.config.relevance_threshold
        candidates = [c for c in candidates if c.score >= threshold]

        # Stage 8: Apply unified scoring adjustments
        for candidate in candidates:
            candidate.score = apply_score_adjustments(
                candidate,
                inferred_category=inferred_category,
                project=project,
            )

        # Re-sort by adjusted score
        candidates.sort(key=lambda x: x.score, reverse=True)

        total_candidates = sum(len(r) for r in query_results)
        filtered_count = len(candidates)

        # Stage 9: Token-budgeted assembly
        selected, tokens_used = assemble_context(candidates, max_tokens)

        log.info(f"[RETRIEVAL] Final: {filtered_count} candidates -> {len(selected)} selected, {tokens_used} tokens")

        return {
            "results": [
                {
                    "memory_id": r.memory_id,
                    "content": r.content,
                    "score": r.score,
                    "scope": r.scope.value,
                    "category": r.metadata.category if r.metadata else None,
                    "source_ref": r.metadata.source_ref if r.metadata else None,
                    "tags": r.metadata.tags if r.metadata else [],
                }
                for r in selected
            ],
            "tokens_used": tokens_used,
            "formatted_context": format_memory_context(selected),
            "queries_used": queries_to_search,
            "total_candidates": total_candidates,
            "filtered_count": filtered_count,
            "mode": selected_mode,
            "intent": intent,
        }

    def _search_raw(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Raw search without score adjustments - for use in retrieve_for_inference.

        This returns base vector scores only, allowing unified scoring later.

        Args:
            query: Search query
            scope: Which namespace(s) to search
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of SearchResult with base vector scores
        """
        self._ensure_initialized()
        assert self._vectorstore is not None
        assert self._embedder is not None

        # Generate query embedding
        query_embedding = self._embedder.embed(query)

        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        # Use vector search only (raw scores)
        raw_results = _run_async(
            self._vectorstore.search(
                query_embedding=query_embedding,
                user_id=user_id,
                team_id=team_id,
                scope=scope,
                category=category,
                limit=limit,
            )
        )

        # Convert to SearchResult objects
        results: list[SearchResult] = []
        for mem in raw_results:
            memory_scope = MemoryScope(mem.get("scope", "personal"))

            metadata = MemoryMetadata(
                memory_id=mem["id"],
                user_id=mem.get("user_id", user_id),
                scope=memory_scope,
                category=mem.get("category", "general"),
                source=mem.get("source"),
                source_ref=mem.get("source_ref"),
                tags=mem.get("tags", []),
                priority=mem.get("priority", 1.0),
                pinned=mem.get("pinned", False),
                pin_reason=mem.get("pin_reason"),
                archived=mem.get("archived", False),
                access_count=mem.get("access_count", 0),
                created_at=mem.get("created_at", datetime.now(UTC)),
                updated_at=mem.get("updated_at", datetime.now(UTC)),
                last_accessed=mem.get("last_accessed", datetime.now(UTC)),
                expires_at=mem.get("expires_at"),
            )

            results.append(
                SearchResult(
                    memory_id=mem["id"],
                    content=mem.get("content", ""),
                    score=mem.get("score", 0.0),
                    scope=memory_scope,
                    metadata=metadata,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def _search_raw_async(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Async version of _search_raw()."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None
        assert self._embedder is not None

        query_embedding = self._embedder.embed(query)
        user_id = self.config.user_id
        team_id = self.config.team_id if scope in ("shared", "both") else None

        raw_results = await self._vectorstore.search(
            query_embedding=query_embedding,
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            category=category,
            limit=limit,
        )

        results: list[SearchResult] = []
        for mem in raw_results:
            memory_scope = MemoryScope(mem.get("scope", "personal"))
            metadata = MemoryMetadata(
                memory_id=mem["id"],
                user_id=mem.get("user_id", user_id),
                scope=memory_scope,
                category=mem.get("category", "general"),
                source=mem.get("source"),
                source_ref=mem.get("source_ref"),
                tags=mem.get("tags", []),
                priority=mem.get("priority", 1.0),
                pinned=mem.get("pinned", False),
                pin_reason=mem.get("pin_reason"),
                archived=mem.get("archived", False),
                access_count=mem.get("access_count", 0),
                created_at=mem.get("created_at", datetime.now(UTC)),
                updated_at=mem.get("updated_at", datetime.now(UTC)),
                last_accessed=mem.get("last_accessed", datetime.now(UTC)),
                expires_at=mem.get("expires_at"),
            )
            results.append(
                SearchResult(
                    memory_id=mem["id"],
                    content=mem.get("content", ""),
                    score=mem.get("score", 0.0),
                    scope=memory_scope,
                    metadata=metadata,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def get_related_memories_async(
        self,
        memory_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Async version of get_related_memories()."""
        await self._ensure_initialized_async()
        assert self._vectorstore is not None

        return await self._vectorstore.get_related_memories(memory_id, limit=limit)

    async def retrieve_for_inference_async(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        max_tokens: int = 2000,
        enable_query_synthesis: bool = True,
        enable_graph: bool = True,
        project: str | None = None,
        mode: Literal["auto", "vector", "hybrid"] = "auto",
        enable_hyde: bool = True,
        enable_rerank: bool = True,
    ) -> dict[str, Any]:
        """Async version of retrieve_for_inference(). Use from HTTP server."""
        from cems.retrieval import (
            apply_score_adjustments,
            assemble_context,
            deduplicate_results,
            extract_query_intent,
            format_memory_context,
            generate_hypothetical_memory,
            reciprocal_rank_fusion,
            rerank_with_llm,
            route_to_strategy,
            synthesize_query,
        )

        log = logger
        log.info(f"[RETRIEVAL] Starting async retrieve_for_inference: query='{query[:50]}...'")

        client = None
        try:
            from cems.llm import get_client
            client = get_client()
        except Exception as e:
            log.warning(f"[RETRIEVAL] Could not get LLM client: {e}")

        intent = None
        selected_mode = mode
        if mode == "auto" and client:
            try:
                intent = extract_query_intent(query, client)
                selected_mode = route_to_strategy(intent)
                log.info(f"[RETRIEVAL] Auto mode selected: {selected_mode}")
            except Exception as e:
                log.warning(f"[RETRIEVAL] Query understanding failed: {e}")
                selected_mode = "hybrid"

        inferred_category = self._infer_category_from_query(query)

        queries_to_search = [query]
        if enable_query_synthesis and self.config.enable_query_synthesis and client:
            try:
                expanded = synthesize_query(query, client)
                queries_to_search = [query] + expanded[:3]
                log.info(f"[RETRIEVAL] Query synthesis: {len(queries_to_search)} queries")
            except Exception as e:
                log.warning(f"[RETRIEVAL] Query synthesis failed: {e}")

        if enable_hyde and selected_mode == "hybrid" and client:
            try:
                hypothetical = generate_hypothetical_memory(query, client)
                if hypothetical:
                    queries_to_search.append(hypothetical)
                    log.info(f"[RETRIEVAL] HyDE generated")
            except Exception as e:
                log.warning(f"[RETRIEVAL] HyDE generation failed: {e}")

        query_results: list[list[SearchResult]] = []
        for search_query in queries_to_search:
            vector_results = await self._search_raw_async(
                search_query, scope, limit=self.config.max_candidates_per_query
            )
            query_results.append(vector_results)

        if enable_graph and query_results and query_results[0]:
            relation_results: list[SearchResult] = []
            for top_result in query_results[0][:5]:
                try:
                    related = await self.get_related_memories_async(top_result.memory_id, limit=8)
                    for rel in related:
                        metadata = self.get_metadata(rel["id"])
                        if metadata:
                            base_score = rel.get("relation_similarity", 0.5) or 0.5
                            relation_results.append(
                                SearchResult(
                                    memory_id=rel["id"],
                                    content=rel.get("content", ""),
                                    score=base_score,
                                    scope=metadata.scope,
                                    metadata=metadata,
                                )
                            )
                except Exception as e:
                    log.debug(f"[RETRIEVAL] Relation traversal error: {e}")

            if relation_results:
                query_results.append(relation_results)

        if len(query_results) > 1:
            candidates = reciprocal_rank_fusion(query_results)
            log.info(f"[RETRIEVAL] RRF fusion: {sum(len(r) for r in query_results)} -> {len(candidates)} results")
        else:
            candidates = query_results[0] if query_results else []

        candidates = deduplicate_results(candidates)

        if enable_rerank and selected_mode == "hybrid" and client and len(candidates) > 3:
            try:
                candidates = rerank_with_llm(
                    query, candidates, client,
                    top_k=self.config.rerank_output_limit,
                    config=self.config
                )
                log.info(f"[RETRIEVAL] LLM reranking complete: {len(candidates)} results")
            except Exception as e:
                log.warning(f"[RETRIEVAL] LLM reranking failed: {e}")

        threshold = self.config.relevance_threshold
        candidates = [c for c in candidates if c.score >= threshold]

        for candidate in candidates:
            candidate.score = apply_score_adjustments(
                candidate,
                inferred_category=inferred_category,
                project=project,
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        total_candidates = sum(len(r) for r in query_results)
        filtered_count = len(candidates)

        selected, tokens_used = assemble_context(candidates, max_tokens)
        log.info(f"[RETRIEVAL] Final: {filtered_count} candidates -> {len(selected)} selected, {tokens_used} tokens")

        return {
            "results": [
                {
                    "memory_id": r.memory_id,
                    "content": r.content,
                    "score": r.score,
                    "scope": r.scope.value,
                    "category": r.metadata.category if r.metadata else None,
                    "source_ref": r.metadata.source_ref if r.metadata else None,
                    "tags": r.metadata.tags if r.metadata else [],
                }
                for r in selected
            ],
            "tokens_used": tokens_used,
            "formatted_context": format_memory_context(selected),
            "queries_used": queries_to_search,
            "total_candidates": total_candidates,
            "filtered_count": filtered_count,
            "mode": selected_mode,
            "intent": intent,
        }
