"""CEMS Memory wrapper around Mem0 with namespace isolation."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from mem0 import Memory

from cems.config import CEMSConfig
from cems.models import (
    MemoryMetadata,
    MemoryScope,
    SearchResult,
)

if TYPE_CHECKING:
    from cems.db.metadata_store import PostgresMetadataStore
    from cems.graph import KuzuGraphStore


class CEMSMemory:
    """Memory system with personal/shared namespace isolation.

    Built on top of Mem0, this class provides:
    - Namespace isolation (personal vs shared memories)
    - Extended metadata tracking (access counts, priorities)
    - Unified search across namespaces
    """

    def __init__(self, config: CEMSConfig | None = None):
        """Initialize CEMS memory.

        Args:
            config: CEMS configuration. If None, loads from environment.
        """
        self.config = config or CEMSConfig()

        # Initialize Mem0 with local Qdrant
        mem0_config = self._build_mem0_config()
        self._memory = Memory.from_config(mem0_config)

        # Initialize metadata store (PostgreSQL only - Docker/server mode)
        if not self.config.database_url:
            raise ValueError(
                "CEMS_DATABASE_URL is required. "
                "CEMS runs in Docker/server mode only (no local SQLite mode)."
            )

        from cems.db.database import init_database, is_database_initialized
        from cems.db.metadata_store import PostgresMetadataStore

        # Initialize database if not already done (e.g., by server.py)
        if not is_database_initialized():
            init_database(self.config.database_url)
        self._metadata = PostgresMetadataStore()

        # Initialize graph store (optional)
        self._graph: KuzuGraphStore | None = None
        if self.config.enable_graph and self.config.graph_store == "kuzu":
            try:
                from cems.graph import KuzuGraphStore

                self._graph = KuzuGraphStore(self.config.kuzu_storage_path)
            except ImportError:
                import logging

                logging.getLogger(__name__).warning(
                    "Kuzu not installed. Graph features disabled. "
                    "Install with: pip install kuzu"
                )

    def _build_mem0_config(self) -> dict[str, Any]:
        """Build Mem0 configuration dict.

        Uses OpenRouter for ALL operations (LLM + embeddings) via single API key.

        LLM: OpenAI provider with openrouter_base_url parameter
        Embeddings: OpenAI embedder with openai_base_url pointing to OpenRouter

        Required env vars:
            - OPENROUTER_API_KEY: For all LLM and embedding calls

        See:
            - https://github.com/mem0ai/mem0/blob/main/mem0/llms/openai.py
            - https://github.com/mem0ai/mem0/blob/main/mem0/embeddings/openai.py
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        model = self.config.get_mem0_model()

        return {
            "llm": {
                "provider": "openai",  # Use OpenAI provider with OpenRouter base URL
                "config": {
                    "model": model,
                    "api_key": api_key,
                    "openrouter_base_url": "https://openrouter.ai/api/v1",
                    "site_url": "https://github.com/cems",
                    "app_name": "CEMS Memory Server",
                },
            },
            "embedder": {
                "provider": "openai",  # OpenAI embedder with OpenRouter base URL
                "config": {
                    "model": self.config.embedding_model,
                    "api_key": api_key,
                    "openai_base_url": "https://openrouter.ai/api/v1",
                },
            },
            "vector_store": {
                "provider": self.config.vector_store,
                "config": self._get_vector_store_config(),
            },
            "version": "v1.1",
            "custom_fact_extraction_prompt": """Extract actionable facts from developer conversations.

For each fact, capture:
- SPECIFIC commands, file paths, URLs (not "uses scripts" but "uses ~/.claude/servers/gsc/seo_daily.sh")
- CONCRETE preferences (not "likes Python" but "prefers Python 3.12+ with strict type hints")
- WORKFLOW steps (not "does SEO" but "runs GSC scripts daily, updates GitHub issue #579 weekly")
- DECISIONS with context (not "uses Coolify" but "deploys EpicPxls via Coolify GitHub integration, never CLI")

Rules:
- Be specific: include file paths, commands, URLs, version numbers
- Be actionable: facts should tell an AI what to DO, not just what exists
- Preserve context: "for EpicPxls", "in production", "when debugging"
- Skip vague context markers like "Context: X" - extract the actual workflow instead

Return JSON: {"facts": ["specific actionable fact 1", "specific actionable fact 2"]}""",
        }

    def _get_vector_store_config(self) -> dict[str, Any]:
        """Get vector store configuration.

        Returns URL-based config if qdrant_url is set, otherwise local path config.
        """
        if self.config.qdrant_url:
            return {"url": self.config.qdrant_url}
        return {"path": str(self.config.qdrant_storage_path)}

    def _get_mem0_user_id(self, scope: MemoryScope) -> str:
        """Get the Mem0 user_id for a scope."""
        if scope == MemoryScope.PERSONAL:
            return f"personal:{self.config.user_id}"
        else:
            if not self.config.team_id:
                raise ValueError("Cannot use shared scope without team_id configured")
            return f"shared:{self.config.team_id}"

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
            infer: If True (default), use LLM for fact extraction and deduplication.
                   If False, store raw content directly (100-200ms vs 1-10s per memory).
                   Use infer=False for bulk imports where speed matters.
            source_ref: Optional project reference for scoped recall (e.g., "project:org/repo")
            ttl_hours: Optional TTL in hours. If set, memory expires after this time.
                       Use for short-term session memories. None = permanent memory.
            pinned: If True, memory is pinned and never auto-pruned. Useful for
                    important guidelines or gate rules.
            pin_reason: Optional reason for pinning the memory.

        Returns:
            Dict with memory operation results
        """
        memory_scope = MemoryScope(scope)
        user_id = self._get_mem0_user_id(memory_scope)

        # Add to Mem0
        # infer=False skips LLM fact extraction and deduplication (much faster)
        result = self._memory.add(
            content,
            user_id=user_id,
            metadata={
                "category": category,
                "source": source,
                "source_ref": source_ref,
                "tags": tags or [],
            },
            infer=infer,
        )

        # Track extended metadata for each created memory
        if result and "results" in result:
            for mem_result in result["results"]:
                if mem_result.get("event") in ("ADD", "UPDATE"):
                    memory_id = mem_result.get("id")
                    if memory_id:
                        # Calculate expires_at if TTL is set
                        expires_at = None
                        if ttl_hours:
                            from datetime import timedelta
                            expires_at = datetime.now(UTC) + timedelta(hours=ttl_hours)

                        metadata = MemoryMetadata(
                            memory_id=memory_id,
                            user_id=self.config.user_id,
                            scope=memory_scope,
                            category=category,
                            source=source,
                            source_ref=source_ref,
                            tags=tags or [],
                            expires_at=expires_at,
                            pinned=pinned,
                            pin_reason=pin_reason,
                        )
                        self._metadata.save_metadata(metadata)

                        # Add to graph store if enabled
                        if self._graph:
                            # Find similar memories for RELATES_TO edges
                            # This is the FIX: we now auto-populate graph relationships!
                            similar_memories: list[tuple[str, float]] = []
                            try:
                                # Search for similar memories using the content
                                similar_results = self._memory.search(
                                    content[:500],  # Use first 500 chars for similarity search
                                    user_id=user_id,
                                    limit=5,
                                )
                                for sim_mem in similar_results.get("results", []):
                                    sim_id = sim_mem.get("id")
                                    sim_score = sim_mem.get("score", 0.0)
                                    if sim_id and sim_id != memory_id:
                                        similar_memories.append((sim_id, sim_score))
                            except Exception as e:
                                import logging
                                logging.getLogger(__name__).debug(f"Similar memory search failed: {e}")

                            self._graph.process_memory(
                                memory_id=memory_id,
                                content=content,
                                scope=scope,
                                user_id=self.config.user_id,
                                category=category,
                                tags=tags,
                                similar_memories=similar_memories,  # Pass similar memories for RELATES_TO edges
                            )

        return result

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
        results: list[SearchResult] = []

        # Determine which scopes to search
        scopes_to_search = []
        if scope in ("personal", "both"):
            scopes_to_search.append(MemoryScope.PERSONAL)
        if scope in ("shared", "both") and self.config.team_id:
            scopes_to_search.append(MemoryScope.SHARED)

        # Collect all results first, then batch fetch metadata
        raw_results: list[tuple[dict, MemoryScope]] = []
        
        for memory_scope in scopes_to_search:
            user_id = self._get_mem0_user_id(memory_scope)

            # Search in Mem0
            mem0_results = self._memory.search(
                query,
                user_id=user_id,
                limit=limit,
            )

            for mem in mem0_results.get("results", []):
                raw_results.append((mem, memory_scope))

        # Batch fetch all metadata in a single query
        memory_ids = [mem.get("id") for mem, _ in raw_results if mem.get("id")]
        metadata_map: dict[str, MemoryMetadata] = {}
        if memory_ids:
            metadata_map = self._metadata.get_metadata_batch(memory_ids)

        # Build results and collect IDs to record access
        access_ids: list[str] = []
        for mem, memory_scope in raw_results:
            memory_id = mem.get("id")
            metadata = metadata_map.get(memory_id) if memory_id else None

            # Apply category filter if metadata exists
            if category and metadata and metadata.category != category:
                continue

            if memory_id:
                access_ids.append(memory_id)

            results.append(
                SearchResult(
                    memory_id=memory_id or "",
                    content=mem.get("memory", ""),
                    score=mem.get("score", 0.0),
                    scope=memory_scope,
                    metadata=metadata,
                )
            )

        # Batch record access in a single query
        if access_ids:
            self._metadata.record_access_batch(access_ids)

        # Apply priority boost and time decay to scores
        now = datetime.now(UTC)
        inferred_category = self._infer_category_from_query(query)

        for result in results:
            if result.metadata:
                # Priority boost (1.0 default, up to 2.0 for hot memories)
                result.score *= result.metadata.priority

                # Time decay: 50% penalty per month since last access
                # More recently accessed memories get higher scores
                days_since_access = (now - result.metadata.last_accessed).days
                time_decay = 1.0 / (1.0 + (days_since_access / 30))
                result.score *= time_decay

                # Boost pinned memories slightly (they're important)
                if result.metadata.pinned:
                    result.score *= 1.1

                # Cross-category penalty: 20% reduction if memory category
                # doesn't match the inferred query category
                if inferred_category and result.metadata.category:
                    if result.metadata.category.lower() != inferred_category:
                        result.score *= 0.8

        # Sort by adjusted score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Memory dict or None if not found
        """
        result = self._memory.get(memory_id)
        if result:
            self._metadata.record_access(memory_id)
        return result

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
        all_memories: list[dict[str, Any]] = []

        scopes_to_get = []
        if scope in ("personal", "both"):
            scopes_to_get.append(MemoryScope.PERSONAL)
        if scope in ("shared", "both") and self.config.team_id:
            scopes_to_get.append(MemoryScope.SHARED)

        # Get archived IDs in a single batch query (if filtering)
        archived_ids: set[str] = set()
        if not include_archived and hasattr(self._metadata, "get_archived_memory_ids"):
            archived_ids = self._metadata.get_archived_memory_ids()

        for memory_scope in scopes_to_get:
            user_id = self._get_mem0_user_id(memory_scope)
            result = self._memory.get_all(user_id=user_id)

            if result and "results" in result:
                for mem in result["results"]:
                    memory_id = mem.get("id")
                    # Fast O(1) set lookup instead of N database queries
                    if memory_id and not include_archived and memory_id in archived_ids:
                        continue
                    mem["scope"] = memory_scope.value
                    all_memories.append(mem)

        return all_memories

    def update(self, memory_id: str, content: str) -> dict[str, Any]:
        """Update a memory's content.

        Args:
            memory_id: The memory ID to update
            content: New content

        Returns:
            Update result dict
        """
        result = self._memory.update(memory_id, content)

        # Update metadata timestamp
        metadata = self._metadata.get_metadata(memory_id)
        if metadata:
            metadata.updated_at = datetime.now(UTC)
            self._metadata.save_metadata(metadata)

        return result

    def delete(self, memory_id: str, hard: bool = False) -> dict[str, Any]:
        """Delete or archive a memory.

        Args:
            memory_id: The memory ID to delete
            hard: If True, permanently delete. If False, archive.

        Returns:
            Delete result dict
        """
        if hard:
            result = self._memory.delete(memory_id)
            self._metadata.delete_metadata(memory_id)
            # Also remove from graph
            if self._graph:
                self._graph.delete_memory_node(memory_id)
        else:
            self._metadata.archive_memory(memory_id)
            result = {"status": "archived", "memory_id": memory_id}

        return result

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

        Args:
            memory_id: The memory ID

        Returns:
            List of history entries
        """
        return self._memory.history(memory_id)

    def get_stale_memories(self, days: int | None = None) -> list[str]:
        """Get memories that haven't been accessed in N days.

        Args:
            days: Days threshold. Uses config default if not specified.

        Returns:
            List of stale memory IDs
        """
        days = days or self.config.stale_days
        return self._metadata.get_stale_memories(self.config.user_id, days)

    def get_hot_memories(self, threshold: int | None = None) -> list[str]:
        """Get frequently accessed memories.

        Args:
            threshold: Access count threshold. Uses config default if not specified.

        Returns:
            List of hot memory IDs
        """
        threshold = threshold or self.config.hot_access_threshold
        return self._metadata.get_hot_memories(self.config.user_id, threshold)

    def get_recent_memories(self, hours: int = 24) -> list[str]:
        """Get memories created in the last N hours.

        Args:
            hours: Hours to look back

        Returns:
            List of memory IDs
        """
        return self._metadata.get_recent_memories(self.config.user_id, hours)

    def get_old_memories(self, days: int = 30) -> list[str]:
        """Get memories older than N days.

        Args:
            days: Days threshold

        Returns:
            List of memory IDs
        """
        return self._metadata.get_old_memories(self.config.user_id, days)

    def promote_memory(self, memory_id: str, boost: float = 0.1) -> None:
        """Increase a memory's priority.

        Args:
            memory_id: The memory ID
            boost: Priority boost amount
        """
        self._metadata.increase_priority(memory_id, boost)

    def archive_memory(self, memory_id: str) -> None:
        """Archive a memory (soft delete).

        Args:
            memory_id: The memory ID
        """
        self._metadata.archive_memory(memory_id)

    def get_metadata(self, memory_id: str) -> MemoryMetadata | None:
        """Get extended metadata for a memory.

        Args:
            memory_id: The memory ID

        Returns:
            MemoryMetadata or None
        """
        return self._metadata.get_metadata(memory_id)

    def get_category_counts(self, scope: Literal["personal", "shared"] | None = None) -> dict[str, int]:
        """Get memory counts grouped by category using a single efficient query.

        Args:
            scope: Optional filter by scope ("personal" or "shared")

        Returns:
            Dict mapping category name to count
        """
        from cems.models import MemoryScope
        
        scope_enum = MemoryScope(scope) if scope else None
        
        return self._metadata.get_category_counts(self.config.user_id, scope_enum)

    @property
    def metadata_store(self) -> "PostgresMetadataStore":
        """Access the metadata store directly for maintenance operations."""
        return self._metadata

    @property
    def mem0(self) -> Memory:
        """Access the underlying Mem0 instance."""
        return self._memory

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

    def get_recently_accessed(self, limit: int = 10) -> list[dict]:
        """Get recently accessed memories.

        Args:
            limit: Maximum number of results

        Returns:
            List of dicts with memory info and access timestamps
        """
        return self._metadata.get_recently_accessed(self.config.user_id, limit)

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
        return self._metadata.get_category_summary(
            self.config.user_id, category, scope
        )

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

    # =========================================================================
    # Graph Store Methods
    # =========================================================================

    @property
    def graph_store(self) -> KuzuGraphStore | None:
        """Access the graph store directly for graph queries.

        Returns:
            KuzuGraphStore instance or None if graph is disabled
        """
        return self._graph

    def get_related_memories(
        self,
        memory_id: str,
        max_depth: int = 2,
        limit: int = 10,
    ) -> list[dict]:
        """Find memories related to a given memory via the knowledge graph.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum path length to traverse
            limit: Maximum results

        Returns:
            List of related memories, or empty list if graph disabled
        """
        if not self._graph:
            return []
        return self._graph.get_related_memories(memory_id, max_depth, limit)

    def get_memories_by_entity(
        self,
        entity_name: str,
        entity_type: str = "tool",
        limit: int = 20,
    ) -> list[dict]:
        """Find memories that mention a specific entity.

        Args:
            entity_name: Entity name (e.g., "Python", "Docker")
            entity_type: Entity type (e.g., "tool", "concept")
            limit: Maximum results

        Returns:
            List of memories mentioning the entity
        """
        if not self._graph:
            return []
        entity_id = f"{entity_type}:{entity_name.lower()}"
        return self._graph.get_memories_by_entity(entity_id, limit)

    def get_graph_stats(self) -> dict[str, int]:
        """Get statistics about the knowledge graph.

        Returns:
            Dict with node and edge counts, or empty dict if graph disabled
        """
        if not self._graph:
            return {}
        return self._graph.get_graph_stats()

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
        1. Query understanding (intent, domains, entities) - NEW
        2. Query synthesis (LLM expansion)
        3. HyDE (hypothetical document generation) - NEW
        4. Candidate retrieval (vector + graph)
        5. RRF fusion (combine multi-query results) - NEW
        6. LLM re-ranking (smarter relevance) - NEW
        7. Relevance filtering (threshold)
        8. Scoring adjustments (unified: time decay, priority, project)
        9. Token-budgeted assembly

        This is the primary search method for LLM context injection.

        Args:
            query: User's search query
            scope: "personal", "shared", or "both"
            max_tokens: Token budget for results
            enable_query_synthesis: Use LLM to expand query
            enable_graph: Include graph traversal
            project: Optional project ID for project-scoped scoring
            mode: Retrieval mode - "auto" (smart routing), "vector" (fast), "hybrid" (full)
            enable_hyde: Use HyDE for better vector matching
            enable_rerank: Use LLM re-ranking for smarter results

        Returns:
            Dict with results, tokens_used, metadata
        """
        import logging
        log = logging.getLogger(__name__)
        
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
                log.info(f"[RETRIEVAL] Auto mode selected: {selected_mode} (intent={intent.get('primary_intent')}, complexity={intent.get('complexity')})")
            except Exception as e:
                log.warning(f"[RETRIEVAL] Query understanding failed: {e}")
                selected_mode = "hybrid"

        # Infer category from query for scoring
        inferred_category = self._infer_category_from_query(query)
        log.debug(f"[RETRIEVAL] Inferred category: {inferred_category}")

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
                    log.info(f"[RETRIEVAL] HyDE generated: '{hypothetical[:50]}...'")
            except Exception as e:
                log.warning(f"[RETRIEVAL] HyDE generation failed: {e}")

        # Stage 4: Candidate retrieval (collect per-query results for RRF)
        query_results: list[list[SearchResult]] = []

        for search_query in queries_to_search:
            # Vector search - use raw search to get base scores
            vector_results = self._search_raw(search_query, scope, limit=20)
            query_results.append(vector_results)
            log.debug(f"[RETRIEVAL] Vector search for '{search_query[:30]}...': {len(vector_results)} results")

        # Category summaries integration - use summaries to find relevant categories
        # then boost memories from those categories
        category_boost_map: dict[str, float] = {}
        if selected_mode == "hybrid" and client:
            try:
                summaries = self.get_all_category_summaries(scope=scope)
                if summaries:
                    # Find categories whose summaries are relevant to the query
                    summary_text = "\n".join([
                        f"- {s.category}: {s.summary[:100]}..."
                        for s in summaries[:20]  # Top 20 categories
                    ])
                    cat_prompt = f"""Given this search query, which categories are most relevant?

Query: {query}

Available categories:
{summary_text}

Return a JSON object with category names as keys and relevance scores (0.0-1.0) as values.
Only include categories with relevance >= 0.3.
Example: {{"debugging": 0.9, "deployment": 0.5}}

JSON:"""
                    response = client.complete(cat_prompt, max_tokens=200, temperature=0.1)
                    response = response.strip()
                    if response.startswith("```"):
                        import re
                        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
                        if match:
                            response = match.group(1).strip()
                    category_boost_map = json.loads(response)
                    log.info(f"[RETRIEVAL] Category summaries matched: {list(category_boost_map.keys())}")
            except Exception as e:
                log.debug(f"[RETRIEVAL] Category summary matching failed: {e}")

        # Graph traversal (if enabled) - add as separate result list for RRF
        if enable_graph and self._graph and query_results and query_results[0]:
            graph_results: list[SearchResult] = []
            for top_result in query_results[0][:3]:  # Top 3 from primary query
                try:
                    related = self._graph.get_related_memories(
                        top_result.memory_id, max_depth=2, limit=5
                    )
                    for rel in related:
                        mem = self.get(rel["id"])
                        if mem:
                            metadata = self._metadata.get_metadata(rel["id"])
                            # Use actual edge similarity instead of fixed 0.5
                            base_score = rel.get("similarity", 0.5)
                            # Apply distance penalty
                            distance = rel.get("distance", 1)
                            base_score *= 1.0 / distance
                            graph_results.append(
                                SearchResult(
                                    memory_id=rel["id"],
                                    content=mem.get("memory", ""),
                                    score=base_score,
                                    scope=MemoryScope(metadata.scope) if metadata else MemoryScope.PERSONAL,
                                    metadata=metadata,
                                )
                            )
                except Exception as e:
                    log.debug(f"[RETRIEVAL] Graph traversal error: {e}")
            
            if graph_results:
                query_results.append(graph_results)
                log.info(f"[RETRIEVAL] Graph traversal: {len(graph_results)} results")

        # Stage 5: RRF Fusion (combine all query results)
        if len(query_results) > 1:
            candidates = reciprocal_rank_fusion(query_results)
            log.info(f"[RETRIEVAL] RRF fusion: {sum(len(r) for r in query_results)} -> {len(candidates)} results")
        else:
            candidates = query_results[0] if query_results else []

        # Deduplicate
        candidates = deduplicate_results(candidates)

        # Stage 6: LLM Re-ranking (if enabled and in hybrid mode)
        if enable_rerank and selected_mode == "hybrid" and client and len(candidates) > 3:
            try:
                candidates = rerank_with_llm(query, candidates, client, top_k=15)
                log.info(f"[RETRIEVAL] LLM reranking complete: {len(candidates)} results")
            except Exception as e:
                log.warning(f"[RETRIEVAL] LLM reranking failed: {e}")

        # Stage 7: Relevance filtering
        threshold = self.config.relevance_threshold
        before_filter = len(candidates)
        candidates = [c for c in candidates if c.score >= threshold]
        log.debug(f"[RETRIEVAL] Relevance filter (threshold={threshold}): {before_filter} -> {len(candidates)}")

        # Stage 8: Apply unified scoring adjustments
        for candidate in candidates:
            candidate.score = apply_score_adjustments(
                candidate,
                inferred_category=inferred_category,
                project=project,
            )
            # Apply category boost from summary matching
            if category_boost_map and candidate.metadata:
                cat = candidate.metadata.category
                if cat and cat.lower() in category_boost_map:
                    boost = 1.0 + (category_boost_map[cat.lower()] * 0.3)  # Up to 30% boost
                    candidate.score *= boost
                    log.debug(f"[RETRIEVAL] Category boost for '{cat}': {boost:.2f}")

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
        results: list[SearchResult] = []

        # Determine which scopes to search
        scopes_to_search = []
        if scope in ("personal", "both"):
            scopes_to_search.append(MemoryScope.PERSONAL)
        if scope in ("shared", "both") and self.config.team_id:
            scopes_to_search.append(MemoryScope.SHARED)

        # Collect raw results
        raw_results: list[tuple[dict, MemoryScope]] = []

        for memory_scope in scopes_to_search:
            user_id = self._get_mem0_user_id(memory_scope)
            mem0_results = self._memory.search(query, user_id=user_id, limit=limit)
            for mem in mem0_results.get("results", []):
                raw_results.append((mem, memory_scope))

        # Batch fetch metadata
        memory_ids = [mem.get("id") for mem, _ in raw_results if mem.get("id")]
        metadata_map: dict[str, MemoryMetadata] = {}
        if memory_ids:
            metadata_map = self._metadata.get_metadata_batch(memory_ids)

        # Build results with RAW scores (no adjustments here!)
        for mem, memory_scope in raw_results:
            memory_id = mem.get("id")
            metadata = metadata_map.get(memory_id) if memory_id else None

            # Skip expired memories (not in metadata_map due to TTL filtering)
            if memory_id and memory_id not in metadata_map:
                continue

            # Apply category filter if provided
            if category and metadata and metadata.category != category:
                continue

            results.append(
                SearchResult(
                    memory_id=memory_id or "",
                    content=mem.get("memory", ""),
                    score=mem.get("score", 0.0),  # Raw vector score!
                    scope=memory_scope,
                    metadata=metadata,
                )
            )

        # Sort by raw score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
