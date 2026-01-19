"""CEMS Memory wrapper around Mem0 with namespace isolation."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from mem0 import Memory

from cems.config import CEMSConfig
from cems.models import (
    MemoryMetadata,
    MemoryScope,
    MetadataStore,
    SearchResult,
)

if TYPE_CHECKING:
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
        self.config.ensure_dirs()

        # Initialize Mem0 with local Qdrant
        mem0_config = self._build_mem0_config()
        self._memory = Memory.from_config(mem0_config)

        # Initialize metadata store
        self._metadata = MetadataStore(self.config.metadata_db_path)

        # Initialize graph store (optional)
        self._graph: "KuzuGraphStore | None" = None
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

        Note:
            Mem0 requires direct provider access (OpenAI/Anthropic), not OpenRouter.
            The OPENAI_API_KEY or ANTHROPIC_API_KEY must be set for Mem0 to work.
        """
        provider = self.config.get_mem0_provider()
        model = self.config.get_mem0_model()

        return {
            "llm": {
                "provider": provider,
                "config": {
                    "model": model,
                },
            },
            "embedder": {
                "provider": provider,
                "config": {
                    "model": self.config.embedding_model,
                },
            },
            "vector_store": {
                "provider": self.config.vector_store,
                "config": self._get_vector_store_config(),
            },
            "version": "v1.1",
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

    def add(
        self,
        content: str,
        scope: Literal["personal", "shared"] = "personal",
        category: str = "general",
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add a memory to the specified namespace.

        Args:
            content: The content to remember
            scope: "personal" or "shared"
            category: Category for organization
            source: Optional source identifier
            tags: Optional tags for organization

        Returns:
            Dict with memory operation results
        """
        memory_scope = MemoryScope(scope)
        user_id = self._get_mem0_user_id(memory_scope)

        # Add to Mem0
        result = self._memory.add(
            content,
            user_id=user_id,
            metadata={
                "category": category,
                "source": source,
                "tags": tags or [],
            },
        )

        # Track extended metadata for each created memory
        if result and "results" in result:
            for mem_result in result["results"]:
                if mem_result.get("event") in ("ADD", "UPDATE"):
                    memory_id = mem_result.get("id")
                    if memory_id:
                        metadata = MemoryMetadata(
                            memory_id=memory_id,
                            user_id=self.config.user_id,
                            scope=memory_scope,
                            category=category,
                            source=source,
                            tags=tags or [],
                        )
                        self._metadata.save_metadata(metadata)

                        # Add to graph store if enabled
                        if self._graph:
                            self._graph.process_memory(
                                memory_id=memory_id,
                                content=content,
                                scope=scope,
                                user_id=self.config.user_id,
                                category=category,
                                tags=tags,
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

        for memory_scope in scopes_to_search:
            user_id = self._get_mem0_user_id(memory_scope)

            # Build filters
            filters = {}
            if category:
                filters["category"] = category

            # Search in Mem0
            mem0_results = self._memory.search(
                query,
                user_id=user_id,
                limit=limit,
            )

            # Convert to SearchResult and track access
            for mem in mem0_results.get("results", []):
                memory_id = mem.get("id")

                # Get extended metadata
                metadata = self._metadata.get_metadata(memory_id) if memory_id else None

                # Apply category filter if metadata exists
                if category and metadata and metadata.category != category:
                    continue

                # Record access
                if memory_id:
                    self._metadata.record_access(memory_id)

                results.append(
                    SearchResult(
                        memory_id=memory_id or "",
                        content=mem.get("memory", ""),
                        score=mem.get("score", 0.0),
                        scope=memory_scope,
                        metadata=metadata,
                    )
                )

        # Apply priority boost and time decay to scores
        now = datetime.now(UTC)
        for result in results:
            if result.metadata:
                # Priority boost (1.0 default, up to 2.0 for hot memories)
                result.score *= result.metadata.priority

                # Time decay: 10% penalty per month since last access
                # More recently accessed memories get higher scores
                days_since_access = (now - result.metadata.last_accessed).days
                time_decay = 1.0 / (1.0 + (days_since_access / 30) * 0.1)
                result.score *= time_decay

                # Boost pinned memories slightly (they're important)
                if result.metadata.pinned:
                    result.score *= 1.1

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

        for memory_scope in scopes_to_get:
            user_id = self._get_mem0_user_id(memory_scope)
            result = self._memory.get_all(user_id=user_id)

            if result and "results" in result:
                for mem in result["results"]:
                    memory_id = mem.get("id")
                    if memory_id and not include_archived:
                        metadata = self._metadata.get_metadata(memory_id)
                        if metadata and metadata.archived:
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

    @property
    def metadata_store(self) -> MetadataStore:
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
    def graph_store(self) -> "KuzuGraphStore | None":
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
    # V2 Advanced Search Methods
    # =========================================================================

    def hybrid_search(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 10,
        graph_weight: float = 0.3,
    ) -> list[SearchResult]:
        """Hybrid search combining vector similarity and graph relationships.

        This method performs:
        1. Vector similarity search (via Mem0)
        2. Graph traversal to find related memories
        3. Merges and re-ranks results

        Args:
            query: Search query
            scope: Which namespace(s) to search
            category: Optional category filter
            limit: Maximum results to return
            graph_weight: Weight for graph-based results (0.0-1.0)

        Returns:
            List of SearchResult objects with combined ranking
        """
        # Step 1: Get vector search results
        vector_results = self.search(
            query=query,
            scope=scope,
            category=category,
            limit=limit * 2,  # Get more to allow for merging
        )

        if not self._graph or not vector_results:
            return vector_results[:limit]

        # Step 2: Get graph-related memories for top vector results
        graph_memory_ids: set[str] = set()
        for result in vector_results[:3]:  # Use top 3 as seeds
            related = self._graph.get_related_memories(
                result.memory_id, max_depth=2, limit=5
            )
            for rel in related:
                graph_memory_ids.add(rel["id"])

        # Step 3: Also search by entities mentioned in query
        entities = self._graph.extract_entities_from_content(query)
        for entity in entities[:3]:  # Limit to 3 entities
            entity_memories = self._graph.get_memories_by_entity(entity["id"], limit=5)
            for mem in entity_memories:
                graph_memory_ids.add(mem["id"])

        # Step 4: Fetch graph-discovered memories not in vector results
        vector_ids = {r.memory_id for r in vector_results}
        new_ids = graph_memory_ids - vector_ids

        graph_results: list[SearchResult] = []
        for memory_id in new_ids:
            mem = self.get(memory_id)
            if mem:
                metadata = self._metadata.get_metadata(memory_id)
                # Apply category filter
                if category and metadata and metadata.category != category:
                    continue

                # Assign a base score for graph-discovered memories
                base_score = 0.5  # Lower than vector similarity

                # Apply priority/decay adjustments
                adjusted_score = base_score
                if metadata:
                    adjusted_score *= metadata.priority
                    days_old = (datetime.now(UTC) - metadata.last_accessed).days
                    time_decay = 1.0 / (1.0 + (days_old / 30) * 0.1)
                    adjusted_score *= time_decay
                    if metadata.pinned:
                        adjusted_score *= 1.1

                graph_results.append(
                    SearchResult(
                        memory_id=memory_id,
                        content=mem.get("memory", ""),
                        score=adjusted_score * graph_weight,
                        scope=MemoryScope(metadata.scope) if metadata else MemoryScope.PERSONAL,
                        metadata=metadata,
                    )
                )

        # Step 5: Merge results - vector results keep their scores,
        # graph results are weighted
        combined = vector_results + graph_results
        combined.sort(key=lambda x: x.score, reverse=True)

        # Deduplicate (prefer higher-scored version)
        seen: set[str] = set()
        unique_results: list[SearchResult] = []
        for result in combined:
            if result.memory_id not in seen:
                seen.add(result.memory_id)
                unique_results.append(result)

        return unique_results[:limit]

    def smart_search(
        self,
        query: str,
        scope: Literal["personal", "shared", "both"] = "both",
        category: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Tiered retrieval: check category summaries first, then drill down.

        This implements the "sufficiency check" pattern from the original plan:
        1. First, check if category summaries can answer the query
        2. If summaries seem sufficient, return them
        3. Otherwise, drill down into individual memories

        Args:
            query: Search query
            scope: Which namespace(s) to search
            category: Optional specific category to search
            limit: Maximum memory results if drilling down

        Returns:
            Dict with:
                - tier: "summary" or "memories"
                - summaries: List of relevant category summaries (if tier=summary)
                - results: List of SearchResult objects (if tier=memories)
                - categories_checked: Categories that were evaluated
        """
        # Step 1: Get all category summaries
        if category:
            # Single category requested
            summary = self.get_category_summary(
                category,
                scope=scope if scope != "both" else "personal",  # type: ignore
            )
            summaries = [{"category": category, "scope": scope, **summary}] if summary else []
        else:
            summaries = self.get_all_category_summaries(scope=scope)

        if not summaries:
            # No summaries exist - go straight to memory search
            results = self.search(query=query, scope=scope, category=category, limit=limit)
            return {
                "tier": "memories",
                "summaries": [],
                "results": results,
                "categories_checked": [],
            }

        # Step 2: Check which summaries might be relevant
        # Simple heuristic: check if query terms appear in category name or summary
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        relevant_summaries = []
        for s in summaries:
            cat_name = s.get("category", "").lower()
            summary_text = s.get("summary", "").lower()

            # Check for term overlap
            relevance_score = 0
            for term in query_terms:
                if len(term) > 2:  # Skip very short terms
                    if term in cat_name:
                        relevance_score += 2
                    if term in summary_text:
                        relevance_score += 1

            if relevance_score > 0:
                s["_relevance"] = relevance_score
                relevant_summaries.append(s)

        # Sort by relevance
        relevant_summaries.sort(key=lambda x: x.get("_relevance", 0), reverse=True)

        # Step 3: Determine if summaries are sufficient
        # Heuristic: If we have high-relevance summaries, use them
        if relevant_summaries and relevant_summaries[0].get("_relevance", 0) >= 2:
            # Summaries seem sufficient
            # Clean up internal fields
            for s in relevant_summaries:
                s.pop("_relevance", None)

            return {
                "tier": "summary",
                "summaries": relevant_summaries[:3],  # Top 3 relevant summaries
                "results": [],
                "categories_checked": [s["category"] for s in summaries],
            }

        # Step 4: Summaries not sufficient - drill down into memories
        # Use the most relevant category if we found any
        search_category = category
        if relevant_summaries and not category:
            search_category = relevant_summaries[0]["category"]

        results = self.search(
            query=query,
            scope=scope,
            category=search_category,
            limit=limit,
        )

        return {
            "tier": "memories",
            "summaries": relevant_summaries[:2] if relevant_summaries else [],
            "results": results,
            "categories_checked": [s["category"] for s in summaries],
        }

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
    ) -> dict[str, Any]:
        """The definitive inference retrieval pipeline.

        Implements 5 stages:
        1. Query synthesis (LLM expansion)
        2. Candidate retrieval (vector + graph)
        3. Relevance filtering (threshold)
        4. Temporal ranking (time decay)
        5. Token-budgeted assembly

        This is the primary search method for LLM context injection.

        Args:
            query: User's search query
            scope: "personal", "shared", or "both"
            max_tokens: Token budget for results
            enable_query_synthesis: Use LLM to expand query
            enable_graph: Include graph traversal

        Returns:
            {
                "results": [...],
                "tokens_used": int,
                "formatted_context": str,
                "queries_used": [...],
                "total_candidates": int,
                "filtered_count": int,
            }
        """
        from cems.retrieval import (
            assemble_context,
            deduplicate_results,
            format_memory_context,
            synthesize_query,
        )

        # Stage 1: Query synthesis
        queries_to_search = [query]
        if enable_query_synthesis and self.config.enable_query_synthesis:
            try:
                from cems.llm import get_client

                client = get_client()
                expanded = synthesize_query(query, client)
                queries_to_search = [query] + expanded[:3]  # Original + up to 3 expansions
            except Exception:
                pass  # Fallback to original query

        # Stage 2: Candidate retrieval (top_k=20 per query)
        all_candidates: list[SearchResult] = []

        for search_query in queries_to_search:
            # Vector search via existing method
            vector_results = self.search(search_query, scope, limit=20)
            all_candidates.extend(vector_results)

            # Graph traversal (if enabled)
            if enable_graph and self._graph and vector_results:
                for top_result in vector_results[:3]:
                    related = self._graph.get_related_memories(
                        top_result.memory_id, max_depth=2, limit=5
                    )
                    # Convert graph results to SearchResult objects
                    for rel in related:
                        mem = self.get(rel["id"])
                        if mem:
                            metadata = self._metadata.get_metadata(rel["id"])
                            # Assign moderate base score for graph-discovered memories
                            base_score = 0.5
                            all_candidates.append(
                                SearchResult(
                                    memory_id=rel["id"],
                                    content=mem.get("memory", ""),
                                    score=base_score,
                                    scope=MemoryScope(metadata.scope) if metadata else MemoryScope.PERSONAL,
                                    metadata=metadata,
                                )
                            )

        # Deduplicate candidates
        candidates = deduplicate_results(all_candidates)

        # Stage 3: Relevance filtering (threshold)
        threshold = self.config.relevance_threshold
        candidates = [c for c in candidates if c.score >= threshold]

        # Stage 4: Temporal ranking (already applied in search() method)
        # Re-sort by final score
        candidates.sort(key=lambda x: x.score, reverse=True)

        total_before_filter = len(all_candidates)
        filtered_count = len(candidates)

        # Stage 5: Token-budgeted assembly
        selected, tokens_used = assemble_context(candidates, max_tokens)

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
            "total_candidates": total_before_filter,
            "filtered_count": filtered_count,
        }
