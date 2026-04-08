"""Entity page compilation job — synthesize knowledge from memory clusters.

Uses the relations graph to find clusters of related memories, then generates
LLM-synthesized "entity pages" that represent a concept or topic.

Entity pages are stored as memory_documents with category='entity-page'.
They participate in search naturally and are protected from maintenance pruning.

Follows the standard async maintenance job pattern.
"""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Minimum cluster size to generate an entity page
MIN_CLUSTER_SIZE = 2
# Maximum memories to include in a single entity page synthesis
MAX_CLUSTER_CONTENT = 15


class CompilationJob:
    """Generate entity pages from memory clusters.

    Discovers clusters via the relations graph (connected components),
    synthesizes each cluster into an entity page, and stores the result.
    """

    def __init__(self, memory: "CEMSMemory"):
        self.memory = memory
        self.config = memory.config

    async def run_async(
        self,
        *,
        limit: int = 20,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        force: bool = False,
    ) -> dict:
        """Run the compilation job.

        Args:
            limit: Max entity pages to generate per run
            min_cluster_size: Min memories in a cluster to compile
            force: If True, recompile even if entity page exists

        Returns:
            Dict with: clusters_found, pages_created, pages_updated
        """
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        # Step 1: Discover clusters from the relations graph
        clusters = await self._discover_clusters(doc_store, user_id, min_cluster_size)
        logger.info(f"CompilationJob: found {len(clusters)} clusters >= {min_cluster_size} members")

        # Step 2: For each cluster, check if entity page exists, then compile
        pages_created = 0
        pages_updated = 0

        for cluster_docs in clusters[:limit]:
            try:
                result = await self._compile_cluster(
                    doc_store, user_id, cluster_docs, force
                )
                if result == "created":
                    pages_created += 1
                elif result == "updated":
                    pages_updated += 1
            except Exception as e:
                logger.warning(f"Failed to compile cluster: {e}")

        report = {
            "clusters_found": len(clusters),
            "pages_created": pages_created,
            "pages_updated": pages_updated,
        }
        logger.info(f"CompilationJob completed: {report}")
        return report

    async def _discover_clusters(
        self, doc_store, user_id: str, min_size: int
    ) -> list[list[dict]]:
        """Discover connected components in the relations graph.

        Returns list of clusters, each cluster is a list of document dicts.
        Sorted by cluster size descending.
        """
        pool = await doc_store._get_pool()
        user_uuid = UUID(user_id)

        # Get only 'similar' relations (not compiled_from/assigned_to which are
        # compilation artifacts, not semantic edges for clustering)
        async with pool.acquire() as conn:
            edges = await conn.fetch(
                """
                SELECT r.source_id, r.target_id
                FROM memory_relations r
                JOIN memory_documents s ON r.source_id = s.id
                WHERE s.user_id = $1
                  AND s.deleted_at IS NULL
                  AND s.category != 'entity-page'
                  AND r.relation_type = 'similar'
                """,
                user_uuid,
            )

        if not edges:
            return []

        # Build adjacency list for connected components
        adj: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            src = str(edge["source_id"])
            tgt = str(edge["target_id"])
            adj[src].add(tgt)
            adj[tgt].add(src)

        # Find connected components via BFS
        visited: set[str] = set()
        components: list[list[str]] = []

        for node in adj:
            if node in visited:
                continue
            # BFS
            component = []
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) >= min_size:
                components.append(component)

        # Sort by size descending (biggest clusters first)
        components.sort(key=len, reverse=True)

        # Fetch document content for each cluster (exclude entity pages)
        result = []
        for component in components:
            docs = []
            for doc_id in component[:MAX_CLUSTER_CONTENT]:
                doc = await doc_store.get_document(doc_id, user_id=user_id)
                if doc and doc.get("category") != "entity-page":
                    docs.append(doc)
            if len(docs) >= min_size:
                result.append(docs)

        return result

    async def _compile_cluster(
        self, doc_store, user_id: str, cluster_docs: list[dict], force: bool
    ) -> str | None:
        """Compile a cluster of memories into an entity page.

        Returns: "created", "updated", "skipped", or None on error.
        """
        # Generate a stable cluster tag from sorted member IDs
        doc_ids = sorted([d["id"] for d in cluster_docs])
        # Use hash of all member IDs for stable identification
        import hashlib
        cluster_hash = hashlib.md5("".join(doc_ids).encode()).hexdigest()[:12]
        cluster_tag = f"entity-cluster:{cluster_hash}"

        # Check if entity page already exists for this cluster
        existing = await doc_store.get_documents_by_tag(
            user_id=user_id,
            tag=cluster_tag,
            limit=1,
        )

        if existing and not force:
            return "skipped"

        # Extract content from cluster docs (needed for both dedup recompile and fresh compile)
        contents = []
        categories = set()
        source_refs = set()
        for doc in cluster_docs:
            content = doc.get("content_detailed") or doc.get("content", "")
            if content:
                contents.append(content)  # Full content — no truncation
            cat = doc.get("category", "general")
            categories.add(cat)
            ref = doc.get("source_ref")
            if ref:
                source_refs.add(ref)

        if not contents:
            return None

        # Dedup: check if a similar entity page already exists (cosine similarity)
        # This prevents near-duplicate entity pages about the same concept
        await self.memory._ensure_initialized_async()
        if self.memory._async_embedder and not force:
            # Use the first source doc's content as a proxy for the topic
            sample = (cluster_docs[0].get("content", "") or "")[:500]
            if sample:
                probe_emb = await self.memory._async_embedder.embed_batch([sample])
                if probe_emb:
                    existing_entities = await doc_store.search_chunks(
                        query_embedding=probe_emb[0],
                        user_id=user_id,
                        category="entity-page",
                        limit=3,
                    )
                    for ent in existing_entities:
                        if ent.get("score", 0) > 0.85:
                            existing_entity_id = ent.get("document_id")
                            # Check if cluster has new members not in existing entity's sources
                            existing_sources = await doc_store.get_related_documents(
                                existing_entity_id, user_id=user_id,
                                relation_type="compiled_from", limit=100,
                            )
                            existing_source_ids = {str(s["id"]) for s in existing_sources}
                            cluster_ids = {str(d["id"]) for d in cluster_docs}
                            new_members = [d for d in cluster_docs
                                           if str(d["id"]) not in existing_source_ids]
                            if not new_members:
                                return "skipped"  # No new content
                            # Require source overlap: at least 30% of existing sources
                            # must be in current cluster to confirm same topic
                            overlap = existing_source_ids & cluster_ids
                            if existing_source_ids and len(overlap) / len(existing_source_ids) < 0.3:
                                continue  # Different topic, check next entity
                            # Recompile: update existing entity page with full cluster
                            entity_content = await self._synthesize_entity(contents, categories)
                            if entity_content:
                                lines = entity_content.strip().split("\n")
                                title = lines[0].lstrip("# ").strip()[:100]
                                await self.memory.update_async(
                                    existing_entity_id, entity_content,
                                )
                                await self._update_title(doc_store, existing_entity_id, title)
                                # Add relations for new members (both directions)
                                for d in new_members:
                                    await doc_store.add_relations(
                                        existing_entity_id,
                                        [{"target_id": d["id"], "relation_type": "compiled_from",
                                          "similarity": 1.0}],
                                    )
                                    await doc_store.add_relations(
                                        d["id"],
                                        [{"target_id": existing_entity_id,
                                          "relation_type": "compiled_from",
                                          "similarity": 1.0}],
                                    )
                                logger.info(
                                    f"Updated entity page {existing_entity_id[:8]} "
                                    f"with {len(new_members)} new members"
                                )
                                return "updated"
                            return "skipped"

        # LLM synthesis
        entity_content = await self._synthesize_entity(contents, categories)
        if not entity_content:
            return None

        # Determine the dominant category
        category_counts: dict[str, int] = defaultdict(int)
        for doc in cluster_docs:
            category_counts[doc.get("category", "general")] += 1
        dominant_cat = max(category_counts, key=category_counts.get)

        # Build title from LLM content (first line)
        lines = entity_content.strip().split("\n")
        title = lines[0].lstrip("# ").strip()[:100] if lines else f"Entity: {dominant_cat}"

        # Store or update the entity page
        if existing and force:
            doc_id = existing[0]["id"]
            await self.memory.update_async(doc_id, entity_content)
            await self._update_title(doc_store, doc_id, title)
            return "updated"

        # Create new entity page
        tags = [
            "entity-page",
            cluster_tag,
            f"cluster-size:{len(cluster_docs)}",
        ]
        for ref in list(source_refs)[:3]:
            tags.append(f"source:{ref}")

        result = await self.memory.add_async(
            content=entity_content,
            scope="personal",
            category="entity-page",
            source="compiler",
            tags=tags,
            title=title,
            source_ref=list(source_refs)[0] if source_refs else None,
        )

        # Link entity page to its source memories via relations
        entity_doc_id = result.get("results", [{}])[0].get("id")
        if entity_doc_id:
            relations = [
                {"target_id": d["id"], "relation_type": "compiled_from", "similarity": 1.0}
                for d in cluster_docs if d.get("id")
            ]
            if relations:
                await doc_store.add_relations(entity_doc_id, relations)
                # Reverse: source → entity
                for rel in relations:
                    await doc_store.add_relations(
                        rel["target_id"],
                        [{"target_id": entity_doc_id, "relation_type": "compiled_from", "similarity": 1.0}],
                    )

        return "created"

    @staticmethod
    async def _update_title(doc_store, document_id: str, title: str) -> None:
        """Update a document's title via direct SQL."""
        pool = await doc_store._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_documents SET title = $1 WHERE id = $2",
                title, UUID(document_id),
            )

    async def _synthesize_entity(
        self, contents: list[str], categories: set[str]
    ) -> str | None:
        """Use LLM to synthesize a cluster into an entity page."""
        try:
            from cems.llm import get_client
            client = get_client()
            if not client:
                return self._fallback_synthesis(contents, categories)

            cat_str = ", ".join(sorted(categories))
            memories_text = "\n\n---\n\n".join(
                f"Memory {i+1}:\n{c}" for i, c in enumerate(contents)
            )

            prompt = f"""You are compiling a knowledge wiki page from {len(contents)} related memories.
Categories: {cat_str}

{memories_text}

Write a COMPLETE, DETAILED knowledge page. This is a permanent reference document — be thorough, not brief.

## Required structure:

# [Clear descriptive title]

## Overview
A comprehensive 2-3 paragraph summary of the topic. What is this about? Why does it matter?

## Key Decisions & Patterns
Every decision, pattern, convention, and working solution from the memories. Use bullet points.
Include specific commands, file paths, config values, and code patterns verbatim.

## Technical Details
Implementation specifics, architecture notes, API details. Be precise — include exact values.

## Contradictions & Evolution
How understanding evolved over time. Note any conflicting information between memories.

## Important Notes
Warnings, gotchas, things to remember. Include any "never do X" or "always do Y" patterns.

---

RULES:
- Include ALL information from the memories. Do NOT summarize away details.
- Use exact values: file paths, commands, config keys, error messages.
- Every bullet point should be actionable or informative — no filler.
- Minimum 500 words. This is a reference document, not a summary.
- Use markdown formatting: headers, bullet points, code blocks, bold for emphasis.

Output:"""

            response = client.complete(prompt, max_tokens=4000)
            return response if response and len(response) > 50 else None
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return self._fallback_synthesis(contents, categories)

    def _fallback_synthesis(
        self, contents: list[str], categories: set[str]
    ) -> str:
        """Simple fallback when LLM is unavailable."""
        cat_str = ", ".join(sorted(categories))
        header = f"# Knowledge Cluster ({cat_str})\n\n"
        header += f"Compiled from {len(contents)} related memories.\n\n"
        body = "\n\n---\n\n".join(contents[:5])
        return header + body
