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

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)

# Minimum cluster size to generate an entity page
MIN_CLUSTER_SIZE = 3
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
        from uuid import UUID
        pool = await doc_store._get_pool()
        user_uuid = UUID(user_id)

        # Get all relations for this user's documents
        async with pool.acquire() as conn:
            edges = await conn.fetch(
                """
                SELECT r.source_id, r.target_id
                FROM memory_relations r
                JOIN memory_documents s ON r.source_id = s.id
                WHERE s.user_id = $1 AND s.deleted_at IS NULL
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

        # Fetch document content for each cluster
        result = []
        for component in components:
            docs = []
            for doc_id in component[:MAX_CLUSTER_CONTENT]:
                doc = await doc_store.get_document(doc_id, user_id=user_id)
                if doc:
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

        # Check if entity page already exists for this exact cluster
        existing = await doc_store.get_documents_by_tag(
            user_id=user_id,
            tag=cluster_tag,
            limit=1,
        )

        if existing and not force:
            return "skipped"

        # Synthesize the entity page content
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
            await doc_store.update_document(
                doc_id,
                content=entity_content,
                title=title,
                user_id=user_id,
            )
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
