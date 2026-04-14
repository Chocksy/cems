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

        # Step 3: Merge any duplicate entity pages that already exist
        pages_merged = await self._merge_duplicate_entities(doc_store, user_id)

        report = {
            "clusters_found": len(clusters),
            "pages_created": pages_created,
            "pages_updated": pages_updated,
            "pages_merged": pages_merged,
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

        Dedup strategy (checked in order):
        1. Tag match — exact cluster hash → skip
        2. Synthesize content to get a stable title
        3. Title embedding similarity — find existing entity page about same topic
        4. Content embedding fallback — lower threshold, synthesized content as probe

        Returns: "created", "updated", "skipped", or None on error.
        """
        import hashlib

        # Step 1: Tag-based check (fast, exact match)
        doc_ids = sorted([d["id"] for d in cluster_docs])
        cluster_hash = hashlib.md5("".join(doc_ids).encode()).hexdigest()[:12]
        cluster_tag = f"entity-cluster:{cluster_hash}"

        existing = await doc_store.get_documents_by_tag(
            user_id=user_id, tag=cluster_tag, limit=1,
        )
        if existing and not force:
            return "skipped"

        # Step 2: Extract content from cluster docs
        contents = []
        categories = set()
        source_refs = set()
        for doc in cluster_docs:
            content = doc.get("content_detailed") or doc.get("content", "")
            if content:
                contents.append(content)
            categories.add(doc.get("category", "general"))
            ref = doc.get("source_ref")
            if ref:
                source_refs.add(ref)

        if not contents:
            return None

        # Step 3: Synthesize — we need the title for reliable dedup
        entity_content = await self._synthesize_entity(contents, categories)
        if not entity_content:
            return None

        lines = entity_content.strip().split("\n")
        category_counts: dict[str, int] = defaultdict(int)
        for doc in cluster_docs:
            category_counts[doc.get("category", "general")] += 1
        dominant_cat = max(category_counts, key=category_counts.get)
        title = lines[0].lstrip("# ").strip()[:100] if lines else f"Entity: {dominant_cat}"

        # Step 4: Title + content dedup against existing entity pages
        if not force:
            match = await self._find_existing_entity(
                doc_store, user_id, title, entity_content, cluster_docs,
            )
            if match:
                entity_id, action = match
                if action == "skipped":
                    return "skipped"
                # Update existing entity page with new synthesis
                await self.memory.update_async(entity_id, entity_content)
                await self._update_title(doc_store, entity_id, title)
                await self._link_new_sources(doc_store, entity_id, cluster_docs, user_id)
                logger.info(f"Updated entity page {entity_id[:8]} via dedup")
                return "updated"

        # Step 5: Force-update existing tag match
        if existing and force:
            doc_id = existing[0]["id"]
            await self.memory.update_async(doc_id, entity_content)
            await self._update_title(doc_store, doc_id, title)
            return "updated"

        # Step 6: Create new entity page
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
                for rel in relations:
                    await doc_store.add_relations(
                        rel["target_id"],
                        [{"target_id": entity_doc_id, "relation_type": "compiled_from", "similarity": 1.0}],
                    )

        return "created"

    async def _find_existing_entity(
        self, doc_store, user_id: str, title: str, content: str,
        cluster_docs: list[dict],
    ) -> tuple[str, str] | None:
        """Find an existing entity page covering the same topic.

        Two-phase search:
          Phase A: Embed the synthesized title, search entity-page chunks (>0.80)
          Phase B: Embed first 1000 chars of synthesized content (>0.78)

        Returns (entity_id, "update"|"skipped") or None.
        """
        await self.memory._ensure_initialized_async()
        if not self.memory._async_embedder:
            return None

        # Phase A: Title-based similarity
        title_embs = await self.memory._async_embedder.embed_batch([title])
        if title_embs:
            candidates = await doc_store.search_chunks(
                query_embedding=title_embs[0],
                user_id=user_id,
                category="entity-page",
                limit=5,
            )
            for cand in candidates:
                if cand.get("score", 0) > 0.80:
                    entity_id = cand.get("document_id")
                    has_new = await self._has_new_sources(
                        doc_store, entity_id, cluster_docs, user_id,
                    )
                    return (entity_id, "update" if has_new else "skipped")

        # Phase B: Content-based similarity (synthesized content, not raw source)
        content_sample = content[:1000]
        content_embs = await self.memory._async_embedder.embed_batch([content_sample])
        if content_embs:
            candidates = await doc_store.search_chunks(
                query_embedding=content_embs[0],
                user_id=user_id,
                category="entity-page",
                limit=3,
            )
            for cand in candidates:
                if cand.get("score", 0) > 0.78:
                    entity_id = cand.get("document_id")
                    has_new = await self._has_new_sources(
                        doc_store, entity_id, cluster_docs, user_id,
                    )
                    return (entity_id, "update" if has_new else "skipped")

        return None

    async def _has_new_sources(
        self, doc_store, entity_id: str, cluster_docs: list[dict], user_id: str,
    ) -> bool:
        """Check if cluster has source memories not yet linked to the entity page."""
        existing_sources = await doc_store.get_related_documents(
            entity_id, user_id=user_id,
            relation_type="compiled_from", limit=200,
        )
        existing_ids = {str(s["id"]) for s in existing_sources}
        return any(str(d["id"]) not in existing_ids for d in cluster_docs)

    async def _link_new_sources(
        self, doc_store, entity_id: str, cluster_docs: list[dict], user_id: str,
    ) -> int:
        """Link new source memories to an existing entity page. Returns count linked."""
        existing_sources = await doc_store.get_related_documents(
            entity_id, user_id=user_id,
            relation_type="compiled_from", limit=200,
        )
        existing_ids = {str(s["id"]) for s in existing_sources}
        linked = 0
        for d in cluster_docs:
            if str(d["id"]) not in existing_ids:
                await doc_store.add_relations(
                    entity_id,
                    [{"target_id": d["id"], "relation_type": "compiled_from", "similarity": 1.0}],
                )
                await doc_store.add_relations(
                    d["id"],
                    [{"target_id": entity_id, "relation_type": "compiled_from", "similarity": 1.0}],
                )
                linked += 1
        return linked

    async def _merge_duplicate_entities(
        self, doc_store, user_id: str, limit: int = 50,
    ) -> int:
        """Find and merge duplicate entity pages about the same topic.

        Compares entity page titles via embedding similarity. When two pages
        are >0.80 similar, keeps the one with higher shown_count and
        soft-deletes the other, transferring its compiled_from relations.

        Returns count of pages merged (soft-deleted).
        """
        await self.memory._ensure_initialized_async()
        if not self.memory._async_embedder:
            return 0

        entities = await doc_store.get_documents_by_category(
            user_id=user_id, category="entity-page", limit=200,
        )
        if len(entities) < 2:
            return 0

        # Embed all titles
        titles = [e.get("title") or (e.get("content", "")[:80]) for e in entities]
        embeddings = await self.memory._async_embedder.embed_batch(titles)
        if not embeddings or len(embeddings) != len(entities):
            return 0

        # Find duplicate pairs (greedy: first match wins)
        merged_ids: set[str] = set()
        merged_count = 0

        for i in range(len(entities)):
            if str(entities[i]["id"]) in merged_ids or merged_count >= limit:
                break
            for j in range(i + 1, len(entities)):
                if str(entities[j]["id"]) in merged_ids:
                    continue
                # Cosine similarity
                dot = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
                norm_i = sum(a * a for a in embeddings[i]) ** 0.5
                norm_j = sum(a * a for a in embeddings[j]) ** 0.5
                if norm_i == 0 or norm_j == 0:
                    continue
                sim = dot / (norm_i * norm_j)
                if sim > 0.80:
                    # Keep the one with higher shown_count
                    shown_i = entities[i].get("shown_count", 0) or 0
                    shown_j = entities[j].get("shown_count", 0) or 0
                    if shown_i >= shown_j:
                        keep, discard = entities[i], entities[j]
                    else:
                        keep, discard = entities[j], entities[i]

                    keep_id = str(keep["id"])
                    discard_id = str(discard["id"])

                    # Transfer compiled_from relations from discard → keep
                    discard_sources = await doc_store.get_related_documents(
                        discard_id, user_id=user_id,
                        relation_type="compiled_from", limit=200,
                    )
                    keep_sources = await doc_store.get_related_documents(
                        keep_id, user_id=user_id,
                        relation_type="compiled_from", limit=200,
                    )
                    keep_source_ids = {str(s["id"]) for s in keep_sources}

                    for src in discard_sources:
                        src_id = str(src["id"])
                        if src_id not in keep_source_ids:
                            await doc_store.add_relations(
                                keep_id,
                                [{"target_id": src_id, "relation_type": "compiled_from",
                                  "similarity": 1.0}],
                            )
                            await doc_store.add_relations(
                                src_id,
                                [{"target_id": keep_id, "relation_type": "compiled_from",
                                  "similarity": 1.0}],
                            )

                    # Soft-delete the duplicate
                    await doc_store.delete_document(discard_id, hard=False, user_id=user_id)
                    merged_ids.add(discard_id)
                    merged_count += 1
                    logger.info(
                        f"Merged entity page {discard_id[:8]} into {keep_id[:8]} "
                        f"(title sim={sim:.2f})"
                    )
                    break  # Move to next i

        return merged_count

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
