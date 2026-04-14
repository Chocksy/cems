---
title: "fix: Compilation entity page deduplication"
type: fix
date: 2026-04-14
brainstorm: docs/brainstorms/2026-04-08-entity-aware-maintenance-brainstorm.md
---

# Fix Compilation Entity Page Deduplication

## Overview

The `CompilationJob` creates duplicate entity pages about the same topic. On
`cems.chocksy.com`: 86 entity pages, ~30+ duplicate/near-duplicate pairs. On
`cems.ai.hbstf.co` (Hubstaff): two wiki pages about "Simplify Pay Rate Guardrails".

The worst case: **4 entity pages** about "Fiscal Printer Integration and Z-Report
Reprint" — each with a different `entity-cluster:` hash, created on different days.

Root cause: three gaps in the dedup logic in `_compile_cluster()` at
`src/cems/maintenance/compilation.py:163-324`.

## Problem Analysis

The current dedup has three layers, each with a gap:

### Gap 1: Cluster hash changes when cluster membership changes (line 171-175)

```python
cluster_hash = hashlib.md5("".join(doc_ids).encode()).hexdigest()[:12]
cluster_tag = f"entity-cluster:{cluster_hash}"
```

The tag is an MD5 of exact member IDs. When the relations graph grows (new memories,
new `similar` edges), connected components change shape. The same topic gets a new hash,
the tag-based lookup finds nothing, and a new entity page is created.

**Evidence**: Fiscal Printer pages have 3 different cluster hashes:
- `entity-cluster:8b48711cdb13` (13 members, Apr 9)
- `entity-cluster:40798be532ef` (13 members, Apr 10 — different members!)
- `entity-cluster:109f7a238c7a` (8 members, Apr 12)

### Gap 2: Embedding probe uses only first doc's first 500 chars (line 209)

```python
sample = (cluster_docs[0].get("content", "") or "")[:500]
```

This is fragile. Different clusters about the same topic have different first documents.
The probe embedding may not match the existing entity page above the 0.85 threshold.

### Gap 3: 30% source overlap requirement is too strict (line 236)

```python
if existing_source_ids and len(overlap) / len(existing_source_ids) < 0.3:
    continue  # Different topic, check next entity
```

When the graph restructures, a new cluster may share <30% of members with the
existing entity's sources — even though it's clearly the same topic. The overlap
check was meant to prevent merging unrelated entities, but it's too aggressive.

### Compounding factor: 6-hour schedule

`CompilationJob` runs every 6 hours (`scheduler.py:119`). `RelationBuilderJob` also
runs every 6 hours (`scheduler.py:111`). Between runs, new edges reshape clusters,
producing new hashes for the same topics.

## Proposed Solution

A **title-based dedup** layer before the embedding check. Entity page titles are
short, descriptive, and stable — if two entity pages have highly similar titles,
they're about the same topic. This is the most reliable signal and the cheapest check.

### Strategy: Multi-layer dedup with title matching as primary

1. **Tag match** (existing, fast) — exact cluster hash → skip/update
2. **Title similarity** (NEW, cheap) — embed the candidate title against existing
   entity page titles. High similarity → merge into existing page
3. **Embedding probe** (existing, improved) — use multiple samples instead of one,
   lower threshold
4. **Merge existing duplicates** — one-time cleanup of existing dupes on `cems.chocksy.com`

## What Changes

### 1. `src/cems/maintenance/compilation.py` — Rewrite dedup in `_compile_cluster()`

**Replace the single-sample embedding probe with a two-phase approach:**

Phase A: After synthesizing the entity content, extract the title. Embed the title
and search against existing entity page chunks with `category='entity-page'`.
If any existing entity page scores > 0.80 on the title embedding, treat it as the
same topic.

Phase B: If title match fails, use the full synthesized content (not just first 500
chars of first source doc) as the probe. Check with a lower threshold (0.78 instead
of 0.85).

**Remove the 30% source overlap requirement.** The title/content similarity check
is a much stronger signal than source overlap. Two clusters about the same topic
may share zero source memories if the graph restructured.

**Key change in `_compile_cluster()`:**

```python
async def _compile_cluster(self, doc_store, user_id, cluster_docs, force):
    # 1. Tag-based check (existing, unchanged)
    cluster_tag = self._compute_cluster_tag(cluster_docs)
    existing = await doc_store.get_documents_by_tag(user_id=user_id, tag=cluster_tag, limit=1)
    if existing and not force:
        return "skipped"

    # 2. Extract content from cluster docs (existing, unchanged)
    contents, categories, source_refs = self._extract_cluster_content(cluster_docs)
    if not contents:
        return None

    # 3. Synthesize first — we need the title for dedup
    entity_content = await self._synthesize_entity(contents, categories)
    if not entity_content:
        return None
    lines = entity_content.strip().split("\n")
    title = lines[0].lstrip("# ").strip()[:100]

    # 4. NEW: Title-based dedup against existing entity pages
    if not force:
        match = await self._find_existing_entity(
            doc_store, user_id, title, entity_content, cluster_docs
        )
        if match:
            entity_id, action = match
            if action == "skipped":
                return "skipped"
            # Update existing entity page
            await self.memory.update_async(entity_id, entity_content)
            await self._update_title(doc_store, entity_id, title)
            await self._link_new_sources(doc_store, entity_id, cluster_docs, user_id)
            return "updated"

    # 5. Create new entity page (existing logic, unchanged)
    ...
```

**New method: `_find_existing_entity()`:**

```python
async def _find_existing_entity(self, doc_store, user_id, title, content, cluster_docs):
    """Find an existing entity page that covers the same topic.

    Uses title embedding similarity as primary signal,
    falls back to content embedding.

    Returns: (entity_id, "update"|"skipped") or None
    """
    await self.memory._ensure_initialized_async()
    if not self.memory._async_embedder:
        return None

    # Phase A: Title-based search
    title_emb = await self.memory._async_embedder.embed_batch([title])
    if title_emb:
        candidates = await doc_store.search_chunks(
            query_embedding=title_emb[0],
            user_id=user_id,
            category="entity-page",
            limit=5,
        )
        for cand in candidates:
            if cand.get("score", 0) > 0.80:
                entity_id = cand.get("document_id")
                has_new = await self._has_new_sources(
                    doc_store, entity_id, cluster_docs, user_id
                )
                return (entity_id, "update" if has_new else "skipped")

    # Phase B: Content-based search (use first ~1000 chars of synthesized content)
    content_sample = content[:1000]
    content_emb = await self.memory._async_embedder.embed_batch([content_sample])
    if content_emb:
        candidates = await doc_store.search_chunks(
            query_embedding=content_emb[0],
            user_id=user_id,
            category="entity-page",
            limit=3,
        )
        for cand in candidates:
            if cand.get("score", 0) > 0.78:
                entity_id = cand.get("document_id")
                has_new = await self._has_new_sources(
                    doc_store, entity_id, cluster_docs, user_id
                )
                return (entity_id, "update" if has_new else "skipped")

    return None
```

**New helper: `_has_new_sources()`:**

```python
async def _has_new_sources(self, doc_store, entity_id, cluster_docs, user_id):
    """Check if cluster has source memories not yet linked to entity page."""
    existing_sources = await doc_store.get_related_documents(
        entity_id, user_id=user_id,
        relation_type="compiled_from", limit=200,
    )
    existing_ids = {str(s["id"]) for s in existing_sources}
    return any(str(d["id"]) not in existing_ids for d in cluster_docs)
```

**New helper: `_link_new_sources()`:**

```python
async def _link_new_sources(self, doc_store, entity_id, cluster_docs, user_id):
    """Link any new source memories to the entity page."""
    existing_sources = await doc_store.get_related_documents(
        entity_id, user_id=user_id,
        relation_type="compiled_from", limit=200,
    )
    existing_ids = {str(s["id"]) for s in existing_sources}
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
```

**Trade-off**: Synthesizing before dedup means we pay the LLM cost even for duplicates.
But synthesis is cheap (~$0.005 per page with Gemini Flash) and gives us the title for
accurate matching. The alternative — matching raw source content — is what fails today.

### 2. `src/cems/maintenance/compilation.py` — Add duplicate entity merge method

For cases where duplicates already exist, add a post-compilation cleanup step:

```python
async def _merge_duplicate_entities(self, doc_store, user_id, limit=10):
    """Find and merge duplicate entity pages (same topic, different cluster tags).

    Runs after compilation. Compares entity page titles via embedding similarity.
    When two entity pages are >0.80 similar, keeps the one with more sources
    and soft-deletes the other, transferring its source relations.
    """
```

This runs at the end of `run_async()` as a cleanup sweep. It catches any duplicates
that slipped through from previous runs.

### 3. `src/cems/maintenance/lint.py` — Add duplicate entity detection

Add a `_detect_duplicate_entities()` check to the lint job so the dashboard surfaces
duplicate entity pages:

```python
async def _detect_duplicate_entities(self, doc_store, user_id, limit=50):
    """Detect entity pages with high title similarity (potential duplicates)."""
    entities = await doc_store.get_documents_by_category(
        user_id=user_id, category="entity-page", limit=200
    )
    # Embed all titles, compute pairwise similarity
    # Flag pairs with similarity > 0.80 as potential duplicates
```

### 4. One-time cleanup script — `scripts/merge_duplicate_entities.py`

A standalone script to merge existing duplicates on `cems.chocksy.com`:

- Fetch all entity pages
- Group by title similarity (>0.80 cosine)
- For each group: keep the page with highest shown_count, soft-delete others,
  transfer `compiled_from` relations to the kept page
- Dry-run mode by default, `--execute` to apply

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/cems/maintenance/compilation.py` | Modify | Rewrite dedup: title-based matching, remove source overlap check, add merge step |
| `src/cems/maintenance/lint.py` | Modify | Add `_detect_duplicate_entities()` check |
| `scripts/merge_duplicate_entities.py` | Create | One-time cleanup of existing duplicates |
| `tests/test_maintenance.py` | Modify | Add tests for title-based dedup and merge logic |

## Acceptance Criteria

- [x] CompilationJob never creates a new entity page when an existing one has >0.80
      title similarity
- [x] CompilationJob updates existing entity pages when clusters grow (adds new source
      relations, recompiles content)
- [ ] Existing duplicate entity pages on `cems.chocksy.com` merged via cleanup script
- [x] Lint job detects and reports duplicate entity pages
- [x] Existing compilation tests still pass
- [x] New tests cover: title dedup hit, content dedup fallback, merge of existing dupes

## Verification Plan

1. Run cleanup script in dry-run on `cems.chocksy.com` — verify it identifies the known
   duplicates (Fiscal Printer 4x, eGalax 2x, SSRF 3x, etc.)
2. Run cleanup script with `--execute` — verify duplicates merged, shown_count preserved
3. Trigger compilation job — verify no new duplicates created
4. Wait 24h (4 compilation cycles) — verify entity page count stays stable or decreases

## References

- Research from this session: identified 30+ duplicate pairs on cems.chocksy.com
- Brainstorm: `docs/brainstorms/2026-04-08-entity-aware-maintenance-brainstorm.md` (section 4)
- Previous plan: `docs/plans/2026-04-08-feat-entity-aware-maintenance-plan.md` (partially implemented)
- CompilationJob: `src/cems/maintenance/compilation.py`
- LintJob: `src/cems/maintenance/lint.py`
- Scheduler: `src/cems/scheduler.py:117-123` (6-hourly compilation)
