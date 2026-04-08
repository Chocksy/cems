---
title: "feat: Entity-Aware Maintenance — Compile First, Archive Later"
type: feat
date: 2026-04-08
brainstorm: docs/brainstorms/2026-04-08-entity-aware-maintenance-brainstorm.md
---

# Entity-Aware Maintenance

## Overview

Shift maintenance from "aggressively prune old memories" to "compile memories into
entity pages first, then archive sources." Entity pages become the permanent knowledge
store. Individual memories are raw material that flows through: link → compile → archive.

Three changes: (1) reduce orphans via lower thresholds + LLM orphan assignment,
(2) simplify SummarizationJob to only archive compiled memories, (3) fix CompilationJob
staleness so entity pages update when clusters grow.

## What Changes

### 1. `src/cems/maintenance/relation_builder.py` — Lower similarity threshold

**Current**: `SIMILARITY_THRESHOLD = 0.75` (line 17)
**New**: `SIMILARITY_THRESHOLD = 0.65`

This catches more semantic connections. Memories about "Docker build errors" and
"container image fails on Mac" may only have 0.68 cosine similarity but clearly
belong together.

### 2. `src/cems/maintenance/compilation.py` — Lower min cluster + fix staleness

**Lower minimum cluster size:**
```python
# Current
MIN_CLUSTER_SIZE = 3

# New
MIN_CLUSTER_SIZE = 2  # Even 2 memories can make a slim entity page
```

**Fix staleness detection (lines 186-204):**

Currently when cosine dedup finds a similar entity page (>0.85), it returns
`"skipped"`. Instead: check if the cluster has NEW source memories not yet
compiled into that entity page. If yes, recompile and update it.

```python
# Current (line 199-203):
for ent in existing_entities:
    if ent.get("score", 0) > 0.85:
        return "skipped"

# New:
for ent in existing_entities:
    if ent.get("score", 0) > 0.85:
        existing_entity_id = ent.get("document_id")
        # Check if cluster has new members not in existing entity's sources
        existing_sources = await doc_store.get_related_documents(
            existing_entity_id, user_id=user_id,
            relation_type="compiled_from", limit=100
        )
        existing_source_ids = {str(s["id"]) for s in existing_sources}
        new_members = [d for d in cluster_docs
                       if str(d["id"]) not in existing_source_ids]
        if not new_members:
            return "skipped"  # No new content
        # Recompile: update existing entity page with full cluster
        entity_content = await self._synthesize_entity(contents, categories)
        if entity_content:
            lines = entity_content.strip().split("\n")
            title = lines[0].lstrip("# ").strip()[:100]
            await doc_store.update_document(
                existing_entity_id, content=entity_content,
                title=title, user_id=user_id,
            )
            # Add relations for new members
            for d in new_members:
                await doc_store.add_relations(
                    existing_entity_id,
                    [{"target_id": d["id"], "relation_type": "compiled_from",
                      "similarity": 1.0}]
                )
            return "updated"
        return "skipped"
```

### 3. `src/cems/maintenance/orphan_assigner.py` — NEW: LLM orphan assignment

New maintenance job that runs daily. Finds memories with no `compiled_from` relation
and asks an LLM which existing entity page they belong to.

```python
class OrphanAssignerJob:
    """Assign orphan memories to existing entity pages via LLM."""

    async def run_async(self, limit: int = 50) -> dict:
        """Find orphan memories and assign to entity pages."""
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        # 1. Get entity page titles + IDs
        entity_pages = await self._get_entity_index(doc_store, user_id)
        if not entity_pages:
            return {"orphans_found": 0, "assigned": 0, "message": "no entity pages"}

        # 2. Get orphan memories (no compiled_from relation)
        orphans = await self._get_orphan_memories(doc_store, user_id, limit)
        if not orphans:
            return {"orphans_found": 0, "assigned": 0}

        # 3. For each orphan, ask LLM which entity it belongs to
        assigned = 0
        for orphan in orphans:
            entity_id = await self._assign_orphan(orphan, entity_pages)
            if entity_id:
                # Add compiled_from relation
                await doc_store.add_relations(
                    entity_id,
                    [{"target_id": orphan["id"],
                      "relation_type": "compiled_from",
                      "similarity": 0.8}]
                )
                assigned += 1

        return {"orphans_found": len(orphans), "assigned": assigned}
```

**Orphan query** (memories with no `compiled_from` relation as target):
```sql
SELECT d.id, d.content, d.category, d.source_ref
FROM memory_documents d
WHERE d.user_id = $1
  AND d.deleted_at IS NULL
  AND d.category NOT IN ('entity-page', 'gate-rules', 'guidelines',
                          'preferences', 'category-summary')
  AND NOT EXISTS (
      SELECT 1 FROM memory_relations r
      WHERE r.target_id = d.id
        AND r.relation_type = 'compiled_from'
  )
ORDER BY d.created_at ASC
LIMIT $2
```

**LLM assignment prompt:**
```
Given this memory:
"{memory_content}"

Which of these knowledge topics does it belong to?
{entity_titles_and_summaries}

Return the topic ID if it clearly belongs to one. Return "none" if it doesn't
fit any topic. Return at most 1 ID.
```

### 4. `src/cems/maintenance/summarization.py` — Simplify: archive compiled only

**Remove `_compress_by_category()`** — entity pages replace category summaries.

**Simplify `_consolidate_never_shown()`** — only archive memories that:
- Have a `compiled_from` relation (knowledge preserved in entity page)
- Are 14+ days old
- Are not hot (`shown_count < 20`)

**Keep `_prune_chronically_noisy()`** — noisy memories should still be pruned
regardless of compilation status.

```python
async def run_async(self):
    """Simplified maintenance: archive compiled memories, prune noise."""
    report = {}

    # 1. Archive compiled memories (knowledge preserved in entity pages)
    report["archived_compiled"] = await self._archive_compiled_sources()

    # 2. Prune chronically noisy memories
    report["noise_pruned"] = await self._prune_chronically_noisy()

    return report

async def _archive_compiled_sources(self) -> int:
    """Archive memories that have been compiled into entity pages."""
    doc_store = await self.memory._ensure_document_store()
    user_id = self.config.user_id

    # Find compiled source memories older than 14 days
    compiled_sources = await doc_store.get_compiled_sources(
        user_id=user_id,
        min_age_days=14,
        max_shown_count=20,  # Don't archive hot memories
        limit=100,
    )

    archived = 0
    for doc in compiled_sources:
        await doc_store.delete_document(doc["id"], hard=False)
        archived += 1

    return archived
```

**New DocumentStore method: `get_compiled_sources()`**
```sql
SELECT d.id, d.content, d.category
FROM memory_documents d
WHERE d.user_id = $1
  AND d.deleted_at IS NULL
  AND d.category NOT IN ('entity-page', 'gate-rules', 'guidelines',
                          'preferences', 'category-summary')
  AND d.created_at < NOW() - INTERVAL '1 day' * $2
  AND COALESCE(d.shown_count, 0) < $3
  AND EXISTS (
      SELECT 1 FROM memory_relations r
      WHERE r.target_id = d.id
        AND r.relation_type = 'compiled_from'
  )
ORDER BY d.created_at ASC
LIMIT $4
```

### 5. `src/cems/scheduler.py` — Update schedule

```python
# Remove or simplify:
# - _run_summarization: change from weekly to weekly, simplified
# - Add: _run_orphan_assigner (daily)

# New job:
self._scheduler.add_job(
    self._run_orphan_assigner,
    CronTrigger(hour=self.config.nightly_hour, minute=45),
    id="daily_orphan_assigner",
    name="Daily Orphan Assigner",
    replace_existing=True,
)
```

Update `_run_job_for_memory` and `valid_jobs` to include `"orphan_assigner"`.

### 6. `src/cems/maintenance/__init__.py` — Register new job

```python
from cems.maintenance.orphan_assigner import OrphanAssignerJob
__all__ = [..., "OrphanAssignerJob"]
```

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/cems/maintenance/relation_builder.py` | Modify | Lower threshold 0.75 → 0.65 |
| `src/cems/maintenance/compilation.py` | Modify | MIN_CLUSTER_SIZE 3 → 2, fix staleness recompilation |
| `src/cems/maintenance/orphan_assigner.py` | Create | New LLM-based orphan assignment job |
| `src/cems/maintenance/summarization.py` | Modify | Remove `_compress_by_category`, simplify to archive compiled only |
| `src/cems/maintenance/__init__.py` | Modify | Register OrphanAssignerJob |
| `src/cems/scheduler.py` | Modify | Add orphan assigner schedule, update valid_jobs |
| `src/cems/db/document_store.py` | Modify | Add `get_compiled_sources()` method |
| `src/cems/api/handlers/memory.py` | Modify | Add "orphan_assigner" to API maintenance handler |

## Acceptance Criteria

### Functional Requirements

- [x] `MIN_CLUSTER_SIZE` = 2 — even 2-memory clusters produce entity pages
- [x] Similarity threshold lowered to 0.65 for relation building
- [x] CompilationJob updates existing entity pages when clusters grow (staleness fix)
- [x] OrphanAssignerJob assigns orphan memories to entity pages via LLM
- [x] SummarizationJob only archives memories with `compiled_from` relation
- [x] SummarizationJob no longer creates category summaries
- [x] Hot memories (`shown_count >= 20`) never archived regardless of compilation
- [x] All jobs registered in scheduler with correct intervals
- [x] `POST /api/memory/maintenance {job_type: "orphan_assigner"}` works

### Non-Functional Requirements

- [ ] OrphanAssignerJob costs < $0.05 per run (50 orphans × ~$0.001 each)
- [x] All existing tests pass
- [x] New tests for orphan assigner and simplified summarization

### Quality Gates

- [ ] Run compilation on local Docker — verify entity pages created with MIN_CLUSTER_SIZE=2
- [ ] Run orphan assigner — verify orphans get assigned to entity pages
- [ ] Run summarization — verify it only archives compiled memories
- [ ] Verify orphan count drops significantly (from ~1,600 to ~100-200)

## References

- Brainstorm: `docs/brainstorms/2026-04-08-entity-aware-maintenance-brainstorm.md`
- Knowledge Engine brainstorm: `docs/brainstorms/2026-04-06-knowledge-engine-brainstorm.md`
- Current maintenance: `src/cems/maintenance/`
- CompilationJob: `src/cems/maintenance/compilation.py`
- SummarizationJob: `src/cems/maintenance/summarization.py`
- RelationBuilder: `src/cems/maintenance/relation_builder.py`
- Scheduler: `src/cems/scheduler.py`
