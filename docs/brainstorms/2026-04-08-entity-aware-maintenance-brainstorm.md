# Entity-Aware Maintenance: Compile First, Prune Later

**Date**: 2026-04-08
**Status**: Brainstorm
**Previous**: [Knowledge Engine Brainstorm](2026-04-06-knowledge-engine-brainstorm.md)

## What We're Building

Reverse the maintenance philosophy: instead of "delete old memories to save space,"
shift to **"compile memories into entity pages first, then it's safe to prune."**
Entity pages become the permanent knowledge store. Individual memories are raw
material that gets compiled, then can be safely demoted.

## Why This Approach

### The Problem: Maintenance Kills Knowledge Before It's Preserved

Current maintenance jobs aggressively prune memories:

| Job | What it deletes | Checks entity relations? |
|-----|----------------|------------------------|
| SummarizationJob `_compress_by_category` | Memories 14+ days old with shown_count < 20 | **NO** |
| SummarizationJob `_consolidate_never_shown` | Groups of 20+ never-shown memories | **NO** |
| SummarizationJob `_prune_chronically_noisy` | Memories with >50% noise ratio | **NO** |
| ConsolidationJob | Duplicate memories (cosine >0.98 auto-merge) | **NO** |

**The race condition**: SummarizationJob runs nightly at 3 AM. CompilationJob runs
every 10 min. But if a memory is 14+ days old with `shown_count=0` and hasn't been
compiled yet, summarization deletes it first. Its knowledge is lost.

**Real numbers**: We have ~2,580 memories. SummarizationJob `_compress_by_category`
targets anything 14+ days old with shown_count < 20. That's probably the majority
of memories. The entity page system has only compiled ~126 pages so far —
there are ~1,600 orphan memories not in any cluster.

### The Insight: Entity Pages Change Everything

With entity pages:
- **Old memories don't need to be individually preserved** — their knowledge lives
  in entity pages
- **Aggressive summarization is less important** — entity pages ARE the summaries
- **The priority flips**: compile first, prune later
- **A memory's value is**: (1) its knowledge contribution to entity pages,
  (2) its direct recall value. Once compiled, #1 is preserved.

### How Karpathy's System Handles This

Karpathy's wiki: raw notes → compiled wiki pages → notes can be archived.
The wiki IS the knowledge. Notes are source material.

Same principle: **entity pages are the knowledge. Memories are source material.**
Once compiled, the source material can be safely demoted without losing knowledge.

## Key Decisions

### 1. Compile-then-prune ordering

Before any maintenance job soft-deletes a memory, check:
- Is this memory a source for an entity page? (`compiled_from` relation exists)
  - **YES** → safe to prune (knowledge preserved in entity page)
  - **NO** → is it part of a cluster (has `similar` relations)?
    - **YES** → don't prune yet, let CompilationJob compile it first
    - **NO** (orphan) → apply current rules (age + shown_count based)

**Implementation**: Add a `_is_entity_source(doc_id)` check in SummarizationJob
and ConsolidationJob before soft-deleting.

### 2. Reduce maintenance aggression

Now that entity pages preserve knowledge, we can relax pruning:

| Current | Proposed | Why |
|---------|----------|-----|
| Compress at 14 days | Compress at 30 days | Give CompilationJob more time |
| Never-shown consolidation at 20+ group | Keep as-is | These truly add noise |
| Auto-merge at cosine >0.98 | Keep but protect entity sources | Duplicates should merge |

### 3. Entity source protection via relation check

Before soft-deleting, query:
```sql
SELECT COUNT(*) FROM memory_relations
WHERE target_id = $1 AND relation_type = 'compiled_from'
```

If count > 0 → the memory is a source for an entity page → skip pruning.
If count = 0 AND memory has `similar` relations → it's in a cluster, let
CompilationJob process it first.

### 4. Staleness detection for entity pages

When CompilationJob finds a cluster whose members have changed since the entity
page was last compiled, mark it stale and recompile:

**Current**: Cluster hash is based on sorted member IDs. If members change,
a NEW entity page gets created (different hash) → duplicates accumulate.

**Fix**: Instead of only checking the cluster tag, also check cosine similarity
against existing entity pages (already done at lines 186-204). If similar entity
page exists (>0.85), UPDATE it instead of creating a new one.

The current code already does cosine dedup but returns "skipped" instead of
updating. Change to: if similar entity exists AND cluster has new members,
recompile and update.

### 5. Memory state machine (the "demotion" model)

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  ACTIVE  │────▶│ COMPILED │────▶│ DEMOTED  │────▶│ ARCHIVED │
│          │     │          │     │          │     │          │
│ Normal   │     │ Source   │     │ Distilled│     │ Soft-    │
│ memory   │     │ for an   │     │ content  │     │ deleted  │
│          │     │ entity   │     │ kept in  │     │          │
│          │     │ page     │     │ detailed │     │          │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                │                │                │
     │  CompilationJob│  Distillation  │ Summarization  │
     │  creates       │  Job           │ Job (only after│
     │  relation      │  compresses    │ compiled)      │
```

- **ACTIVE**: Normal memory, full content, participates in search
- **COMPILED**: Has `compiled_from` relation to an entity page. Knowledge preserved.
- **DEMOTED**: Distilled (content_detailed has original, content has summary).
  Still searchable but smaller footprint.
- **ARCHIVED**: Soft-deleted. Knowledge fully preserved in entity page.
  Only accessible via deep recall mode.

**Key rule**: A memory cannot move from ACTIVE → ARCHIVED without going through
COMPILED first (unless it's an orphan with no relations and truly low value).

## What Changes

### SummarizationJob
- `_compress_by_category()`: Before soft-deleting, check `_is_entity_source(doc_id)`.
  If yes, skip. If no but has `similar` relations, skip (let CompilationJob work).
- `_consolidate_never_shown()`: Same protection — check relations before pruning.
- Increase stale_days from 14 to 30 (give compilation more time).

### ConsolidationJob
- Before auto-merging (Tier 1) or LLM-classified-duplicate (Tier 2), check if
  either document is an entity source. If yes, skip the merge — the entity page
  already consolidates the knowledge.

### CompilationJob
- **Staleness**: When cosine dedup finds a similar entity page (>0.85), instead of
  "skipped", check if the cluster has new members not in the existing page's sources.
  If yes, recompile and update the existing page.
- **Force recompile on schedule**: Track `last_compiled_at` in entity page metadata.
  If entity page is older than 7 days and source cluster has grown, recompile.

### DistillationJob
- No changes needed — already safe (preserves original in `content_detailed`).
- Actually helps: distilled memories are smaller in search results but full content
  is available for entity page compilation.

## Final Decisions (from discussion)

### Entity pages REPLACE category summaries
- SummarizationJob stops creating category summaries
- Instead: if memory has `compiled_from` relation, it's safe to archive
- If not compiled yet, leave it alone
- CompilationJob is the primary knowledge preserver

### Orphan reduction: two-phase approach
- **Phase 1** (every 10 min): Lower similarity threshold to 0.65, `MIN_CLUSTER_SIZE` to 2
- **Phase 2** (daily): New OrphanAssignerJob — LLM assigns remaining orphans to entities
- Result: maybe ~100 truly unlinked memories (one-off things)

### No "protection" — compile first, then archive is natural
- Don't protect source memories — just ensure compilation happens before archival
- Once compiled into entity page, source memory can be freely archived
- Entity page IS the preservation

### Slim entity pages are OK
- A 2-memory entity page is like a Wikipedia stub — name, one paragraph, that's it
- Better to have 200 slim entity pages than 1,600 orphan memories

### New maintenance flow
1. RelationBuilderJob (10min) — link new memories
2. CompilationJob (10min) — create/update entity pages
3. OrphanAssignerJob (daily) — LLM assigns orphans to entities
4. SummarizationJob (weekly, simplified) — archives compiled memories 14+ days old
5. DistillationJob (nightly) — compress verbose memories (unchanged)
6. ConsolidationJob (nightly) — merge exact duplicates (unchanged)
