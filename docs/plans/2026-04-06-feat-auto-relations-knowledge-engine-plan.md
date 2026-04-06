---
title: "feat: Auto-Relations Layer for Knowledge Engine"
type: feat
date: 2026-04-06
brainstorm: docs/brainstorms/2026-04-06-knowledge-engine-brainstorm.md
---

# Auto-Relations Layer for CEMS Knowledge Engine

## Overview

Add automatic relation-building to CEMS: every time a new memory is stored via
`add_async()`, find and link related memories using the already-computed embeddings.
This populates the empty `memory_relations` table and forms the foundation for the
full Knowledge Engine (entity pages, lint, wiki dashboard).

## Problem Statement / Motivation

CEMS has a `memory_relations` table (schema in `init.sql:105-116`, runtime in
`database.py:242-255`) with read methods (`get_related_documents()` at
`document_store.py:1366`) — but **zero write methods**. The table is permanently empty.
This means:
- Memories are isolated islands — no navigation between related content
- The retrieval pipeline can't leverage graph traversal for better recall
- Future features (entity pages, contradictions, wiki) all depend on relations

## Proposed Solution

### Phase 1: DocumentStore.add_relations() — The Write Method

Add a new method to `DocumentStore` that inserts relation rows.

**File**: `src/cems/db/document_store.py`
**Location**: After `get_related_documents()` (line ~1435)

```python
async def add_relations(
    self,
    source_id: str,
    relations: list[dict],  # [{"target_id": str, "relation_type": str, "similarity": float}]
) -> int:
    """Insert relations between documents.
    
    Uses INSERT ... ON CONFLICT DO UPDATE to handle re-runs gracefully.
    PK is (source_id, target_id, relation_type) — natural dedup.
    
    Args:
        source_id: The source document ID
        relations: List of target docs with type and similarity
        
    Returns:
        Number of relations created
    """
    pool = await self._get_pool()
    source_uuid = UUID(source_id)
    created = 0
    
    async with pool.acquire() as conn:
        for rel in relations:
            target_uuid = UUID(rel["target_id"])
            if target_uuid == source_uuid:
                continue  # CHECK constraint prevents self-relation
            try:
                await conn.execute("""
                    INSERT INTO memory_relations (source_id, target_id, relation_type, similarity)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (source_id, target_id, relation_type)
                    DO UPDATE SET similarity = EXCLUDED.similarity
                """, source_uuid, target_uuid, rel["relation_type"], rel.get("similarity"))
                created += 1
            except Exception as e:
                logger.warning(f"Failed to add relation {source_id[:8]}→{rel['target_id'][:8]}: {e}")
    
    return created
```

**Key design decisions:**
- `ON CONFLICT DO UPDATE` — idempotent, safe for backfill re-runs
- PK is `(source_id, target_id, relation_type)` — defined in `init.sql:111`
- Individual inserts in loop (not batch) — safer for partial failures, relation count per add is small (5-10)
- Self-relation check via `if target_uuid == source_uuid` (belt) + DB constraint (suspenders)

### Phase 2: Auto-Link in add_async() — The Integration

Extend `add_async()` in `src/cems/memory/write.py` to auto-link after storing.

**Extension point**: After line 174 (`if is_new:`), before the return statement.

```python
# Step 4: Auto-link to related memories (fire-and-forget, don't block the add)
if is_new and embeddings:
    try:
        await self._auto_link_relations(doc_store, doc_id, embeddings[0], user_id, team_id, scope)
    except Exception as e:
        # Relations are best-effort — never fail the add
        logger.warning(f"[WRITE] Auto-link failed for {doc_id[:8]}: {e}")
```

**New method on WriteMixin** (or a new `RelationBuildMixin`):

```python
async def _auto_link_relations(
    self: "CEMSMemory",
    doc_store: "DocumentStore",
    doc_id: str,
    embedding: list[float],
    user_id: str,
    team_id: str | None,
    scope: str,
) -> int:
    """Find and link related memories for a newly added document.
    
    Reuses the already-computed first-chunk embedding.
    Only runs on new documents (not duplicates).
    """
    # Search for similar chunks (same user, all scopes)
    neighbors = await doc_store.search_chunks(
        query_embedding=embedding,
        user_id=user_id,
        limit=10,
    )
    
    # Build relations for neighbors above threshold
    SIMILARITY_THRESHOLD = 0.75
    relations = []
    seen_docs = {doc_id}  # Skip self
    
    for neighbor in neighbors:
        neighbor_doc_id = neighbor["document_id"]
        score = neighbor.get("score", 0)
        
        if neighbor_doc_id in seen_docs:
            continue
        if score < SIMILARITY_THRESHOLD:
            continue
            
        seen_docs.add(neighbor_doc_id)
        relations.append({
            "target_id": neighbor_doc_id,
            "relation_type": "similar",
            "similarity": score,
        })
    
    if relations:
        created = await doc_store.add_relations(doc_id, relations)
        # Also add reverse relations (bidirectional graph)
        reverse = [
            {"target_id": doc_id, "relation_type": r["relation_type"], "similarity": r["similarity"]}
            for r in relations
        ]
        for rel in reverse:
            await doc_store.add_relations(rel["target_id"], [{"target_id": doc_id, **{k: v for k, v in rel.items() if k != "target_id"}}])
        
        logger.info(f"[WRITE] Auto-linked {doc_id[:8]} to {created} neighbors")
        return created
    
    return 0
```

**Critical constraints:**
- Only runs when `is_new == True` — never on duplicates
- Uses first chunk's embedding — already computed, zero extra cost
- Wrapped in try/except — relations are best-effort, never fail the add
- Bidirectional: if A→B exists, also create B→A
- Filters to same user_id (ownership boundary)

### Phase 3: Backfill Job — Populate Relations for Existing Memories

New maintenance job to populate relations for all existing memories.

**File**: `src/cems/maintenance/relation_builder.py`

```python
class RelationBuilderJob:
    """Backfill job to populate memory_relations for existing documents.
    
    Follows the standard maintenance job pattern:
    Job(memory).run_async() → uses _ensure_document_store()
    
    Processes documents in batches, finding and linking neighbors.
    Safe to re-run (upsert semantics via ON CONFLICT).
    """
    
    async def run_async(
        self,
        limit: int = 100,
        offset: int = 0,
        force: bool = False,
    ) -> dict:
        """Run the backfill.
        
        Args:
            limit: Batch size (Coolify proxy timeout ~60s, keep batches small)
            offset: Starting offset for pagination
            force: If True, re-process documents that already have relations
        """
```

**Design:**
- Follows existing job pattern (`Job(memory).run_async()`)
- Batch processing with limit/offset (Coolify proxy timeout ~60s)
- Skip documents that already have relations (unless force=True)
- API endpoint: `POST /api/memory/maintenance {"job_type": "relations", "limit": 50, "offset": 0}`

### Phase 4: Heat Score Integration

Add heat-based decay floor to `apply_score_adjustments()` in `src/cems/retrieval.py`.

**Location**: After time decay calculation (line ~728), before project scoring (line ~746).

```python
# Heat-based decay floor (implements documented but missing adaptive ceiling)
shown = getattr(result, "shown_count", 0) or 0
if shown >= 20:
    # Hot: frequently surfaced, almost no decay
    time_decay = max(time_decay, 0.95)
elif shown >= 5:
    # Warm: moderately used, slow decay
    time_decay = max(time_decay, 0.80)
# Cool/Cold: no floor, current behavior
```

**Why shown_count and not a computed heat_score:**
- `shown_count` is already on every search result (CHUNK_WITH_DOC_COLUMNS line 46)
- No extra DB query needed
- Simple, debuggable, easy to tune thresholds
- Heat score formula from brainstorm can be added later if needed

### Phase 5: Maintenance Protection

Add `entity-page` to `PROTECTED_CATEGORIES` in `src/cems/maintenance/__init__.py`.

```python
PROTECTED_CATEGORIES = {
    "gate-rules", "guidelines", "preferences", 
    "category-summary", "session-summary",
    "entity-page",  # NEW: Knowledge Engine entity pages
}
```

Add heat-based protection to `SummarizationJob._compress_by_category()`:

```python
# Before compressing, skip hot memories
hot_threshold = 20  # shown_count
cold_docs = [d for d in docs if (d.get("shown_count", 0) or 0) < hot_threshold]
if not cold_docs or len(cold_docs) < 3:
    continue  # Not enough cold docs to summarize
```

## Technical Approach

### Architecture

```
add_async() pipeline (write.py)
    │
    ├─ [1] chunk_document(content)      ← existing
    ├─ [2] embed_batch(chunk_texts)     ← existing  
    ├─ [3] add_document(...)            ← existing
    │       returns (doc_id, is_new)
    │
    └─ [4] _auto_link_relations(...)    ← NEW (only if is_new)
            ├─ search_chunks(embedding) ← reuses embedding from [2]
            ├─ filter by threshold
            └─ add_relations(...)       ← NEW method in DocumentStore
```

### Database Schema

No migrations needed — `memory_relations` already exists with the right schema:

```sql
-- From init.sql:105-116
CREATE TABLE IF NOT EXISTS memory_relations (
    source_id UUID REFERENCES memory_documents(id) ON DELETE CASCADE,
    target_id UUID REFERENCES memory_documents(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) DEFAULT 'similar',
    similarity REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (source_id, target_id, relation_type)
);
```

Also note: `memory_conflicts` table already exists (`init.sql:324-337`) for future
contradiction detection — has `doc_a_id`, `doc_b_id`, `explanation`, `status`.

### Schema Discrepancy

Two schemas define `memory_relations`:

| Column | `database.py:242` | `init.sql:105` |
|--------|-------------------|----------------|
| PK | Single `id UUID` | Composite `(source_id, target_id, relation_type)` |
| `weight` | Yes (FLOAT) | No |
| `metadata` | Yes (JSONB) | No |
| `similarity` | Added via migration | Yes (REAL) |

**Decision:** Use the `init.sql` schema as truth — it's what Docker deployments use.
The `database.py` SQLAlchemy schema is legacy (used for local dev only).
Code should not reference `weight`, `metadata`, or `id` columns — use the composite PK.

### Implementation Phases

#### Phase 1: DocumentStore.add_relations() [Foundation]

**Files to modify:**
- `src/cems/db/document_store.py` — add `add_relations()` method

**Files to read first:**
- `src/cems/db/document_store.py:1366-1435` — `get_related_documents()` for pattern
- `scripts/migrate_relations_fk.sql` — FK contract reference

**Tests:**
- `tests/test_document_store.py` — add test for `add_relations()` (upsert, self-relation skip, FK cascade)

#### Phase 2: Auto-Link in add_async() [Core Feature]

**Files to modify:**
- `src/cems/memory/write.py` — extend `add_async()` with auto-link step

**Files to read first:**
- `src/cems/memory/write.py:86-198` — full add_async pipeline
- `src/cems/db/document_store.py:1189-1230` — `search_chunks()` signature
- `src/cems/memory/relations.py` — existing RelationsMixin pattern

**Tests:**
- `tests/test_write.py` or `tests/test_auto_relations.py` — test that adding a memory creates relations
- `test_integration.py` — end-to-end test against Docker

#### Phase 3: Backfill Job [Existing Data]

**Files to create:**
- `src/cems/maintenance/relation_builder.py` — new job class

**Files to modify:**
- `src/cems/maintenance/__init__.py` — register new job
- `src/cems/api/handlers/maintenance.py` — expose via API

**Files to read first:**
- `src/cems/maintenance/consolidation.py` — job pattern reference
- `src/cems/api/handlers/maintenance.py` — API handler pattern

**Tests:**
- `tests/test_relation_builder.py` — backfill creates expected relations

#### Phase 4: Heat Score [Protection]

**Files to modify:**
- `src/cems/retrieval.py:693-744` — add decay floor in `apply_score_adjustments()`
- `src/cems/maintenance/__init__.py` — add `entity-page` to PROTECTED_CATEGORIES
- `src/cems/maintenance/summarization.py:89-140` — skip hot memories in compression

**Files to read first:**
- `src/cems/retrieval.py:660-770` — full scoring function
- `src/cems/maintenance/summarization.py:39-88` — current summarization flow

**Tests:**
- `tests/test_retrieval.py` — verify hot memories get decay floor
- `tests/test_summarization.py` — verify hot memories skip compression

## Acceptance Criteria

### Functional Requirements

- [x] `DocumentStore.add_relations()` inserts rows into `memory_relations` with upsert semantics
- [x] `add_async()` auto-links new memories to existing neighbors (threshold 0.75)
- [x] Relations are bidirectional (A→B and B→A)
- [x] Duplicate memories (`is_new == False`) do NOT trigger relation building
- [x] Relations are best-effort — failures never block the add operation
- [x] `get_related_documents()` returns auto-linked neighbors correctly
- [x] Backfill job processes existing memories in batches
- [x] Backfill is idempotent (safe to re-run)
- [x] Heat-based decay floor: shown_count ≥ 20 → min 0.95 decay, ≥ 5 → min 0.80
- [x] Hot memories (shown_count ≥ 20) skip summarization compression
- [x] `entity-page` category is protected from maintenance

### Non-Functional Requirements

- [ ] Auto-link adds < 50ms latency to `add_async()` (one pgvector search + ~5 INSERTs)
- [ ] Backfill processes 50 docs per batch without Coolify timeout
- [ ] No extra embedding API calls (reuses existing embeddings)
- [ ] No schema migrations needed (table already exists)

### Quality Gates

- [ ] All existing tests pass (577 tests)
- [ ] New unit tests for add_relations, auto-link, backfill, heat score
- [ ] Integration test against Docker validates end-to-end relation building
- [ ] Backfill tested on prod data dump in local Docker

## Success Metrics

- `memory_relations` table goes from 0 rows to proportional to memory count
- Average relations per memory: 3-8 (tunable via threshold)
- `add_async()` latency increase: < 50ms
- Backfill completes for ~2,580 existing memories in < 5 minutes

## Dependencies & Prerequisites

- Docker environment running with `init.sql` schema (memory_relations table)
- No schema migrations needed
- No new Python dependencies
- pgvector HNSW index on `memory_chunks.embedding` (already exists)

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Schema mismatch (database.py vs init.sql) | Relations fail to insert | Use init.sql composite PK, test against Docker |
| Threshold too low → noise | Too many relations, slow queries | Start at 0.75, tune with prod data |
| Threshold too high → sparse | Few relations, weak graph | Monitor average relations/doc, lower if < 2 |
| Auto-link slows add_async | User-facing latency | try/except + fire-and-forget + monitor timing |
| Backfill timeout on large batch | Job fails | Limit=50, offset pagination, no Coolify timeout |
| Bidirectional doubles storage | 2x relation rows | Acceptable at current scale (<50K relations) |

## Future Considerations

This is Layer 1 of the Knowledge Engine. Subsequent layers build on it:

- **Layer 2: Lint & Contradictions** — scan relations for contradictions, use existing `memory_conflicts` table
- **Layer 3: Entity Pages** — cluster related memories via graph traversal, compile into wiki pages
- **Layer 4: Wiki Dashboard** — browse entity pages, visualize graph, resolve lint issues

## Testing Plan

1. **Unit tests**: `DocumentStore.add_relations()`, auto-link logic, heat score, backfill
2. **Integration test**: add memory via API → verify relations created in DB
3. **Prod data validation**: dump prod → local Docker → run backfill → inspect relation quality
4. **Scale test**: synthetically expand to 50K memories → measure search/add latency

## References & Research

### Internal References

- Schema: `scripts/init.sql:105-116` (memory_relations), `scripts/init.sql:324-337` (memory_conflicts)
- Runtime schema: `src/cems/db/database.py:242-255` (SQLAlchemy version — **legacy, use init.sql**)
- Read path: `src/cems/db/document_store.py:1366-1435` (`get_related_documents()`)
- Write pipeline: `src/cems/memory/write.py:86-198` (`add_async()`)
- Search: `src/cems/db/document_store.py:1189-1230` (`search_chunks()`)
- Scoring: `src/cems/retrieval.py:693-765` (`apply_score_adjustments()`)
- Maintenance pattern: `src/cems/maintenance/consolidation.py:30-69`
- Protected categories: `src/cems/maintenance/__init__.py:4`
- FK migration: `scripts/migrate_relations_fk.sql`
- Brainstorm: `docs/brainstorms/2026-04-06-knowledge-engine-brainstorm.md`

### Key Learnings from Research

- Never use `update_document()` in maintenance — bumps `updated_at`, breaks age-based pruning
- Only create relations when `is_new == True` (not on content-hash duplicates)
- `PROTECTED_CATEGORIES` in `maintenance/__init__.py` exempts categories from all pruning
- Scoring: `apply_score_adjustments()` is the single source of truth — no scoring logic elsewhere
- Adaptive ceiling documented at `retrieval.py:706` but NOT implemented — we implement it here
- `SummarizationJob` orders `ASC` for old docs (fixed in recent audit, was DESC)
