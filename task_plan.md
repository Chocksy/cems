# Phase 4+5: Smart Dedup + Conflict Detection — Implementation Plan

**Status:** IN PROGRESS
**Phases:** 4 (LLM Smart Dedup) + 5 (Conflict Detection) — combined into one
**Pattern:** TDD — tests first, then implementation

---

## Step 1: Add `classify_memory_pair()` LLM function

**File:** `src/cems/llm/dedup.py` (new)

New function that classifies two memories as `duplicate | related | conflicting | distinct`:
- Uses Gemini 2.5 Flash (`google/gemini-2.5-flash`)
- `fast_route=False` (not on Cerebras/Groq)
- Returns `{"classification": str, "explanation": str, "confidence": float}`
- Structured JSON output with fallback parsing
- Temperature 0.0 for deterministic classification

**Also update:** `src/cems/llm/__init__.py` — export `classify_memory_pair`

**Tests:** Unit test with mocked LLM client verifying JSON parsing, fallback on bad response, confidence field.

---

## Step 2: Create `memory_conflicts` table migration

**File:** `scripts/migrate_conflicts.sql` (new)

```sql
CREATE TABLE IF NOT EXISTS memory_conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    doc_a_id UUID NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,
    doc_b_id UUID NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,
    explanation TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(doc_a_id, doc_b_id)
);
CREATE INDEX IF NOT EXISTS idx_conflicts_user_status ON memory_conflicts(user_id, status);
```

**Also update:** `scripts/init.sql` — add table creation so fresh deploys get it

---

## Step 3: Add conflict CRUD to DocumentStore

**File:** `src/cems/db/document_store.py`

Add 4 methods:
- `add_conflict(user_id, doc_a_id, doc_b_id, explanation) -> str` — INSERT ... ON CONFLICT DO NOTHING
- `get_open_conflicts(user_id, limit=5) -> list[dict]` — SELECT with status='open', join memory_documents for content
- `resolve_conflict(conflict_id, resolution='resolved') -> bool` — UPDATE status + resolved_at
- `dismiss_conflict(conflict_id) -> bool` — shorthand for resolve with status='dismissed'

**Tests:** Unit tests with mocked asyncpg pool.

---

## Step 4: Rewrite ConsolidationJob with three-tier logic

**File:** `src/cems/maintenance/consolidation.py`

New three-tier `_merge_duplicates()`:
- **Tier 1 (0.98+):** Auto-merge — near-identical, skip LLM classification
- **Tier 2 (0.80-0.98):** LLM classification via `classify_memory_pair()`
  - `duplicate` + confidence >= 0.7 → merge using `merge_memory_contents()`
  - `conflicting` → store conflict via `doc_store.add_conflict()`
  - `related` / `distinct` → skip
- **Tier 3 (<0.80):** Skip entirely

**Metadata guards** (before LLM call):
- Different `category` → skip (different kinds of knowledge)
- Different `source_ref` → skip (different projects)

**New config fields** in `config.py`:
- `dedup_automerge_threshold: float = 0.98` — auto-merge threshold
- `dedup_llm_threshold: float = 0.80` — lower bound for LLM classification

**Return shape update:**
```python
{
    "duplicates_merged": int,
    "conflicts_found": int,
    "llm_classifications": int,
    "memories_checked": int,
}
```

**Tests:** Mock LLM to return each classification type, verify merge/conflict/skip behavior.

---

## Step 5: Surface conflicts in profile endpoint

**File:** `src/cems/api/handlers/memory.py`

In `api_memory_profile()`, after existing sections, add:
```python
# 6. Memory conflicts (unresolved)
conflicts = await doc_store.get_open_conflicts(user_id, limit=3)
if conflicts:
    conflict_lines = []
    for c in conflicts:
        conflict_lines.append(f"- **Conflict:** {c['explanation']}")
    context_parts.append(
        "## Memory Conflicts Detected\n"
        + "\n".join(conflict_lines)
        + "\nUse `cems memory resolve <id>` to resolve."
    )
```

**Tests:** Integration test — add conflicting memories, run consolidation, verify profile shows conflicts.

---

## Step 6: Add conflict resolution API endpoint

**File:** `src/cems/api/handlers/memory.py`

New endpoint `POST /api/memory/conflict/resolve`:
```python
{"conflict_id": "uuid", "resolution": "keep_a|keep_b|merge|dismiss"}
```

Actions:
- `keep_a` → soft-delete doc_b, resolve conflict
- `keep_b` → soft-delete doc_a, resolve conflict
- `merge` → LLM merge both, update doc_a, soft-delete doc_b, resolve conflict
- `dismiss` → mark conflict as dismissed (both docs stay)

**Also:** Register route in `src/cems/api/routes.py`

**Tests:** Unit test for each resolution action.

---

## Step 7: Update scheduler and maintenance API

**File:** `src/cems/scheduler.py`
- No new scheduled job needed — consolidation already runs nightly
- Update `valid_jobs` set if needed

**File:** `src/cems/api/handlers/memory.py`
- The maintenance API dispatch already calls `ConsolidationJob(memory).run_async` — returns updated shape
- Update the "all" branch result to show new fields

---

## Step 8: Run full test suite, rebuild Docker, integration test

- Run unit tests: `.venv/bin/python3 -m pytest tests/ -x -q`
- Docker rebuild: `docker compose build cems-server && docker compose up -d`
- Integration test via API:
  1. Add two similar but conflicting memories
  2. Run consolidation: `POST /api/memory/maintenance {"job_type": "consolidation"}`
  3. Verify `conflicts_found > 0` in response
  4. Check profile: `GET /api/memory/profile` shows conflict section
  5. Resolve conflict: `POST /api/memory/conflict/resolve`
  6. Verify profile no longer shows resolved conflict

---

## Step 9: Codex-investigator validation

Final validation pass:
- Three-tier thresholds work correctly
- Metadata guards prevent cross-category/cross-project merges
- Confidence threshold prevents hallucinated merges
- Conflicts persisted and surfaced in profile
- Resolution endpoint works for all 4 actions
- No regressions in existing maintenance tests
- Full pipeline tested end-to-end

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/cems/llm/dedup.py` | **NEW** — `classify_memory_pair()` |
| `src/cems/llm/__init__.py` | **UPDATE** — export new function |
| `scripts/migrate_conflicts.sql` | **NEW** — conflicts table |
| `scripts/init.sql` | **UPDATE** — add conflicts table |
| `src/cems/db/document_store.py` | **UPDATE** — conflict CRUD methods |
| `src/cems/maintenance/consolidation.py` | **REWRITE** — three-tier logic |
| `src/cems/config.py` | **UPDATE** — new threshold fields |
| `src/cems/api/handlers/memory.py` | **UPDATE** — profile + resolution endpoint |
| `src/cems/api/routes.py` | **UPDATE** — new route |
| `tests/test_maintenance.py` | **UPDATE** — tests for three-tier dedup + conflicts |
| `tests/test_dedup.py` | **NEW** — tests for `classify_memory_pair()` |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |
