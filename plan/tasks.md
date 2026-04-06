# CEMS Improvement Plan

## Context

Research analysis revealed 9 issues across the CEMS hook pipeline and maintenance system. The most critical discovery: **3 of 4 maintenance jobs were broken** — they read from the orphaned `memory_metadata` table while all data lives in `memory_documents`. Additionally, the hook pipeline adds ~2.5s latency per prompt with a redundant HTTP call, search results are dominated by massive session summaries, and there's no observability into what Claude actually receives.

**User's direction:** No category-based score demotion. Instead:
- Extract the most relevant **chunk** from large documents (forward-looking for when repos/docs will be stored)
- Fix and improve **maintenance jobs** with LLM-powered semantic dedup
- Add **conflict detection** between memories, surfaced via daily report
- Fix the **broken maintenance pipeline** first (prerequisite for everything else)

**Related research:** `docs/query-decomposition-research.md` — query decomposition for complex multi-part questions

---

## Phase 1: Fix Broken Maintenance Jobs `[COMPLETED 2026-02-24]`

All 3 broken maintenance jobs (ConsolidationJob, SummarizationJob, ReindexJob) were rewritten to async + DocumentStore pattern, following the working ObservationReflector. Dead code removed from AnalyticsMixin and PostgresMetadataStore. 526 tests passing. Docker verified.

**What was done:**
- Rewrote `consolidation.py`, `summarization.py`, `reindex.py` to async + DocumentStore
- Removed `_promote_hot_memories()`, standalone functions, dead AnalyticsMixin methods
- Updated scheduler to run all jobs async via `asyncio.new_event_loop()`
- Updated API handler for maintenance endpoint
- Stripped 10 dead methods from `PostgresMetadataStore`
- TDD: wrote tests first, then implementation
- Docker rebuild + all 5 maintenance endpoints verified (consolidation, summarization, reindex, reflect, all)

---

## Phase 2: Hook Pipeline Quality + Latency `[COMPLETED 2026-02-25]` (P0)

### 2a. Delete `fetch_recent_observations()` — redundant with SessionStart

**File:** `hooks/cems_user_prompts_submit.py`

- Delete the function (lines 163-215)
- Delete its call (lines 569-572)
- Also delete from: `src/cems/data/claude/hooks/cems_user_prompts_submit.py`

**Why safe:** SessionStart hook already injects "Recent Context (last 24h)" via `/api/memory/profile` (`src/cems/api/handlers/memory.py:668`). The UserPromptSubmit version is a second search call returning overlapping session summaries (~6000 chars of noise, ~1.25s latency per prompt).

### 2b. Penalize null-project items when project filter is active

**File:** `src/cems/retrieval.py` — `apply_score_adjustments()`

Add `else` branch after existing project conditions:
```python
if source_ref.startswith(f"project:{project}"):
    score *= boost          # Same project: 1.3x
elif source_ref.startswith("project:"):
    score *= penalty         # Different project: 0.8x
else:
    score *= 0.9             # No project tag: mild penalty (NEW)
```

Currently items with no `source_ref` escape both boost and penalty, letting unrelated session summaries dominate.

### 2c. Remove the 1.0 upper clamp

**File:** `src/cems/retrieval.py`

Change `score = max(0.0, min(1.0, score))` to `score = max(0.0, score)`

The 1.0 clamp defeats project boosts: a same-project item at `0.85 * 1.3 = 1.105` gets clamped to 1.0, same as an unrelated summary. Scores >1.0 are fine for ranking.

### 2d. Raise relevance threshold

**File:** `src/cems/config.py`

Change `relevance_threshold` from `0.3` to `0.4`.

Also add client-side filtering in `hooks/cems_user_prompts_submit.py` — in `search_cems()` after receiving results:
```python
results = [r for r in results if r.get("score", 0) >= 0.4]
```

### 2e. Client-side session deduplication

**File:** `hooks/cems_user_prompts_submit.py` — in `search_cems()` after receiving results

If multiple results share the same session tag prefix (e.g., `session:abc123`), keep only the highest-scoring one. Prevents a session from consuming 2-3 result slots with both its summary and observations.

---

## Phase 3: Chunk-Level Retrieval for Large Documents `[NOT NEEDED]` (P1)

> **Finding (2026-02-25):** `SearchResult.content` is already chunk-level content. The full chain (document_store → search.py → retrieval.py → API → hook) returns chunk text, not full document text. The plan was based on a false premise. See findings.md for proof chain.

### 3a. Return most relevant chunk instead of deduping to one-per-document

**File:** `src/cems/memory/search.py` — `_dedupe_by_document()`

Currently keeps only the single best-scoring chunk per document. For large docs (session summaries, future repo/doc ingestion), this discards relevant chunks.

**Change:** For documents with multiple chunks, return the **top-scoring chunk's content** (as now), but also concatenate adjacent high-scoring chunks from the same document when they'd provide coherent context:
- If a document has >1 chunk and the top chunk scores well, include up to 2 chunks (sorted by `seq` position) to provide surrounding context
- This directly addresses session-summary dominance: instead of returning the entire 4000-char summary, we return the ~800-char chunk that's actually relevant to the query

### 3b. Expose chunk-level content in API response

**File:** `src/cems/api/handlers/memory.py` — search endpoint response

Add `chunk_content` field alongside `content` in search results:
- `content`: full document content (as now)
- `chunk_content`: the specific chunk(s) that matched the query

The hook can then display `chunk_content` in `<memory-recall>` instead of the full document, dramatically reducing context token usage for large documents.

---

## Phase 4+5: LLM Smart Dedup + Conflict Detection `[COMPLETED 2026-02-25]` (P1)

Phases 4 and 5 were combined into a single implementation (per codex-investigator recommendation). Conflicts are detected as a byproduct of the consolidation scan, not a separate O(N²) job.

**What was done:**
- Created `src/cems/llm/dedup.py` — `classify_memory_pair()` LLM function (Gemini 2.5 Flash)
- Rewrote `src/cems/maintenance/consolidation.py` — three-tier dedup:
  - Tier 1 (>= 0.98): Auto-merge near-identical memories
  - Tier 2 (0.80-0.98): LLM classifies as duplicate/related/conflicting/distinct
  - Tier 3 (< 0.80): Skip
- Metadata guards: skip LLM when different category or source_ref
- MIN_CONFIDENCE = 0.7 double gate on LLM classification
- Created `memory_conflicts` table (`scripts/migrate_conflicts.sql` + `deploy/init.sql`)
- Added conflict CRUD to DocumentStore: `add_conflict()`, `get_conflict()`, `get_open_conflicts()`, `resolve_conflict()`
- Profile endpoint shows unresolved conflicts
- New `POST /api/memory/conflict/resolve` endpoint (keep_a/keep_b/merge/dismiss)
- Config fields: `dedup_automerge_threshold` (0.98), `dedup_llm_threshold` (0.80)
- Bug fixes from codex-investigator review:
  - Stale content in multi-merge chains (content variable updated after merge)
  - Merge-then-delete guards (don't delete when LLM merge returns empty)
  - Direct conflict lookup by ID (replaces list scan, no limit bug)
  - Accurate conflict count (checks add_conflict return value)
- 562 tests passing (13 dedup + 13 consolidation + 536 existing), 20/20 integration tests
- Docker verified with real conflicting memories (tabs vs spaces indentation test)

---

## Phase 6: Observability `[COMPLETED 2026-02-25]` (P1)

### 6a. Log hook OUTPUT (what Claude receives)

**File:** `hooks/utils/hook_logger.py`

Add an `output_text` parameter to `log_hook_event()`:
```python
def log_hook_event(event, session_id, extra=None, input_data=None, output_text=None):
```

In the verbose log section, if `output_text` is provided, write it:
```python
if output_text and short_sid:
    verbose_entry["output"] = output_text[:50000]  # cap at 50KB
```

**File:** `hooks/cems_user_prompts_submit.py`

Before `output_result()`, log the combined output:
```python
combined = '\n\n'.join(output_parts)
log_hook_event("HookOutput", session_id, {
    "output_len": len(combined),
    "has_recall": "<memory-recall>" in combined,
    "has_observations": "<recent-observations>" in combined,
}, output_text=combined)
output_result(combined, is_cursor)
```

### 6b. Write MemoryRetrieval to verbose log too

**File:** `hooks/cems_user_prompts_submit.py`

Add `input_data={"details": score_details}` to the MemoryRetrieval log call so it appears in verbose logs and the debug dashboard can show full details per session.

---

## Phase 7: Hygiene `[pending]` (P2)

### 7a. Log rotation

**File:** `hooks/utils/hook_logger.py`

Add a `_rotate_if_needed()` function called at the start of `log_hook_event()`:
- Lean log: rotate at 10MB -> rename to `.1`, delete `.2` if exists
- Verbose dir: delete files older than 7 days
- Prevents unbounded disk growth on developer machines

### 7b. Fix `memory_relations` FK

**File:** SQL migration

The `memory_relations` table FK still points to `memories.id` (legacy PgVectorStore table), not `memory_documents.id`. Since `memories` is fully orphaned, this FK is dead. Options:
- Drop and recreate pointing to `memory_documents.id`
- Or drop `memory_relations` entirely if the graph store is being rebuilt

---

## Bonus: Query Decomposition Research

**File:** `docs/query-decomposition-research.md`

Research on decomposing complex multi-part queries into sub-queries for better retrieval. Approaches analyzed:
- LLM-based decomposition (expensive but accurate)
- Rule-based decomposition (cheap, handles common patterns)
- Hybrid approach

**Status:** Research complete, not yet implemented. Would benefit from Phase 3 (chunk-level retrieval) being done first.

---

## Priority Summary

| Phase | Priority | Status | Dependencies |
|-------|----------|--------|--------------|
| 1. Fix Maintenance Jobs | P0 | **COMPLETED** | None |
| 2. Hook Pipeline Quality | P0 | **COMPLETED** | None |
| 3. Chunk-Level Retrieval | P1 | **NOT NEEDED** | None |
| 4+5. LLM Smart Dedup + Conflicts | P1 | **COMPLETED** | Phase 1 (done) |
| 6. Observability | P1 | **COMPLETED** | None |
| 7. Hygiene | P2 | **7a DONE** (rotation) | None |
