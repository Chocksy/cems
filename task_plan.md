# Task Plan: Memory Quality Overhaul + Always-On Context Search

## Goal
Fix two fundamental problems with CEMS:
1. **Bad data**: ~85% of 200 production memories have NULL source_ref, many are duplicates or noise from legacy migration
2. **Limited context**: The UserPromptSubmit hook should ALWAYS use conversation context to improve search — not just for confirmatory prompts

## Current Problems

### Problem 1: Production Memory Quality
- **200 total memories** in production DB
- **~170 (85%) have NULL source_ref** — no project association, can't scope results
- **Duplicate content**: Same learnings stored multiple times (e.g., "output files stored in /private/tmp/..." appears 3+ times)
- **Noisy categories**: "unknown", "general", overlapping categories like "hooks" vs "hooks-logging" vs "hook logging"
- **Low relevance threshold**: 0.005 — basically returns everything, no quality filter
- **Root cause**: Older memories predate source_ref tracking (added later). Tool learning and session analysis now pass source_ref, but legacy data doesn't have it.

### Problem 2: Search Only Uses User Prompt Text
- The hook searches CEMS using `extract_intent(prompt)` — raw user text
- User text like "yes" or "refactor auth" has NO context about what Claude was doing
- The ASSISTANT's last message (the proposal) is the real context, but it's only used for confirmatory prompts
- Result: Even non-confirmatory prompts get poor search results because the user's text alone isn't enough

## Approach

### Phase 1: Always-On Context-Enriched Search (Code Change)
**Expand transcript context usage to ALL prompts**, not just confirmatory ones.

For every prompt, optionally enrich the search query with keywords from the last assistant message. This gives CEMS "what is Claude doing?" context alongside "what did the user ask?".

**File**: `hooks/user_prompts_submit.py`
**Change**: In the main search path (line ~526), after extracting intent from user prompt, also read the last assistant message and blend keywords from it into the search query.

### Phase 2: Production Data Cleanup (DB Operations)
Three options (present for user decision):

**Option A: Surgical Cleanup** (Conservative)
- Export all production memories via API
- Analyze locally: deduplicate, tag with project IDs where inferrable from session context
- Delete confirmed noise/duplicates via soft-delete
- Backfill source_ref where session_id can map to a project
- Pro: No data loss. Con: Labor-intensive, may not fix category mess

**Option B: Fresh Start** (Aggressive)
- Export production memories as backup
- Wipe `memory_documents` table
- Re-ingest from session transcripts (stored in CEMS via /api/session/analyze)
- All new memories get proper source_ref from day one
- Pro: Clean slate. Con: Loses any manually-added memories, gate rules

**Option C: Hybrid** (Recommended)
- Export and backup everything
- Keep gate rules and manually-added memories (category = "gate-rules" or no session tag)
- Wipe auto-generated learnings (those with "session-learning" or "tool-learning" tags)
- Raise relevance threshold from 0.005 to something meaningful (0.3-0.4)
- Improve deduplication: tighten content_hash + add semantic dedup in consolidation job
- Pro: Keeps important manual memories, eliminates noise. Con: Moderate complexity

### Phase 3: Ingestion Quality Gates (Going Forward)

Prevent the data quality issues from recurring by tightening the ingestion pipeline.

#### 3a. Raise Confidence Threshold (learning_extraction.py)
**File**: `src/cems/llm/learning_extraction.py`

**Current**: Line 234 — `confidence >= 0.3` for session learnings, line 513 — `confidence < 0.5` for tool learnings.

**Change**:
- Session learnings: `0.3` → `0.6` (line 234 and line 299 for chunk path)
- Tool learning already at `0.5` — raise to `0.6` for consistency (line 513)

#### 3b. Minimum Content Length (learning_extraction.py)
**File**: `src/cems/llm/learning_extraction.py`

**Current**: `_parse_learnings_response()` at line 404 caps content at 500 chars but has no minimum.

**Change**: In `_parse_learnings_response()`, after validation, skip learnings where `len(content) < 80`.

#### 3c. Noise Content Filtering (learning_extraction.py)
**File**: `src/cems/llm/learning_extraction.py`

**Current**: No content filtering — `/private/tmp/claude` paths, "background command", exit codes all stored.

**Change**: Add `_is_noise()` check in `_parse_learnings_response()`:
- Skip if content contains `/private/tmp/claude`
- Skip if content matches `^(background command|exit code \d)`
- Skip if content length < 80 chars (from 3b)

#### 3d. Controlled Category Vocabulary (learning_extraction.py)
**File**: `src/cems/llm/learning_extraction.py`

**Current**: Category is freeform LLM text — produces 500 unique categories, case variants, singletons.

**Change**:
1. Define `CANONICAL_CATEGORIES` list (the 33 categories from Phase 2 cleanup)
2. In the LLM system prompt, list these as the ONLY valid categories
3. In `_parse_learnings_response()`, normalize the category:
   - lowercase + strip
   - if not in `CANONICAL_CATEGORIES`, map to closest match or "general"

#### 3e. Raise Relevance Threshold (config.py)
**File**: `src/cems/config.py`

**Current**: `relevance_threshold: float = 0.005` (line 194-196)

**Change**: `0.005` → `0.3` — this is the scored threshold after all adjustments. With 584 quality memories instead of 2,499, we can afford to be selective.

**Risk**: May reduce recall for some queries. Mitigated by Phase 1's always-on context enrichment.

#### 3f. Semantic Dedup on Ingestion (document_store.py)
**File**: `src/cems/db/document_store.py`

**Current**: Dedup is content-hash only (exact match). Near-duplicates with `(session: XXXX)` suffix get through.

**Change**: In `add_document()`, after the content-hash check, do a quick vector search for the new chunk's embedding against existing chunks. If cosine similarity > 0.92 (from `config.duplicate_similarity_threshold`), skip insertion and return the existing document ID.

**Timing budget**: One extra vector query (~5ms) per document add — negligible.

---

**Implementation order**: 3a → 3b → 3c → 3d → 3e → 3f (each builds on the previous)

---

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Always-on context search | `complete` | Enriches all prompts with assistant context (5af1f05) |
| Phase 2: Production data cleanup | `complete` | Local DB cleaned, ready to push to prod |
| Phase 3: Ingestion quality gates | `complete` | All 6 sub-tasks implemented, 251 tests pass |

### Phase 2 Results
- **Before**: 2,499 memory_documents, 500 categories, 92.9% missing source_ref
- **After**: 584 active memories, 33 canonical categories, 33.7% have source_ref
- **Soft-deleted**: 1,915 memories (noise, legacy patterns, never-shown/no-ref)
- **Category normalization**: 500 → 33 canonical categories
- **source_ref normalization**: `project:pxls` → `project:EpicCoders/pxls`, etc.
- **Cleanup is local** — needs pg_dump + restore to production

## Research Findings

### Production Data Analysis (2026-02-10) — CORRECTED by codex-investigator
- **Total memories: 2,453** (200 was just the search candidate limit!)
- With source_ref: ~175 (7.1%) — only the newest ~300 entries
- Without source_ref: ~2,278 (92.9%) — everything before hooks added source_ref
- **477 unique categories** — 241 are singletons (50.5%), 36 duplicate groups
- "patterns" category: 216 legacy entries from old Mem0 migration
- Relevance threshold: 0.005 (effectively no filtering)
- 68.5% of recent memories have shown_count=0 (never surfaced)
- Ingestion confidence threshold: 0.3 (too low — lets noise through)
- Gate rules: ~5-6 memories with category "gate-rules"

### How source_ref flows
- `hooks/stop.py` → sends `source_ref: project:org/repo` to `/api/session/analyze` ✅
- `hooks/cems_post_tool_use.py` → sends `source_ref` to `/api/tool/learning` ✅
- `hooks/pre_compact.py` → sends `source_ref` to `/api/session/analyze` ✅
- Server stores `source_ref` in `memory_documents` table ✅
- **Gap**: Older memories were created BEFORE these hooks passed source_ref → NULL forever

### Why search returns irrelevant results
1. No source_ref → can't boost current-project memories
2. Low threshold (0.005) → everything passes
3. User prompt alone is thin context → "yes" or "refactor" matches too broadly
4. No "no results" path → always returns top-10 regardless of actual relevance
