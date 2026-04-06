# LongMemEval Investigation Report

## Executive Summary

Our LongMemEval implementation has **three critical flaws** that make the reported 98% Recall@5 misleading:

1. **Uses the Oracle variant** (1.9 sessions/question) — retrieval is trivially easy with no distractors
2. **Bypasses the observer pipeline** — stores raw transcripts instead of LLM-compressed summaries
3. **Doesn't test the real system** — hooks, consolidation, category normalization, MCP all skipped

Mastra achieves 84.23% macro accuracy by running their full Observer+Reflector pipeline on the **S variant** (50 sessions/question). Our 98% Recall@5 measures something entirely different: raw text retrieval on pre-filtered data.

---

## 1. Original LongMemEval Benchmark

**Paper**: [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) (ICLR 2025)
**Repo**: [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)

### What it tests

Five long-term memory abilities:
1. **Information Extraction** — recall specific details from distant sessions
2. **Multi-Session Reasoning** — synthesize facts scattered across ≤6 sessions
3. **Temporal Reasoning** — understand chronological order and timestamps
4. **Knowledge Updates** — track changed/updated information
5. **Abstention** — correctly refuse when info was never discussed

### Dataset variants

| Variant | Sessions/question | Unique sessions | Purpose |
|---------|------------------|----------------|---------|
| **Oracle** | 1-6 (avg 1.9) | 940 | Sanity check / ceiling |
| **S** | ~50 (avg 53) | 19,195 | Real retrieval test |
| **M** | ~500 | Enormous | Extreme stress test |

### Metrics

- **Retrieval**: Recall@{5,10,50}, NDCG@{5,10,50} at session or turn level
- **QA**: Macro accuracy (average of per-type accuracies), GPT-4o judge with official prompts
- **Headline metric**: Task-averaged (macro) accuracy on S variant

### How Mastra runs it

Mastra's [Observational Memory](https://mastra.ai/research/observational-memory) achieves:
- 84.23% (GPT-4o reader) / 94.87% (GPT-5-mini reader) on S variant
- Sessions ingested **chronologically** through Observer+Reflector pipeline
- Observer compresses sessions into observations during ingestion
- Full pipeline is tested: ingestion → compression → retrieval → reading

---

## 2. Our Implementation

### 2a. Retrieval-Only Eval (`longmemeval.py`)

**File**: `src/cems/eval/longmemeval.py`

**Flow** (lines 491-639):
1. Parse CLI args (line 663): `--questions N`, `--api-url`, `--api-key`, etc.
2. Health check → clean stale data → download dataset
3. Load questions from **oracle** variant, filter out abstention questions (line 429)
4. **Bulk ingest**: `collect_all_sessions()` (line 461) extracts unique sessions, formats as flat text (`"role: content\nrole: content"`), sends to `/api/memory/add_batch`
5. **Search & score**: For each question, calls `/api/memory/search` with limit=10, extracts session IDs from `source_ref`, computes Recall@5
6. Cleanup → save results

**Ingestion details** (lines 117-125):
- `category`: `"eval-session"` (hardcoded, bypasses normalization)
- `source_ref`: `"project:longmemeval:{session_id}"`
- `tags`: `["longmemeval", session_id]`
- `infer`: `False` (no LLM extraction)

### 2b. End-to-End Eval (`longmemeval_e2e.py`)

**File**: `src/cems/eval/longmemeval_e2e.py`

Same ingestion, plus:
1. After search, feeds context to GPT-4o reader (line 286)
2. GPT-4o judge evaluates answer correctness (lines 55-113)
3. Reports macro accuracy per type

**Supports both oracle and S variants** via `--dataset` flag (line 705).

Uses **custom judge prompts** (lines 55-113) with `max_tokens=256`, not the official LongMemEval prompts (`max_tokens=10`, "yes or no only").

---

## 3. Critical Issues

### Issue 1: Oracle Dataset = Trivially Easy (SEVERITY: HIGH)

The retrieval-only eval uses **only** the Oracle variant.

**Why this matters**: With avg 1.9 sessions per question (all containing the answer), even naive retrieval gets near-perfect recall. The 98% Recall@5 is meaningless as a benchmark. The S variant with ~50 sessions per question (mostly distractors) is the actual test.

**Location**: `longmemeval.py:38` (`LONGMEMEVAL_URL` points to oracle JSON)

### Issue 2: Raw Transcripts Instead of Observer Pipeline (SEVERITY: HIGH)

The eval stores verbatim conversation transcripts via `/api/memory/add_batch`. In production, CEMS processes sessions through:

```
Session transcript → Observer daemon → LLM summarization → Observation extraction → Store
```

**What the eval skips**:
- `extract_session_summary()` (Gemini 2.5 Flash → 200-400 word narrative)
- `observation_extraction.py` (atomic fact extraction)
- Epoch model (incremental observation with `session:{id[:12]}:e{N}` tags)
- Category normalization (all gets "eval-session")

**Impact**: Raw transcripts contain enormous noise (generic advice, filler content). The chunker splits long sessions into many chunks, and vector search may return irrelevant chunks. In production, the observer would compress these into tight, searchable summaries.

**Comparison to Mastra**: Mastra explicitly runs their full pipeline during ingestion. Their 84.23% reflects compression quality + retrieval quality. Our 98% reflects raw text retrieval on pre-filtered data. These numbers are not comparable.

### Issue 3: Custom Judge Prompts (SEVERITY: MEDIUM)

The e2e eval uses custom prompts (lines 55-113) with `max_tokens=256` and "explain briefly" instructions. The original benchmark uses standardized prompts with `max_tokens=10` and strict "yes or no only". This makes results non-comparable to published numbers.

### Issue 4: Project Boost Inflates Scores (SEVERITY: MEDIUM)

All eval memories get `project="longmemeval"` and all searches pass `project="longmemeval"`, giving a uniform 1.3x boost to ALL results. In real usage, only current-project memories get boosted. Since all results are boosted equally, this doesn't change ranking, but it does inflate raw scores past the relevance threshold.

### Issue 5: No Recall@10, No NDCG (SEVERITY: LOW)

Only Recall@5 is reported. The original benchmark reports Recall@{5,10,50} and NDCG@{5,10,50}. Position-sensitive NDCG reveals whether the right session is ranked high vs barely in top-K.

---

## 4. What the Eval Does NOT Test

| CEMS Component | Tested? | Impact |
|---------------|---------|--------|
| Memory add/search API | Yes | Core flow works |
| Observer daemon | **No** | Most critical gap |
| Session summarization | **No** | LLM compression quality untested |
| Observation extraction | **No** | Atomic fact extraction untested |
| Epoch model | **No** | Incremental observation untested |
| Hooks (all 6) | **No** | Real-world trigger pipeline untested |
| Tool learning | **No** | `extract_tool_learning()` skipped |
| Category normalization | **No** | Hardcoded "eval-session" |
| Gate rules | **No** | Not exercised |
| Profile generation | **No** | Profile API not called |
| MCP wrapper | **No** | Port 8766 not exercised |
| Consolidation job | **No** | Dedup/merge untested |
| Summarization job | **No** | Summary generation untested |
| Reindex job | **No** | Reindexing untested |
| ObservationReflector | **No** | Reflection untested |
| Soft-delete/feedback | **No** | shown_count, deleted_at unused |
| Hybrid search | **No** | mode="auto" defaults to vector |
| Multi-user isolation | **No** | Single user context |
| Knowledge update API | **No** | No `update_document` calls |

---

## 5. Observer System Overview

The observer daemon (`src/cems/observer/daemon.py`) is architecturally similar to Mastra's Observer:

1. **Adapters** discover sessions (Claude, Codex, Cursor, Goose) by scanning tool-specific directories
2. **Growth detection**: When a session file grows by 10KB+ raw / 3K+ extracted chars
3. **Summary extraction**: Calls `/api/session/summarize` with `mode="incremental"` → Gemini 2.5 Flash produces 200-400 word narrative
4. **Atomic upsert**: `upsert_document_by_tag()` with `session:{id[:12]}` tag, uses `SELECT ... FOR UPDATE`
5. **Signal IPC**: Hooks write signal files (`compact`/`stop`) to `~/.cems/observer/signals/`
6. **Epoch model**: Each epoch gets its own doc tagged `session:{id[:12]}:e{N}`
7. **Staleness**: 5min idle → auto-finalize

---

## 6. Proposed Solutions

### Option A: Quick Fix — Switch to S Variant + Observer Ingestion

**Effort**: Moderate. Changes to `longmemeval.py` and `longmemeval_e2e.py`.

1. **Switch retrieval eval to S variant** — change `LONGMEMEVAL_URL` at line 38
2. **Replace `add_batch` with `/api/session/summarize`** — instead of raw transcript insert:
   ```python
   for session_id, session_content in sessions.items():
       client.post("/api/session/summarize", json={
           "session_id": session_id,
           "content": format_as_transcript(session_content),
           "mode": "finalize",
           "project": "longmemeval"
       })
   ```
3. **Use official judge prompts** — replace custom prompts with original LongMemEval prompts
4. **Add NDCG and Recall@10** — expand metrics

**What this tests**: Session summarization quality, embedding of compressed content, retrieval from summaries. Doesn't test the full observer daemon loop but tests the LLM extraction + storage + retrieval.

### Option B: Full Pipeline — LongMemEval Observer Adapter

**Effort**: Significant. New adapter + eval orchestration.

1. **Create `LongMemEvalAdapter`** implementing `SessionAdapter`:
   - `discover_sessions()` → returns list of simulated sessions from LongMemEval data
   - `extract_text()` → formats session turns as Claude Code transcript
   - `enrich_metadata()` → sets project context and timestamps

2. **Simulate chronological ingestion**:
   - Write simulated session JSONL files to temp directory
   - Run observer `run_cycle()` to trigger growth detection
   - Sessions arrive in timestamp order, with epoch boundaries

3. **Run eval after observer processing**:
   - Wait for all summaries to be stored
   - Run questions against observer-produced content
   - This tests the FULL pipeline end-to-end

**What this tests**: Everything. Observer discovery → transcript extraction → LLM summarization → observation extraction → epoch model → upsert → search → reading.

### Option C: Hybrid — Observer Ingestion + Maintenance Pipeline

Same as Option A, plus:
1. After all sessions are ingested via `/api/session/summarize`, run consolidation
2. Re-evaluate after consolidation to see if it preserves or destroys information
3. Run ObservationReflector to generate higher-order observations
4. Re-evaluate to see if reflections help retrieval

---

## 7. Recommended Implementation Order

### Phase 1: Fix the basics (Option A)
1. Switch retrieval eval to S variant
2. Replace direct add with `/api/session/summarize` for observer-style ingestion
3. Use official judge prompts
4. Add NDCG and Recall@10 metrics
5. Re-run eval, establish new baseline

### Phase 2: Full observer integration (Option B)
1. Build `LongMemEvalAdapter` for the observer daemon
2. Simulate chronological session arrival
3. Test full pipeline end-to-end
4. Compare scores with Mastra (directly comparable at this point)

### Phase 3: Maintenance pipeline testing (Option C addition)
1. Run consolidation after observer ingestion
2. Run ObservationReflector
3. Measure impact of each stage on retrieval quality

---

## 8. Key Code Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| Retrieval eval main | `src/cems/eval/longmemeval.py` | `main()` at 663, `run_eval()` at 491 |
| E2E eval main | `src/cems/eval/longmemeval_e2e.py` | `main()` at 693, `run_eval()` at 467 |
| Session formatting | `src/cems/eval/longmemeval.py` | `format_session_content()` at 451 |
| Oracle URL | `src/cems/eval/longmemeval.py` | Line 38 |
| Recall computation | `src/cems/eval/longmemeval.py` | Lines 594-598 |
| Judge prompts | `src/cems/eval/longmemeval_e2e.py` | Lines 55-113 |
| Observer daemon | `src/cems/observer/daemon.py` | `run_cycle()`, `process_session_growth()` at 252 |
| Session summarize API | `src/cems/api/handlers/session.py` | Summarize endpoint |
| Observation extraction | `src/cems/llm/observation_extraction.py` | Main extraction logic |
| Observer adapters | `src/cems/observer/adapters/base.py` | `SessionAdapter` interface |

---

## Sources

- [LongMemEval Paper (arXiv)](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
- [Mastra Observational Memory](https://mastra.ai/research/observational-memory)
- [Mastra LongMemEval Blog](https://mastra.ai/blog/observational-memory)
