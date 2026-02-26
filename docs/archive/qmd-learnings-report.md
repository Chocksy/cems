# QMD Learnings Report for CEMS Retrieval

Date: 2026-01-31
Owner: CEMS retrieval
Scope: LongMemEval recall improvements + production-safe reranking

## Executive summary (decisive)

You can likely push past 81% recall by **adopting QMD's guardrails around reranking** and **bringing lexical retrieval into the inference pipeline**. The two biggest gaps vs QMD are:

1) **CEMS inference search is vector-only** (hybrid BM25 is implemented but not used in `retrieve_for_inference`).
2) **CEMS reranker replaces ordering too aggressively** (no top-rank protection or position-aware blending).

QMD's pipeline keeps exact matches at the top while still improving mid-rank ordering. That is precisely the failure mode you saw when removing the reranker.

If you implement the P0 items in this report, you should beat 81% without losing the speed improvements from the refactor.

---

## What QMD does better (mechanics that matter)

From QMD's README (now verified via GitHub API), the pipeline has these key properties:

- **Hybrid retrieval per query (BM25 + vector)** across original + expanded queries, using SQLite FTS5 for BM25 and vector embeddings for semantic search.
- **Original query is weighted x2**, so expansions cannot overrule the core intent.
- **RRF fusion with a top-rank bonus** (rank #1 gets +0.05, #2-3 gets +0.02), which protects exact matches from being drowned out.
- **Only top 30 candidates** go to reranking, limiting noise.
- **Specialized reranker (qwen3-reranker-0.6B)** and **position-aware blending** (top 1-3: 75% retrieval / 25% reranker; top 4-10: 60/40; top 11+: 40/60).
- **Score normalization** to a shared 0-1 scale before fusion (FTS abs(score), vector 1/(1+distance), reranker score/10).
- **Dedicated query-expansion model** (qmd-query-expansion-1.7B) instead of a generic LLM prompt.

These are *guardrails* around reranking; they prevent the reranker from destroying strong retrieval hits.

---

## Deep QMD code review (exact mechanics from source)

This section is based on the QMD source code in `src/qmd.ts`, `src/store.ts`, and `src/llm.ts`.

### Query expansion (structured, typed output)

QMD does not use a plain prompt for expansion. It enforces a grammar and types:

- File: `src/llm.ts`
- Function: `expandQuery(...)`
- Output types: `lex`, `vec`, `hyde` lines (GBNF grammar)
- Behavior: returns 1-3 lexical queries, 1-3 vector queries, and max 1 HyDE line
- Context: optional extra context string can bias expansion

Why it matters: this gives QMD a deterministic split of lexical and semantic expansions, which are routed to the appropriate search backend without mixing intents.

### Strong-signal detection to skip expansion

QMD does a quick BM25 probe and skips expansion if the signal is strong:

- File: `src/qmd.ts`
- Function: `querySearch(...)`
- Logic: if top BM25 score >= 0.85 and gap to #2 is >= 0.15, it skips LLM expansion

Why it matters: this keeps recall high when a clear lexical match exists and avoids polluting the candidate set with expansion noise.

### Lexical search (FTS5) and score conversion

Lexical retrieval uses SQLite FTS5 and converts bm25 to a stable 0-1 score:

- File: `src/store.ts`
- Function: `searchFTS(...)`
- bm25 lower is better, converted to `1 / (1 + bm25)`
- Not per-query normalized, so strong-signal logic stays stable

Why it matters: this keeps lexical signal strong and comparable without damping the best hits.

### Vector search (sqlite-vec) and chunk selection

Vector search is chunk-level, with best chunk per document:

- File: `src/store.ts`
- Function: `searchVec(...)`
- Two-step query due to sqlite-vec JOIN limitations
- Chooses best chunk (min distance) per file
- Score = `1 - distance` (cosine similarity)

Why it matters: vector results are already de-duplicated at the document level and biased to the strongest chunk.

### RRF fusion with weighting and top-rank bonus

QMD fuses all ranked lists and applies guardrails:

- File: `src/store.ts`
- Function: `reciprocalRankFusion(...)`
- List weights: original query lists are weighted 2.0; expansions 1.0
- Top-rank bonus: +0.05 for rank 1, +0.02 for ranks 2-3

Why it matters: exact matches are preserved even when expansion introduces noise.

### Reranking (cross-encoder, cached, best chunk only)

Reranking is done per document with one selected chunk:

- File: `src/qmd.ts`
- Rerank cap: `RERANK_DOC_LIMIT = 40`
- Chunk selection: choose chunk with most query term hits
- Rerank model: qwen3-reranker via `node-llama-cpp`
- Cache: query + file + model key in `llm_cache`

Why it matters: reranker is applied only where it adds value, and it sees the best chunk instead of arbitrary prefixes.

### Position-aware blending (protects top hits)

QMD blends reranker scores with retrieval rank:

- File: `src/qmd.ts`
- Uses RRF rank position, not RRF score
- RRF rank 1-3: 75% retrieval, 25% reranker
- RRF rank 4-10: 60% retrieval, 40% reranker
- RRF rank 11+: 40% retrieval, 60% reranker

Why it matters: it avoids reranker damage to the top results, which is what you observed in CEMS.

---

## Current CEMS inference pipeline (observed in code)

From `src/cems/memory/retrieval.py` + `src/cems/retrieval.py`:

- Query synthesis and HyDE are optional and currently disabled in eval via API defaults.
- Candidate retrieval uses `_search_raw`, which is **vector-only**.
- RRF fuses only those vector results (+ relation traversal results), with no lexical stream.
- Reranking is LLM prompt-based, takes up to 100 candidates, and **reorders top-k with a 70/30 blend** that can wipe out strong top hits.
- No top-rank bonus, no position-aware blending, no score normalization across sources.

This explains why removing reranking improved eval: the reranker is too powerful and not constrained by retrieval confidence.

---

## Gap analysis (QMD vs CEMS)

| Area | QMD | CEMS (current) | Impact |
|------|-----|----------------|--------|
| Hybrid retrieval | BM25 + vector for every query | Vector only in inference | Misses lexical matches, weaker precision |
| Query weighting | Original x2 | No weighting | Expansion can dilute intent |
| Fusion | RRF + top-rank bonus | RRF only | Exact matches get diluted |
| Reranker input | Top 30 | Up to 100 | More noise, worse stability |
| Reranker blending | Position-aware | Global 70/30 | Overwrites top hits |
| Reranker model | Specialized cross-encoder | Generic LLM prompt | Inconsistent relevance judgments |
| Score normalization | 0-1 for all | Mixed | One scorer can dominate |

---

## P0 (highest-impact) changes to implement

These changes directly mirror QMD's guardrails and should increase recall without reintroducing reranker regressions.

### P0-1: Add lexical retrieval into inference pipeline

**Why:** CEMS already has BM25 (full-text) and hybrid search in `PgVectorStore`, but `retrieve_for_inference` uses only vector search.

**Implementation:**
- Add a `_search_lexical_raw` using `vectorstore.full_text_search()`.
- For each query in `queries_to_search`, run both vector and lexical.
- Feed all lists into RRF, not just vector lists.
- Treat lexical lists as first-class inputs (do not down-weight them by default).

**Files:**
- `src/cems/memory/search.py` (add raw lexical search helper)
- `src/cems/memory/retrieval.py` (collect both lists per query)

### P0-2: Add top-rank bonus + original query weight to RRF

**Why:** This is the core guardrail QMD uses to preserve exact matches.

**Implementation:**
- Modify `reciprocal_rank_fusion()` to accept per-list weights and top-rank bonus.
- Use weights: original query list weight 2.0, expansions weight 1.0.
- Apply bonus: rank #1 +0.05, #2-3 +0.02.

**Files:**
- `src/cems/retrieval.py` (RRF function)
- `src/cems/memory/retrieval.py` (pass list weights and bonuses)

### P0-3: Limit reranking to top 30 + position-aware blending

**Why:** QMD shows reranking only helps once it stops overthrowing top results.

**Implementation:**
- Set `rerank_input_limit=30` for eval.
- Compute blended score using retrieval rank buckets:
  - ranks 1-3: 75% retrieval, 25% reranker
  - ranks 4-10: 60% retrieval, 40% reranker
  - ranks 11+: 40% retrieval, 60% reranker
- Keep retrieval ordering for documents not returned by reranker.

**Files:**
- `src/cems/retrieval.py` (rerank / blending logic)

---

## P1 (strong improvements, slightly more work)

### P1-1: Replace prompt-reranker with a proper cross-encoder

**Why:** Prompt-reranking is noisy and brittle. A cross-encoder is deterministic and aligns with QMD's approach.

**Implementation options:**
- Local: qwen3-reranker-0.6B (if you can run it locally in docker)
- Hosted: reranker models via OpenRouter (if available)
- If LLM reranker stays, include full memory content or a short deterministic summary, not first 200 chars.

### P1-2: Normalize vector scores to 0-1

**Why:** Your vector similarity (`1 - distance`) can be negative and not comparable to other scores.

**Implementation:**
- Replace score computation with `1 / (1 + distance)` or min-max normalize within a query list before RRF.

### P1-3: Restrict query expansion

**Why:** Expansions can swamp intent. QMD uses only 1-2 and weights original x2.

**Implementation:**
- Cap to 1-2 expansions.
- Weight original x2 in RRF.
- Disable expansion for temporal questions if it hurts precision.

---

## P2 (state-of-the-art, longer term)

- **Dual-encoder + reranker distillation:** Train a small reranker or teacher-student pair on LongMemEval to stabilize relevance judgments.
- **Domain-specific lexical boosts:** Add token-level boosts for file paths, code symbols, and config keys when they match query terms.
- **Query-type routing:** Keep your query understanding but only enable rerank for "complex" or "multi-session" types.
- **Better snippet selection:** Use a fast sentence selection model to feed reranker the most relevant spans instead of truncating by 200 chars.

---

## Proposed evaluation plan (fast, decisive)

1) **Baseline (current eval run)**
   - Keep as is for comparison.

2) **P0-Only (expected immediate gain)**
   - Hybrid lexical + vector in inference
   - RRF top-rank bonus + original weight
   - Rerank top 30 + position-aware blending

3) **P0 + P1-1 (if you can plug a reranker fast)**
   - Swap LLM prompt reranker with qwen3-reranker or similar cross-encoder

4) **Tune weights**
   - If results still dip: lower reranker influence in top-10 (e.g., 85/15, 70/30, 55/45)

Success criteria: +2-4% overall Recall@All with no regression on temporal-reasoning.

---

## Why this should push you >81%

Your best gains came from *removing* a destructive reranker. QMD's pipeline is proof that reranking works when it is bounded by retrieval confidence. The P0 changes above implement those exact guardrails. In practice, this yields higher recall without sacrificing precise top results.

---

## Implementation checklist (code-level, minimal churn)

### Retrieval input streams

- Add `_search_lexical_raw()` in `src/cems/memory/search.py` alongside `_search_raw`.
- In `src/cems/memory/retrieval.py`, for each `search_query`, collect:
  - vector list (existing)
  - lexical list (new)
  - relation list (existing, optional)
- Pass list weights to RRF:
  - original query lists: weight 2.0
  - expansion lists: weight 1.0

### RRF fusion improvements

- Update `reciprocal_rank_fusion()` in `src/cems/retrieval.py`:
  - Accept `list_weights` and `top_rank_bonus` parameters.
  - Apply top-rank bonus to ranks 1 and 2-3.
  - Keep current RRF constant k=60.

### Reranker guardrails

- Set `rerank_input_limit=30` in eval or via a dedicated env flag.
- Add position-aware blend in `src/cems/retrieval.py`:
  - For each candidate, compute `retrieval_rank`.
  - Blend with reranker score using rank bucket weights.
  - Keep retrieval ordering for candidates missing reranker scores.

### Scoring normalization

- Normalize vector scores (1 / (1 + distance)) in `src/cems/vectorstore.py`.
- Normalize FTS scores to a 0-1 range before blending if you use raw BM25 scores later.

---

## Exact change map (CEMS: where and how)

This is a precise placement guide for each change based on the refactored layout.

### 1) Add lexical retrieval to inference

**Files**: `src/cems/memory/search.py`, `src/cems/memory/retrieval.py`

- Add a new helper in `src/cems/memory/search.py`:
  - `_search_lexical_raw(...)` that calls `self._vectorstore.full_text_search(...)`
  - Convert rows using `_make_search_result(...)`
  - Sort by score desc and return list

- Update `retrieve_for_inference(...)` in `src/cems/memory/retrieval.py`:
  - In Stage 4, for each `search_query`, call both:
    - `_search_raw(...)` (vector)
    - `_search_lexical_raw(...)` (BM25)
  - Append both lists to `query_results`
  - Keep relation traversal as its own list (optional)

Why here: this keeps the inference pipeline aligned with the refactor and gives RRF more diverse candidates.

### 2) Add list weights + top-rank bonus to RRF

**File**: `src/cems/retrieval.py`

- Update `reciprocal_rank_fusion(...)` signature to accept:
  - `list_weights: list[float] | None`
  - `top_rank_bonus: tuple[float, float] | None` (for rank 1 and ranks 2-3)
- Apply weights per list; apply bonus based on best rank across lists (same as QMD).

**File**: `src/cems/memory/retrieval.py`

- Build `list_weights` in parallel with `query_results`:
  - For original query lists (lex + vec): weight 2.0
  - For expansion lists: weight 1.0
  - For relation list: consider weight 0.5-1.0 (start at 1.0)
- Call `reciprocal_rank_fusion(query_results, list_weights=..., top_rank_bonus=(0.05, 0.02))`

Why here: this is the exact spot where QMD protects exact matches and prevents expansion noise from dominating.

### 3) Position-aware rerank blending

**File**: `src/cems/retrieval.py`

- Modify `rerank_with_llm(...)`:
  - Capture `retrieval_rank` before reranking (index in `candidates` list).
  - When reranker outputs indices, compute a rerank score (e.g., `1 / (1 + rank)`).
  - Blend with retrieval rank buckets:
    - rank 1-3: 75% retrieval / 25% reranker
    - rank 4-10: 60% retrieval / 40% reranker
    - rank 11+: 40% retrieval / 60% reranker
  - Keep candidates missing rerank output in retrieval order (do not drop them).

Why here: this prevents the reranker from overriding top hits, which is the main regression you saw.

### 4) Candidate cap and best-chunk selection for reranking

**Files**: `src/cems/retrieval.py`, `src/cems/memory/retrieval.py`

- Reduce `rerank_input_limit` for eval to 30-40 (QMD uses 40).
- Add a small helper that selects the best chunk for reranking:
  - Split content into chunks (use existing `tiktoken` estimate or sentence split).
  - Score each chunk by query term overlap (simple, fast).
  - Feed the best chunk to reranker instead of the first 200 chars.

Why here: QMD reranks the most relevant chunk, not an arbitrary prefix.

### 5) Strong-signal skip expansion

**Files**: `src/cems/memory/retrieval.py`, `src/cems/vectorstore.py` (optional)

- Run a fast BM25 probe (top 2) on the original query.
- If top score is strong and separated (similar to QMD's 0.85 and 0.15 gap), skip query synthesis and HyDE.
- This requires BM25 scores to be stable (normalize if needed).

Why here: this avoids expansion noise when a lexical match is already clear.

---

## Effort estimate

- P0 changes (lexical stream, RRF weights/bonus, position-aware blend): 0.5-1 day
- Chunk selection for reranking: 0.5 day
- Strong-signal skip expansion: 0.25 day
- Swap to cross-encoder reranker (optional): 1-2 days depending on infra

---

## Notes from recent refactor and commits

- The architectural refactor split retrieval into `src/cems/memory/retrieval.py` and `src/cems/retrieval.py`, which makes these changes low-risk and localized.
- The latest eval branch defaults to vector mode, disables query synthesis and HyDE, and allows reranking via API flags.
- This report assumes you keep that simplified eval path and only add the QMD guardrails above.

---

## Appendix: Code touchpoints

- Retrieval pipeline: `src/cems/memory/retrieval.py`
- Fusion + rerank logic: `src/cems/retrieval.py`
- Vector + lexical search: `src/cems/vectorstore.py`
- Eval harness: `src/cems/eval/longmemeval.py`
- Config defaults: `src/cems/config.py`
