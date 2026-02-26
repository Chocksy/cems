# CEMS Retrieval Improvements Plan: QMD Learnings Adaptation

**Goal:** Push past 81% recall by adopting QMD's guardrails around reranking and adding lexical retrieval to the inference pipeline.

**Current baseline:** 74% Recall@5 (with project penalties working)

---

## Codex Evaluation Summary

**Rating:** Architecture 8/10, Execution Plan 5/10 (critical math errors found)
**Expected result as-is:** 75-77% (gains suppressed by errors)
**Expected result with fixes:** 80-82% (target achieved)

### Critical Gaps Identified

| Gap | Problem | Impact | Fix |
|-----|---------|--------|-----|
| **#1 Reranker scoring** | Using `1/retrieval_rank` instead of RRF score | Boosts low-rank items too much | Use `1/(k+rank)` for retrieval score |
| **#2 Reranker normalization** | Qwen3 outputs 0-10+ logits, not 0-1 | Magnitude mismatch breaks blending | Divide by 10 to normalize |
| **#3 Skip expansion missing** | Will expand every query, adding noise | Cancels lexical wins | Add BM25 probe FIRST |
| **#4 Lexical score normalization** | BM25 returns 0-5+, vector returns 0-1 | BM25 dominates RRF | Min-max normalize lexical |
| **#5 Best-chunk not implemented** | `select_best_chunk()` referenced but missing | Arbitrary truncation | Implement term overlap scoring |
| **#6 RRF weights not in code** | Plan describes but retrieval.py lacks them | No list weighting | Add to `reciprocal_rank_fusion()` |

---

## Decisions Made (Updated)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Reranker** | Qwen3-Reranker via Ollama | State-of-the-art, deterministic, 50-200ms vs 2-3s LLM |
| **Implementation** | Incremental with evals | Add each change separately, run eval after each |
| **DB-side RRF** | Keep Python for now | Easier to iterate on weights/bonuses |
| **Chunk selection** | Query term overlap | Select chunk with highest query term matches |
| **Lexical stream** | Yes + normalize scores | Add BM25 alongside vector, min-max normalize |
| **RRF weights** | Yes + correct math | Original 2.0x, use `1/(k+rank)` not `1/rank` |
| **Score normalization** | Yes + reranker normalization | Clamp all to 0-1, divide Qwen3 output by 10 |
| **Skip expansion** | Yes - do FIRST | BM25 probe before synthesis to avoid noise |

---

## Implementation Phases (Reordered Based on Codex Review)

### Phase 0: Infrastructure - Ollama Setup
**Goal:** Get Qwen3-Reranker running locally/on Hetzner

```yaml
# Add to docker-compose.yml
ollama:
  image: ollama/ollama
  ports:
    - "11434:11434"
  volumes:
    - ollama_data:/root/.ollama
  restart: unless-stopped
```

```bash
# Pull the reranker model (639MB)
docker compose up -d ollama
docker compose exec ollama ollama pull sam860/qwen3-reranker:0.6b-Q8_0
```

**Verify:** `curl http://localhost:11434/api/tags` should list the model

---

### Phase 1: Strong-Signal Skip Expansion (DO FIRST - Critical Fix)
**Goal:** Avoid expansion noise that cancels lexical wins

**Why first:** Codex identified this as the #1 priority fix. Without it, expansions pollute candidate sets and undo any gains from lexical/reranker improvements.

**Changes:**
| File | Description |
|------|-------------|
| `src/cems/memory/search.py` | Add `_search_lexical_raw_async()` helper |
| `src/cems/memory/retrieval.py` | Add BM25 probe BEFORE query synthesis |

**Implementation:**
```python
# In memory/search.py - add new helper
async def _search_lexical_raw_async(
    self, query: str, scope: str, limit: int = 5
) -> list[SearchResult]:
    """BM25 search without score adjustments."""
    raw = await self._vectorstore.full_text_search(
        query=query, user_id=self.config.user_id,
        team_id=self.config.team_id if scope in ("shared", "both") else None,
        scope=scope, limit=limit,
    )
    return [_make_search_result(mem, self.config.user_id) for mem in raw]

# In memory/retrieval.py - add before query synthesis (around line 100)
# Stage 2: Query synthesis with strong-signal skip
queries_to_search = [query]
if enable_query_synthesis and self.config.enable_query_synthesis and client:
    # BM25 probe FIRST - skip expansion if signal is strong
    lexical_probe = await self._search_lexical_raw_async(query, scope, limit=2)
    skip_expansion = False
    if lexical_probe:
        top_score = lexical_probe[0].score
        second_score = lexical_probe[1].score if len(lexical_probe) > 1 else 0.0
        gap = top_score - second_score
        # QMD threshold: top >= 0.85 AND gap >= 0.15
        if top_score >= 0.85 and gap >= 0.15:
            log.info(f"Strong signal (score={top_score:.2f}, gap={gap:.2f}), skipping expansion")
            skip_expansion = True

    if not skip_expansion:
        expanded = synthesize_query(query, client)
        queries_to_search = [query] + expanded[:3]
```

**Eval after Phase 1:** Run 100 questions WITH synthesis=ON to measure noise reduction

---

### Phase 2: Lexical Stream + Score Normalization (Critical Fix)
**Goal:** Add BM25 alongside vector, properly normalized

**Why:** Codex found BM25 returns 0-5+ while vector returns 0-1. Without normalization, BM25 dominates RRF.

**Changes:**
| File | Description |
|------|-------------|
| `src/cems/memory/retrieval.py` | Add lexical search per query + normalize |

**Implementation:**
```python
# In Stage 4 of retrieve_for_inference_async (around line 300)
for search_query in queries_to_search:
    # Vector search (0-1 scores)
    vector_results = await self._search_raw_async(
        search_query, scope, limit=self.config.max_candidates_per_query,
        query_embedding=embedding,
    )
    query_results.append(vector_results)

    # Lexical search (needs normalization!)
    lexical_results = await self._search_lexical_raw_async(
        search_query, scope, limit=50
    )
    # CRITICAL: Normalize BM25 scores to 0-1
    if lexical_results:
        max_score = max(r.score for r in lexical_results)
        if max_score > 0:
            for r in lexical_results:
                r.score = r.score / max_score
    query_results.append(lexical_results)
```

**Eval after Phase 2:** Measure lexical stream contribution

---

### Phase 3: RRF Weights + Top-Rank Bonus (Critical Fix)
**Goal:** Protect original query, add rank bonus

**Why:** Codex confirmed the formula but noted it's not in the actual code yet.

**Changes:**
| File | Description |
|------|-------------|
| `src/cems/retrieval.py` | Add `list_weights` + `top_rank_bonus` to RRF |
| `src/cems/memory/retrieval.py` | Build and pass weights |

**Implementation:**
```python
# In retrieval.py - update reciprocal_rank_fusion signature
def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    list_weights: list[float] | None = None,
    top_rank_bonus: tuple[float, float] = (0.05, 0.02),
    k: int = 60,
    rrf_weight: float = 0.5,
) -> list[SearchResult]:
    """RRF with QMD-style guardrails."""
    if list_weights is None:
        list_weights = [1.0] * len(result_lists)

    bonus_r1, bonus_r23 = top_rank_bonus
    rrf_scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, SearchResult] = {}

    for weight, results in zip(list_weights, result_lists):
        for rank, result in enumerate(results, 1):
            # RRF score with weight
            base = weight / (k + rank)
            # Top-rank bonus (per-list, stacks if in multiple lists)
            bonus = bonus_r1 if rank == 1 else (bonus_r23 if rank <= 3 else 0.0)
            rrf_scores[result.memory_id] += base + bonus

            if result.memory_id not in result_map or result.score > result_map[result.memory_id].score:
                result_map[result.memory_id] = result
    # ... rest unchanged (normalization + blending)

# In memory/retrieval.py - build weights when calling RRF
# For each query: [vector_list, lexical_list]
# Original query gets 2.0x, expansions get 1.0x
list_weights = []
for i, _ in enumerate(queries_to_search):
    if i == 0:  # Original query
        list_weights.extend([2.0, 2.0])  # vector, lexical
    else:  # Expansions
        list_weights.extend([1.0, 1.0])
# Relations list
if relation_results:
    list_weights.append(0.5)

candidates = reciprocal_rank_fusion(
    query_results,
    list_weights=list_weights,
    top_rank_bonus=(0.05, 0.02),
)
```

**Eval after Phase 3:** Measure RRF improvements

---

### Phase 4: Qwen3-Reranker with CORRECT Math (Critical Fix)
**Goal:** Deterministic reranking with proper score handling

**Why:** Codex found the blending formula was WRONG. Must use:
- `1/(k+rank)` for retrieval score (not `1/rank`)
- Normalize Qwen3 output by dividing by 10 (it outputs 0-10+ logits)

**Changes:**
| File | Description |
|------|-------------|
| `src/cems/retrieval.py` | New `rerank_with_qwen()` + `select_best_chunk()` |
| `src/cems/config.py` | Add Ollama config |
| `pyproject.toml` | Add `ollama` dependency |

**Implementation:**
```python
def select_best_chunk(content: str, query: str, max_chars: int = 300) -> str:
    """Select chunk with highest query term overlap."""
    sentences = content.split('. ')
    query_terms = set(query.lower().split())

    best_chunk = ""
    best_score = 0
    current_chunk = ""
    current_score = 0

    for sent in sentences:
        sent_lower = sent.lower()
        term_matches = sum(1 for term in query_terms if term in sent_lower)
        current_chunk += sent + ". "
        current_score += term_matches

        if len(current_chunk) > max_chars:
            if current_score > best_score:
                best_score = current_score
                best_chunk = current_chunk[:max_chars]
            current_chunk = ""
            current_score = 0

    # Handle remaining chunk
    if current_score > best_score:
        best_chunk = current_chunk[:max_chars]

    return best_chunk or content[:max_chars]


async def rerank_with_qwen(
    query: str,
    candidates: list[SearchResult],
    top_k: int = 10,
    config: CEMSConfig = None,
    k: int = 60,  # RRF constant
) -> list[SearchResult]:
    """Rerank using Qwen3-Reranker with CORRECT position-aware blending."""
    import ollama

    client = ollama.AsyncClient(host=config.ollama_url or "http://localhost:11434")

    # Cap at 40 candidates (QMD's limit)
    candidates_to_rank = candidates[:40]

    # Score each candidate with Qwen3-Reranker
    scored = []
    for cand in candidates_to_rank:
        chunk = select_best_chunk(cand.content, query, max_chars=300)
        prompt = f"Query: {query}\nDocument: {chunk}"

        try:
            response = await client.generate(
                model=config.reranker_model or "sam860/qwen3-reranker:0.6b-Q8_0",
                prompt=prompt,
            )
            raw_score = float(response.response.strip())
        except Exception as e:
            logger.warning(f"Reranker failed for candidate: {e}")
            raw_score = 5.0  # Neutral score

        scored.append((cand, raw_score))

    # CORRECT position-aware blending (Codex fix)
    for retrieval_rank, (cand, raw_rerank_score) in enumerate(scored, 1):
        # Normalize reranker output (Qwen3 outputs 0-10+)
        reranker_normalized = raw_rerank_score / 10.0
        reranker_normalized = max(0.0, min(1.0, reranker_normalized))  # Clamp

        # Retrieval score uses RRF formula (NOT 1/rank!)
        retrieval_score = 1.0 / (k + retrieval_rank)

        # Position-aware blend (QMD's approach)
        if retrieval_rank <= 3:
            # Top 3: Trust retrieval more (75%)
            cand.score = 0.75 * retrieval_score + 0.25 * reranker_normalized
        elif retrieval_rank <= 10:
            # Ranks 4-10: Balanced (60/40)
            cand.score = 0.60 * retrieval_score + 0.40 * reranker_normalized
        else:
            # Ranks 11+: Trust reranker more (40/60)
            cand.score = 0.40 * retrieval_score + 0.60 * reranker_normalized

    # Re-sort by blended score
    reranked = sorted([c for c, _ in scored], key=lambda x: x.score, reverse=True)
    return reranked[:top_k]
```

**Eval after Phase 4:** Full comparison with all fixes

---

### Phase 5: Score Normalization Cleanup
**Goal:** Fix negative vector scores, ensure all scores 0-1

**Changes:**
| File | Description |
|------|-------------|
| `src/cems/retrieval.py` | Add score clamping in `apply_score_adjustments()` |

**Implementation:**
```python
# At the end of apply_score_adjustments():
score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
return score
```

---

### Phase 6: Full Eval + Tuning
**Goal:** Validate on 500 questions, tune parameters

```bash
python -m cems.eval.longmemeval --questions 500 -o full_results.json
```

**Parameters to tune if needed:**
| Parameter | Default | Tune If |
|-----------|---------|---------|
| `top_rank_bonus` | (0.05, 0.02) | Top hits still getting overridden |
| `list_weights` | original 2.0x | Expansions hurting more than helping |
| Position blend | 75/25, 60/40, 40/60 | Reranker still damaging top results |
| `strong_signal_threshold` | 0.85 | Too many/few queries skipping expansion |
| `rerank_input_limit` | 40 | Latency issues (reduce) or recall issues (increase) |

---

## Verification Strategy (Per-Phase Evals)

### After Each Phase
```bash
# Run 100-question eval
python -m cems.eval.longmemeval --questions 100 --api-url http://localhost:8765 -o phase{N}_results.json -v
```

### Results Tracking (Updated with Codex Estimates)

| Phase | Change | Expected (Codex) | Actual |
|-------|--------|-----------------|--------|
| Baseline | Current (74%) | 74% | 74% |
| Phase 1 | Strong-signal skip expansion | 75-76% | TBD |
| Phase 2 | Lexical stream + normalize | 76-78% | TBD |
| Phase 3 | RRF weights + top-rank bonus | 77-79% | TBD |
| Phase 4 | Qwen3-Reranker (correct math) | 79-81%+ | TBD |
| Phase 5 | Score normalization cleanup | ~same | TBD |
| **Final** | All changes | **79-82%** | TBD |

### Full Eval (After All Phases)
```bash
python -m cems.eval.longmemeval --questions 500 -o final_results.json
```

---

## Infrastructure Changes

### Docker Compose Addition
```yaml
# Add to docker-compose.yml
services:
  ollama:
    image: ollama/ollama
    container_name: cems-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  ollama_data:
```

### Model Download (One-time)
```bash
docker compose up -d ollama
docker compose exec ollama ollama pull sam860/qwen3-reranker:0.6b-Q8_0
```

---

## Config Changes

New fields in `src/cems/config.py`:

```python
# Ollama / Reranker Settings
ollama_url: str = Field(
    default="http://ollama:11434",
    description="Ollama server URL for reranking",
)
reranker_model: str = Field(
    default="sam860/qwen3-reranker:0.6b-Q8_0",
    description="Qwen3-Reranker model name in Ollama",
)
reranker_backend: str = Field(
    default="qwen",  # "qwen" | "llm" | "disabled"
    description="Which reranker to use",
)

# RRF Settings
enable_lexical_in_inference: bool = Field(
    default=True,
    description="Add BM25 stream alongside vector in inference",
)
rrf_original_weight: float = Field(
    default=2.0,
    description="Weight for original query lists in RRF",
)
rrf_expansion_weight: float = Field(
    default=1.0,
    description="Weight for expansion query lists in RRF",
)
rrf_top_rank_bonus_r1: float = Field(
    default=0.05,
    description="Bonus for rank 1 in RRF",
)
rrf_top_rank_bonus_r23: float = Field(
    default=0.02,
    description="Bonus for ranks 2-3 in RRF",
)

# Reranker Position Blending
rerank_top3_retrieval_weight: float = Field(
    default=0.75,
    description="Retrieval weight for top 3 results (reranker gets 1 - this)",
)
rerank_top10_retrieval_weight: float = Field(
    default=0.60,
    description="Retrieval weight for ranks 4-10",
)
rerank_rest_retrieval_weight: float = Field(
    default=0.40,
    description="Retrieval weight for ranks 11+",
)

# Strong Signal Detection
strong_signal_threshold: float = Field(
    default=0.85,
    description="Skip expansion if top BM25 score >= this",
)
strong_signal_gap: float = Field(
    default=0.15,
    description="AND gap to second result >= this",
)

# Chunk Selection
rerank_chunk_size: int = Field(
    default=300,
    description="Max chars to send to reranker per candidate",
)
```

---

## Files to Modify (Updated Phase Order)

| File | Phase | Changes |
|------|-------|---------|
| `docker-compose.yml` | 0 | Add Ollama service |
| `src/cems/memory/search.py` | 1 | Add `_search_lexical_raw_async()` |
| `src/cems/memory/retrieval.py` | 1 | Add BM25 probe for strong-signal skip |
| `src/cems/memory/retrieval.py` | 2 | Add lexical stream + normalization |
| `src/cems/retrieval.py` | 3 | Enhance RRF with weights + bonus |
| `src/cems/memory/retrieval.py` | 3 | Build and pass list weights |
| `pyproject.toml` | 4 | Add `ollama` dependency |
| `src/cems/config.py` | 4 | Add Ollama + reranker config |
| `src/cems/retrieval.py` | 4 | Add `rerank_with_qwen()`, `select_best_chunk()` |
| `src/cems/memory/retrieval.py` | 4 | Use Qwen reranker |
| `src/cems/retrieval.py` | 5 | Score normalization clamping |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Ollama container issues | Low | Well-documented, simple setup |
| Qwen3-Reranker slow on CPU | Low | 0.6B model is efficient, ~50-200ms |
| Lexical adds latency | Low | Run vector + lexical in parallel |
| RRF weights hurt some queries | Low | Configurable, can tune |
| Breaking existing behavior | Low | All changes behind config flags |

---

## Success Criteria

**Minimum success:** 78% Recall@5 (+4% from baseline)
**Target:** 81%+ Recall@5 (+7% from baseline)
**No regressions:** Temporal reasoning should not decrease

---

## Summary

This plan implements QMD's proven retrieval guardrails with **Codex-reviewed corrections**:

### Key Fixes from Codex Review
1. **Reordered phases** - Strong-signal skip FIRST to avoid expansion noise
2. **Fixed reranker math** - Use `1/(k+rank)` for retrieval score, not `1/rank`
3. **Normalize Qwen3 output** - Divide by 10 (outputs 0-10+ logits)
4. **Normalize BM25 scores** - Min-max normalize before RRF (0-5+ → 0-1)
5. **Implement best-chunk** - Query term overlap, not arbitrary truncation

### Phase Order (Codex Recommended)
1. **Strong-signal skip** - Avoid expansion noise FIRST
2. **Lexical stream + normalize** - BM25 alongside vector, properly scaled
3. **RRF weights + bonus** - Original 2x, top-rank protection
4. **Qwen3-Reranker** - Deterministic, position-aware blending
5. **Score cleanup** - Clamp all to 0-1
6. **Full eval + tuning** - 500 questions, adjust parameters

### Expected Outcomes (Codex Estimates)
- **Conservative:** 79% Recall@5 (+5% from baseline)
- **Optimistic:** 82% Recall@5 (+8% from baseline)
- **No regressions:** Temporal reasoning monitored per-phase

Each phase is incremental with eval runs to isolate impact. All changes are configurable and can be tuned.
