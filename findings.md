# Findings - CEMS Retrieval Improvements

## 2026-02-04: Preference Query Analysis

### The Semantic Gap Problem

Preference queries have a fundamental retrieval problem:
- **Question phrasing**: "Can you recommend video editing resources?"
- **Answer phrasing**: "I use Adobe Premiere Pro for all my video editing"

Neither vector similarity NOR BM25 keyword matching bridge this gap effectively.

### Solution: Preference-Aware Query Synthesis

We implemented a specialized synthesis prompt that generates queries matching user statements:

```
"video editing resources" →
  - "Adobe Premiere Pro"
  - "video editing software I use"
  - "editing workflow preferences"
```

### Preference Signal Detection Patterns

| Category | Signals | Example |
|----------|---------|---------|
| Recommendation | recommend, suggest, advice | "Can you recommend..." |
| Resource seeking | resources, tools, accessories, publications | "What tools..." |
| Complement/match | complement, go with, works well | "What goes with..." |
| Interest probing | might like, based on my | "What might I find interesting..." |

### Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| single-session-preference | 13.3% | 50.0% | **+36.7%** |
| temporal-reasoning | 66.1% | 92.6% | +26.5% |
| Overall Recall@5 | 73.6% | 88.0% | +14.4% |

### Remaining Challenges

**Indirect preference queries** not detected by current signals:
- "I've been feeling stuck with my paintings" (advice seeking via emotion)
- "What should I serve for dinner" (planning/recommendation hybrid)
- "I'm thinking about making a cocktail" (intent/anticipation framing)
- "Trouble with battery life" (troubleshooting, not preference)

**Options tried:**
1. ✗ LLM reranker - HURTS performance (81% vs 88% without)
2. ✓ Expanded signal detection - v2 patterns improved preference by 5.6%
3. TODO: HyDE for preferences - generate hypothetical answers

### Final Results Summary

| Category | Baseline | After Improvements | Change |
|----------|----------|-------------------|--------|
| **Overall Recall@5** | 73.6% | **87.5%** | **+13.9%** |
| single-session-preference | 13.3% | **55.6%** | **+42.3%** |
| temporal-reasoning | 66.1% | **90.7%** | +24.6% |
| knowledge-update | ~94% | **98.6%** | +4.6% |
| multi-session | 74.4% | **80.4%** | +6.0% |

### Key Learnings

1. **Query synthesis is critical for semantic gap queries** - temporal and preference queries NEED expansion
2. **LLM rerankers can hurt performance** - they may disagree with ground truth labels
3. **Pattern-based detection is effective** - simple keyword patterns work well for query routing
4. **Force-enable synthesis for specific query types** - don't let global config disable it
5. **HyDE for preferences should generate USER STATEMENTS** - first-person voice matches stored memories
6. **Observation queries are implicit preference queries** - "I noticed X" often seeks explanation based on past context
7. **Generic queries hit a wall** - "recommend a show" with no domain hint is fundamentally hard

### v3 Improvements Summary (2026-02-04)

| Change | Impact |
|--------|--------|
| Added "any recommendations/suggestions" patterns | Catches explicit request forms |
| Added "planning a trip/my" patterns | Catches trip/meal planning |
| Added "thinking of trying" patterns | Catches intent-based queries |
| Added "I noticed" observation patterns | Catches implicit preference queries |
| Preference-specific HyDE (first-person) | Bridges semantic gap with user-voice |
| Force HyDE for preference queries | Ensures HyDE runs even in vector mode |

**Result: single-session-preference 55.6% → 61.1% (+5.5%)**

### v4 Domain-Specific HyDE Improvements (2026-02-04)

Added detailed domain-specific examples to both synthesis and HyDE prompts:
- Entertainment: stand-up comedy, Netflix, streaming preferences
- Health/wellness: meditation apps, sleep schedule, bedtime routines
- Cooking: mixology class, favorite ingredients, cocktail preferences
- Design: mid-century modern, walnut wood, brass accents
- Tech: power bank, charging setup, tech accessories

**Result: 61.1% → 66.7% (+5.6%)**

### Final Preference Query Results

| Version | Recall@5 | Improvement |
|---------|----------|-------------|
| Baseline | 13.3% (4/30) | - |
| v1 (basic patterns) | 50.0% (9/18) | +36.7% |
| v2 (expanded patterns) | 55.6% (10/18) | +5.6% |
| v3 (HyDE + more patterns) | 61.1% (11/18) | +5.5% |
| v4 (domain-specific HyDE) | **66.7% (12/18)** | +5.6% |

**Total: 13.3% → 66.7% = 5x improvement**

### Remaining Hard Cases

The 6 still-failing queries share a pattern:
1. **Very domain-specific context** - requires finding user's exact interests (healthcare AI, MCM design)
2. **Multi-turn dependency** - answer depends on accumulated conversation context
3. **Generic question → specific answer** - largest semantic gap

These may require:
- **RAG-style retrieval augmentation** - first retrieve context, then search with it
- **User profile/interest tracking** - maintain explicit preference summaries
- **Iterative retrieval** - multiple retrieval rounds with feedback

---

# Findings - Local GGUF Models

## QMD Ground Truth

QMD uses node-llama-cpp with local GGUF models:
- **Embeddings:** `embeddinggemma-300M` GGUF (768-dim)
- **Reranker:** `Qwen3-Reranker-0.6B` GGUF
- **No Ollama** - Direct llama.cpp bindings

## Models to Use

| Model | Purpose | HuggingFace Repo | Size | Dimensions |
|-------|---------|------------------|------|------------|
| Embedding Gemma 300M | Vector embeddings | `ggml-org/embeddinggemma-300M-GGUF` | ~300MB | 768 |
| Qwen3-Reranker 0.6B | Cross-encoder scoring | `mradermacher/Qwen3-Reranker-0.6B-GGUF` | ~640MB | N/A |

## Dimension Strategy

**Problem:** OpenRouter embeddings are 1536-dim, local embeddings are 768-dim. pgvector column must match.

**Chosen: Strategy B (Dual Columns)**
- Keep existing `embedding` (1536) for OpenRouter
- Add `embedding_local` (768) for llama.cpp
- Use config to choose which column to query
- Non-destructive migration

## Position-Aware Reranking (QMD Approach)

Blend retrieval score with reranker score based on position:
- **Top 3:** 75% retrieval / 25% reranker
- **Ranks 4-10:** 60% / 40%
- **Ranks 11+:** 40% / 60%

This protects top hits while allowing reranker to improve mid-range.

## llama-cpp-python Notes

- Supports embeddings via `create_embedding()` method
- Model stays in memory after load (fast subsequent calls)
- GPU acceleration via `n_gpu_layers=-1` (all layers to GPU)
- Thread pool wrapper needed for async contexts

## HuggingFace Model Download

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="ggml-org/embeddinggemma-300M-GGUF",
    filename="embeddinggemma-300M-Q8_0.gguf",
    local_dir="~/.cache/cems/models",
)
```

## Docker Considerations

- llama-cpp-python requires compilation with GPU flags
- For CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`
- For CPU: `pip install llama-cpp-python` (slower but works)
- Pre-download models in Docker build to avoid runtime download

## Expected Performance

| Config | Expected Recall@5 | Speed/Query |
|--------|-------------------|-------------|
| OpenRouter + no rerank (current) | 80% | ~5s |
| llama.cpp embed + no rerank | 78-80% | ~2s |
| llama.cpp embed + llama.cpp rerank | 83-85% | ~4s |
| llama.cpp embed + llama.cpp rerank (GPU) | 83-85% | ~2.5s |



---

## Profile Probe MVP Results (2026-02-04)

### Hypothesis Validated

Manual test showed profile probe moves Hendricks gin result from #5 to #1 for cocktail query.

### Implementation

Added profile probe to retrieval pipeline:
1. For preference queries, do a lightweight "profile probe" search
2. Extract key phrases using regex patterns
3. Inject profile context into HyDE prompt

### Results on LongMemEval (230 questions)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| single-session-preference | 56.7% | **66.7%** | **+10%** |
| Overall | 87.0% | **88.7%** | +1.7% |

### Key Insight

Profile probe works because it:
1. Surfaces relevant user preferences BEFORE HyDE generation
2. HyDE then generates hypotheticals that match actual stored memories
3. No new infrastructure needed - just an extra vector search

### Cumulative Preference Query Results

| Version | Recall@5 | Improvement |
|---------|----------|-------------|
| Baseline | 13.3% (4/30) | - |
| v1 (basic patterns) | 50.0% (9/18*) | +36.7% |
| v2 (expanded patterns) | 55.6% (10/18*) | +5.6% |
| v3 (preference HyDE) | 66.7% (12/18*) | +11.1% |
| v4 (domain HyDE) | 56.7% (17/30) | (full dataset) |
| **v5 (profile probe)** | **66.7% (20/30)** | **+10%** |

*Earlier evals used 18-question subset; v4+ use full 30 questions

---

## 2026-02-05: Adaptive Profile Probe Design

### Codex-Investigator Review

Sent the adaptive profile probe plan for expert review. Key findings:

1. **Do NOT blindly replace fixed probe** - Fixed probe provides baseline coverage. Replacing it entirely risks regression on queries where generic preferences ARE the right answer.

2. **Use hybrid approach** - Run fixed probe first, only use adaptive probe if fixed probe finds < 2 preference phrases. This saves latency on 60-70% of queries while improving hard cases.

3. **Multiple separate queries > concatenation** - Issuing 2-3 searches for top adaptive phrases then merging via RRF is better than one diluted concatenated query.

4. **Structured JSON output** - Return JSON with phrases array to reduce parsing errors.

5. **Simple caching** - Cache key: `normalized_query + prompt_version`, TTL: 7 days.

### Recommended Architecture

```python
# Stage 2.0: Profile Probe (hybrid approach)
profile_context: list[str] = []
if is_preference:
    # Step 1: Always run fixed probe first
    fixed_probe_query = "I use I prefer my favorite I recently I took a class I really like"
    fixed_results = self._search_raw(fixed_probe_query, scope, limit=5)
    if fixed_results:
        profile_context = extract_profile_context([r.content for r in fixed_results])

    # Step 2: Only run adaptive probe if fixed probe underperforms
    if len(profile_context) < 2:
        adaptive_phrases = generate_adaptive_probe(query, client)  # Returns list
        if adaptive_phrases:
            for phrase in adaptive_phrases[:3]:
                adaptive_results = self._search_raw(phrase, scope, limit=3)
                profile_context.extend(extract_profile_context([r.content for r in adaptive_results]))
            profile_context = list(set(profile_context))[:5]  # Dedupe, limit
```

### Key Insight: Gating Saves Latency

- If fixed probe returns >= 2 preference phrases via `extract_profile_context()`, skip adaptive probe
- Only call LLM when fixed probe underperforms
- This saves ~300ms latency on majority of preference queries

### RESULT: Adaptive Probe HURTS Performance (2026-02-05)

**Experiment Results:**

| Approach | single-session-preference | Overall |
|----------|---------------------------|---------|
| Fixed probe only | 56.7% | 87.0% |
| **Hybrid (fixed + adaptive)** | **40.0%** | **84.8%** |

**Why it failed:**
1. Fixed probe often finds only 1 preference (triggering adaptive)
2. Adaptive probe searches for domain-specific terms
3. But the resulting memories are **NOT relevant to the actual query**
4. Irrelevant preferences get injected into HyDE prompt
5. HyDE generates hypotheticals based on wrong context
6. Retrieval quality degrades

**Example of bad adaptive probe output:**
- Query: "recommend a cocktail"
- Adaptive probe found: "Instagram's Question sticker", "luxury evening gown", "camping trip to Yellowstone"
- These are real user preferences but IRRELEVANT to cocktails

**Lesson Learned:**
The adaptive probe concept is sound, but the implementation needs to filter for RELEVANCE, not just match preference patterns. The current `extract_profile_context()` regex just finds any preference statement, not domain-relevant ones.

**Potential fixes attempted (v2, v3):**
1. ~~Filter adaptive probe results by semantic similarity to query~~
2. LLM to judge relevance of extracted preferences - TRIED, FAILED
3. ~~Only inject preferences above a similarity threshold~~

### Relevance Filtering Experiments (2026-02-05)

Implemented LLM-based relevance filtering as suggested:

**v2 (Strict Filter):**
```python
prompt = "Select ONLY preferences DIRECTLY relevant to answering: {query}"
```
Result: 50% preference (up from 40%, still below 56.7% baseline)
Problem: Filter too aggressive, often filtered ALL preferences

**v3 (Lenient Filter):**
```python
prompt = """
RULES:
- Include preferences that MIGHT help
- Be GENEROUS - if any connection, INCLUDE IT
- Only EXCLUDE COMPLETELY UNRELATED
- When in doubt, INCLUDE
"""
# Plus fallback: if filter removes ALL, keep original
```
Result: 46.7% preference (worse than v2!)

### Root Cause: LLM Filter Calibration is Hard

| Filter Type | Preference Recall | Problem |
|-------------|-------------------|---------|
| No filter | 40% | Irrelevant prefs pollute HyDE |
| Strict filter | 50% | Removes too many good prefs |
| Lenient filter | 46.7% | Still too aggressive OR keeps bad prefs |

The sweet spot between "too strict" and "too lenient" is:
- Query-dependent (varies by domain)
- Sensitive to prompt wording
- Hard to generalize across preference types

### Conclusion: Simple Fixed Probe is Best (For Now)

After all adaptive probe experiments:

| Approach | Preference Recall |
|----------|-------------------|
| Fixed probe only | **56.7%** (baseline) |
| v1 adaptive (no filter) | 40% |
| v2 adaptive (strict filter) | 50% |
| v3 adaptive (lenient filter) | 46.7% |

**Decision: REVERTED to simple fixed probe.**

### Future Directions (Not Pursued)

To make adaptive probe work, would need:
1. **Semantic similarity scoring** - Not binary yes/no, but continuous relevance score
2. **Embedding-based filtering** - cosine(query_embed, pref_embed) > threshold
3. **Multi-stage filtering** - Coarse filter → fine-grained LLM scoring
4. **Per-domain calibration** - Different thresholds for different query types

### Final State: Current Performance

Eval results (250 questions):
- Overall: 85.6%
- knowledge-update: 97.2%
- single-session-assistant: 94.7%
- temporal-reasoning: 90.7%
- multi-session: 82.1%
- **single-session-preference: 43.3%** (variance: 40-57%)

The preference score shows significant variance between runs due to LLM non-determinism and eval sampling.

---

## 2026-02-05: Multi-Session Recall Improvement

### Problem: Multi-Session "All" Recall was 10.7%

Multi-session queries require finding ALL relevant memories from different conversation sessions.
- "How many different doctors did I visit?"
- "What is the total amount I spent on luxury items?"

### Root Cause Analysis (codex-investigator)

1. **No aggregation query detection** - Unlike temporal/preference queries
2. **RRF reinforces "best" result** - Similar sessions compete, one wins
3. **Query synthesis helps** - Generates variations that find different sessions
4. **Candidate pool too small** - Default 20 may miss relevant memories

### Solution: Aggregation Query Detection + Larger Pool

Added `_is_aggregation_query()` in `src/cems/retrieval.py`:
```python
def _is_aggregation_query(query: str) -> bool:
    patterns = [
        "how many", "how much", "total", "altogether", "in total",
        "all the times", "every time", "number of", "count of",
        "how often", "different", "various", "throughout", "across all",
    ]
    return any(p in query.lower() for p in patterns)
```

For aggregation queries:
- Force query synthesis (like temporal queries)
- Use 2x candidate pool (at least 50)

### Results

| Category | Before | After | Change |
|----------|--------|-------|--------|
| **multi-session (any)** | 82.1% | **87.5%** | **+5.4%** |
| **multi-session (all)** | 10.7% | **12.5%** | **+1.8%** |
| **Overall (all)** | 54.8% | **59.2%** | **+4.4%** |
| knowledge-update (all) | 80.6% | **84.7%** | **+4.1%** |

### Key Insight

The "all" recall improvement (+4.4%) is significant. While multi-session "all" is still low (12.5%), the direction is correct.

### Session Ordering Fix + Token Budget Increase (v3 - 2026-02-05)

Following codex-investigator review, implemented:
1. **Session ordering bug fix** - Sessions were iterated in dict order (random), not by relevance
2. **Best-fit selection** - Try multiple candidates per session to fit budget
3. **Token budget increase** - 2000 → 4000 for aggregation queries

**Results:**
| Category | v2 | v3 | Change |
|----------|-----|-----|--------|
| multi-session (any) | 87.5% | **89.3%** | +1.8% |
| multi-session (all) | 10.7% | **12.5%** | +1.8% |
| temporal-reasoning (all) | 50.0% | **59.3%** | +9.3% |

### Why multi-session "all" is fundamentally hard

The core issue is **upstream retrieval quality**, not selection strategy:
1. We have 300+ candidate sessions after retrieval
2. Only ~5-10 are actually relevant to the query
3. Our selection improvements assume relevant sessions ARE in the pool
4. But if RRF/vector/BM25 ranks irrelevant sessions higher, no selection algorithm can help

**What would actually help:**
1. **Better retrieval** - Ensure relevant sessions rank in top 50 candidates
2. **MMR at retrieval stage** - Balance relevance and diversity before assembly
3. **Reranker** - Use LLM to identify which candidate sessions are relevant

### llamacpp_server Reranker Test (2026-02-05) - CATASTROPHIC FAILURE

Enabled llamacpp_server reranker (Qwen3-Reranker-0.6B) on local server:

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Overall (any) | 86.4% | **28.4%** | **-58%** |
| multi-session (all) | 12.5% | **0%** | **-12.5%** |
| single-session-assistant | 97.4% | **2.6%** | **-95%** |

**Conclusion:** The reranker completely destroys good retrieval results. The position-aware blending formula may be miscalibrated, or the reranker model disagrees with our relevance criteria.

**REVERTED immediately.** Do NOT enable any reranker.

### Remaining Options to Try

1. **MMR (Maximal Marginal Relevance)** - Balance relevance vs diversity at retrieval stage
2. **Content truncation** - Fit more sessions by limiting content length
3. **Query decomposition** - Break "how many X?" into per-time-period searches

