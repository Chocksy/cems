# Task Plan: Retrieval Improvements Round 2

## Goal

Improve LongMemEval scores by trying multiple approaches:
1. **llamacpp_server reranker** - Use local llama.cpp server for reranking (not Docker)
2. **Multi-session recall** - Fix the 10.7% "all" recall problem
3. **Relaxed deduplication** - Allow near-duplicates for multi-session queries

## Current Baseline (2026-02-05)

| Category | Recall@5 (any) | Recall@5 (all) |
|----------|----------------|----------------|
| knowledge-update | 97.2% | 80.6% |
| single-session-assistant | 94.4% | 94.4% |
| temporal-reasoning | 90.7% | 61.1% |
| multi-session | 82.1% | **10.7%** |
| single-session-preference | 43.3% | 43.3% |
| **Overall** | **85.6%** | **54.8%** |

## Problem Analysis

### Multi-Session "All" Recall is Very Low (10.7%)
Multi-session questions require retrieving MULTIPLE relevant memories from different sessions.
- "Any" recall (82.1%) is decent - we find at least ONE relevant memory
- "All" recall (10.7%) is terrible - we almost never find ALL relevant memories

**Root causes:**
1. **Deduplication too aggressive** - Similar memories from different sessions get merged
2. **Candidate pool too small** - Need more candidates to find all relevant memories
3. **No special handling** for multi-session query type

### Local Reranker Configuration
The `llamacpp_server` reranker is implemented but:
- Docker version is slow (no GPU)
- Need to point at local llama.cpp server for speed
- Need to benchmark impact on eval scores

## Implementation Phases

| Phase | Task | Status |
|-------|------|--------|
| 0 | Consult codex-investigator for approach review | `complete` |
| 1 | Add `_is_aggregation_query()` detection | `complete` |
| 2 | Force synthesis + larger candidate pool for aggregation | `complete` |
| 3 | Run eval (v2 with proper code) | `complete` |
| 3b | Session ordering fix + token budget increase | `complete` |
| 4 | Implement diversity-aware selection (MMR) | `pending` |
| 5 | Test llamacpp_server reranker with local server | `complete` (**FAILED** - 86%→28%) |
| 6 | Document findings and commit | `in_progress` |

## Latest Results (v3 - Session Ordering + 4000 Token Budget)

| Category | Recall@5 (any) | Recall@5 (all) |
|----------|----------------|----------------|
| knowledge-update | 98.6% | 80.6% |
| single-session-assistant | 97.4% | 97.4% |
| temporal-reasoning | 88.9% | **59.3%** |
| multi-session | **89.3%** | **12.5%** |
| single-session-preference | 33.3% | 33.3% |
| **Overall** | **86.4%** | **57.6%** |

**Changes from v2:**
- multi-session (any): 87.5% → 89.3% (+1.8%)
- multi-session (all): 10.7% → 12.5% (+1.8%)
- temporal-reasoning (all): 50.0% → 59.3% (+9.3%)

**Observation:** Session ordering and larger token budget helped temporal-reasoning significantly but multi-session "all" is still stuck at ~12%. The issue is upstream retrieval quality, not selection strategy.

---

## Phase 0: Codex-Investigator Review ✓ COMPLETE

### Key Findings:

1. **Deduplication is NOT the problem** - memory_id dedup doesn't merge different sessions
2. **API defaults disable query synthesis** - multi-session queries need expansion
3. **RRF reinforces "best" result** - similar sessions compete, one wins, others lost
4. **Token budget exhausts before 5 results** - large documents fill budget
5. **No aggregation query detection exists** - unlike temporal/preference
6. **LLM reranker alone won't help** - optimizes for relevance, not diversity

### Recommended Actions (Priority Order):

1. Add `_is_aggregation_query()` detection (how many, total, all the times)
2. Force query synthesis for aggregation queries
3. Implement MMR (Maximal Marginal Relevance) for diversity
4. Guarantee minimum 5 results by truncating content
5. Increase candidate pool for aggregation queries

---

## Phase 1: llamacpp_server Reranker

### Current Config Options
```python
reranker_backend: Literal["llamacpp_server", "llm", "disabled"] = "disabled"
```

### Local Server Setup
User has local llama.cpp server running for embeddings.
Need to configure reranker to use same server.

### Files to Check/Modify
- `src/cems/config.py` - Reranker config settings
- `src/cems/llamacpp_server.py` - Client implementation
- `src/cems/memory/retrieval.py` - Reranker integration

---

## Phase 2: Relaxed Deduplication

### Current Deduplication Logic
```python
def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: dict[str, SearchResult] = {}
    for result in results:
        if result.memory_id not in seen:
            seen[result.memory_id] = result
        elif result.score > seen[result.memory_id].score:
            seen[result.memory_id] = result
    return list(seen.values())
```

This dedupes by memory_id only - not by content similarity.

### Proposed Enhancement
For multi-session queries, use content-based deduplication with HIGH threshold (0.95+):
- Keep memories that are only slightly different
- Different sessions about same topic should NOT be merged

### Implementation Ideas
1. Add `deduplicate_relaxed()` function with similarity threshold
2. Detect multi-session queries (count/aggregation patterns)
3. Use relaxed dedup only for multi-session

---

## Phase 3: Multi-Session Query Detection

### Patterns That Indicate Multi-Session
- "How many..." (counting across sessions)
- "total", "altogether", "in total"
- "all the times", "every time"
- "across all", "throughout"

### Special Handling
- Larger candidate pool (limit=50 instead of 20)
- Relaxed deduplication
- Skip aggressive score filtering

---

## Success Criteria

- [ ] llamacpp_server reranker working with local server
- [ ] Multi-session "all" recall improves from 10.7% to 20%+
- [ ] No regression on other categories
- [ ] Document what works and what doesn't

---

## Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Try local reranker first | User wants to avoid slow Docker, already has local server | 2026-02-05 |
| Focus on multi-session | 10.7% "all" recall is biggest gap | 2026-02-05 |
| Use planning-with-files | Complex multi-step task needs tracking | 2026-02-05 |
