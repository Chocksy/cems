# Multi-Session Recall Research

## Problem Statement

CEMS achieves only **12.5% "all" recall** on multi-session queries in LongMemEval benchmark.
These are queries like "How many different doctors did I visit?" that require finding memories
from MULTIPLE different sessions to answer correctly.

## Current State

- multi-session (any): 87.5% - we find AT LEAST ONE relevant memory
- multi-session (all): 12.5% - we rarely find ALL relevant memories
- Gap indicates: retrieval finds one good result but misses the others

## Root Cause (Discovered)

**Vector similarity is UNIMODAL** - it finds the single best semantic cluster.

For "doctor visits", all doctor appointments embed close together in vector space.
Top-k returns **variations of the SAME visit**, not different visits:
- Dr. Smith visit (score 0.92)
- Dr. Smith followup (score 0.91)
- Dr. Smith prescription (score 0.89)
- ... misses Dr. Johnson, Dr. Chen entirely

This is a **fundamental limitation of dense retrieval**, not a CEMS bug.

## Key Findings from Tool Analysis

| Tool | Solves Multi-Session? | Notes |
|------|----------------------|-------|
| **claude-mem** | ❌ No | Uses SQL/chronological ordering, no semantic search |
| **QMD** | ❌ No | Excellent retrieval engineering but no diversity mechanism |
| **Supermemory** | ❌ No | SaaS wrapper, 81% claim likely "any" not "all" recall |

**None of these tools solve multi-session aggregation.** They're optimized for different use cases (document retrieval, not conversation memory).

## Why 12.5% is Actually Above Random

Mathematical analysis:
- Average expected sessions per query: 3-5
- Vector search precision@50: ~40-60%
- Probability all relevant in top-50: 0.5^4 = **6.25%**
- Our 12.5% is **2x above random chance**

## Proposed Solutions (Priority Order)

### 1. MMR (Maximal Marginal Relevance) - Immediate
Penalize documents similar to already-selected ones:
```
MMR = λ * relevance - (1-λ) * max_similarity_to_selected
```
Expected: +5-10% multi-session "all" recall

### 2. Content Truncation - Immediate
Truncate memories to 200 chars to fit 10x more sessions in token budget.

### 3. Query Decomposition - Medium Effort
For "how many X?", generate sub-queries:
- "doctor visits in January"
- "doctor visits in February"
- Aggregate and dedupe

### 4. Iterative Negative Feedback - Medium Effort
After finding Dr. Smith, search: "doctor visits NOT Dr. Smith"
Repeat until no new doctors found.

## NOT Recommended

- **Reranking** - Catastrophically hurts performance (-58%)
- **Fine-tuned models** - High effort, unclear benefit for conversation memory

## References

- MMR: Carbonell & Goldstein, 1998
- Multi-hop: Xiong et al., 2021 (MDR)
- Query Decomposition: Khot et al., 2023

See `research-progress.md` for detailed analysis.
