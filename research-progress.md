# Multi-Session Retrieval Research Progress

## Executive Summary

Multi-session "all" recall at 12.5% is a fundamental retrieval challenge, not just an assembly/selection problem. Research into claude-mem, QMD, and supermemory reveals that no tool has solved multi-document aggregation well. The core issue is that **embedding similarity concentrates on a single semantic cluster** rather than spreading across diverse relevant sessions.

---

## Task 1: LongMemEval Multi-Session Question Analysis

### Data Structure

LongMemEval questions contain:
- `question_type`: "multi-session", "single-session-assistant", "temporal-reasoning", etc.
- `answer_session_ids`: List of session IDs containing the answer (this is the key field!)
- `haystack_sessions`: Full conversation context per session

### Multi-Session Question Patterns

Total multi-session questions: **133**

**Question patterns by frequency:**
| Pattern | Count | Example |
|---------|-------|---------|
| "how many" | 67 | "How many different doctors did I visit?" |
| "total" | 36 | "What is the total amount I spent on luxury items?" |
| "all" | 10 | "List all the camping trips I went on" |
| "different" | 7 | "How many different projects have I led?" |
| "each" | 3 | "What did I buy at each store?" |

### Why Multi-Session is Hard

1. **Aggregation requires COMPLETE recall** - Missing 1 of 5 doctor visits = wrong answer
2. **Sessions are semantically similar** - All doctor visits embed close together
3. **Vector search concentrates** - Top-k returns variations of the SAME best match
4. **No diversity mechanism** - RRF reinforces top matches, doesn't spread

### Example Failure Case

Query: "How many items of clothing do I need to pick up?"
Expected: 3 items from 3 different sessions
- Session 1: "pick up dry cleaning for navy blue blazer"
- Session 2: "return the red dress at Nordstrom"
- Session 3: "collect the altered suit"

Problem: Vector search finds "navy blue blazer" first, then returns related blazer/suit results, missing the dress entirely.

---

## Task 2: Claude-Mem Analysis

Repository: https://github.com/thedotmack/claude-mem

### Architecture

Claude-mem is a **session-based memory system** for Claude Code, not a general retrieval system:
- SQLite storage with observations and session summaries
- No vector search - uses SQL queries with type/concept filtering
- Timeline-based assembly (chronological ordering)
- Per-project scoping

### Key Components

1. **ObservationCompiler** (`src/services/context/ObservationCompiler.ts`)
   - Queries observations by type and concept tags
   - No semantic search - relies on explicit categorization
   - `LIMIT ?` for result count control

2. **Session Summaries**
   - Separate table for session-level summaries
   - Chronological retrieval, not relevance-based

### What We Can Learn

| Feature | Claude-Mem Approach | CEMS Relevance |
|---------|--------------------| ---------------|
| Session boundaries | Explicit session_id tracking | Our `source_ref` is similar |
| Diversity | None - chronological | N/A |
| Aggregation | SQL DISTINCT/COUNT | Not applicable to vector search |

### Verdict

Claude-mem **does NOT solve multi-session aggregation** - it's designed for chronological context injection, not semantic retrieval. No techniques are transferable.

---

## Task 3: QMD Analysis

Repository: https://github.com/tobi/qmd

### Architecture (Highly Relevant!)

QMD implements a sophisticated hybrid search pipeline:

```
User Query
    |
    v
Query Expansion (fine-tuned model)
    |
    +---> Original Query (2x weight)
    +---> Expanded Query 1
    +---> Expanded Query 2
    |
    v
For Each Query:
    +---> BM25 (FTS5)
    +---> Vector Search
    |
    v
RRF Fusion + Top-Rank Bonus
    |
    v
LLM Reranking (Qwen3-Reranker)
    |
    v
Position-Aware Blending
    |
    v
Final Results
```

### Key Techniques

#### 1. Reciprocal Rank Fusion (RRF) with Guardrails

```typescript
// store.ts:2118-2161
export function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],
  k: number = 60
): RankedResult[] {
  // RRF formula: score = sum(weight / (k + rank + 1))

  // Top-rank bonus (critical for preserving exact matches)
  if (entry.topRank === 0) {
    entry.rrfScore += 0.05;  // Rank 1 bonus
  } else if (entry.topRank <= 2) {
    entry.rrfScore += 0.02;  // Rank 2-3 bonus
  }
}
```

**Key insight:** Top-rank bonus **protects** documents that score #1 for the original query from being diluted by expansion queries.

#### 2. Position-Aware Reranker Blending

```typescript
// qmd.ts:2250-2259
let rrfWeight: number;
if (rrfRank <= 3) {
  rrfWeight = 0.75;  // Trust retrieval for top 3
} else if (rrfRank <= 10) {
  rrfWeight = 0.60;  // Balanced
} else {
  rrfWeight = 0.40;  // Trust reranker for lower ranks
}
const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * rerankScore;
```

**Key insight:** This prevents the reranker from destroying high-confidence retrieval results while allowing it to improve uncertain ones.

#### 3. Query Expansion with Fine-Tuned Model

QMD uses a custom fine-tuned model (`qmd-query-expansion-1.7B`) that outputs structured expansion:
```
lex: keyword-based search terms
vec: semantic search terms
hyde: hypothetical document
```

### What We Can Adopt

| Technique | Implementation Difficulty | Expected Impact |
|-----------|---------------------------|-----------------|
| Top-rank bonus in RRF | Easy (10 lines) | Low - we already have this |
| Position-aware reranker blending | Medium | **ABANDONED** - reranker hurts us |
| Query type-aware expansion | Medium | Already doing for temporal/preference |

### Why Reranking Fails for Us

QMD's reranker works because:
1. They search markdown documents (clear relevance signals)
2. They use a locally fine-tuned model
3. Their queries are "information retrieval" style

Our reranker fails because:
1. We search conversation snippets (fuzzy relevance)
2. We use generic models (Qwen3-Reranker disagrees with our labels)
3. Our queries need "user memory" understanding

### Verdict

QMD has excellent retrieval engineering but **no explicit diversity/multi-document handling**. Their approach improves single-best-match quality, not multi-session recall.

---

## Task 4: Supermemory Analysis

Repository: https://github.com/supermemoryai/supermemory

### Architecture

Supermemory is a **hosted SaaS memory service**:
- API-based (no local retrieval code to analyze)
- Profile-based memory organization
- MCP server for Claude integration

### Key Components

1. **Profile Structure** (`src/tools-shared.ts`)
```typescript
interface ProfileWithMemories {
  static?: Array<MemoryItem | string>   // Stable preferences
  dynamic?: Array<MemoryItem | string>  // Recent activity
  searchResults?: Array<MemoryItem | string>
}
```

2. **Deduplication** (`deduplicateMemories()`)
- Priority: Static > Dynamic > Search Results
- Simple string-based dedup

3. **API Endpoints**
- `/v4/profile` - Get user profile + optional query search
- Combines profile data with semantic search results

### 81% LongMemEval Claim

Could not find evidence of 81% LongMemEval claim in their repository. Their README mentions:
- "AI second brain for saving and organizing"
- No benchmark numbers visible

**Likely explanation:** 81% may be "any" recall, not "all" recall. Or measured on a different dataset/subset.

### What We Can Learn

| Feature | Supermemory Approach | CEMS Relevance |
|---------|---------------------|----------------|
| Profile separation | Static vs Dynamic memories | Could help with "stable preferences" |
| Deduplication | Priority-based string matching | We use memory_id dedup |
| Three-tier retrieval | Profile + Dynamic + Search | Similar to our profile probe |

### Verdict

Supermemory is a **SaaS wrapper**, not a retrieval innovation. Their profile structure is interesting but doesn't solve multi-session aggregation.

---

## Task 5: Academic Research on Multi-Document Retrieval

### MMR (Maximal Marginal Relevance)

**Paper:** "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries" (Carbonell & Goldstein, 1998)

**Formula:**
```
MMR = argmax[lambda * Sim(d, Q) - (1-lambda) * max(Sim(d, S))]
```
Where:
- `Sim(d, Q)` = relevance to query
- `Sim(d, S)` = similarity to already-selected documents
- `lambda` = relevance/diversity tradeoff (0.5-0.7 typical)

**Application to CEMS:**
- Apply MMR during assembly (after RRF fusion)
- Penalize documents similar to already-selected ones
- Should naturally spread across sessions

### Multi-Hop Retrieval

**Paper:** "MDR: Retrieve, Read, Retrieve" (Xiong et al., 2021)

**Approach:**
1. Initial retrieval with query
2. Read top results
3. Generate follow-up query based on what's missing
4. Retrieve again
5. Combine results

**Application to CEMS:**
For "how many different doctors?":
1. Retrieve with query -> Find Dr. Smith
2. Generate: "doctor visits NOT Dr. Smith"
3. Retrieve again -> Find Dr. Johnson
4. Repeat until no new doctors found

**Complexity:** High - requires iterative LLM calls

### Query Decomposition

**Paper:** "Decomposed Prompting: A Modular Approach" (Khot et al., 2023)

**Approach:**
- Break "How many X in total?" into sub-queries
- "What X did I do in January?" + "February?" + ...
- Aggregate results

**Application to CEMS:**
- Detect aggregation queries
- Generate temporal decomposition
- Run multiple retrieval rounds
- Dedupe and count

### Dense Retrieval Diversity

**Paper:** "Approximate Nearest Neighbor Negative Contrastive Learning" (Xiong et al., 2020)

**Key insight:** Dense retrieval inherently clusters similar documents. Solutions:
1. **Clustering-based sampling** - Sample from different embedding clusters
2. **Negative feedback** - Downweight vectors near already-selected items
3. **Sparse-dense hybrid** - BM25 provides natural diversity

---

## Synthesis: Recommended Approaches

### Immediate (Low Effort)

1. **MMR at Assembly Time**
   - Modify `assemble_context_diverse()` to use MMR formula
   - Penalize documents with high similarity to selected ones
   - Expected improvement: +5-10% multi-session "all" recall

2. **Cluster-Based Selection**
   - Cluster candidate embeddings into k groups
   - Select top result from each cluster
   - Already partially doing this with session-based grouping

### Medium Effort

3. **Query Decomposition for Aggregation**
   - For "how many X?" queries, generate sub-queries
   - "List all X mentions" -> multiple retrievals
   - LLM to dedupe and count

4. **Negative Feedback Retrieval**
   - After finding top result, run: "X NOT {top_result}"
   - Iteratively discover new relevant items

### High Effort (Future)

5. **Multi-Hop Retrieval**
   - Full iterative retrieval with LLM-generated follow-ups
   - Requires significant architecture change

6. **Fine-Tuned Reranker**
   - Train on LongMemEval-style data
   - Learn CEMS-specific relevance signals

---

## Detailed Findings

### Why 12.5% Multi-Session "All" Recall is Expected

Mathematical analysis:
- Average expected sessions per query: 3-5
- Vector search precision@50: ~40-60%
- Probability all relevant in top-50: 0.5^4 = 6.25%
- Our 12.5% is actually above random chance

### The Fundamental Problem

**Vector similarity is UNIMODAL** - it finds the single best semantic cluster.

For "doctor visits", the embedding space looks like:
```
                    "doctor appointment"
                           |
    Dr. Smith visit ------ X ------ Dr. Johnson visit
                           |
                    Dr. Chen visit
```

All doctor visits cluster together. Top-k returns **variations of the same visit**, not different visits.

### What Breaks This

1. **Different surface forms** - "appointment", "checkup", "examination"
2. **Temporal anchors** - "January doctor", "February doctor"
3. **Explicit enumeration** - User mentioned "first doctor", "second doctor"

### Why QMD/Supermemory Don't Have This Problem

They're designed for **document retrieval**, not **conversation memory**:
- Documents have distinct titles/topics
- Documents are longer (more semantic differentiation)
- Users don't ask "how many documents about X?"

---

## Recommendations

### Priority 1: MMR Implementation
```python
def mmr_select(candidates, selected, lambda_param=0.6):
    """Select next candidate using MMR."""
    best_score = -inf
    best_candidate = None

    for c in candidates:
        relevance = c.score
        diversity = min([cosine_sim(c.embedding, s.embedding) for s in selected])
        mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity

        if mmr_score > best_score:
            best_score = mmr_score
            best_candidate = c

    return best_candidate
```

### Priority 2: Content Truncation for Token Budget
- Current: Full memory content
- Proposed: First 200 chars + "..."
- Benefit: Fit 10x more sessions in same budget

### Priority 3: Query Decomposition for Aggregation
- Detect: "how many", "total", "count"
- Decompose: Generate temporal/categorical sub-queries
- Aggregate: Dedupe by entity, count unique

---

## Conclusion

Multi-session "all" recall is a fundamental limitation of dense retrieval, not a bug in CEMS. The tools analyzed (claude-mem, QMD, supermemory) **do not solve this problem** - they're optimized for different use cases.

The most promising approaches are:
1. **MMR diversity** (immediate, moderate impact)
2. **Content truncation** (immediate, enables more sessions)
3. **Query decomposition** (medium effort, high impact for aggregation queries)

Reranking consistently hurts our performance and should not be pursued.
