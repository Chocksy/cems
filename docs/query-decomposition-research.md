# Multi-Topic Query Decomposition for CEMS

> Research compiled 2026-02-22. Three-agent parallel investigation.

## Problem Statement

When a user writes a long prompt with multiple distinct topics, CEMS embeds the
ENTIRE prompt as one vector. This dilutes all topic signals — a query about
"Docker config + TypeScript patterns + Coolify setup" produces a single
embedding that captures none of those topics well.

The old `synthesize_query()` did **synonym expansion** ("generate 2-3 queries
about the EXACT SAME TOPIC"), which made results **worse** by adding noise
without covering new topics.

We need **decomposition** — splitting multi-topic prompts into focused
sub-queries, one per distinct topic.

## Current Pipeline Analysis

### Flow (async path — production)

```
API handler (memory.py:277)
  → Truncate query to 2000 chars (line 322)
  → Query type detection: temporal/preference/aggregation (retrieval.py:79-173)
  → Query synthesis: DISABLED (config.py:175, handler line 329)
    - Only force-enables for temporal/preference/aggregation
    - General prompts: single query, no expansion
  → HyDE: disabled by default
  → Batch embed all queries (single API call)
  → Sequential vector + lexical search per query (for loop with await)
  → RRF fusion (if multiple result lists)
  → Reranking, filtering, scoring, assembly
```

### The Gap

| Query Type | Synthesis | Coverage |
|-----------|-----------|----------|
| Temporal ("when did X happen") | Force-enabled, 3-4 expansions | Good |
| Preference ("recommend X") | Force-enabled, 4-5 expansions | Good |
| Aggregation ("how many X") | Force-enabled, 3-4 expansions | Good |
| **General multi-topic** | **Disabled** | **Single diluted vector** |

A prompt like *"fix Docker port binding, also what was that TypeScript pattern,
and remind me about Coolify"* matches NONE of the three detectors. It gets one
embedding, one search, diluted results.

## State of the Art

### Approaches Evaluated

| System | Strategy | Relevant? |
|--------|----------|-----------|
| LangChain MultiQueryRetriever | Paraphrase same query 3 ways | No — synonym expansion, not decomposition |
| LlamaIndex SubQuestionQueryEngine | Decompose into sub-questions + tool routing | Partially — tool routing irrelevant, decomposition pattern useful |
| Haystack Query Decomposition | LLM decompose with passthrough for simple queries | **Yes** — best prompt design pattern |
| Microsoft GraphRAG | Community-based map-reduce, not decomposition | No — different problem |
| CompactRAG (2025) | 2 fixed LLM calls, dependency-ordered DAG | Interesting but over-engineered for our case |
| Tip-of-the-Tongue (EMNLP 2023) | Clue extraction for memory recall | **Yes** — closest to our domain |
| DSPy | Typed signatures, optimizable prompts | Useful pattern for future |
| Mem0 | No decomposition, relies on structured metadata | Shows good metadata reduces need |

### Key Insights

1. **Haystack's passthrough pattern** is critical: few-shot examples showing
   when NOT to decompose prevents wasted LLM calls
2. **Tip-of-the-Tongue** research confirms that memory recall benefits from
   topic-level decomposition (6% Recall@5 improvement)
3. **LangChain's approach is what we already tried** (synonym expansion) and it
   made things worse — validates our diagnosis
4. **CompactRAG** shows bounded-cost is achievable (2 LLM calls max)
5. **All production systems use LLM-based decomposition**, not rule-based —
   rules can't handle the nuance of "is this 1 topic or 3?"

## Proposed Solution

### Architecture: Heuristic Gate + LLM Decomposition

```
User prompt arrives
  ↓
[Stage 0] Heuristic gate (NO LLM call, <1ms)
  - Count topic-shift signals: "also", "and also", "by the way",
    "another thing", "separately", "remind me about"
  - Count question marks, sentence boundaries
  - If clearly single-topic → skip decomposition
  - If possibly multi-topic → proceed to LLM
  ↓
[Stage 1] LLM decomposition (80-120ms via Cerebras/Groq)
  - Extract 1-3 focused sub-queries
  - If returns 1 → single-topic confirmed, proceed normally
  - If returns 2-3 → multi-topic, use sub-queries
  ↓
[Stage 2] Existing pipeline continues
  - queries_to_search = [original] + sub_queries (if decomposed)
  - Skip synthesize_query (decomposed sub-queries are already focused)
  - Batch embed all queries in one API call
  - Search per query, RRF fusion, filter, assemble
```

### Zero-Cost for Simple Queries

The heuristic gate catches the common case (single-topic prompts like "fix the
login bug") with zero LLM overhead. The LLM decomposer only runs when
multi-topic signals are detected.

### The Decomposition Prompt

```python
DECOMPOSE_PROMPT = """Extract distinct search topics from this user prompt.
Return focused search queries for a memory system.

RULES:
- 1 query for single-topic prompts (most prompts are single-topic)
- 2-3 queries ONLY for genuinely DIFFERENT subjects
- Strip filler ("I need help with", "can you remind me about")
- Preserve specifics: tool names, versions, project names, error messages
- Do NOT add new terms or synonyms — only extract what's stated
- Multiple questions about ONE subject = 1 query
- "Docker with nginx and postgres" = 1 topic (one setup)
- "Fix printer. Also monitor is broken" = 2 topics (different devices)

Prompt: {query}

Return ONLY: {{"queries": ["query1", ...]}}"""
```

**Configuration:**
- Temperature: 0.0 (pure extraction, zero creativity)
- Max tokens: 150
- Model: `qwen/qwen3-32b` via Cerebras/Groq (`fast_route=True`)
- Expected latency: ~80-120ms
- Cost: ~$0.000072 per call (<$0.10/day at 1000 queries)

### Test Cases

| Input | Expected Output | Reasoning |
|-------|----------------|-----------|
| "I need help with Docker config. Also, what was that TypeScript pattern we discussed for error handling? And can you remind me about the Coolify setup?" | `["Docker configuration", "TypeScript error handling pattern", "Coolify setup"]` | 3 different tools/topics |
| "what is the proper way to handle big prompts and texts and searches? should we do multiple searches? how does embedding handle long text?" | `["handling big prompts and long text in embedding and search"]` | 1 topic — different angles on same subject |
| "fix the datecs fp-700 printer connection on Windows. also the Elo monitor has washed out colors" | `["datecs fp-700 printer connection Windows", "Elo monitor washed out colors"]` | 2 different hardware devices |
| "fix the bug in the login flow" | `["bug in login flow"]` | Single topic |
| "set up Docker with nginx and postgres" | `["Docker setup with nginx and postgres"]` | 1 topic — one deployment |
| "I've been working on the Docker setup for 3 hours and tried multiple approaches but the port binding keeps failing" | `["Docker port binding failing"]` | 1 topic — context stripped, core problem extracted |

### Fallback

If LLM call fails or returns invalid JSON:
```python
fallback = {"queries": [original_query]}
```
Falls back to current behavior — safe, no regression.

## Integration Plan

### Where in the Pipeline

Insert at `retrieval.py` line ~389 — after query type detection, before profile
probe:

```
Lines 377-388: Query type detection (regex) — UNCHANGED
>>> NEW ~389: Multi-topic decomposition
Lines 390-409: Profile probe (preference) — skip if decomposed
Lines 411-444: Query synthesis — SKIP if decomposed
Lines 446-458: HyDE — runs on ORIGINAL query only, unchanged
Lines 460-466: Batch embed — unchanged (handles any number of queries)
Lines 480-506: Search loop — use decomposition weights
Lines 538+: RRF, rerank, filter, assembly — all unchanged
```

### Interaction with Existing Stages

| Stage | If Decomposed | If Not Decomposed |
|-------|---------------|-------------------|
| Profile probe | Skip (not preference) | Runs as usual |
| synthesize_query | **Skip** (sub-queries already focused) | Runs as usual |
| HyDE | Runs on original query only | Runs as usual |
| Batch embed | Embeds all sub-queries + original + HyDE | Unchanged |
| RRF weights | Sub-queries get 1.5x weight | Unchanged |
| Everything else | Unchanged | Unchanged |

### RRF Weight Scheme

```python
# Existing
rrf_original_weight = 2.0    # Original query
rrf_expansion_weight = 1.0   # Synonym expansions

# New
rrf_decomposition_weight = 1.5  # Decomposed sub-queries
```

Sub-queries are more trustworthy than LLM-generated synonym expansions (they're
reformulations of what the user stated), but the original query should still
dominate (captures cross-topic connections).

### Config Flags

Add to `CEMSConfig`:

```python
enable_query_decomposition: bool = Field(
    default=True,
    description="Decompose multi-topic queries into focused sub-queries",
)
max_decomposed_queries: int = Field(
    default=3,
    description="Maximum sub-queries from decomposition (plus original)",
)
rrf_decomposition_weight: float = Field(
    default=1.5,
    description="RRF weight for decomposed sub-queries",
)
```

API handler accepts: `enable_decomposition = body.get("enable_decomposition", True)`

### Files to Modify

1. **`src/cems/retrieval.py`** — Add `decompose_query()` function + heuristic gate
2. **`src/cems/memory/retrieval.py`** — Insert decomposition stage in both sync/async
3. **`src/cems/config.py`** — Add 3 config flags
4. **`src/cems/api/handlers/memory.py`** — Accept `enable_decomposition` parameter

### Qwen 3 Thinking Tags

Qwen 3-32B may output `<think>...</think>` before JSON. Add stripping to the
JSON parser:

```python
response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip()
```

## Testing Strategy

### Integration Test for `test_integration.py`

```python
def test_multi_topic_decomposition() -> tuple[bool, str]:
    """Test that multi-topic queries decompose and find results for each topic."""
    # 1. Add 2 memories about different topics
    # 2. Search with compound query mentioning both
    # 3. Verify results include matches from BOTH topics
    # 4. Check queries_used in response shows decomposition happened
    # 5. Clean up
```

### Unit Tests

- Heuristic gate: single-topic prompts return False, multi-topic return True
- Decomposition prompt: returns correct sub-queries for test cases
- Fallback: invalid JSON returns original query
- Integration: decomposed queries skip synthesis, non-decomposed use synthesis

### Eval Benchmark

Run LongMemEval after implementation:
```bash
python -m cems.eval.longmemeval --questions 50 --api-url http://localhost:8765
```
Baseline: 98% Recall@5. Must not regress.

## Open Questions

1. **Should decomposition weight be equal to original (2.0) or lower (1.5)?**
   Pipeline analyst suggests 1.5, prompt designer suggests 2.0. Needs
   benchmarking.

2. **Should we parallelize DB searches with `asyncio.gather()`?**
   Currently sequential. With decomposition adding more queries, parallelization
   helps. But v1 can keep sequential and optimize later.

3. **Should temporal/preference/aggregation queries also benefit from
   decomposition?** A prompt could be "when did I set up Docker (temporal) and
   also what printer do I use (preference)". Currently these are mutually
   exclusive type detections.

4. **Should we re-enable synthesize_query for single-topic prompts?** It was
   disabled because synonym expansion hurt results. But with decomposition
   handling multi-topic cases, synthesis only handles single-topic — maybe the
   noise was from multi-topic prompts being synonym-expanded instead of
   decomposed?

5. **Heuristic gate accuracy** — how often will it miss multi-topic prompts or
   false-positive on single-topic? Needs real-world testing with the debug
   dashboard.
