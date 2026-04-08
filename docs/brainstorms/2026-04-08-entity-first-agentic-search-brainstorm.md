# Entity-First Agentic Search

**Date**: 2026-04-08
**Status**: Brainstorm
**Previous**: [Knowledge Engine Brainstorm](2026-04-06-knowledge-engine-brainstorm.md), [Next Steps](../plans/2026-04-06-knowledge-engine-next-steps.md)

## What We're Building

Redesign agentic search mode so **entity pages are the primary knowledge interface**
instead of raw memory snippets. When agentic mode is used (hook, MCP, CLI), the system
returns a structured response: a **topic index** of relevant entity pages + a small
number of individual memories. Claude is prompted to `/recall` entity pages for details
instead of getting answers inlined.

This is NOT an addition to current behavior — it **replaces** the agentic mode output
format. Vector search mode stays unchanged.

## Why This Approach

### The Problem with Current Agentic Mode

Current agentic search loads ALL raw memories (~3,700 docs, up to 700K chars) into
3 LLM agent contexts. This is:
- **Expensive** — 3 LLM calls on 700K chars each
- **Noisy** — agents wade through thousands of individual memories
- **Flat** — no concept hierarchy, just raw snippets
- **Wasteful** — the same knowledge is often spread across many memories

### Why Entity Pages Fix This

Entity pages are pre-synthesized knowledge. A single entity page about "Stripe Integration"
contains the synthesized understanding from 20 individual memories. Instead of the LLM
reading 20 raw snippets about Stripe, it sees one title + summary line and drills deeper
only if needed.

This follows Karpathy's pattern: **index → page → sources**. The wiki IS the knowledge;
individual memories are just raw material underneath.

### Why Not Entity-Only (No Individual Memories)

~1,600 memories are "orphans" — not part of any entity page cluster. They still have
value. Showing 3 individual memories alongside entity topics ensures orphan knowledge
isn't lost.

## Key Decisions

### 1. Agentic mode only — vector search unchanged

- **Vector mode** (default, hook): stays as-is, shows up to 5 memory snippets
- **Agentic mode** (hook with `CEMS_SEARCH_MODE=agentic`, MCP, CLI): entity-first output
- Vector mode is fast, cheap, good for most prompts
- Agentic mode is for when you want the "full knowledge engine" experience

### 2. Output format: separate topic index + memories

```
<memory-recall>
KNOWLEDGE TOPICS matching your query:

1. Stripe Integration (20 sources)
   Webhook-based payment flow with sampled API requests
   → /recall a8f3b201 for full details

2. Docker Deployment (15 sources)
   CI builds amd64 images, never build locally on Mac
   → /recall c4e91bf2 for full details

RELEVANT MEMORIES:
1. [preferences] Keep Stripe requests sampled...
2. [guidelines] Never manually push Docker...
3. [session-summary] Fixed webhook retry logic...
</memory-recall>
```

- Max **3 entity topics** + max **3 individual memories** (down from 5)
- Clear visual separation between topics and memories
- Entity topics include source count and `/recall` instruction
- Promotes drill-down behavior — Claude must read entity pages for detail

### 3. Separate entity picker agent (4 parallel LLM calls)

```
Parallel execution:

[Entity Picker]      → top 3 entities
  Input: ~100 entity summaries (50K chars)
  Cost: ~$0.005
  Time: ~1s

[Direct Seeker]      → top 10 memories
[Inference Engine]   → top 10 memories
[Temporal Navigator] → top 10 memories
  Input: ~3,700 memories (700K chars)
  RRF fusion → top 3 memories
```

- Entity picker is a NEW 4th agent, lightweight, runs in parallel
- Entities can't get drowned out (100 items in separate context, not 100 in 3,800)
- Memory agents stay unchanged — same 3-agent RRF architecture
- Entity picker uses same `gemini-2.5-flash-lite` model (cheap, 1M context)

### 4. Server-side structured response

The `/api/memory/search` endpoint returns structured response when `mode=agentic`:

```json
{
  "entities": [
    {"id": "a8f3b201", "title": "Stripe Integration",
     "summary": "Webhook-based payment flow...", "sources": 20}
  ],
  "memories": [
    {"id": "d4f12a33", "content": "Keep Stripe requests sampled...",
     "category": "preferences", "score": 0.92}
  ],
  "mode": "agentic",
  "total_candidates": 3800,
  "entity_candidates": 126
}
```

- One API response format, consumed by hook, MCP, and CLI
- Each client formats the structured response into its output format
- Hook → `<memory-recall>` XML
- MCP → tool response content
- CLI → terminal output

### 5. Consistent behavior across all clients

Same search API, same response format. The hook, MCP `memory_search` tool, and
`cems search` CLI all call `/api/memory/search` with `mode=agentic` and format the
same structured response. No special-casing per client.

## Architecture: How It Works

```
User types prompt
    │
    ▼
Hook / MCP / CLI calls POST /api/memory/search {mode: "agentic"}
    │
    ▼
Server: agentic_search_async()
    │
    ├──── [Entity Picker Agent]     ← NEW
    │       Input: entity summaries from /api/wiki/index
    │       Output: top 3 entity IDs
    │
    ├──── [Direct Seeker]           ← existing
    ├──── [Inference Engine]        ← existing  
    ├──── [Temporal Navigator]      ← existing
    │       Input: all memories (3 buckets)
    │       Output: top 10 IDs each → RRF → top 3
    │
    ▼
Server merges: {entities: [...], memories: [...]}
    │
    ▼
Client formats for output:
  Hook: <memory-recall> XML with topics + memories
  MCP: tool response
  CLI: terminal output
    │
    ▼
Claude sees: "KNOWLEDGE TOPICS: Stripe, Docker..."
             "RELEVANT MEMORIES: ..."
    │
    ▼
Claude decides to /recall entity page for more detail
    │
    ▼
mcp__cems__memory_get returns full entity page
```

## What Changes vs. What Stays

| Component | Change |
|-----------|--------|
| `src/cems/agentic/search.py` | Add entity picker agent, load entity summaries, return structured response |
| `src/cems/api/handlers/memory.py` | Modify agentic response format to include `entities` + `memories` |
| `hooks/cems_user_prompts_submit.py` | Format agentic response into entity-first `<memory-recall>` |
| `src/cems/mcp_wrapper/` | Format agentic response for MCP tool output |
| `src/cems/cli/` | Format agentic response for terminal |
| Vector search mode | **NO CHANGE** |
| Entity page compilation | **NO CHANGE** |
| `/recall` / `memory_get` | **NO CHANGE** — already works for entity pages |
| 3 memory search agents | **NO CHANGE** — same architecture, same prompts |

## Open Questions

1. **Entity picker prompt**: What should the system prompt be? Probably simple:
   "Given this query, which knowledge topics are most relevant? Return entity IDs."
2. **Empty entities**: What if no entity pages exist yet (fresh install)?
   Graceful fallback — skip entity section, show memories only.
3. **Entity picker model**: Same `gemini-2.5-flash-lite` or could we use something
   even cheaper since the context is only ~50K chars?
4. **Should vector mode ALSO show entities?** Decision: no, keep it simple for now.
   If users want entities, they use agentic mode.

## Testing Plan

1. **Unit test**: Entity picker agent returns valid entity IDs from entity summaries
2. **Integration test**: Full agentic search returns structured `{entities, memories}` response
3. **Hook test**: Verify `<memory-recall>` output has correct topic + memory format
4. **End-to-end**: User prompt → hook → agentic search → entity topics shown →
   `/recall` fetches full entity page → Claude uses the knowledge
5. **Edge cases**: No entities exist, query matches no entities, query matches entities
   but no memories, etc.
