# Unified Search Truncation + Entity Page Separation

**Date**: 2026-04-10
**Status**: Approved
**Problem**: Entity pages leak into memory results as full multi-KB content, and truncation behavior is inconsistent across the three search consumers (hook, CLI, MCP).

## Context

The agentic search path has two bugs:

1. **Double-inclusion**: `_load_context_memories()` loads ALL categories including `entity-page` into the raw memory pool. Entity pages already have a dedicated path via `_load_entity_summaries()` + entity picker agent. Result: the same entity page appears in both the KNOWLEDGE TOPICS section (title + summary) and the RELEVANT MEMORIES section (full content).

2. **No truncation**: The non-agentic path uses `_make_snippet()` (500 chars) and marks `truncated=True`. The agentic path has no equivalent — full document content passes through to consumers.

Combined effect: Claude sees full entity page content inline in `<memory-recall>`, making the "REQUIRED: /recall for details" instruction pointless. Data shows 0% follow-through on 55 entity suggestions in a single day.

### Consumer Inconsistencies

| Consumer | Truncation | Mode Propagation |
|----------|-----------|-----------------|
| Hook | None — dumps full content | Reads `CEMS_SEARCH_MODE` from credentials |
| CLI | Client-side `[:120]` hack | Only via explicit `--mode` flag |
| MCP | None — passes raw JSON | Does NOT pass `mode` at all |
| Server (non-agentic) | `_make_snippet()` at 500 chars | N/A |
| Server (agentic) | None | N/A |

## Design

### Principle

The search API response is a **preview layer**. Full content always requires `GET /api/memory/get`. All consumers are thin formatters of the same server response. Truncation is a server concern.

This mirrors the Karpathy pattern: see document titles → decide to read → read full content.

### Server Changes

#### 1. Exclude entity pages from memory buckets (`agentic/search.py`)

In `_load_context_memories()`, filter out `category='entity-page'` documents from all three buckets (project, profile, recent). Entity pages are already loaded by `_load_entity_summaries()` for the entity picker agent.

```python
# In _load_context_memories, after loading each bucket:
# Filter out entity-page docs — they're handled by entity picker
docs = [d for d in docs if d.get("category") != "entity-page"]
```

This is applied to:
- Bucket 1 (project memories) — after `get_all_documents` call
- Bucket 3 (recent memories) — after `recent_filtered` construction
- Bucket 2 (profile) already filters by specific categories, so no entity pages there

#### 2. Truncate memory content in agentic response (`agentic/search.py`)

After RRF merge picks top memories, apply `_make_snippet()` to each memory's content before returning. Reuse the existing function from `memory/retrieval.py`.

```python
from cems.memory.retrieval import _make_snippet

# In the RRF merge loop (after line ~610):
snippet, truncated = _make_snippet(mem.get("content", ""))
entry = {
    "memory_id": str(mem.get("id", "")),
    "content": snippet,  # Was: mem.get("content", "")
    "category": mem.get("category", ""),
    # ... rest unchanged
}
if truncated:
    entry["truncated"] = True
    entry["full_length"] = len(mem.get("content", ""))
```

#### 3. Entity topics remain metadata-only

The entity topics response stays as `{id, title, summary, sources}` — no content field. This is the pointer that tells Claude what exists. Full content requires `/recall`.

### Consumer Changes

#### Hook (`hooks/cems_user_prompts_submit.py`)

- `_format_agentic_response()` already handles `truncated` flag and `full_length` (lines 131-134) — this code path will now actually trigger since the server sends truncated content
- KNOWLEDGE TOPICS section keeps imperative `/recall` instructions (already implemented in commit 6daf644)
- Add `/recall <id>` hint after truncated memories

#### MCP Wrapper (`mcp-wrapper/src/index.ts`)

- Read `CEMS_SEARCH_MODE` from environment variable
- Pass as `mode` field in the API payload
- No other changes — MCP is already a thin pass-through

#### CLI (`src/cems/commands/memory.py`)

- Remove client-side `content[:120] + "..."` hack (line ~152)
- Use server-provided snippet content directly
- Show `[truncated]` indicator when `truncated=True`
- `--verbose` mode: call `GET /api/memory/get` per result for full content (future enhancement, not in scope)

### Instructive Language

The hook output for agentic search results:

```
KNOWLEDGE TOPICS matching your query:

1. Fiscal Printer Integration (15 sources)
   The core focus is replacing fragmented patterns with unified device_actions...
   → /recall 4fbe2d9d for full details

REQUIRED: Fetch these knowledge pages before responding:
  /recall 4fbe2d9d
These are curated documents selected for your task. Read them first.

RELEVANT MEMORIES:
1. [general] (score: 0.83) PostHog: identifyUser updated to set display... [truncated — 2,400 chars] (id: 4bc620fd)
```

The SessionStart profile already instructs Claude (line ~723 in memory.py):
> "When `<memory-recall>` includes KNOWLEDGE TOPICS with `/recall` commands, you MUST fetch those pages before responding"

### Files Touched

| File | Change |
|------|--------|
| `src/cems/agentic/search.py` | Exclude entity-page from `_load_context_memories`, truncate memory output via `_make_snippet()` |
| `hooks/cems_user_prompts_submit.py` | Add `/recall` hint for truncated memories in `_format_agentic_response()` |
| `mcp-wrapper/src/index.ts` | Read `CEMS_SEARCH_MODE` env var, pass as `mode` in API payload |
| `src/cems/commands/memory.py` | Remove client-side `[:120]` truncation, use server snippets |
| `tests/test_maintenance.py` | May need adjustment if entity-page filtering tests exist |
| `tests/test_hooks.py` | Update assertions for truncated content format |

### What Doesn't Change

- `/api/memory/get` endpoint — returns full content, unchanged
- `/recall` skill — calls `memory_get`, unchanged
- Non-agentic search path — already correct with `_make_snippet()`
- Entity picker agent logic — still searches summaries, returns IDs
- 3 memory search agents — still search full content internally (needed for ranking), output truncated after ranking
- `_make_snippet()` function — reused as-is (500 chars, sentence-boundary aware)

### Additional Fixes (from Codex review)

These are related inconsistencies discovered during code review that should be fixed alongside the main design:

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | `queries_used` type mismatch: int (agentic) vs list[str] (non-agentic) | Medium | Normalize to int count in both paths |
| 2 | CLI pipeline stats check `"unified"` mode — dead code, API never returns "unified" | Low | Change condition to `result_mode not in ("agentic", "raw")` |
| 3 | Agentic `count` excludes entity results | Low | Change to `len(top_memories) + len(top_entities)` |
| 4 | MCP stdio (`mcp_stdio.py`) also needs mode propagation check | Medium | Verify it reads `CEMS_SEARCH_MODE` (Codex found it does — confirm) |
| 5 | Non-agentic results missing `created_at` field in `_serialize_results` | Low | Add `created_at` to serialization |

### Out of Scope (noted for future)

- `enable_query_synthesis` default mismatch (API=False, MCP/CLI=True) — needs design decision on whether hook search should use synthesis
- `enable_hyde` not exposed to MCP/CLI — feature gap, not blocking
- Entity pages appearing in non-agentic vector search — they're valid semantic matches; only the agentic double-inclusion is a bug
- Hook agentic detection via `entities is not None` — code smell but not breaking

### Testing

- Verify agentic search response has truncated memory content
- Verify entity-page docs don't appear in the memories array
- Verify MCP wrapper passes mode through
- Verify CLI displays server snippets without client-side chopping
- Integration test: search with entity pages present, confirm topics section has pointers, memories section has snippets
