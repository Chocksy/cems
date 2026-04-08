---
title: "feat: Entity-First Agentic Search"
type: feat
date: 2026-04-08
brainstorm: docs/brainstorms/2026-04-08-entity-first-agentic-search-brainstorm.md
---

# Entity-First Agentic Search

## Overview

Redesign agentic search mode so entity pages are the primary knowledge interface.
When `mode=agentic`, the search API returns a structured response with **separate
`entities` and `memories` arrays**. A new 4th "Entity Picker" agent searches entity
summaries in parallel with the existing 3 memory agents. Output: max 3 entity topics
+ max 3 individual memories. All clients (hook, MCP, CLI) format this consistently.

Vector search mode is unchanged.

## Problem Statement

Current agentic mode loads ALL raw memories (~3,700 docs, up to 700K chars) into
3 LLM agents. Problems:
- **Expensive**: 3 LLM calls on 700K chars each
- **Flat**: no concept hierarchy — just raw snippets in a sea of text
- **Wasteful**: same knowledge spread across many memories; entity pages already
  synthesize this into ~100 topic pages but agentic mode ignores them
- **No drill-down**: dumps answers inline instead of promoting `/recall` for detail

Entity pages exist but aren't surfaced during search. They're only browsable in the
dashboard. This feature connects them to the search pipeline.

## What Changes

### 1. `src/cems/agentic/search.py` — Add Entity Picker agent

**New function: `_load_entity_summaries()`**

Load entity page summaries from the DB (same query as `/api/wiki/index`):

```python
# src/cems/agentic/search.py

async def _load_entity_summaries(
    document_store, user_id: str, project: str | None = None, limit: int = 200,
) -> list[dict]:
    """Load entity page summaries for entity picker agent.

    Returns compact list: id, title, summary, source count.
    Same query as api_wiki_index but returns dicts for internal use.
    """
    pool = await document_store._get_pool()
    user_uuid = UUID(user_id)

    if project:
        rows = await conn.fetch("""
            SELECT id, title, content, source_ref, tags
            FROM memory_documents
            WHERE user_id = $1 AND category = 'entity-page'
              AND deleted_at IS NULL AND source_ref LIKE $2
            ORDER BY COALESCE(shown_count, 0) DESC
            LIMIT $3
        """, user_uuid, f"project:{project}%", limit)
    else:
        rows = await conn.fetch("""
            SELECT id, title, content, source_ref, tags
            FROM memory_documents
            WHERE user_id = $1 AND category = 'entity-page' AND deleted_at IS NULL
            ORDER BY COALESCE(shown_count, 0) DESC
            LIMIT $2
        """, user_uuid, limit)

    entries = []
    for row in rows:
        title = row["title"] or ""
        content = row["content"] or ""
        sentences = content.replace("\n", " ").split(". ")
        summary = ". ".join(sentences[1:4]).strip()
        if len(summary) > 200:
            summary = summary[:200] + "..."

        cluster_size = ""
        for tag in (row["tags"] or []):
            if tag.startswith("cluster-size:"):
                cluster_size = tag.split(":")[1]

        entries.append({
            "id": str(row["id"]),
            "title": title,
            "summary": summary,
            "sources": cluster_size,
        })
    return entries
```

**New agent prompt: `ENTITY_PICKER_PROMPT`**

```python
ENTITY_PICKER_SYSTEM = (
    "You are a Topic Matcher — a knowledge retrieval specialist.\n\n"
    "You receive a list of knowledge topic pages (entity pages). Each has a title "
    "and a brief summary synthesized from multiple source memories.\n\n"
    "Your task: given a user question, identify which topic pages are most likely "
    "to contain relevant knowledge. Return the IDs of the most relevant topics.\n\n"
    "Prefer topics that DIRECTLY address the question over loosely related ones.\n"
    "Return at most 5 IDs, ranked by relevance."
)
```

**New function: `_run_entity_picker()`**

```python
def _run_entity_picker(
    question: str, entities_text: str, n_entities: int, model: str,
    project: str | None = None,
) -> list[str]:
    """Run entity picker agent. Returns list of entity IDs."""
    # Same pattern as _run_single_agent but with ENTITY_PICKER_SYSTEM prompt
    # Uses a simplified user prompt showing entity summaries
```

**Modify `agentic_search_async()`**

Add entity picker as a 4th parallel agent. Restructure return value:

```python
async def agentic_search_async(...) -> dict:
    # 1. Load entity summaries (parallel with memory loading)
    entities_task = _load_entity_summaries(document_store, user_id, project)
    memories_task = _load_context_memories(document_store, user_id, project, scope)
    entity_summaries, memories = await asyncio.gather(entities_task, memories_task)

    # 2. Run 4 agents in parallel
    #    - Entity Picker: sees entity summaries (~50K chars)
    #    - Direct Seeker, Inference Engine, Temporal Navigator: see raw memories (~700K)

    # 3. Post-process:
    #    - Entity picker results → top 3 entity dicts (id, title, summary, sources)
    #    - Memory RRF fusion → top 3 memory dicts (id, content, category, score)

    return {
        "entities": top_entities,    # NEW: list of entity dicts
        "memories": top_memories,    # RENAMED from "results"
        "results": top_memories,     # KEEP for backward compat
        "mode": "agentic",
        "total_candidates": len(memories),
        "entity_candidates": len(entity_summaries),
        ...
    }
```

### 2. `src/cems/api/handlers/memory.py:377-391` — Structured agentic response

Currently:
```python
if mode == "agentic":
    result = await agentic_search_async(...)
    return JSONResponse({"success": True, **result})
```

After (minimal change — `agentic_search_async` already returns the right shape):
```python
if mode == "agentic":
    result = await agentic_search_async(...)
    logger.info(
        f"[API] Agentic search: {result.get('count', 0)} memories, "
        f"{len(result.get('entities', []))} entities from "
        f"{result.get('entity_candidates', 0)} candidates"
    )
    return JSONResponse({"success": True, **result})
```

No structural change needed — the handler already passes through the dict.
The new `entities` field flows naturally. Clients that don't know about it ignore it.

### 3. `hooks/cems_user_prompts_submit.py` — Entity-first formatting

The hook currently calls `/api/memory/search` without a mode. When agentic mode is
configured (via `CEMS_SEARCH_MODE=agentic` in credentials), it needs to format the
response differently.

**Modify `search_cems()` to detect and format agentic responses:**

```python
def search_cems(client, query, project=None):
    """Search CEMS for relevant memories."""
    payload = {"query": query, "scope": "both", "limit": 5}
    if project:
        payload["project"] = project

    # Pass search mode from credentials if configured
    search_mode = client.get_search_mode()  # NEW method on CEMSClient
    if search_mode:
        payload["mode"] = search_mode

    response = client.post("/api/memory/search", json=payload)
    data = response.json()

    # If agentic mode returned entities, format differently
    if data.get("entities"):
        return _format_agentic_response(data), memory_ids, truncation_flags, score_details
    else:
        # Existing vector search formatting (unchanged)
        return _format_vector_response(data), memory_ids, truncation_flags, score_details
```

**New function: `_format_agentic_response()`**

```python
def _format_agentic_response(data: dict) -> str:
    """Format agentic search results with entity-first layout."""
    parts = []
    entities = data.get("entities", [])
    memories = data.get("memories", data.get("results", []))[:3]

    if entities:
        parts.append("KNOWLEDGE TOPICS matching your query:\n")
        for i, e in enumerate(entities, 1):
            sources = f" ({e['sources']} sources)" if e.get("sources") else ""
            parts.append(f"{i}. {e['title']}{sources}")
            if e.get("summary"):
                parts.append(f"   {e['summary']}")
            parts.append(f"   → /recall {e['id'][:8]} for full details")
        parts.append("")

    if memories:
        parts.append("RELEVANT MEMORIES:")
        for i, m in enumerate(memories, 1):
            category = m.get("category", "general")
            content = m.get("content", "")
            mid = m.get("memory_id", m.get("id", ""))[:8]
            parts.append(f"{i}. [{category}] {content} (id: {mid})")

    return "\n".join(parts)
```

**Also update**: `src/cems/data/claude/hooks/cems_user_prompts_submit.py` (bundled copy).

**Also update**: `hooks/utils/credentials.py` — add `CEMS_SEARCH_MODE` to the
credentials resolver so the hook can read it.

### 4. `src/cems/mcp_stdio.py:85-111` — MCP tool formatting

The MCP `memory_search` tool already passes `SEARCH_MODE` from credentials.
It currently returns raw JSON. Modify to format entity-first when agentic:

```python
@mcp.tool()
def memory_search(query, scope="both", max_results=10, ...):
    payload = {"query": query, "scope": scope, "limit": max_results, ...}
    search_mode = SEARCH_MODE
    if search_mode:
        payload["mode"] = search_mode

    result = _request("POST", "/api/memory/search", payload)

    # Format entity-first for agentic mode
    if result.get("entities"):
        return _format_agentic_for_mcp(result)
    return json.dumps(result)
```

**New function: `_format_agentic_for_mcp()`**

```python
def _format_agentic_for_mcp(data: dict) -> str:
    """Format agentic results as structured text for MCP tool response."""
    # Same format as hook: KNOWLEDGE TOPICS + RELEVANT MEMORIES
    # Use /recall instruction so Claude knows to drill deeper
    parts = []
    # ... same formatting logic as hook's _format_agentic_response()
    return "\n".join(parts)
```

### 5. `src/cems/client.py:180-220` — Add `mode` parameter

The `CEMSClient.search()` method currently doesn't accept a `mode` parameter.
Add it so CLI and other programmatic clients can pass it:

```python
def search(
    self, query, limit=10, scope="both", max_tokens=2000,
    enable_graph=True, enable_query_synthesis=True, raw=False,
    mode=None,  # NEW
):
    payload = {
        "query": query, "limit": limit, "scope": scope,
        "max_tokens": max_tokens, "enable_graph": enable_graph,
        "enable_query_synthesis": enable_query_synthesis, "raw": raw,
    }
    if mode:
        payload["mode"] = mode
    return self._request("POST", "/api/memory/search", json=payload)
```

### 6. `src/cems/commands/memory.py:60-135` — CLI entity-first output

Add `--mode` option and format entity-first output:

```python
@click.command()
@click.argument("query")
@click.option("--mode", "-m", default=None, help="Search mode: vector, agentic")
# ... existing options ...
def search(ctx, query, scope, limit, ..., mode):
    client = get_client(ctx)
    result = client.search(query, ..., mode=mode)

    entities = result.get("entities", [])
    results = result.get("results", [])

    if entities:
        # Print entity topics table
        entity_table = Table(title="Knowledge Topics")
        entity_table.add_column("ID", style="dim", max_width=12)
        entity_table.add_column("Topic", style="bold cyan")
        entity_table.add_column("Sources", style="yellow")
        entity_table.add_column("Summary", style="white")
        for e in entities:
            entity_table.add_row(e["id"][:12], e["title"], ...)
        console.print(entity_table)

    if results:
        # Existing memory table (unchanged)
        ...
```

### 7. `hooks/utils/credentials.py` — Expose search mode

Add `CEMS_SEARCH_MODE` to the credential resolver so the hook's `CEMSClient`
can access it:

```python
class CEMSClient:
    def __init__(self, api_url, api_key, search_mode=None, ...):
        self.search_mode = search_mode
        ...

    def get_search_mode(self):
        return self.search_mode

    @classmethod
    def from_cwd(cls, cwd):
        creds = resolve_credentials(cwd)
        return cls(
            api_url=creds.get("CEMS_API_URL"),
            api_key=creds.get("CEMS_API_KEY"),
            search_mode=creds.get("CEMS_SEARCH_MODE"),
        )
```

## Files Summary

| File | Action | Lines Changed |
|------|--------|--------------|
| `src/cems/agentic/search.py` | Modify | ~80 new (entity picker agent, loader, prompt) |
| `src/cems/api/handlers/memory.py` | Modify | ~5 (logging update) |
| `hooks/cems_user_prompts_submit.py` | Modify | ~40 (search mode pass-through, agentic formatting) |
| `src/cems/data/claude/hooks/cems_user_prompts_submit.py` | Modify | Same as above (bundled copy) |
| `hooks/utils/credentials.py` | Modify | ~10 (search_mode on CEMSClient) |
| `src/cems/mcp_stdio.py` | Modify | ~20 (agentic formatting) |
| `src/cems/client.py` | Modify | ~5 (add mode param) |
| `src/cems/commands/memory.py` | Modify | ~25 (--mode flag, entity table) |

## Acceptance Criteria

### Functional Requirements

- [x] `POST /api/memory/search {mode: "agentic"}` returns `{entities: [...], memories: [...]}`
- [x] Entity picker agent runs in parallel with 3 memory agents
- [x] Entity picker only sees entity summaries (~100 docs, ~50K chars)
- [x] Max 3 entities + max 3 memories in agentic response
- [x] Hook formats agentic response as "KNOWLEDGE TOPICS" + "RELEVANT MEMORIES"
- [x] MCP `memory_search` tool formats same way when `CEMS_SEARCH_MODE=agentic`
- [x] CLI `cems search` accepts `--mode agentic` and shows entity table
- [x] Vector search mode completely unchanged (no regressions)
- [x] `/recall <entity-id>` returns full entity page content (already works)
- [x] Graceful fallback when no entity pages exist (skip entity section)

### Non-Functional Requirements

- [ ] Entity picker LLM call < 2 seconds (small context)
- [ ] Total agentic search < 10 seconds (same as current, entity picker runs in parallel)
- [x] All existing tests pass (712 passed, 0 failed)

### Quality Gates

- [ ] No changes to vector search code path
- [ ] Entity picker prompt tested with real entity summaries
- [ ] Hook output matches the format from brainstorm exactly
- [ ] Bundled hook copy matches installed hook copy

## Edge Cases

1. **No entity pages exist** (fresh install): Entity picker returns empty, response has
   `entities: []`. Clients skip the KNOWLEDGE TOPICS section, show memories only.
2. **Query matches entities but no memories**: Show entities only, memories section empty.
3. **Query matches memories but no entities**: Show memories only, topics section empty.
4. **Entity picker agent fails/times out**: Gracefully degrade — return memories only.
   Same pattern as existing agent failure handling.
5. **All 4 agents fail**: Return empty results (existing behavior).

## Testing Plan

1. **Unit test entity picker**: Mock entity summaries, verify agent returns valid IDs
2. **Unit test `_load_entity_summaries()`**: Verify correct SQL and formatting
3. **Unit test response structure**: `agentic_search_async()` returns `{entities, memories}`
4. **Integration test**: Full agentic search against Docker with real entity pages
5. **Hook test**: Verify `<memory-recall>` output has KNOWLEDGE TOPICS + MEMORIES format
6. **CLI test**: Verify `--mode agentic` shows entity table
7. **Backward compat**: Existing tests pass, `results` key still present in response

## References

- Brainstorm: `docs/brainstorms/2026-04-08-entity-first-agentic-search-brainstorm.md`
- Knowledge Engine brainstorm: `docs/brainstorms/2026-04-06-knowledge-engine-brainstorm.md`
- Next steps plan: `docs/plans/2026-04-06-knowledge-engine-next-steps.md`
- Agentic search: `src/cems/agentic/search.py`
- Wiki index endpoint: `src/cems/api/handlers/wiki.py:29-110`
- Search API handler: `src/cems/api/handlers/memory.py:315-444`
- Hook: `hooks/cems_user_prompts_submit.py`
- MCP stdio: `src/cems/mcp_stdio.py:85-111`
- CLI search: `src/cems/commands/memory.py:60-135`
- Client: `src/cems/client.py:180-220`
