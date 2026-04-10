---
title: Session-Enriched Memory Search
type: feat
date: 2026-04-09
---

# Session-Enriched Memory Search

## Overview

When a user types "yes, let's do this" or any short/context-dependent prompt, the memory search has zero context about the ongoing conversation. The search query is just the raw user text, which returns irrelevant results.

The fix: pass `session_id` with every search request. The **server** looks up the latest observer session summary for that session and prepends it to the search query. This enriches the semantic search with conversation context — automatically, for all clients (hooks, MCP, CLI, Codex skills).

## Problem Statement

- Short prompts ("yes", "go ahead", "let's work on the retry logic") lack context for semantic search
- The confirmatory prompt handler reads the last assistant message, but that's a client-side workaround only available in Claude hooks
- Codex `/recall` skill has no session context at all — it just searches the raw query
- MCP `memory_search` tool has no session context

## Proposed Solution

Add `session_id` as an optional parameter to the search pipeline. When present, the server fetches the latest session summary (already stored by the observer daemon) and uses it to enrich the search query.

### Phase 1: Server — Accept and use session_id in search

**File:** `src/cems/api/handlers/memory.py:315-448`

1. Add `session_id` to accepted params (after `mode`, ~line 367):
   ```python
   session_id = body.get("session_id", "")
   ```

2. When `session_id` is present, look up the latest session summary:
   ```python
   # src/cems/api/handlers/memory.py
   session_context = ""
   if session_id:
       session_context = await _get_session_context(memory, session_id)
   ```

3. New helper function `_get_session_context()`:
   ```python
   async def _get_session_context(memory, session_id: str) -> str:
       """Fetch the latest session summary for search enrichment."""
       doc_store = await memory._ensure_document_store()
       short_id = session_id[:12]
       # Try current epoch first, then scan backwards
       # Use tag prefix search: find most recent doc with tag starting with "session:{short_id}"
       doc = await doc_store.find_document_by_tag(
           f"session:{short_id}", user_id=..., category="session-summary"
       )
       if doc:
           content = doc.get("content", "")
           # Extract just the Context line (last line, ~100 chars) for query enrichment
           # Full content could be 3K chars — too much for query enrichment
           for line in content.splitlines():
               if line.startswith("Context:"):
                   return line
           # Fallback: use title
           return doc.get("title", "")[:200]
       return ""
   ```

4. Prepend session context to search query:
   ```python
   if session_context:
       enriched_query = f"{session_context}\n{query}"
   else:
       enriched_query = query
   ```
   
   Use `enriched_query` for vector search. Keep original `query` for logging.

**Key design decision:** Only use the `Context:` line from the session summary (~100-200 chars), not the full 3K content. This is enough to anchor the semantic search without overwhelming the embedding model.

### Phase 2: Hook — Pass session_id in search calls

**File:** `hooks/cems_user_prompts_submit.py:150-167`

1. Add `session_id` param to `search_cems()`:
   ```python
   def search_cems(client, query, project=None, session_id=""):
       payload = {"query": query, "scope": "both", "limit": 5}
       if project:
           payload["project"] = project
       if session_id:
           payload["session_id"] = session_id
   ```

2. Update callers (~line 565 and ~line 610):
   ```python
   memories, memory_ids, _, _ = search_cems(client, query, project=project, session_id=session_id)
   ```

   The `session_id` is already available at line 516: `session_id = input_data.get('session_id', '')`

### Phase 3: MCP — PPID-based session auto-detection

**File:** `src/cems/mcp_stdio.py`

The MCP server auto-detects the current session ID at startup using a **resolver ladder** — no LLM action needed. Each stdio MCP process is spawned per-session, so session_id can be cached once at module init.

**Resolver ladder** (in priority order):

1. **Env override:** `os.environ.get("CEMS_SESSION_ID")` — explicit override for testing/future-proofing
2. **PPID session file (Claude Code):** Read `~/.claude/sessions/{os.getppid()}.json` → extract `sessionId`
   - Verified 100% success rate on 16 Claude Code processes (both CLI and Desktop)
   - Each Claude Code session spawns its own MCP process; PPID maps to the session file
   - Format: `{"pid": N, "sessionId": "uuid", "cwd": "...", "startedAt": N}`
3. **Codex env:** `os.environ.get("CODEX_COMPANION_SESSION_ID")`
4. **Fallback:** `None` — search works normally without enrichment

```python
import json
from pathlib import Path

def _detect_session_id() -> str:
    """Auto-detect current session ID from Claude/Codex session context.
    
    Resolver ladder:
    1. CEMS_SESSION_ID env var (explicit override)
    2. PPID session file (~/.claude/sessions/{ppid}.json)
    3. CODEX_COMPANION_SESSION_ID env var
    """
    # 1. Explicit override
    sid = os.environ.get("CEMS_SESSION_ID", "")
    if sid:
        return sid
    
    # 2. Claude Code PPID → session file
    session_file = Path.home() / ".claude" / "sessions" / f"{os.getppid()}.json"
    try:
        if session_file.exists():
            data = json.loads(session_file.read_text())
            sid = data.get("sessionId", "")
            if sid:
                return sid
    except (json.JSONDecodeError, OSError):
        pass
    
    # 3. Codex companion
    return os.environ.get("CODEX_COMPANION_SESSION_ID", "")

# Cache at module init — stdio MCP = 1 session per process
_SESSION_ID = _detect_session_id()
```

Then in `memory_search`:
```python
def memory_search(query, ...) -> str:
    if _SESSION_ID:
        payload["session_id"] = _SESSION_ID
```

**Cursor (HTTP MCP):** Doesn't have per-session PPID mapping. Phase 2 future work — hook writes workspace→session mapping, or use StreamableHTTP `mcp-session-id` header when Cursor supports it.

**Caveat:** `~/.claude/sessions/{PID}.json` is an internal Claude Code implementation detail, not a public API. Degradation is clean: if file doesn't exist or format changes, resolver returns `None` and search works without enrichment.

### Phase 4: CLI — Same resolver + explicit override

**File:** `src/cems/commands/memory.py:60-183`

Add `--session-id` as explicit override, but auto-detect by default using the same resolver ladder.

### Phase 5: No skill changes needed

Since session_id is auto-detected by the MCP server and CLI, the LLM doesn't need to know about it. Searches automatically get richer context.

## Acceptance Criteria

- [x] `/api/memory/search` accepts optional `session_id` param
- [x] When `session_id` is present, server enriches query with latest session summary context
- [x] Hook passes `session_id` in all search calls
- [x] MCP `memory_search` accepts optional `session_id`
- [x] CLI `cems search` accepts `--session-id` option
- [x] Short prompts like "yes" return relevant memories when session context is available
- [x] No regression: searches without `session_id` work exactly as before
- [x] Tests pass

## Context

- Brainstorm: `docs/brainstorms/2026-04-09-imperative-recall-and-session-enriched-search-brainstorm.md`
- Session summaries stored in `memory_documents` with `category="session-summary"`, tagged `session:{id[:12]}`
- Lookup via `doc_store.find_document_by_tag(tag, user_id, category)` — returns most recent by `updated_at`
- Content format: `[CATEGORY] fact\n...\nContext: overview sentence` (capped at 3K chars)
- `session_id` available in hook via `input_data.get('session_id')`

## References

- Search handler: `src/cems/api/handlers/memory.py:315-448`
- Session summary storage: `src/cems/api/handlers/session.py:90-170`
- Document lookup: `src/cems/db/document_store.py:639` (`find_document_by_tag`)
- Session tag builder: `src/cems/observer/state.py:39-55`
- MCP search tool: `src/cems/mcp_stdio.py:131-163`
- CLI search: `src/cems/commands/memory.py:60-183`
- Hook search: `hooks/cems_user_prompts_submit.py:150-167`
