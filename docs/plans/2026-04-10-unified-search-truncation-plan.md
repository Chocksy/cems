# Unified Search Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agentic search path return truncated snippets (like the non-agentic path already does), exclude entity pages from the memory pool, and fix mode propagation in the MCP HTTP wrapper.

**Architecture:** Server-side truncation via the existing `_make_snippet()` function applied to agentic search output. Entity pages excluded from `_load_context_memories()` since they're already handled by the entity picker agent. MCP HTTP wrapper reads `CEMS_SEARCH_MODE` from env. CLI removes client-side truncation hack.

**Tech Stack:** Python (agentic search, CLI), TypeScript (MCP wrapper), pytest

---

### File Map

| File | Responsibility | Change |
|------|---------------|--------|
| `src/cems/agentic/search.py` | Agentic search pipeline | Exclude entity-page from memory buckets, truncate output |
| `tests/test_agentic_search.py` | Agentic search tests | Add entity-page exclusion + truncation tests |
| `hooks/cems_user_prompts_submit.py` | Hook consumer formatting | Add `/recall` hint for truncated memories |
| `mcp-wrapper/src/index.ts` | MCP HTTP wrapper | Pass `mode` from env |
| `src/cems/commands/memory.py` | CLI consumer | Remove `[:120]` hack, show truncation indicator, fix dead code |
| `src/cems/memory/retrieval.py` | Non-agentic retrieval | Add `created_at` to `_serialize_results` |

---

### Task 1: Exclude entity pages from agentic memory buckets

**Files:**
- Modify: `src/cems/agentic/search.py:394-448`
- Test: `tests/test_agentic_search.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_agentic_search.py`, add to `TestLoadContextMemories`:

```python
@pytest.mark.asyncio
async def test_excludes_entity_pages_from_buckets(self):
    """Entity-page documents should be filtered out — they're handled by entity picker."""
    from cems.agentic.search import PROFILE_CATEGORIES, _load_context_memories

    entity_doc = {
        "id": "entity-1", "content": "# Big Entity Page\nLots of content...",
        "category": "entity-page",
        "source_ref": "project:chocksy/pos", "created_at": "2026-03-22",
    }
    regular_doc = {
        "id": "regular-1", "content": "Regular memory",
        "category": "general",
        "source_ref": "project:chocksy/pos", "created_at": "2026-03-22",
    }

    # Bucket 1 returns both entity-page and regular docs
    responses = [[entity_doc, regular_doc]]
    for _ in PROFILE_CATEGORIES:
        responses.append([])
    responses.append([])  # recent
    mock_store = self._make_mock_store(responses)

    result = await _load_context_memories(mock_store, user_id="u1", project="chocksy/pos")

    ids = [d["id"] for d in result]
    assert "entity-1" not in ids, "Entity-page docs should be excluded from memory buckets"
    assert "regular-1" in ids, "Regular docs should still be included"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py::TestLoadContextMemories::test_excludes_entity_pages_from_buckets -v`
Expected: FAIL — entity-1 is in the result

- [ ] **Step 3: Add entity-page filter to `_load_context_memories`**

In `src/cems/agentic/search.py`, add a constant near line 323:

```python
# Categories handled separately (entity picker) — exclude from memory buckets
EXCLUDED_MEMORY_CATEGORIES = {"entity-page"}
```

Then modify `_load_context_memories`. After bucket 1 loading (line ~406), filter before sorting:

```python
        project_docs = [d for d in project_docs if d.get("category") not in EXCLUDED_MEMORY_CATEGORIES]
        project_docs.sort(key=_relevance_score, reverse=True)
```

In bucket 3, add the filter inside the existing loop (line ~437-445):

```python
    for d in recent_docs:
        src = d.get("source_ref") or ""
        cat = d.get("category") or ""
        if cat in EXCLUDED_MEMORY_CATEGORIES:
            continue
        if project and src and f"project:{project}".lower() not in src.lower():
            if cat not in PROFILE_CATEGORIES:
                continue
        recent_filtered.append(d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py::TestLoadContextMemories::test_excludes_entity_pages_from_buckets -v`
Expected: PASS

- [ ] **Step 5: Run full agentic test suite**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/cems/agentic/search.py tests/test_agentic_search.py
git commit -m "fix: exclude entity-page docs from agentic memory buckets"
```

---

### Task 2: Truncate memory content in agentic search response

**Files:**
- Modify: `src/cems/agentic/search.py:610-626`
- Test: `tests/test_agentic_search.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_agentic_search.py`, add to `TestAgenticSearchAsync`:

```python
@pytest.mark.asyncio
@patch("cems.agentic.search._load_entity_summaries", new_callable=AsyncMock, return_value=[])
@patch("cems.agentic.search.get_client")
async def test_memory_content_is_truncated(self, mock_get_client, _mock_entities):
    """Agentic search should return snippet content, not full documents."""
    from cems.agentic.search import agentic_search_async

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.complete.return_value = '["abc12345"]'

    long_content = "This is a long memory. " * 100  # ~2300 chars, well over 500

    mock_store = AsyncMock()
    mock_store.get_all_documents = AsyncMock(return_value=[
        {
            "id": "abc12345-full-uuid-here",
            "content": long_content,
            "category": "general",
            "source_ref": "project:test",
            "created_at": "2026-03-22",
            "scope": "personal",
            "tags": [],
        }
    ])

    result = await agentic_search_async(
        document_store=mock_store,
        user_id="user-1",
        query="What is this about?",
        project="test",
    )

    assert len(result["memories"]) >= 1
    mem = result["memories"][0]
    assert len(mem["content"]) <= 600, f"Content should be truncated, got {len(mem['content'])} chars"
    assert mem.get("truncated") is True
    assert mem.get("full_length") == len(long_content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py::TestAgenticSearchAsync::test_memory_content_is_truncated -v`
Expected: FAIL — content is full length, no `truncated` key

- [ ] **Step 3: Apply `_make_snippet` to agentic output**

In `src/cems/agentic/search.py`, add the import near the top (after existing imports, around line 22):

```python
from cems.memory.retrieval import _make_snippet
```

Then modify the RRF merge loop (lines ~610-626). Replace the entry construction:

```python
        for i, short_id in enumerate(merged_short_ids):
            mem = mem_by_id.get(short_id, {})
            score = 1.0 - (i * 0.5 / max(len(merged_short_ids), 1))
            raw_content = mem.get("content", "")
            snippet, truncated = _make_snippet(raw_content)
            entry = {
                "memory_id": str(mem.get("id", "")),
                "content": snippet,
                "category": mem.get("category", ""),
                "scope": mem.get("scope", "personal"),
                "source_ref": mem.get("source_ref", ""),
                "tags": mem.get("tags", []),
                "score": round(score, 3),
                "created_at": str(mem.get("created_at", "")),
            }
            if truncated:
                entry["truncated"] = True
                entry["full_length"] = len(raw_content)
            elif mem.get("content_detailed"):
                entry["has_detailed"] = True
                entry["full_length"] = len(mem["content_detailed"])
            top_memories.append(entry)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py::TestAgenticSearchAsync::test_memory_content_is_truncated -v`
Expected: PASS

- [ ] **Step 5: Also verify short content is NOT truncated**

Add a test that confirms short memories pass through unchanged:

```python
@pytest.mark.asyncio
@patch("cems.agentic.search._load_entity_summaries", new_callable=AsyncMock, return_value=[])
@patch("cems.agentic.search.get_client")
async def test_short_memory_not_truncated(self, mock_get_client, _mock_entities):
    """Short memories should not be marked as truncated."""
    from cems.agentic.search import agentic_search_async

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.complete.return_value = '["abc12345"]'

    mock_store = AsyncMock()
    mock_store.get_all_documents = AsyncMock(return_value=[
        {
            "id": "abc12345-full-uuid-here",
            "content": "Short memory content",
            "category": "general",
            "source_ref": "project:test",
            "created_at": "2026-03-22",
            "scope": "personal",
            "tags": [],
        }
    ])

    result = await agentic_search_async(
        document_store=mock_store,
        user_id="user-1",
        query="Short query",
        project="test",
    )

    if result["memories"]:
        mem = result["memories"][0]
        assert mem["content"] == "Short memory content"
        assert "truncated" not in mem or mem.get("truncated") is False
```

- [ ] **Step 6: Run full agentic test suite**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/cems/agentic/search.py tests/test_agentic_search.py
git commit -m "fix: truncate memory content in agentic search via _make_snippet"
```

---

### Task 3: Fix agentic response metadata bugs

**Files:**
- Modify: `src/cems/agentic/search.py:628-647`
- Test: `tests/test_agentic_search.py`

- [ ] **Step 1: Write the failing test for `count` including entities**

```python
@pytest.mark.asyncio
@patch("cems.agentic.search.get_client")
async def test_count_includes_entities(self, mock_get_client):
    """count field should include both entities and memories."""
    from cems.agentic.search import agentic_search_async

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.complete.side_effect = [
        '["ent12345"]',
        '["mem12345"]',
        '["mem12345"]',
        '["mem12345"]',
    ]

    entity_summaries = [
        {"id": "ent12345-full-uuid", "title": "Test Entity",
         "summary": "Summary", "sources": "5"},
    ]

    mock_store = AsyncMock()
    mock_store.get_all_documents = AsyncMock(return_value=[
        {
            "id": "mem12345-full-uuid",
            "content": "Memory content",
            "category": "general",
            "source_ref": "project:test",
            "created_at": "2026-03-22",
            "scope": "personal",
            "tags": [],
        }
    ])

    with patch("cems.agentic.search._load_entity_summaries",
                new_callable=AsyncMock, return_value=entity_summaries):
        result = await agentic_search_async(
            document_store=mock_store,
            user_id="user-1",
            query="test",
            project="test",
        )

    # count should include both entities and memories
    assert result["count"] == len(result["entities"]) + len(result["memories"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py::TestAgenticSearchAsync::test_count_includes_entities -v`
Expected: FAIL — count only includes memories

- [ ] **Step 3: Fix count and queries_used in the return dict**

In `src/cems/agentic/search.py`, modify the return dict (lines ~636-647):

```python
    return {
        "entities": top_entities,
        "memories": top_memories,
        "results": top_memories,  # backward compat
        "count": len(top_memories) + len(top_entities),
        "mode": "agentic",
        "tokens_used": 0,
        "queries_used": queries_used,  # int — matches count semantics
        "total_candidates": n_memories,
        "entity_candidates": len(entity_summaries),
        "filtered_count": len(top_memories) + len(top_entities),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_agentic_search.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/cems/agentic/search.py tests/test_agentic_search.py
git commit -m "fix: agentic count includes entities, normalize queries_used type"
```

---

### Task 4: Add `/recall` hint for truncated memories in hook

**Files:**
- Modify: `hooks/cems_user_prompts_submit.py:124-139`
- Modify: `src/cems/data/claude/hooks/cems_user_prompts_submit.py` (bundled copy — keep in sync)

- [ ] **Step 1: Modify `_format_agentic_response` to add recall hint for truncated memories**

In `hooks/cems_user_prompts_submit.py`, modify the memories formatting loop (around line 124-139). Replace the memory formatting block:

```python
    if memories:
        parts.append("RELEVANT MEMORIES:")
        for i, m in enumerate(memories, 1):
            category = m.get("category", "general")
            content = m.get("content", m.get("memory", ""))
            mem_id = m.get("memory_id", m.get("id", ""))
            short_id = mem_id[:8] if mem_id else ""
            score = m.get("score", 0.0)
            truncated = m.get("truncated", False)
            full_len = m.get("full_length", 0)
            suffix = f" [truncated — full doc: {full_len} chars]" if truncated else ""
            parts.append(f"{i}. [{category}] (score: {score:.2f}) {content}{suffix} (id: {short_id})")
            if truncated:
                parts.append(f"   Use /recall {short_id} to read the full document.")
            truncation_flags.append(truncated)
            if mem_id:
                memory_ids.append(mem_id)
            score_details.append({"id": short_id, "score": round(score, 3), "category": category, "content": content})
```

The key addition is the `if truncated:` line that appends a `/recall` hint per truncated memory.

- [ ] **Step 2: Copy the change to the bundled hook**

Copy the same change to `src/cems/data/claude/hooks/cems_user_prompts_submit.py` to keep the bundled version in sync.

- [ ] **Step 3: Verify by running hook tests**

Run: `.venv/bin/python3 -m pytest tests/test_hooks.py -v -x`
Expected: All pass (no existing tests for `_format_agentic_response`, so nothing breaks)

- [ ] **Step 4: Commit**

```bash
git add hooks/cems_user_prompts_submit.py src/cems/data/claude/hooks/cems_user_prompts_submit.py
git commit -m "feat: add /recall hint for truncated memories in hook output"
```

---

### Task 5: MCP HTTP wrapper — pass mode from env

**Files:**
- Modify: `mcp-wrapper/src/index.ts:134-153`

- [ ] **Step 1: Read `CEMS_SEARCH_MODE` and pass it through**

In `mcp-wrapper/src/index.ts`, add near line 19 (after the existing env vars):

```typescript
const CEMS_SEARCH_MODE = process.env.CEMS_SEARCH_MODE || "";
```

Then in the search tool handler (around line 143), add mode to the payload:

```typescript
          body: JSON.stringify({
            query: args.query,
            limit: args.max_results,
            scope: args.scope,
            max_tokens: args.max_tokens,
            enable_graph: args.enable_graph,
            enable_query_synthesis: args.enable_query_synthesis,
            raw: args.raw,
            project: args.project,
            ...(CEMS_SEARCH_MODE && { mode: CEMS_SEARCH_MODE }),
          }),
```

- [ ] **Step 2: Verify MCP wrapper builds**

Run: `cd mcp-wrapper && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add mcp-wrapper/src/index.ts
git commit -m "fix: MCP HTTP wrapper passes CEMS_SEARCH_MODE to search API"
```

---

### Task 6: CLI — remove client-side truncation hack, fix dead code

**Files:**
- Modify: `src/cems/commands/memory.py:147-180`

- [ ] **Step 1: Remove the `[:120]` hack and fix mode check**

In `src/cems/commands/memory.py`, replace the results display section (lines ~147-180):

```python
            for r in results:
                content = r.get("content", r.get("memory", ""))
                truncated = r.get("truncated", False)
                if verbose:
                    display_content = content
                else:
                    display_content = content
                    if truncated:
                        display_content += " [truncated]"
                table.add_row(
                    (r.get("memory_id") or r.get("id", "?"))[:12],
                    display_content,
                    f"{r.get('score', 0):.3f}",
                    r.get("scope", "?"),
                )

            console.print(table)

            # Show pipeline stats
            if result_mode == "agentic":
                console.print(
                    f"[dim]Agentic: {result.get('total_candidates', '?')} memories + "
                    f"{result.get('entity_candidates', '?')} entity pages | "
                    f"{len(entities)} topics + {len(results)} memories returned[/dim]"
                )
            elif result_mode != "raw":
                console.print(
                    f"[dim]Pipeline: {result.get('total_candidates', '?')} candidates → "
                    f"{result.get('filtered_count', '?')} after filtering → "
                    f"{len(results)} returned | "
                    f"Tokens: {result.get('tokens_used', '?')} | "
                    f"Queries: {result.get('queries_used', '?')}[/dim]"
                )
```

Key changes:
- Removed `content[:120] + "..."` hack — server sends truncated content
- Added `[truncated]` indicator when server marks content as truncated
- Fixed pipeline stats: changed `result_mode == "unified"` to `result_mode != "raw"` (the API never returns mode "unified")
- Changed `len(result.get('queries_used', []))` to `result.get('queries_used', '?')` — `queries_used` is now always an int

- [ ] **Step 2: Run CLI tests if they exist**

Run: `.venv/bin/python3 -m pytest tests/ -k "memory" -v -x`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/cems/commands/memory.py
git commit -m "fix: CLI uses server snippets, remove client-side truncation hack"
```

---

### Task 7: Add `created_at` to non-agentic `_serialize_results`

**Files:**
- Modify: `src/cems/memory/retrieval.py:62-82`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_retrieval.py`, find the `TestSnippetTruncation` class and add:

```python
def test_serialize_includes_created_at(self):
    """_serialize_results should include created_at from metadata."""
    from cems.memory.retrieval import _serialize_results
    from cems.models import MemoryMetadata, MemoryScope, SearchResult
    from datetime import datetime, UTC

    ts = datetime(2026, 3, 22, 12, 0, 0, tzinfo=UTC)
    result = SearchResult(
        memory_id="test-id",
        content="Short content",
        score=0.9,
        scope=MemoryScope.PERSONAL,
        metadata=MemoryMetadata(
            category="general",
            source_ref="project:test",
            tags=[],
            created_at=ts,
        ),
    )

    serialized = _serialize_results([result])
    assert len(serialized) == 1
    assert "created_at" in serialized[0]
    assert "2026" in serialized[0]["created_at"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_retrieval.py::TestSnippetTruncation::test_serialize_includes_created_at -v`
Expected: FAIL — `created_at` not in serialized dict

- [ ] **Step 3: Add `created_at` to `_serialize_results`**

In `src/cems/memory/retrieval.py`, modify `_serialize_results` (line ~67-75):

```python
        entry: dict[str, Any] = {
            "memory_id": r.memory_id,
            "content": snippet,
            "score": r.score,
            "scope": r.scope.value,
            "category": r.metadata.category if r.metadata else None,
            "source_ref": r.metadata.source_ref if r.metadata else None,
            "tags": r.metadata.tags if r.metadata else [],
            "created_at": str(r.metadata.created_at) if r.metadata and r.metadata.created_at else None,
        }
```

- [ ] **Step 4: Check MemoryMetadata has `created_at`**

Run: `grep -n "created_at" src/cems/models.py | head -5`

If `MemoryMetadata` doesn't have `created_at`, the test needs adjustment to use `last_accessed` instead. Check and adapt.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/test_retrieval.py::TestSnippetTruncation::test_serialize_includes_created_at -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/cems/memory/retrieval.py tests/test_retrieval.py
git commit -m "fix: include created_at in non-agentic search results"
```

---

### Task 8: Full test suite verification

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python3 -m pytest tests/ -x -q`
Expected: All ~730+ tests pass

- [ ] **Step 2: Run integration test if Docker is running**

Run: `/opt/homebrew/bin/python3.12 test_integration.py`
Expected: All 20 integration tests pass

- [ ] **Step 3: Manual smoke test with CLI**

Run: `cems search "fiscal printer" --mode agentic`
Expected:
- Knowledge Topics table shows entity pages with short summaries
- Relevant Memories table shows truncated snippets, NOT full entity pages
- `[truncated]` indicator appears for long memories

- [ ] **Step 4: Final commit (if any test fixups needed)**

```bash
git add -A
git commit -m "test: fix any test adjustments from unified truncation changes"
```
