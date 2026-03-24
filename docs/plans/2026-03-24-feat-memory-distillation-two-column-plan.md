---
title: Memory Distillation — Two-Column Progressive Condensation
type: feat
date: 2026-03-24
---

# Memory Distillation — Two-Column Progressive Condensation

## Overview

Add a `content_detailed` column to `memory_documents`. Nightly distillation condenses `content` to ~500 char observer-style summaries. Original full text lives in `content_detailed`, fetchable on-demand via `/recall`. Agentic search uses condensed content, cutting token costs dramatically (700K → ~100K context).

## Problem Statement

After the cleanup sprint, consolidation creates 3-8K char merged memories. Agentic search loads all of these into a 700K char context window. Most of that text is redundant for retrieval — agents only need key facts to decide relevance. The full text is valuable but only when a user asks for details.

Current state after consolidation: **1,651 memories, 835 never-shown**. The 35 consolidated docs average 5K chars each. A single agentic search loads ~200K chars of context, most of which is wasted tokens.

## Proposed Solution

Every memory gets two versions:
- `content` (~500 chars) — condensed summary, always returned in search/agentic
- `content_detailed` (full original) — fetched on-demand via `/recall <id>`

Nightly `DistillationJob` progressively condenses `content`. Consolidation and category summarization read `content_detailed` when available to avoid summarizing-summaries quality loss.

## Technical Approach

### Phase 1: Schema + Storage Layer

**Migration script: `scripts/migrate_content_detailed.sql`**

```sql
-- Memory distillation: add content_detailed column
-- Run after: migrate_relevance_feedback.sql

ALTER TABLE memory_documents
  ADD COLUMN IF NOT EXISTS content_detailed TEXT;
```

**DocumentStore changes (`src/cems/db/document_store.py`):**

- Add `content_detailed` to `DOCUMENT_COLUMNS` (line 30)
- Add `content_detailed` to `_doc_row_to_dict()` output
- Add `content_detailed` param to `add_document()` INSERT (line 224) — optional, NULL default
- Add `content_detailed` clearing in `update_document()` — set to NULL when content changes (prevents stale data)

**Write path (`src/cems/memory/write.py`):**

- Add `content_detailed: str | None = None` param to `add_async()` (line 86)
- Pass through to `doc_store.add_document()`

### Phase 2: Distillation Job

**New file: `src/cems/maintenance/distillation.py`**

Separate job (not in SummarizationJob — that's already doing 3 things).

```python
# src/cems/maintenance/distillation.py

DISTILLATION_THRESHOLD = 500  # chars — content longer than this gets distilled
DETAILED_CAP = 10_000  # chars — cap content_detailed growth

PROTECTED_CATEGORIES = {
    "gate-rules", "guidelines", "preferences",
    "category-summary", "session-summary",
}

class DistillationJob:
    async def run_async(self) -> dict:
        doc_store = await self.memory._ensure_document_store()
        user_id = self.config.user_id

        all_docs = await doc_store.get_all_documents(user_id, limit=2000, order="asc")

        candidates = [
            d for d in all_docs
            if len(d.get("content", "")) > DISTILLATION_THRESHOLD
            and d.get("category", "general") not in PROTECTED_CATEGORIES
        ]

        distilled = 0
        for doc in candidates:
            original = doc["content"]
            detailed = doc.get("content_detailed") or original

            # Cap content_detailed growth
            if len(detailed) > DETAILED_CAP:
                detailed = await self._condense_detailed(detailed)

            # LLM distillation
            summary = await self._distill(original, doc.get("category"))
            if summary and len(summary) < len(original):
                await doc_store.distill_document(
                    doc["id"], user_id=user_id,
                    content=summary, content_detailed=detailed,
                )
                distilled += 1

        return {"distilled": distilled, "candidates": len(candidates)}
```

**Trigger logic**: `len(content) > 500 AND category NOT IN PROTECTED`. Ignores `content_detailed` state — if content is still long, condense it. This handles both "never distilled" and "distillation failed" cases.

**Distillation prompt** — modeled after `REFLECTOR_SYSTEM_PROMPT`:

```
You are condensing a memory into a terse, fact-dense summary.

Rules:
- Maximum 500 characters
- Preserve exact values: names, numbers, dates, versions, handles
- Use terse language — no filler words
- Frame state changes explicitly ("switched from X to Y")
- Each fact should be a complete standalone sentence
- Preserve project context ("In Chocksy/cems: ...")

Input memory:
{content}

Output: condensed summary (plain text, no JSON, no markdown)
```

**New DocumentStore method: `distill_document()`**

```python
async def distill_document(
    self, document_id: str, user_id: str,
    content: str, content_detailed: str,
) -> bool:
    """Atomically update content + content_detailed in one transaction."""
    # UPDATE memory_documents
    # SET content = $1, content_detailed = $2, content_hash = $3,
    #     content_bytes = $4, updated_at = NOW()
    # WHERE id = $5 AND user_id = $6 AND deleted_at IS NULL
```

This is atomic — both columns update in one SQL statement. No partial state.

**Register in maintenance API**: Add `"distillation"` to the job_type dispatch in `api/handlers/memory.py`.

**Schedule**: Nightly at 3:15 AM (after consolidation at 3:00, before reflection at 3:30).

### Phase 3: Search & Retrieval Updates

**Agentic search (`src/cems/agentic/search.py`):**
- `_format_memories_for_agents()` already uses `mem.get("content")` — post-distillation this is condensed. No change needed.
- Context budget naturally benefits: more memories fit in the 700K budget.

**Retrieval serialization (`src/cems/memory/retrieval.py`):**
- Add `has_detailed: bool` to `_serialize_results()` output when `content_detailed IS NOT NULL`
- Existing `truncated`/`full_length` logic stays for non-distilled memories

**Consolidation quality fix** (`src/cems/maintenance/summarization.py`):
- `_consolidate_never_shown()` line 231: change `d.get("content", "")[:500]` to `d.get("content_detailed", d.get("content", ""))[:500]`
- `_create_category_summary()`: same — prefer `content_detailed` for LLM input
- This prevents summarizing-summaries quality degradation

### Phase 4: /recall + API Updates

**api_memory_get (`src/cems/api/handlers/memory.py` line 775):**
- Return `content_detailed` as a separate field in the response (backwards compatible)
- `/recall` can then show `content_detailed` when present

**Recall skill (`src/cems/data/claude/commands/recall.md`):**
- Update step 5: "For results with `has_detailed: true` OR `truncated: true`, fetch full content via memory_get"
- When displaying memory_get result, prefer `content_detailed` over `content`

**MCP stdio (`src/cems/mcp_stdio.py`):**
- Add `memory_get` tool (currently missing — blocks /recall on stdio transport):

```python
@mcp.tool()
def memory_get(memory_id: str) -> str:
    """Get full document content by ID."""
    return json.dumps(_request("GET", f"/api/memory/get?id={memory_id}"))
```

### Phase 5: /store Big Document Ingest

**Approach: defer to nightly distillation** (simplest, avoids API changes).

Normal `/store` ingestion stores content as-is. The nightly DistillationJob automatically detects content > 500 chars and distills it. This means big docs have full content in `content` for up to 24 hours until the next nightly run.

**Future enhancement** (not in MVP): Add `content_detailed` param to `memory_add` API for immediate two-column storage from `/store`. Skip for now.

### Phase 6: Tests

**New test file: `tests/test_distillation.py`**

- [x] `test_distill_skips_short_content` — content ≤500 chars untouched
- [x] `test_distill_skips_protected_categories` — gate-rules, preferences etc. skipped
- [x] `test_distill_copies_to_content_detailed` — original content preserved
- [x] `test_distill_condenses_content` — content shortened to ~500 chars
- [x] `test_distill_caps_content_detailed` — content_detailed over 10K gets condensed
- [x] `test_update_clears_content_detailed` — updating content NULLs content_detailed
- [x] `test_memory_get_returns_content_detailed` — API includes content_detailed field
- [x] `test_has_detailed_flag_in_search` — search results include has_detailed boolean
- [x] `test_consolidation_reads_content_detailed` — consolidation uses full content not condensed

**Existing test updates:**
- `tests/test_maintenance.py` — update mock docs to include `content_detailed` key
- `tests/test_mcp_stdio.py` — add test for new `memory_get` tool

## Acceptance Criteria

### Functional Requirements
- [x] `content_detailed TEXT` column added to `memory_documents`
- [x] `DistillationJob` condenses content > 500 chars nightly
- [x] Original content preserved in `content_detailed` before condensation
- [x] Protected categories (gate-rules, guidelines, preferences, category-summary, session-summary) are never distilled
- [x] `GET /api/memory/get` returns `content_detailed` field
- [x] Search results include `has_detailed: true` for distilled memories
- [x] `/recall` fetches `content_detailed` when available
- [x] `memory_get` tool added to `mcp_stdio.py`
- [x] Consolidation reads `content_detailed` when available (prevents summarizing-summaries)
- [x] `update_document()` clears `content_detailed` to NULL when content changes
- [x] `content_detailed` growth capped at 10K chars

### Quality Gates
- [x] All existing tests pass (659+)
- [x] New distillation tests pass (9+ new tests)
- [x] Distillation job completes within 60s for batches of 50 docs

## Dependencies & Prerequisites

- Cleanup sprint v0.9.29 (merged) — IDOR fixes, maintenance philosophy, parallel queries
- Migration script must be run on production DB before deploying code
- Docker image rebuild and deploy

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM condenses too aggressively (loses facts) | Medium | High | Test prompt on real memories, tune temperature/prompt iteratively |
| content_hash mismatch causes duplicates post-distill | Low | Medium | Semantic dedup (cosine > 0.92) catches near-dupes; chunks keep original embeddings |
| Consolidation quality degrades (summarizing summaries) | High | High | Phase 3 fix: consolidation reads content_detailed when available |
| Nightly job timeout | Low | Low | Batch with limit/offset, same as consolidation pattern |
| /recall broken post-distill (no truncated flag) | High | High | Phase 4 fix: add has_detailed flag to search results |

## References

### Internal
- Brainstorm: `docs/brainstorms/2026-03-24-memory-distillation.md`
- Cleanup sprint plan: `docs/plans/2026-03-24-refactor-codebase-cleanup-sprint-plan.md`
- Migration pattern: `scripts/migrate_relevance_feedback.sql`
- Observer prompt (style reference): `src/cems/llm/observation_extraction.py:24`
- Reflector prompt (condensation reference): `src/cems/llm/observation_reflection.py`
- Distillation target: `_consolidate_never_shown()` at `src/cems/maintenance/summarization.py:191`
- DOCUMENT_COLUMNS: `src/cems/db/document_store.py:30`
- Agentic context loading: `src/cems/agentic/search.py:200`
- Retrieval serialization: `src/cems/memory/retrieval.py:62`
- API memory_get: `src/cems/api/handlers/memory.py:775`
- Recall skill: `src/cems/data/claude/commands/recall.md`
- Store skill: `src/cems/data/claude/skills/cems/remember.md`

### Key Gotchas (from MEMORY.md + SpecFlow)
- `update_document()` must clear `content_detailed` on content change (stale data risk)
- Consolidation must read `content_detailed` to avoid summarizing-summaries
- `/recall` breaks if only checking `truncated: true` — needs `has_detailed` flag
- `content_hash` changes after distillation — semantic dedup is the reliable fallback
- Chunks retain original embeddings — do NOT re-embed from condensed content
- Use `order="asc"` in get_all_documents to process oldest first
- Respect PROTECTED_CATEGORIES for distillation
- `memory_get` missing from mcp_stdio.py — must add for /recall to work on stdio
