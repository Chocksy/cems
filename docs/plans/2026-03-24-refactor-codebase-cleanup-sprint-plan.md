---
title: Codebase Cleanup Sprint
type: refactor
date: 2026-03-24
---

# Codebase Cleanup Sprint

## Overview

Address 33 findings from a 5-agent codebase review across 6 workstreams: IDOR security fixes, maintenance philosophy change (consolidate-not-delete), query parallelization, dead code removal, code deduplication, and test coverage. Prioritized P1 (security) → P3 (quality).

## Problem Statement

The CEMS memory system has accumulated technical debt across three categories:

1. **Security (P1)**: Three `DocumentStore` methods (`get_document`, `delete_document`, `update_document`) lack `user_id` filtering — any authenticated user who guesses a UUID can read/modify/delete another user's documents.
2. **Philosophy mismatch (P1)**: Maintenance jobs delete memories that could be consolidated, losing information permanently.
3. **Performance/Quality (P2-P3)**: Sequential queries where parallel is safe, 5+ dead code modules, duplicated agentic search logic, and zero test coverage on `mcp_stdio.py`.

## Proposed Solution

Six workstreams executed in dependency order. Each is independently shippable.

## Technical Approach

### Architecture

```
DocumentStore methods gain user_id parameter
         ↓
CRUD mixins pull user_id from self.config.user_id
         ↓
API handlers already have memory.config.user_id — no change
         ↓
Maintenance jobs thread user_id from self.memory.config.user_id
```

### Implementation Phases

#### Phase 1: Dead Code Removal (P2 — unblocks cleaner diffs)

Remove confirmed-dead modules to reduce noise in subsequent diffs.

**Files to delete:**
- `src/cems/db/constants.py` — maps orphaned `memories` table columns (`pinned`, `archived`, `priority`, `access_count`, `expires_at`)
- `src/cems/db/metadata_store.py` — `PostgresMetadataStore` class, zero imports, reads orphaned `memory_metadata` table
- `src/cems/db/row_mapper.py` — `row_to_dict()`, zero imports, maps removed columns

**Files to edit:**
- `src/cems/db/__init__.py` — remove imports of dead modules (lines 10, 13) and dead `__all__` exports (lines 16-20). Keep `FilterBuilder`, `DocumentStore`, `PgVectorStore` exports.
- `src/cems/llm/observation_extraction.py` — remove `extract_observations()` (line 133+), dead since `/api/session/observe` was removed 2026-02-17
- `src/cems/llm/__init__.py` — remove `extract_observations` from imports (line 46) and `__all__` (line 74)

**Verification:**
```bash
# Confirm zero imports before deleting
colgrep -e "from cems.db.constants import" -e "from cems.db import.*MEMORY_COLUMNS"
colgrep -e "from cems.db.metadata_store import" -e "PostgresMetadataStore"
colgrep -e "from cems.db.row_mapper import" -e "from cems.db import.*row_to_dict"
colgrep -e "extract_observations" --exclude="*observation_extraction*" --exclude="*__init__*"

# Run tests
.venv/bin/python3 -m pytest tests/ -x -q
```

#### Phase 2: Maintenance Philosophy — Consolidate, Never Delete (P1)

Change `SummarizationJob` to consolidate instead of prune.

**File: `src/cems/maintenance/summarization.py`**

1. **Delete functions:**
   - `_prune_stale()` (line 199) — soft-deletes docs older than `stale_days`
   - `_prune_never_shown()` (line 249) — soft-deletes docs with `shown_count=0` older than 7d
   - `_recently_shown()` helper (line 42) — only used inside `_prune_stale()`
   - `NEVER_SHOWN_MIN_AGE_DAYS` constant (line 239) — only used in `_prune_never_shown()`

2. **Keep functions:**
   - `_prune_chronically_noisy()` (line 382) — soft-deletes docs with >50% noise ratio, proven signal
   - `_consolidate_never_shown()` (line 290) — LLM-consolidates groups of >20 never-shown docs

3. **Update orchestrator** `run_async()` (line 69):
   - Remove calls at lines 92-93
   - Keep `_prune_chronically_noisy()` call (line 94)
   - Keep `_consolidate_never_shown()` call (line 95)
   - Return dict: set `memories_pruned: 0` and `never_shown_pruned: 0` for backwards compat (MCP tool consumers parse this)

4. **Update docstrings:**
   - Class docstring (line 62): reflect "consolidate, never delete" philosophy
   - Module docstring (line 1): same

**Verification:**
```bash
.venv/bin/python3 -m pytest tests/test_maintenance.py -x -q
```

#### Phase 3: Security — IDOR Fixes with TDD (P1)

Add `user_id` filtering to prevent cross-user document access.

**Design decisions:**
- `user_id` is **required** on DocumentStore methods (no optional bypass)
- Return 404 (not 403) when `user_id` doesn't match — no information leakage
- CRUD mixins pull `user_id` from `self.config.user_id` internally — no caller signature changes
- Maintenance jobs thread `self.memory.config.user_id` to DocumentStore calls

**Step 1: Write failing tests (TDD)**

Create `tests/test_idor_security.py`:

```python
# tests/test_idor_security.py
"""IDOR boundary tests — verify cross-user access is blocked."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

class TestIDORGetDocument:
    """get_document() must filter by user_id."""

    async def test_get_own_document_succeeds(self):
        """User A can read their own document."""
        # Setup: create doc for user_a, fetch as user_a → success

    async def test_get_other_users_document_returns_none(self):
        """User A cannot read User B's document — returns None (404)."""
        # Setup: create doc for user_b, fetch as user_a → None

class TestIDORDeleteDocument:
    """delete_document() must filter by user_id."""

    async def test_delete_own_document_succeeds(self):
        """User A can delete their own document."""

    async def test_delete_other_users_document_returns_false(self):
        """User A cannot delete User B's document — returns False."""

class TestIDORUpdateDocument:
    """update_document() must filter by user_id."""

    async def test_update_own_document_succeeds(self):
        """User A can update their own document."""

    async def test_update_other_users_document_returns_false(self):
        """User A cannot update User B's document — returns False."""
```

**Step 2: Fix DocumentStore methods**

File: `src/cems/db/document_store.py`

| Method | Line | Change |
|--------|------|--------|
| `get_document(self, document_id)` | 302 | Add `user_id: str` param, `AND user_id = $2` |
| `delete_document(self, document_id, hard=False)` | 317 | Add `user_id: str` param, `AND user_id = $2` |
| `update_document(self, document_id, content, chunks, embeddings)` | 542 | Add `user_id: str` param, `AND user_id = $5` |

Reference implementations that already have `user_id`:
- `restore_document()` line 339: `WHERE id = $1 AND user_id = $2`
- `promote_document()` line 355: `WHERE id = $2 AND user_id = $3`

**Step 3: Update CRUD mixins to pass user_id**

File: `src/cems/memory/crud.py`

| Method | Line | Change |
|--------|------|--------|
| `_get_async()` | 54 | Pass `user_id=self.config.user_id` to `doc_store.get_document()` |
| `update_async()` | 128 | Pass `user_id=self.config.user_id` to `doc_store.update_document()` |
| `_delete_async_internal()` | 158 | Pass `user_id=self.config.user_id` to `doc_store.delete_document()` |

**Step 4: Update API handlers**

File: `src/cems/api/handlers/memory.py`

| Handler | Line | Change |
|---------|------|--------|
| `api_memory_get()` | 790 | Pass `user_id=memory.config.user_id` to `doc_store.get_document()` |
| Conflict resolution | 1160-1180 | Pass `user_id=memory.config.user_id` to `delete_document()` and `memory.update_async()` |

**Step 5: Update maintenance job callers**

All maintenance jobs access `self.memory.config.user_id`:

| File | Lines | Method called |
|------|-------|---------------|
| `maintenance/consolidation.py` | 164, 180, 212 | `get_document()`, `delete_document()` |
| `maintenance/summarization.py` | 145, 228, 277, 367, 423 | `delete_document()` |
| `maintenance/observation_reflector.py` | 141 | `delete_document()` |
| `maintenance/reindex.py` | 186 | `delete_document()` |

**Step 6: Update metadata mixin**

File: `src/cems/memory/metadata.py` line 52 — pass `user_id=self.config.user_id` to `get_document()`.

**Verification:**
```bash
# All 6 IDOR tests pass
.venv/bin/python3 -m pytest tests/test_idor_security.py -x -q

# Full suite still passes (no regressions)
.venv/bin/python3 -m pytest tests/ -x -q
```

#### Phase 4: Performance — Parallel Queries (P2)

Parallelize independent database queries with pool-safe concurrency.

**4a. Agentic search bucket queries**

File: `src/cems/agentic/search.py`, function `_load_context_memories()` (line 213)

Current: 6 sequential queries (1 project + 4 profile categories + 1 recent).

Change: Gather the 4 profile category queries (Bucket 2, lines 276-283). Keep Buckets 1, 2, 3 sequential to cap at 4 concurrent connections per request:

```python
# Before (lines 276-283): sequential loop
for cat in PROFILE_CATEGORIES:
    docs = await document_store.get_all_documents(user_id, category=cat, limit=200)
    profile_docs.extend(docs)

# After: parallel gather
profile_results = await asyncio.gather(*[
    document_store.get_all_documents(user_id, category=cat, limit=200)
    for cat in PROFILE_CATEGORIES
])
profile_docs = [doc for result in profile_results for doc in result]
```

**4b. Profile endpoint queries**

File: `src/cems/api/handlers/memory.py`, function `api_memory_profile()` (line 558)

Current: 4-5 sequential queries (preferences, guidelines, recent, gate-rules, project).

Change: Gather all 5 queries:

```python
prefs, guidelines, recent, gate_rules, project_docs = await asyncio.gather(
    doc_store.get_documents_by_category(user_id, category="preferences", limit=10),
    doc_store.get_documents_by_category(user_id, category="guidelines", limit=25),
    doc_store.get_recent_documents(user_id, hours=24, limit=15, ...),
    doc_store.get_documents_by_category(user_id, category="gate-rules", limit=50),
    doc_store.get_documents_by_category(user_id, category="project", ...),
)
```

Profile is called once per session — pool exhaustion risk is negligible.

**4c. Push time filter to SQL**

File: `src/cems/agentic/search.py`, lines 289-308

Current: Fetches 500 docs then filters by `created_at >= cutoff` in Python.

Change: Add `created_after: datetime | None = None` parameter to `get_all_documents()` in `document_store.py`. Add `AND created_at >= $N` to SQL when provided. Pass `cutoff` from `_load_context_memories()`.

Note: The project-scope filter (lines 302-307) stays in Python — it's complex conditional logic that doesn't map cleanly to SQL.

**Verification:**
```bash
.venv/bin/python3 -m pytest tests/test_agentic_search.py tests/test_server.py -x -q
```

#### Phase 5: Code Deduplication (P3)

**5a. Extract shared agentic code**

Create `src/cems/agentic/rrf.py`:

```python
# src/cems/agentic/rrf.py
"""Reciprocal Rank Fusion — shared between production search and eval."""

RRF_K = 60

def reciprocal_rank_fusion(rankings: list[list], k: int = RRF_K) -> list:
    """Merge multiple ranked lists using RRF scoring."""
    # Extract identical logic from agentic/search.py:174 and eval/longmemeval_agentic.py:435
```

Update imports in both files:
- `src/cems/agentic/search.py` — `from cems.agentic.rrf import reciprocal_rank_fusion, RRF_K`
- `src/cems/eval/longmemeval_agentic.py` — `from cems.agentic.rrf import reciprocal_rank_fusion, RRF_K`

**Do NOT extract:**
- `_parse_agent_response()` — parses different ID formats (hex UUIDs vs session IDs)
- Agent prompts — intentionally different ("memories" vs "sessions" language)

**5b. Extract terminal UI helpers**

File: `src/cems/commands/setup.py`

Extract shared terminal rendering logic (raw mode, cursor movement, key handling) from `_multiselect()` (line 32) and `_single_select()` (line 126) into a common `_render_menu()` helper. Keep the two entry points with their distinct signatures and return types.

**Verification:**
```bash
# Eval benchmark regression check
python -m cems.eval.longmemeval --questions 10 --api-url http://localhost:8765

# Full test suite
.venv/bin/python3 -m pytest tests/ -x -q
```

#### Phase 6: Test Coverage (P3)

**6a. MCP stdio tests**

Create `tests/test_mcp_stdio.py`:

The module has import-time side effects (`_get_config()` and `_fetch_profile()` run at import). Tests must mock these before importing:

```python
# tests/test_mcp_stdio.py
import pytest
from unittest.mock import patch, AsyncMock

@pytest.fixture(autouse=True)
def mock_config():
    with patch("cems.mcp_stdio._get_config", return_value=("http://test:8765", "test-key")):
        with patch("cems.mcp_stdio._fetch_profile", return_value="test profile"):
            yield

class TestMemoryAdd:
    """Test memory_add tool handler."""

class TestMemorySearch:
    """Test memory_search tool handler."""

class TestMemoryForget:
    """Test memory_forget tool handler."""

class TestMemoryUpdate:
    """Test memory_update tool handler."""

class TestMemoryMaintenance:
    """Test memory_maintenance tool handler."""

class TestResources:
    """Test MCP resource handlers (status, personal summary, shared summary)."""

class TestCredentialReading:
    """Test _get_config() reads from env vars and ~/.cems/credentials."""
```

**6b. IDOR boundary tests** — already covered in Phase 3.

**6c. Weak assertion audit** — scan tests for `assert True`, bare `assert response`, and missing assertions on response content. Fix the most impactful ones.

**Verification:**
```bash
.venv/bin/python3 -m pytest tests/test_mcp_stdio.py -x -q
.venv/bin/python3 -m pytest tests/ -x -q
```

## Acceptance Criteria

### Functional Requirements

- [x] `get_document()`, `delete_document()`, `update_document()` require and filter by `user_id`
- [x] Cross-user document access returns None/False (404 behavior, no 403 info leak)
- [x] `_prune_stale()` and `_prune_never_shown()` are removed from codebase
- [x] `_consolidate_never_shown()` is the primary maintenance path
- [x] Maintenance API response preserves `memories_pruned` and `never_shown_pruned` keys (set to 0)
- [x] Profile category queries in `_load_context_memories()` run in parallel
- [x] Profile endpoint queries run in parallel
- [x] Bucket 3 time filter pushed to SQL `created_after` parameter
- [x] Dead code files deleted: `db/constants.py`, `db/metadata_store.py`, `db/row_mapper.py`
- [x] Dead exports removed from `db/__init__.py` and `llm/__init__.py`
- [x] `reciprocal_rank_fusion()` extracted to `agentic/rrf.py`, imported by both search and eval
- [x] `mcp_stdio.py` has unit tests for all 5 tools and 3 resources

### Non-Functional Requirements

- [x] No pool exhaustion: max 4 concurrent connections per request
- [ ] Eval benchmark: no regression on Recall@5 (baseline 98%) — requires live Docker
- [x] Full test suite passes (659 tests, up from 636)

### Quality Gates

- [x] IDOR tests written before fix (TDD red-green)
- [x] No new modules created except `agentic/rrf.py` and `tests/test_idor_security.py` and `tests/test_mcp_stdio.py`
- [x] Zero `colgrep` hits for deleted symbols after cleanup

## Dependencies & Prerequisites

- Docker environment running for integration tests
- Existing test infrastructure (`_run()` async helper, `_make_doc()` factory, `AsyncMock` patterns)
- No database migrations required — all changes are application-level

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| IDOR fix breaks maintenance jobs | Medium | High | Thread `user_id` through all 9 maintenance call sites; existing tests catch regressions |
| Pool exhaustion from `asyncio.gather()` | Low | Medium | Cap at 4 concurrent queries; profile endpoint called once/session |
| Eval benchmark regression from dedup | Low | High | Only extract `reciprocal_rank_fusion()` (byte-identical); run eval after |
| Removing pruning causes memory bloat | Medium | Medium | `_consolidate_never_shown()` handles accumulation; `_prune_chronically_noisy()` handles bad memories |
| Dead code deletion breaks hidden import | Very Low | Low | Triple-verify with `colgrep` before each deletion |

## References & Research

### Internal References

- Brainstorm: `docs/brainstorms/2026-03-24-codebase-cleanup.md`
- Maintenance audit: `docs/maintenance-audit-2026-03-03.md`
- IDOR-vulnerable methods: `src/cems/db/document_store.py:302,317,542`
- Already-fixed reference: `src/cems/db/document_store.py:339,355` (restore/promote)
- CRUD callers: `src/cems/memory/crud.py:54,128,158`
- API handlers: `src/cems/api/handlers/memory.py:790,747,812,1160`
- Maintenance orchestrator: `src/cems/maintenance/summarization.py:69-107`
- Agentic bucket queries: `src/cems/agentic/search.py:213-318`
- Profile queries: `src/cems/api/handlers/memory.py:558-744`
- RRF duplicates: `src/cems/agentic/search.py:174` and `src/cems/eval/longmemeval_agentic.py:435`
- Dead code: `src/cems/db/constants.py`, `src/cems/db/metadata_store.py`, `src/cems/db/row_mapper.py`
- MCP stdio (untested): `src/cems/mcp_stdio.py`

### Key Gotchas (from MEMORY.md)

- `update_document()` must NEVER be used for reindex — bumps `updated_at`, defeats age-based pruning
- `upsert_document_by_tag()` uses `SELECT...FOR UPDATE` — preserve this pattern
- Use `AsyncMock(return_value=...)` not `MagicMock()` for async mocking
- `created_at` (not `updated_at`) for age calculations in maintenance
