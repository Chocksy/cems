# CEMS Improvement Tasks

Generated from codebase review on 2026-02-26. Priorities 1-6 for immediate implementation.

## Phase 1: Security Fixes (Critical)

### 1.1 Race condition in `add_document` (TOCTOU)
- **File:** `src/cems/db/document_store.py:209-244`
- **Problem:** Hash check + semantic dedup + insert run without a row lock. Two concurrent requests can both pass dedup and insert duplicates.
- **Fix:** Move hash check + semantic dedup INSIDE the transaction block. Use advisory lock on content_hash or `SELECT ... FOR UPDATE` pattern (like `upsert_document_by_tag` already does correctly).

### 1.2 DB error details leaked in auth middleware
- **File:** `src/cems/server.py:147`
- **Problem:** `f"Database error: {e}"` exposes connection strings/schema to unauthenticated users.
- **Fix:** Return generic `"Database unavailable"` message. Log the full error server-side.

### 1.3 TrustedHostMiddleware with wildcard
- **File:** `src/cems/server.py:220`
- **Problem:** `allowed_hosts=["*"]` negates the middleware entirely.
- **Fix:** Remove the middleware (it does nothing with wildcard).

### 1.4 Exception strings returned to clients
- **File:** `src/cems/api/handlers/memory.py` (many handlers)
- **Problem:** Nearly every handler returns `str(e)` in 500 responses, leaking internal details.
- **Fix:** Return generic "Internal server error" to clients. Keep `logger.error(...)` for server-side logging (already present in most handlers).

## Phase 2: Dead Code Removal

### 2.1 Delete orphaned `vectorstore.py`
- **File:** `src/cems/vectorstore.py` (14KB, 400+ lines)
- **Also:** `tests/test_vectorstore.py`
- **Verification:** Nothing in `src/cems/` imports from it. Only the test file imports it.

### 2.2 Remove empty `AnalyticsMixin`
- **File:** `src/cems/memory/analytics.py`
- **Also:** Remove from `CEMSMemory` class hierarchy in `src/cems/memory/core.py:68`
- **Also:** Remove from `src/cems/memory/__init__.py` if exported

### 2.3 Remove orphaned FilterBuilder methods
- **File:** `src/cems/db/filter_builder.py:92-123`
- **Methods:** `add_not_archived()` and `add_scope_filter()` — only called from dead `vectorstore.py`

### 2.4 Remove deprecated `_infer_category_from_query`
- **File:** `src/cems/memory/core.py:171-179` — the method (always returns None)
- **Also:** `src/cems/memory/search.py:166` — call site
- **Also:** `src/cems/memory/retrieval.py:160,431` — call sites
- **Also:** Remove unused `inferred_category` param from `_apply_score_adjustments` in `search.py:90`

### 2.5 Remove duplicate `_ensure_document_store_search`
- **File:** `src/cems/memory/search.py:112-124`
- **Also:** `src/cems/memory/search.py:112` — remove `_document_store` class variable
- **Fix:** Replace 3 calls in `search.py` (lines 147, 206, 242) with `self._ensure_document_store()`

### 2.6 Fix fragile `"body" in dir()` pattern
- **File:** `src/cems/api/handlers/memory.py:917`
- **Fix:** Initialize `job_type = "unknown"` at top of function, update after parsing.

## Phase 3: Performance — Batch Consolidation Embeddings

### 3.1 Batch-embed documents in consolidation
- **File:** `src/cems/maintenance/consolidation.py:116`
- **Problem:** Embeds 1 doc at a time inside loop = N HTTP roundtrips for N docs.
- **Fix:** Pre-embed all documents in batches (100 at a time) before the dedup loop. Cache in dict keyed by doc_id.

### 3.2 Batch chunk inserts in document_store
- **File:** `src/cems/db/document_store.py:267-282`
- **Problem:** Individual INSERT per chunk in a loop.
- **Fix:** Use `conn.executemany()` for bulk chunk inserts within the transaction.

## Phase 4: Extract Shared Utilities

### 4.1 Extract `_run_async` to shared module
- **Duplicated in:** `core.py`, `write.py`, `search.py`, `crud.py`, `metadata.py`, `relations.py`, `retrieval.py`, `scheduler.py`
- **Fix:** Create `src/cems/lib/async_utils.py` with the function. Import from there in all 8 files.

### 4.2 Extract `get_project_id` to hooks shared utils
- **Duplicated in:** `hooks/cems_user_prompts_submit.py`, `hooks/cems_pre_tool_use.py`, `hooks/cems_session_start.py`, `hooks/cems_post_tool_use.py`
- **Also:** Bundled copies in `src/cems/data/claude/hooks/`
- **Fix:** Move to `hooks/utils/project.py`, import from there.

## Phase 5: Rewrite DEPLOYMENT.md

- **File:** `docs/DEPLOYMENT.md`
- **Problem:** References Qdrant, Celery, Redis, cems-worker — all removed infrastructure.
- **Fix:** Rewrite to reflect current Docker Compose setup (PostgreSQL + pgvector + CEMS server).

## Phase 6: Archive Completed Docs

Move the following to `docs/archive/`:
- `docs/code-architecture-refactoring-plan.md` (all 6 refactors completed)
- `docs/reranker-bottleneck-solutions.md` (decision made, reranker disabled)
- `docs/retrieval-changes-analysis.md` (analysis complete)
- `docs/qmd-implementation-plan.md` (never implemented, 98% recall reached)
- `docs/ollama-removal-plan.md` (Ollama removed)
- `docs/website-research.md` (website already built and deployed)
- `research/option-d-observer-plan.md` (Observer V2 implemented)
- `.cursor/plans/memory_system_tech_spec_0563f5b1.plan.md` (architecture diverged)
- `.cursor/plans/simplify_mcp_server_b4442391.plan.md` (all todos completed)
- Root: `progress.md`, `task_plan.md`, `findings.md` (Phase 4+5 snapshots, done)

## Deferred (not in this session)

- **7. Query decomposition** — implement separately (research doc ready)
- **8. LongMemEval 500q** — after eval system overhaul
- **9. Codex/Cursor adapters** — after cleanup done
- **10. getcems.com** — already deployed
