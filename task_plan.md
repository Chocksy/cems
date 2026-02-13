# Fix Observer Daemon Document Duplication

## Goal
Fix the document duplication bugs identified by codex-investigator audit. Production DB shows 6 duplicate session-summary docs for session `b40eb706` and 9+ duplicate learnings.

## Status: IN PROGRESS

## Root Causes Identified
1. **CRITICAL**: `api_session_summarize()` upsert is TOCTOU race (check-then-act without transaction)
2. **MODERATE**: `save_state()` not atomic (crash mid-write → state reset → re-send)
3. **LOW**: `_spawn_daemon()` PID file race between concurrent hooks
4. **CLEANUP**: Existing duplicates need deduplication in production

---

## Phase 1: Atomic Upsert in `api_session_summarize()` `pending`
**Files**: `src/cems/api/handlers/session.py`, `src/cems/db/document_store.py`

### Problem
`find_document_by_tag()` → check → `add_async()` is not transactional. Two concurrent requests both see "no existing doc" and both create.

### Solution
Add `find_or_create_by_tag()` method to DocumentStore that uses `SELECT ... FOR UPDATE` within a transaction. The handler calls this single atomic method instead of separate find+create.

### Changes
- `document_store.py`: Add `upsert_document_by_tag()` — SELECT FOR UPDATE in transaction
- `session.py`: Replace find+create/update with atomic upsert call

---

## Phase 2: Atomic `save_state()` `pending`
**Files**: `src/cems/observer/state.py`

### Problem
`save_state()` writes directly to file. Crash mid-write → corrupted JSON → fresh state → re-send from byte 0.

### Solution
Use tmp+rename pattern (same as `write_signal()`).

### Changes
- `state.py`: Write to `.tmp` then `os.rename()`

---

## Phase 3: Spawn Lock in `_spawn_daemon()` `pending`
**Files**: `hooks/utils/observer_manager.py`, `src/cems/data/claude/hooks/utils/observer_manager.py`

### Problem
Two hooks calling `ensure_daemon_running()` simultaneously can both spawn daemons. The flock in `__main__.py` makes only one survive, but PID file points to wrong one temporarily.

### Solution
Add `fcntl.flock` around spawn operation in `_spawn_daemon()`.

### Changes
- `observer_manager.py`: Add file lock around spawn (both canonical + bundled)

---

## Phase 4: Production Data Cleanup `pending`
**Target**: Production DB via CEMS API

### Problem
6 duplicate session-summary docs for `session:b40eb706`, 9+ duplicate learnings.

### Solution
Use CEMS API to soft-delete duplicates, keeping only the most recent/comprehensive one per session tag.

---

## Phase 5: Tests `pending`
**Files**: `tests/test_observer.py`, `tests/test_integration.py`

### Changes
- Test atomic upsert (concurrent calls produce single document)
- Test atomic save_state (verify tmp+rename)
- Run full test suite to verify no regressions

---

## Key Decisions
1. Use `SELECT ... FOR UPDATE` (PostgreSQL row-level lock) rather than application-level lock
2. Keep the upsert logic in DocumentStore (not handler) for reusability
3. Spawn lock uses same `fcntl.flock` pattern as daemon singleton
