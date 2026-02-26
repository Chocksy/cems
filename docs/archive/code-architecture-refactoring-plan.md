# CEMS Code Architecture Refactoring Plan

**Date:** 2026-01-30
**Goal:** Split large files (>400 lines) into focused modules following Python best practices

---

## Executive Summary

The CEMS codebase has 6 files exceeding recommended size limits (150-500 lines for AI-assisted development). This document provides a detailed refactoring plan with specific file splits and task lists.

### Current State

| File | Lines | Chars | Status |
|------|-------|-------|--------|
| memory.py | **1,662** | 60,800 | ✅ REFACTORED → memory/ package (8 mixin files, core.py 211 lines) |
| server.py | **1,191** | 47,501 | ✅ PARTIAL - deps extracted to api/deps.py (1118 lines remaining) |
| vectorstore.py | **1,116** | 37,225 | ✅ REFACTORED → 660 lines + db/ package |
| llm.py | **922** | ~31,000 | ✅ REFACTORED → llm/ package (3 files) |
| cli.py | **821** | 27,195 | ✅ REFACTORED → cli_utils.py + commands/ (8 files) |
| graph.py | **750** | 24,197 | ✅ ARCHIVED → _archive/graph.py |

### Architecture Health
- **Circular dependencies:** None detected
- **Import structure:** Clean 4-layer hierarchy
- **Foundation layers:** `llm.py` and `vectorstore.py` have zero CEMS imports

---

## File-by-File Refactoring Plans

### 1. memory.py (1,662 lines → 8 files)

**Current:** Single monolithic `CEMSMemory` class handling all operations.

**Final Structure (Actual):**
```
src/cems/
├── memory.py                    (facade - re-exports public API)
└── memory/
    ├── __init__.py              (16 lines)   - Re-exports
    ├── core.py                  (211 lines)  - CEMSMemory class, init, config
    ├── write.py                 (259 lines)  - WriteMixin: add(), add_async()
    ├── search.py                (232 lines)  - SearchMixin: search(), _search_raw()
    ├── crud.py                  (220 lines)  - CRUDMixin: get, update, delete, forget
    ├── analytics.py             (120 lines)  - AnalyticsMixin: stale/hot/recent
    ├── metadata.py              (199 lines)  - MetadataMixin: categories, summaries
    ├── relations.py             (112 lines)  - RelationsMixin: graph-like queries
    └── retrieval.py             (388 lines)  - RetrievalMixin: retrieve_for_inference()
                                 (1757 total)
```

**Tasks:** ✅ ALL COMPLETE (mixin-based refactor)
- [x] Create `src/cems/memory/` directory structure
- [x] Move full class to `memory/core.py` (1662 lines)
- [x] Create `memory/__init__.py` with re-exports
- [x] Create facade in `memory.py` that re-exports public API
- [x] Run tests to verify no regressions (151 unit, 20 integration)
- [x] Split core.py into mixins: WriteMixin, SearchMixin, CRUDMixin, AnalyticsMixin, MetadataMixin, RelationsMixin, RetrievalMixin
- [x] Final core.py: 211 lines (down from 1662), all files under 400 lines

---

### 2. server.py (1,191 lines → 6 files)

**Current:** 14 route handlers defined inside `create_http_app()` function.

**Final Structure (Actual):**
```
src/cems/
├── server.py                    (264 lines) - Entry point, middleware, route registration
└── api/
    ├── __init__.py              (17 lines)  - Re-exports from deps
    ├── deps.py                  (105 lines) - Shared state, get_memory, get_scheduler
    └── handlers/
        ├── __init__.py          (41 lines)  - Handler exports
        ├── health.py            (35 lines)  - ping, health_check
        ├── memory.py            (493 lines) - 10 memory endpoints
        ├── session.py           (108 lines) - session/analyze
        └── tool.py              (131 lines) - tool/learning
                                 (1194 total)
```

**Tasks:** ✅ ALL COMPLETE
- [x] Create `src/cems/api/` directory structure
- [x] Extract shared state to `api/deps.py` (get_memory, get_scheduler, context vars)
- [x] Update server.py to import from `api/deps.py`
- [x] Extract handlers to `api/handlers/` (health, memory, session, tool)
- [x] Update server.py to import handlers and register routes
- [x] Update tests to patch correct modules
- [x] Verify all 151 unit tests pass
- [x] Verify all 20 integration tests pass

**Result:** server.py reduced from 1191 to 264 lines. Handlers extracted to domain-specific files under `api/handlers/`. All files under 500 lines. Total: 1194 lines split across 7 files.

---

### 3. vectorstore.py (1,116 lines → 4 files)

**Current:** Single `PgVectorStore` class with mixed SQL building, execution, and result processing.

**Proposed Structure:**
```
src/cems/
├── vectorstore.py               (~200 lines) - Public API facade
└── db/
    ├── query_builder.py         (~250 lines) - FilterBuilder, SQL templates
    ├── vectorstore_ops.py       (~400 lines) - CRUD operations
    └── row_mapper.py            (~80 lines)  - Result processing
```

**Key Improvements:**
- Extract duplicate SELECT columns to constant
- Centralize WHERE clause building in `FilterBuilder`
- Automate parameter indexing

**Tasks:** ✅ ALL COMPLETE
- [x] Create `MEMORY_COLUMNS` constant for 19 repeated columns
- [x] Extract `FilterBuilder` class for WHERE clause generation
- [x] Create `db/` package with filter_builder.py, row_mapper.py, constants.py
- [x] Extract `_row_to_dict()` to `db/row_mapper.py`
- [x] Keep `vectorstore.py` as public facade (reduced from 1116 → 660 lines)

---

### 4. llm.py (940 lines → 5 files)

**Current:** OpenRouterClient + summarization + learning extraction all in one file.

**Proposed Structure:**
```
src/cems/
└── llm/
    ├── __init__.py              (~50 lines)  - Public API, get_client()
    ├── client.py                (~150 lines) - OpenRouterClient
    ├── summarization.py         (~200 lines) - summarize_memories, merge
    ├── learning_extraction.py   (~450 lines) - session learnings
    └── prompts.py               (~100 lines) - System prompts (optional)
```

**Tasks:** ✅ ALL COMPLETE
- [x] Create `src/cems/llm/` directory
- [x] Move `OpenRouterClient` to `llm/client.py`
- [x] Move `summarize_memories`, `merge_memory_contents` to `llm/summarization.py`
- [x] Move `extract_session_learnings`, `extract_tool_learning` to `llm/learning_extraction.py`
- [x] Create `llm/__init__.py` with public exports and `get_client()`
- [x] Update all imports across codebase (backward compatible via __init__.py re-exports)

---

### 5. cli.py (821 lines → 8 files)

**Current:** All 19 commands in single file.

**Proposed Structure:**
```
src/cems/
├── cli.py                       (~100 lines) - Entry point only
├── cli_utils.py                 (~80 lines)  - Shared utilities
└── commands/
    ├── __init__.py
    ├── status.py                (~50 lines)  - status, health
    ├── memory.py                (~200 lines) - add, search, list, delete, update
    ├── maintenance.py           (~50 lines)  - maintenance run
    └── admin/
        ├── __init__.py
        ├── users.py             (~180 lines) - 5 user commands
        └── teams.py             (~200 lines) - 6 team commands
```

**Tasks:** ✅ ALL COMPLETE
- [x] Create `src/cems/commands/` directory structure
- [x] Extract shared utilities to `cli_utils.py`
- [x] Move commands to domain-specific files
- [x] Keep `cli.py` as entry point with command registration
- [x] Verify all CLI commands work

---

### 6. graph.py (750 lines) - DEAD CODE

**Finding:** `KuzuGraphStore` is defined but **never instantiated anywhere** in the codebase.

**Evidence:**
- grep for `KuzuGraphStore(` returns zero results outside graph.py
- memory.py uses PostgreSQL relations, not Kuzu
- README mentions "knowledge graph" but implementation uses pgvector

**Options:**
1. **Delete entirely** - Clean up 750 lines of unused code
2. **Archive** - Move to `_archive/` for potential future use
3. **Keep** - If there are plans to integrate Kuzu

**Recommendation:** Delete or archive. Currently adds maintenance burden with zero value.

**Tasks:** ✅ ALL COMPLETE
- [x] Verify no runtime usage of KuzuGraphStore
- [x] Either delete `graph.py` or move to `_archive/graph.py`
- [x] Remove any TYPE_CHECKING imports of graph types (verified: none exist)

---

## Shared Utilities to Extract

Create `src/cems/lib/` for cross-cutting concerns:

```
src/cems/lib/
├── __init__.py
├── json_parsing.py      - extract_json_from_markdown(), parse_json_list()
├── error_handling.py    - Exception classes, @handle_exceptions decorator
├── config_helpers.py    - get_required_env(), lazy singleton pattern
└── string_utils.py      - strip_markdown_code_blocks()
```

**High-Impact Extractions:**
1. **JSON parsing** - Used in llm.py, retrieval.py, fact_extraction.py (60+ duplicate lines) ✅ DONE
2. **Error handling** - Standardize try/except patterns across 20+ files
3. **Config caching** - Improve testability of singleton patterns

---

## Implementation Order (Recommended)

### Phase 1: Foundation (Low Risk) ✅ COMPLETE
1. Extract `lib/` utilities ✅ DONE (lib/json_parsing.py)
2. Refactor `llm.py` → `llm/` package ✅ DONE (llm/client.py, summarization.py, learning_extraction.py)
3. Delete/archive `graph.py` (dead code) ✅ DONE (moved to _archive/)

### Phase 2: API Layer
4. Refactor `server.py` → `api/` package ✅ PARTIAL (deps extracted, 1118 lines remaining)
5. Refactor `cli.py` → `commands/` package ✅ DONE (cli.py 50 lines, cli_utils.py 56 lines, commands/ 8 files)

### Phase 3: Core (Higher Risk) ✅ COMPLETE
6. Refactor `vectorstore.py` with query builders ✅ DONE (db/ package + FilterBuilder)
7. Refactor `memory.py` → `memory/` package ✅ DONE (safe refactor preserving public API)

---

## Validation Checklist

After each file split:
- [x] All imports resolve correctly
- [x] No circular dependencies introduced
- [x] Unit tests pass (151/151)
- [x] Integration tests pass (20/20)
- [x] Docker build succeeds
- [ ] LongMemEval benchmark unchanged (not re-run)

---

## Best Practices Applied

Based on 2025-2026 Python standards:

1. **File size:** Target 150-500 lines (AI-assisted development sweet spot)
2. **Organization:** Module-by-functionality, not file-type
3. **Cohesion:** Related code stays together
4. **Coupling:** Clear interfaces between modules
5. **Pipeline design:** Swappable components (LangChain pattern)

---

## Risk Assessment

| Refactor | Risk | Mitigation |
|----------|------|------------|
| memory.py split | Medium | 7 files depend on it - careful API preservation |
| server.py split | Low | Entry point only, clear route separation |
| vectorstore.py | Low | Only memory.py imports it |
| llm.py split | Low | Zero CEMS imports, clean foundation |
| cli.py split | Low | Entry point, no upstream dependencies |
| graph.py delete | None | Dead code, never instantiated |

---

## Summary

**Status:** ✅ 6/6 COMPLETE

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| memory.py | 1,662 lines | 8 mixin files, core.py 211 lines | ✅ Complete |
| vectorstore.py | 1,116 lines | 660 lines + db/ package | ✅ Complete |
| llm.py | 922 lines | llm/ package (3 files) | ✅ Complete |
| cli.py | 821 lines | 50 lines + commands/ (8 files) | ✅ Complete |
| graph.py | 750 lines | Archived to _archive/ | ✅ Complete |
| server.py | 1,191 lines | 264 lines + api/ package (7 files) | ✅ Complete |

**Results:**
- 151 unit tests passing
- 20 integration tests passing
- memory.py reduced from 1,662 to 211 lines (core)
- server.py reduced from 1,191 to 264 lines (handlers extracted)
- No circular dependencies
- Docker build/deploy working

**Primary benefit:** Improved maintainability, testability, and AI-assisted development efficiency

---

## Post-Refactor Findings (Review on 2026-01-31)

Notes captured for later follow-up.

### Doc/Plan Drift
- Phase 2 "API Layer" marked PARTIAL in the plan, but code is complete (server split + handlers present).
- "No circular dependencies" is not strictly true at static-import level (TYPE_CHECKING references create core/mixin import loops).
- "Foundation layers: llm.py and vectorstore.py have zero CEMS imports" is no longer true (llm imports cems.lib; vectorstore imports cems.db/models).

### Incomplete Shared Utilities
- Only `src/cems/lib/json_parsing.py` exists.
- Planned files missing: `error_handling.py`, `config_helpers.py`, `string_utils.py`.

### Structure/Consistency Risks
- Duplicate `_run_async` helpers exist across memory mixins + core (drift risk).
- ~~`src/cems/api/handlers/memory.py` reads `_scheduler_cache` directly; `get_scheduler_state()` in `src/cems/api/deps.py` appears unused.~~ ✅ FIXED: `get_scheduler_state()` deleted
- Several large files remain (>500 lines): `src/cems/vectorstore.py`, `src/cems/api/handlers/memory.py`, `src/cems/admin/routes.py`, `src/cems/client.py`, `src/cems/retrieval.py`, `src/cems/llm/learning_extraction.py`, `src/cems/db/metadata_store.py`, `src/cems/eval/longmemeval.py`.

### Likely Unused (No Internal References)
- ~~Retrieval: `calculate_relevance_score` (`src/cems/retrieval.py`)~~ ✅ FIXED: Deleted (superseded by `apply_score_adjustments`)
- Pattern extraction module: `src/cems/pattern_extraction.py` (all functions)
- Fact extraction helper: `extract_facts` (`src/cems/fact_extraction.py`)
- Vectorstore helper: `get_vectorstore` (`src/cems/vectorstore.py`)
- Scheduler helper: `create_scheduler` (`src/cems/scheduler.py`)
- Indexer helper: `create_indexer` (`src/cems/indexer/indexer.py`)
- Embedding helpers: `get_embedding_client`, `get_async_embedding_client` (`src/cems/embedding.py`)
- Maintenance wrappers: `archive_dead`, `rebuild_embeddings`, `compress_old_memories`, `prune_stale`, `merge_duplicates`, `promote_hot_memories`
- ~~API deps helper: `get_scheduler_state` (`src/cems/api/deps.py`)~~ ✅ FIXED: Deleted
- Model/enums: `MemoryCategory`, `PinCategory` (`src/cems/models.py`)
- Tool schemas: `src/cems/tools.py` Pydantic models appear unused internally

---

## Bug Fixes Applied (2026-01-31)

### Critical Fixes
1. **Missing `import re` in `src/cems/fact_extraction.py`** ✅ FIXED
   - Lines 213-214 used `re.sub()` without importing `re`
   - Would crash at runtime when JSON parsing failed

### Dead Code Removed
2. **`calculate_relevance_score()` in `src/cems/retrieval.py`** ✅ DELETED
   - Superseded by `apply_score_adjustments()` (unified scoring)
   - Used 30-day half-life (inconsistent with current 60-day)

3. **`get_scheduler_state()` in `src/cems/api/deps.py`** ✅ DELETED
   - Never called (handlers access `_scheduler_cache` directly)

### Retrieval Enhancements
4. **Restored configurable cross-category and project penalties** ✅ ADDED
   - New config options in `src/cems/config.py`:
     - `enable_cross_category_penalty` (default: True)
     - `enable_project_penalty` (default: True)
     - `cross_category_penalty_factor` (default: 0.8)
     - `project_penalty_factor` (default: 0.8)
     - `project_boost_factor` (default: 1.3)
   - Updated `apply_score_adjustments()` to use config
   - For eval: Set `CEMS_ENABLE_CROSS_CATEGORY_PENALTY=false` and `CEMS_ENABLE_PROJECT_PENALTY=false`
