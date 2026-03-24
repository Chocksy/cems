---
date: 2026-03-24
topic: codebase-cleanup-sprint
---

# CEMS Codebase Cleanup Sprint

## What We're Building

A cleanup sprint addressing 33 findings from a 5-agent codebase review. Covers security fixes, maintenance philosophy change, performance optimizations, dead code removal, and deduplication.

## Key Decisions

- **IDOR fix (P1)**: Add user_id filter to get_document, delete_document, update_document. TDD approach — failing tests first.
- **Maintenance philosophy**: CONSOLIDATE, never DELETE (except proven noise). Remove `_prune_never_shown` and `_prune_stale`. Keep `_prune_chronically_noisy`. Make `_consolidate_never_shown` the primary path.
- **Distillation feature**: Saved for later — daily aggressive consolidation into same-sized summaries, detailed content in separate column, LLMs request details on-demand.
- **Performance**: Parallelize bucket queries, merge profile queries, push time filter to SQL.
- **Dead code**: Delete 10+ confirmed dead files/functions.
- **Deduplication**: Merge agentic/search.py and eval/longmemeval_agentic.py shared code. Merge _multiselect/_single_select.

## Workstreams (in priority order)

1. **Security** — IDOR fixes with TDD (P1, blocks everything)
2. **Maintenance** — Remove deletion, keep consolidation (P1)
3. **Performance** — Parallel queries, SQL filters (P2)
4. **Dead code** — Delete orphaned modules (P2)
5. **Code quality** — Merge duplicates, fix slop (P3)
6. **Test coverage** — mcp_stdio.py, API handlers, weak assertions (P3)

## Next Steps
→ Implement in a new session using `/workflows:plan`
