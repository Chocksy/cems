# Progress: Memory Quality Overhaul

## Session 2026-02-10

### Phase 1: Always-on context search — COMPLETE
- Added `read_last_assistant_message()` to `hooks/utils/transcript.py`
- Modified `hooks/user_prompts_submit.py` to enrich ALL search queries with assistant context
- Committed as `5af1f05`, hooks installed

### Phase 2: Production data cleanup — COMPLETE (local)
- pg_dump from production (Hetzner `pc0c44088c48gog8cg4wck84` container)
- Restored 32MB dump to local `cems-postgres` Docker
- Soft-deleted 1,915 memories (noise + legacy patterns + never-shown/no-ref)
- Normalized 500 → 33 canonical categories
- Normalized source_ref short names to full org/repo
- Final: 584 active memories, 33 categories
- **NOT yet pushed to production** — user wants to review first
- Backups: `cems_prod_backup.dump` (original), `cems_cleaned.dump` (cleaned)

### Phase 3: Ingestion quality gates — COMPLETE
- **3a**: Confidence threshold raised 0.3 → 0.6 (session + tool learnings)
- **3b**: Min content length 80 chars (short vague learnings rejected)
- **3c**: Noise filtering (`/private/tmp/claude`, `background command`, `exit code`)
- **3d**: Controlled category vocabulary (30 canonical + alias mapping)
- **3e**: Relevance threshold raised 0.005 → 0.3 in config.py
- **3f**: Semantic dedup on ingestion (cosine > 0.92 = skip) in document_store.py
- All 251 tests pass, sanity checks pass

## Files Modified
- `hooks/utils/transcript.py` — added `read_last_assistant_message()`
- `hooks/user_prompts_submit.py` — always-on context search + confirmatory handling
- `task_plan.md` — full plan with Phase 3 details
- `findings.md` — production data analysis + cleanup buckets
- `progress.md` — this file

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| test_short_prompts_skip_search failed | 1 | Added `len(prompt) < 15` skip after confirmatory block for non-confirmatory short prompts |
| Production memory count wrong (200 vs 2,453) | 1 | 200 was search API's candidate limit; actual count from `SELECT count(*)` |
