# Phase 4+5 Research Findings: Smart Dedup + Conflict Detection

## Key Architecture Decision: Combine Phases 4 and 5

**Codex-investigator recommendation (adopted):** Conflicts should be detected as a byproduct of the consolidation scan, not via a separate O(N^2) pass. Phase 5's original design of scanning ALL pairs in 0.60-0.92 range has quadratic complexity (500 docs = 124,750 pairs). Combined approach keeps cost bounded at ~30 LLM calls per nightly run.

---

## Finding 1: Auto-Merge at 0.92 Is Dangerous

The current binary threshold (`>0.92 = auto-merge`) can destroy distinct information:
- "User deployed CEMS to Coolify on Hetzner" (0.93 similarity with)
- "User deployed CEMS to Coolify on Railway"

**Decision:** Raise auto-merge to 0.98+ (near-exact only). Use LLM classification for 0.80-0.98.

---

## Finding 2: Metadata Guards Prevent LLM Hallucination

Before calling LLM to classify a pair, check metadata:
- Different `category` → never merge (different kinds of knowledge)
- Different `source_ref` → never merge (different projects)
- Different `tags` containing "session:" → related but distinct session observations

This prevents the highest-risk failure mode: LLM incorrectly classifying related-but-distinct memories as "duplicate".

---

## Finding 3: Don't Store Conflicts in `memory_documents`

Codex-investigator correctly identified three problems with using `category="memory-conflict"`:
1. Shows up in `get_all_documents()` used by SummarizationJob/ReindexJob
2. Conflict text would match vector search queries (contains both memories' content)
3. Conflict lifecycle (resolve/dismiss) differs from memory lifecycle (archive/decay)

**Decision:** Use a separate `memory_conflicts` table with FK cascading. Simple schema, purpose-built.

---

## Finding 4: 30-Day Window Is Unnecessary

If two memories have coexisted for 30 days without being caught by the 7-day window, they are either:
(a) from different time periods and already processed, or
(b) genuinely distinct.

**Decision:** Keep the existing 7-day window for consolidation. The LLM tier doesn't benefit from wider scope.

---

## Finding 5: LLM Call Cost Is Negligible

- Classification call: ~50 input tokens per pair via Gemini 2.5 Flash
- Cost per call: ~$0.00003
- Typical pairs per nightly run: 10-30
- Total cost: <$0.001/run
- Real concern is latency, not cost: keep calls sequential to avoid rate limits

---

## Finding 6: Profile Endpoint Is the Right Place for Conflicts

`api_memory_profile` at `memory.py:549` already assembles session context. Adding a "Memory Conflicts" section here means:
- No new hook complexity
- Conflicts appear automatically in every SessionStart
- Profile already queries multiple document categories — one more query is trivial

---

## LLM Model Choice

Use `google/gemini-2.5-flash` for classification, consistent with:
- `observation_extraction.py` (`OBSERVER_MODEL = "google/gemini-2.5-flash"`)
- `observation_reflection.py` (`REFLECTOR_MODEL = "google/gemini-2.5-flash"`)

Must use `fast_route=False` — Gemini not available on Cerebras/Groq/SambaNova.

---

## Existing Code Reference Points

| What | Where | Notes |
|------|-------|-------|
| ConsolidationJob | `maintenance/consolidation.py` | Rewrite `_merge_duplicates()` to three-tier |
| `merge_memory_contents()` | `llm/summarization.py:86` | Keep for actual merging |
| `search_chunks()` | `db/document_store.py` | Returns `score` (cosine, 0-1) |
| `embed_batch()` | `memory._async_embedder` | Returns `list[list[float]]` |
| `client.complete()` | `llm/client.py:111` | Sync call, `fast_route=False` for Gemini |
| `config.duplicate_similarity_threshold` | `config.py:286` | Currently 0.92 |
| Maintenance API | `api/handlers/memory.py:771` | Job dispatch dict |
| Test pattern | `tests/test_maintenance.py` | `_run()` + `mock_memory` fixture |
| Profile endpoint | `api/handlers/memory.py:549` | Add conflict section here |
