# Task Plan: Option B — End-to-End LongMemEval (Leaderboard-Comparable)

## Goal
Upgrade CEMS's LongMemEval eval from retrieval-only (Recall@5 on Oracle) to a full end-to-end benchmark (answer accuracy on S variant) that produces scores directly comparable to the Mastra leaderboard (84.23% GPT-4o).

## Architecture

### Current Eval (longmemeval.py)
```
Oracle dataset (1-6 sessions) → Bulk ingest → Search → Recall@5
```

### New Eval (longmemeval_e2e.py)
```
S dataset (40 sessions) → Bulk ingest → Search → LLM answers → LLM judges → Accuracy
```

### Key Design Decisions
1. **New file, not modification** — keep `longmemeval.py` for retrieval-only eval (backward compat)
2. **Reuse shared infra** — import `CEMSEvalClient`, `collect_all_sessions`, `format_session_content` from existing eval
3. **OpenRouter for all LLM calls** — GPT-4o via OpenRouter (same client as CEMS uses everywhere)
4. **Shared store** (not per-question isolation) — matches CEMS production behavior. If cross-contamination is an issue, we'll add isolation later.
5. **S variant first** — 277 MB, ~40 sessions per question, ~115k tokens total. Leaderboard standard.

## Phases

### Phase 1: Dataset + Data Structures
**Status:** pending

**Files to create:**
- `src/cems/eval/longmemeval_e2e.py` — main end-to-end eval module

**Work:**
- Add `LONGMEMEVAL_S_URL` pointing to `xiaowu0162/longmemeval-cleaned/longmemeval_s_cleaned.json`
- Add `download_longmemeval_s()` — downloads 277 MB S variant
- Define `E2EResult` dataclass: question_id, question_type, question, ground_truth, generated_answer, judge_verdict (bool), judge_explanation, retrieved_session_ids, correct_session_ids, recall_any, search_time_ms, answer_time_ms, judge_time_ms
- Define `E2ESummary` dataclass: total_questions, correct_count, accuracy, by_type (with per-type accuracy), macro_accuracy (average of per-type accuracies — this is the leaderboard metric)
- Import shared utilities from `longmemeval.py`: CEMSEvalClient, format_session_content, collect_all_sessions

### Phase 2: Answer Generation
**Status:** pending

**Work in `longmemeval_e2e.py`:**
- Create `generate_answer()` function:
  - Input: question (str), context (str, formatted retrieved memories), model (str)
  - System prompt: `"You are a helpful assistant with access to extensive conversation history. Use the provided context to answer the user's question. If the information is not available in the context, say so."`
  - Uses OpenRouterClient with `temperature=0`, `fast_route=False` (GPT-4o not on fast providers)
  - Returns generated answer string
- Context formatting: take search results, format each memory's content with its source_ref/session_id
- Handle empty search results gracefully (let the model say "I don't know")

### Phase 3: LLM Judge
**Status:** pending

**Work in `longmemeval_e2e.py`:**
- Create `judge_answer()` function:
  - Input: question, correct_answer, generated_answer, question_type, model
  - Returns (verdict: bool, explanation: str)
- Type-specific judge prompts (adapted from Mastra/LongMemEval paper):
  - **Standard** (single-session-user, single-session-assistant, multi-session): "Does the response contain the correct answer?"
  - **Temporal-reasoning**: "Does the response contain the correct answer? Allow minor date discrepancies (off by one day/time)."
  - **Knowledge-update**: "Does the response contain the UPDATED answer? It's acceptable if both old and new answers appear, as long as the updated answer is present."
  - **Single-session-preference**: "Does the response correctly reflect the user's preference or personal information? Be lenient — partial matches are acceptable."
  - **Abstention**: "Did the model correctly identify that this information is not available? The model should NOT fabricate an answer."
- Parse YES/NO from judge response (with fallback regex)
- Default to NO on parse failure (conservative)

### Phase 4: Main Eval Loop + Scoring
**Status:** pending

**Work in `longmemeval_e2e.py`:**
- Create `run_e2e_eval()` function:
  1. Collect all unique sessions → bulk ingest (reuse existing logic)
  2. For each question:
     a. Search CEMS (same as current eval)
     b. Format retrieved context
     c. Generate answer via LLM
     d. Judge answer via LLM
     e. Record result
  3. Compute per-type accuracy + macro accuracy
- Progress reporting: `[42/500] temporal-reasoning: ✓ (search 120ms, answer 1.2s, judge 0.8s)`
- Handle API errors gracefully (retry once, then mark as incorrect)

### Phase 5: CLI + Entry Point
**Status:** pending

**Files to create:**
- `src/cems/eval/longmemeval_e2e.py` — add `main()` with CLI (same file)

**CLI arguments:**
- `--questions N` — number of questions (default: 500 for full, use 5 for testing)
- `--api-url` — CEMS API URL (default: http://localhost:8765)
- `--api-key` — CEMS API key
- `--reader-model` — model for answer generation (default: `openai/gpt-4o`)
- `--judge-model` — model for judging (default: `openai/gpt-4o`)
- `--output FILE` — JSON output file with detailed results
- `--verbose` — show detailed per-question output
- `--no-cleanup` — keep eval memories after run
- `--dataset` — `oracle` or `s` (default: `s`)
- `--no-clean-stale` — skip stale data cleanup

**Entry point:** `python -m cems.eval.longmemeval_e2e --questions 5`

### Phase 6: Unit Tests
**Status:** pending

**Files to create:**
- `tests/test_longmemeval_e2e.py` — unit tests for new module

**Tests:**
- `test_generate_answer_formats_context` — verify context formatting
- `test_judge_standard_yes` / `test_judge_standard_no` — binary parsing
- `test_judge_type_specific_prompts` — each type gets correct prompt
- `test_e2e_result_scoring` — macro accuracy calculation
- `test_download_s_variant` — URL is correct, file saves
- `test_context_truncation` — long results get truncated to token budget

---

## Validation Plan (for validator agent)

1. Run unit tests: `.venv/bin/python3 -m pytest tests/test_longmemeval_e2e.py -xvs`
2. Run full test suite: `.venv/bin/python3 -m pytest tests/ -x -q` (ensure no regressions)
3. Rebuild Docker: `docker compose build cems-server && docker compose up -d cems-server`
4. Wait for healthy: `curl http://localhost:8765/health`
5. Run mini eval (5 questions): `cd /Users/razvan/Development/cems && .venv/bin/python3 -m cems.eval.longmemeval_e2e --questions 5 --api-url http://localhost:8765 --api-key <key> --verbose --output /tmp/eval_e2e_test.json --no-cleanup`
6. Check Docker logs: `docker logs cems-server --tail 100`
7. Read output: `cat /tmp/eval_e2e_test.json`
8. Report results back

## Expected Output

```
LONGMEMEVAL E2E BENCHMARK FOR CEMS
============================================================
Dataset: S variant (40 sessions/question)
Reader model: openai/gpt-4o
Judge model: openai/gpt-4o
Questions: 500

Phase 1: Collecting unique sessions...
  Found ~20,000 unique sessions across 500 questions
Phase 2: Bulk ingesting...
  Ingested 20,000 sessions in ~300s
Phase 3: Running end-to-end eval...

[1/500] temporal-reasoning: ✓ (search 120ms, answer 1.2s, judge 0.8s)
[2/500] multi-session: ✗ (search 85ms, answer 1.5s, judge 0.9s)
...

============================================================
EVALUATION SUMMARY
============================================================
Overall: 420/500 = 84.0%
Macro accuracy (leaderboard metric): 82.5%

By type:
  single-session-user:       62/70  = 88.6%
  single-session-assistant:  48/56  = 85.7%
  single-session-preference: 26/30  = 86.7%
  multi-session:            108/133 = 81.2%
  knowledge-update:          59/78  = 75.6%
  temporal-reasoning:       105/133 = 78.9%

Comparison to leaderboard:
  Mastra OM (gpt-4o): 84.23%
  CEMS:               82.5%  ← us
  Supermemory:        81.6%
  Mastra RAG:         80.0%

Total time: 45min | Cost: ~$35
```

## Cost Estimate (500 questions)

| Component | Calls | Cost |
|-----------|-------|------|
| Session ingestion (~20K) | 20K embeddings | ~$2 |
| Search (500 questions) | 500 hybrid searches | ~$0.50 |
| Answer generation (500) | 500 GPT-4o calls | ~$15-25 |
| Judge (500) | 500 GPT-4o calls | ~$10-15 |
| **Total** | | **~$30-45** |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| | | |
