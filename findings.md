# Fix Observer Daemon Document Duplication — Findings

## Production Evidence (2026-02-13)

### Duplicate Session Summaries
- 6 docs all tagged `session:b40eb706`, all category `session-summary`
- Each contains different LLM-generated text about the same session
- Daemon log shows only 2 summary calls from current daemon instance
- Remaining 4 likely from pre-refactor daemon or hook overlap

### Duplicate Learnings
- "Mastra LongMemEval": 9 matches
- "haystack_sessions": 7 matches
- "LongMemEval dataset": 4 matches
- Caused by both stop + pre-compact hooks calling `analyze_session()` with same transcript

### Daemon State for b40eb706
```json
{
    "observation_count": 41,
    "epoch": 0,
    "last_finalized_at": 1771005988.741567,
    "is_done": false
}
```
41 incremental observations, no epoch bumps, session still active.

## Code Analysis

### BUG-1: TOCTOU in api_session_summarize (CRITICAL)
- `find_document_by_tag()` and `add_async()` are separate, non-transactional calls
- Two concurrent requests can both see "no existing" and both INSERT
- content_hash dedup doesn't help: LLM generates different text each time
- Fix: `SELECT ... FOR UPDATE` in a single transaction

### BUG-2: save_state() not atomic (MODERATE)
- Writes directly to file (no tmp+rename)
- Crash mid-write → truncated JSON → fresh state → re-send everything from byte 0
- Fix: tmp+rename pattern (same as write_signal)

### BUG-3: _spawn_daemon() PID race (LOW)
- Two hooks can both see "daemon not running" and both spawn
- flock ensures only one survives, but PID file briefly wrong
- Self-heals via pgrep fallback
- Fix: file lock around spawn operation

### NOT A BUG: Daemon singleton
- fcntl.flock works correctly
- Only 1 daemon running (PID 94206)
- flock released automatically on process exit (even SIGKILL)
