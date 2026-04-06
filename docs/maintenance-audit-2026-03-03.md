# CEMS Maintenance System Audit — 2026-03-03

## Server Status
- **Containers**: Both `cems-server` and `cems-mcp` running healthy
- **Database**: 4,936 active docs, 4,842 soft-deleted, 45 conflicts tracked
- **Container**: Fresh deploy (restart count 0), up since 18:12 UTC Mar 2

---

## Job-by-Job Assessment

### 1. ConsolidationJob (Nightly @ 3 AM) — WORKING WELL

Evidence from soft-delete timestamps at hour 3:

| Date | Deletes at 3AM | Conflicts Found |
|------|---------------|-----------------|
| Feb 14-19 | 500 -> 10 (declining as dupes cleared) | — |
| Feb 20-24 | **GAP** (container down/redeployed) | — |
| Feb 26-27 | 152, 159 | 6, 11 |
| Mar 1 | 197 | 4 |
| Mar 2 | 69 | 1 |

2,004 total soft-deletes at hour 3. Three-tier dedup working correctly. 45 conflicts stored.

**Verdict**: Solid.

---

### 2. SummarizationJob (Weekly Sunday @ 4 AM) — BROKEN BY DESIGN

**0 category summaries have ever been produced.**

**Root cause — critical query bug** at `summarization.py:57`:
```python
all_docs = await doc_store.get_all_documents(user_id, limit=500)
old_docs = [d for d in all_docs if _doc_age_exceeds(d, days=30)]
```

`get_all_documents()` (`document_store.py:780`) uses `ORDER BY created_at DESC LIMIT 500`.
With 4,936 total docs, this returns only the **500 newest documents**.
The 185 documents that are 30+ days old are at the tail and **never get fetched**.

**Secondary issue**: `_compress_by_category()` creates summary docs but **never soft-deletes the originals**.

**Verdict**: Dead code. Will never produce output while doc count > 500.

---

### 3. ReindexJob (Monthly 1st @ 5 AM) — NEVER RAN

System started ~Feb 10. March 1 at 5 AM should have fired but container was redeployed.
No evidence of re-embedding activity in any logs.

**Concerns**:
- Same `ORDER BY created_at DESC` issue at scale (limit=5000 vs 4936 docs — currently safe but won't scale)
- Re-embeds ALL docs including fresh ones — wasteful
- Sequential `update_async()` on 4,936 docs could exceed 768MB container memory or take hours
- No job history tracking to verify completion

**Verdict**: Likely never executed.

---

### 4. ObservationReflector (Nightly @ 3:30 AM) — WORKING, ALWAYS NO-OP

Threshold `MIN_OBSERVATIONS_THRESHOLD = 10` is never met:

| Project | Observations |
|---------|-------------|
| project:Chocksy/cems | 8 |
| project:EpicCoders/pxls | 7 |
| no-source | 6 |
| project:Chocksy/pos | 2 |

Max per project is 8, threshold is 10. Reflector runs, finds nothing, exits cleanly.

**Verdict**: Working as designed, just not active yet.

---

## Other Issues Found

### Embedding Failures (Active)
```
2026-03-02 18:28:29 - cems.embedding - ERROR - Embedding request failed:
```
Empty error message — `httpx.RequestError` has no string repr. Logging bug at `embedding.py:169`.
Should use `f"Embedding request failed: {type(e).__name__}: {e!r}"`. Transient — subsequent calls succeed.

### No Job Execution History
No table, no log marker, no API to answer "when did consolidation last run?" or "did summarization succeed?"
Only evidence is forensic analysis of soft-delete timestamps.

### Container Restarts Kill Scheduler
Gap Feb 20-24 shows maintenance stops during redeploys. APScheduler uses in-memory store.

### "all" Job Type Not Fault-Isolated
In `api_memory_maintenance`, if consolidation throws, summarization/reindex/reflect never run.

### CLI Missing `reflect`
`commands/maintenance.py` exposes consolidation, summarization, reindex, all — but not reflect.

---

## Category Breakdown (Production)

| Category | Count | Last Updated |
|----------|-------|-------------|
| general | 2,151 | 2026-03-02 |
| deployment | 336 | 2026-03-02 |
| development | 304 | 2026-03-02 |
| testing | 288 | 2026-03-02 |
| frontend | 249 | 2026-03-02 |
| cems | 212 | 2026-03-02 |
| infrastructure | 168 | 2026-03-02 |
| workflow | 167 | 2026-03-02 |
| debugging | 145 | 2026-03-02 |
| database | 124 | 2026-03-02 |
| session-summary | 106 | 2026-03-02 |
| refactoring | 99 | 2026-03-02 |
| ui | 89 | 2026-03-02 |
| configuration | 86 | 2026-03-02 |
| api | 80 | 2026-03-02 |
| architecture | 72 | 2026-03-02 |
| security | 61 | 2026-03-02 |
| project-management | 55 | 2026-03-02 |
| preferences | 35 | 2026-03-02 |
| environment | 35 | 2026-03-02 |
| observation | 23 | 2026-03-02 |
| performance | 19 | 2026-03-02 |
| monitoring | 18 | 2026-03-02 |
| documentation | 11 | 2026-03-02 |
| networking | 2 | 2026-03-01 |
| gate-rules | 1 | 2026-03-01 |
| authentication | 1 | 2026-03-01 |

**Total active**: 4,936 | **Soft-deleted**: 4,842

---

## Priority Fixes

1. **Fix SummarizationJob query ordering** — use `ORDER BY created_at ASC` or remove limit
2. **Add job execution logging** — `INSERT INTO maintenance_log (job_type, user_id, result, ran_at)`
3. **Fix empty embedding error messages** — use `repr(e)`
4. **Lower ObservationReflector threshold** — 10 is too high, consider 5
5. **Add original doc cleanup to SummarizationJob** — soft-delete originals after summarizing

---

## Observer System Investigation — CATASTROPHIC FAILURE

### Executive Summary

The observer daemon has **9.8% coverage** — it has observed 49 MB out of 503 MB of session data.
2,386 out of 2,471 session JSONL files have **never been tracked** by the daemon.
137 sessions have state files; of those, 26 were marked done with **zero observations**.

### The Numbers

| Metric | Value |
|--------|-------|
| Session JSONL files on disk | 2,471 |
| Sessions with observer state | 137 |
| Sessions never tracked | 2,386 (96.5%) |
| Total JSONL size | 503.1 MB |
| Total observed | 49.2 MB |
| **Data missed** | **453.9 MB (90.2%)** |
| Sessions done with 0 observations | 26 |
| Sessions done with >0 observations | 63 |
| Active sessions paused >2h | 47 |
| Unprocessed signals | 104 (84 stop, 18 compact) |

### Root Cause #0: Stop Hook Fires on EVERY Assistant Turn (THE BIGGEST BUG)

The Stop hook (`cems_stop.py:116`) writes a `"stop"` signal **every time it fires**.
Claude Code fires the Stop hook after **every assistant turn completion**, not just on session exit.

Verified from hook event log: session `fe51cadf` received **9 unique Stop events in 2 hours**
while the user was actively working.

```python
# cems_stop.py line 116 — no guard, fires unconditionally
if session_id:
    write_signal(session_id, "stop", "claude")
```

The daemon reads the first stop signal and marks `is_done=True`. The session is dead
after the very first assistant response. Every session is killed within seconds of starting.

### Root Cause #0b: Single Document Per Epoch + Rolling Window = Information Loss

Within a single epoch, ALL incremental observations update the **SAME document** via
`upsert_document_by_tag()`. The server appends new facts with `---` separator.

When the document exceeds **10K chars**, the server **truncates from the HEAD** — early facts
are permanently deleted, replaced with only the most recent ~5-7 observation windows.

For a 4-hour session generating 20+ observations without a compact signal, **60-70% of
extracted facts are irrecoverably lost**.

There is **NO auto-epoch mechanism**. New documents are only created when a compact signal
bumps the epoch. Since manual `/compact` doesn't fire the hook, and the stop hook kills
sessions on the first turn, most sessions get at most 1 document.

### Root Cause #0c: Duplicate Hook Execution

Every hook event is registered twice in `settings.json` — once via `run_with_uv.sh` wrapper
and once via bare `uv run`. Every hook fires twice. Every stop signal is written twice
(last-write-wins, so functionally identical, but wasteful).

---

### Root Cause #1: `is_done` Check Before Signal Check (daemon.py:348)

```python
# Line 348 — runs FIRST
if state.is_done:
    continue  # ← skips EVERYTHING including signal check

# Line 356 — NEVER REACHED for done sessions
sig = read_signal(session.session_id)
```

Once a session is marked `is_done=True` (by staleness or a stop signal), **ALL future signals
are orphaned**. The daemon never reads them. 104 signals are currently sitting unprocessed.

This means:
- Session goes idle for 5 min → staleness marks `is_done=True`
- User comes back, continues working → file grows to 50+ MB
- Stop hook fires → writes signal to `~/.cems/observer/signals/`
- Daemon sees `is_done=True` → **skips** → signal never read

### Root Cause #2: No Session Resurrection

There is **no mechanism** to reopen a "done" session. Once `is_done=True`, the session
is permanently invisible. The file can grow from 640KB to 58MB and the daemon will never
notice because it checks `is_done` before checking anything else.

Example from production — session `8b473f7b`:
- Started: Feb 25 20:12
- Marked done: Feb 25 20:18 (staleness after 6 min, 0 observations)
- JSONL grew to: **58 MB**
- Observer saw: **0 bytes** (never observed, marked done immediately)

### Root Cause #3: `max_age_hours=2` Discovery Filter

`daemon.py:331`: `adapter.discover_sessions(max_age_hours=2)`

Sessions whose JSONL file wasn't modified in the last 2 hours are **invisible** to the daemon.
Long-running sessions that pause for a lunch break or overnight become undiscoverable.

Of 2,471 total sessions, only 137 were ever tracked — the rest were either:
- Created before the daemon started (daemon started ~Feb 21)
- Not modified within the 2-hour discovery window during daemon polling

### Root Cause #4: `/clear` Writes No Signal

`/clear` resets the conversation but does NOT fire any Claude Code hook.
No stop, no compact, no signal. The observer has no idea `/clear` happened.

After `/clear`, the session continues with the same JSONL file (new messages appended).
The daemon, if it had the session tracked, continues watching file growth — but if the
session was already `is_done`, it's invisible.

### Root Cause #5: Manual `/compact` Doesn't Trigger PreCompact Hook

The PreCompact hook matcher is set to `"auto"` — it only fires on **automatic** compaction.
Manual `/compact` does NOT write any signal to the observer.

Signal breakdown: 84 stop signals, 18 compact signals, **0 clear signals**.
Only 5 sessions ever had an epoch bump (from compact signals).

### Root Cause #6: `extract_observations()` Is Dead Code

`extract_observations()` in `observation_extraction.py:133` is defined and exported but
**never called from any production code path**. The only caller is `tests/test_observation.py`.

The production path is:
1. Daemon detects growth/signal → calls `send_summary()` to `/api/session/summarize`
2. Server runs `extract_session_summary()` → creates `category="session-summary"` docs
3. There is **NO production path** that creates `category="observation"` docs

The 23 existing observations came from an **old code path** (`/api/session/observe`) that
was removed on Feb 17, 2026. No new observations have been created since.

### Root Cause #7: Staleness + Zero-Observation Trap

```python
def check_staleness(state):
    if state.observation_count == 0:
        return False  # never observed, don't finalize empty
```

Staleness only triggers for sessions with `observation_count > 0`. So sessions that never
met the 10KB growth threshold (many short sessions) are never finalized by staleness.

However, the **stop hook** can still mark them `is_done=True`:
```python
def handle_signal(sig, session, state, ...):
    if state.observation_count > 0:
        # finalize...
    if sig.type == "stop":
        state.is_done = True  # ← even with 0 observations
```

This creates 26 "zombie" sessions: done but never observed.

### HTTP Error Spam in Daemon Log

- 778 HTTP 500 errors from stuck sessions (019ca153, 019ca354, etc.)
- 102 HTTP 503 errors during container redeploys
- 13 HTTP 401 errors (early Feb, before API key configured)

The 500 errors come from sessions whose finalize calls fail server-side, but the daemon
doesn't retry intelligently — it just logs and moves on.

### Impact Assessment

With 9.8% coverage, the observer is essentially non-functional for its core purpose.
500+ MB of rich session transcripts are sitting on disk, never analyzed.

The 106 session-summary documents in production represent the **only** successful
observations — most from short sessions that happened to grow past 10KB in a single
burst before being finalized.

### What Would Fix This

1. **Process signals before `is_done` check** — move `read_signal()` above line 348
2. **Resurrect done sessions** — if JSONL file grew since `last_observed_bytes`,
   reset `is_done=False` and re-observe
3. **Increase discovery window** — `max_age_hours=24` or remove limit entirely
   (use state files for lifecycle, not file mtime for visibility)
4. **Add `/clear` signal** — write a signal from a new hook (or piggyback on SessionStart)
5. **Fix manual `/compact` hook matcher** — change from `"auto"` to `["auto", "manual"]`
   or remove the matcher entirely
6. **Wire up observation extraction** — either call `extract_observations()` from the
   summarize endpoint, or create a separate pipeline
7. **Process backlog signals** — one-time cleanup: read all 104 orphaned signals
   and process the corresponding sessions

---

## Updated Priority Fixes

### P0 — Observer Daemon (Session-Killing Bugs)
1. **Stop writing stop signals on every turn** — the Stop hook must distinguish "turn completed"
   from "session actually exiting". Options: remove signal write from stop hook entirely and
   rely on staleness, or detect a real exit condition.
2. **Make staleness non-terminal** — `handle_finalize()` should NOT set `is_done=True`. Only
   a real session exit (if detectable) should be terminal. Add guard: don't re-finalize if
   `last_finalized_at > last_growth_seen_at`.
3. **Add session resurrection** — in `run_cycle()`, before the `is_done` skip: if
   `session.file_size > state.last_observed_bytes`, reset `is_done=False`, bump epoch, re-observe.
4. **Add auto-epoch after N observations** — without compact signals, one document grows
   forever and old facts are lost via 10K char truncation. Add epoch bump every ~5 observations
   within an epoch. Requires an `epoch_observation_count` field that resets on epoch bump.
5. **Fix manual `/compact` hook matcher** — change from `"auto"` to match-all so manual
   `/compact` triggers PreCompact and bumps epoch.

### P1 — Observer Daemon (Data Loss)
6. **Process signals before `is_done` check** — move `read_signal()` above line 348
7. **Increase discovery window** — `max_age_hours=24` minimum
8. **Process 104 orphaned signals** — one-time cleanup script
9. **Fix duplicate hook registrations** — remove either `run_with_uv.sh` or bare `uv run` entries
10. **One-time backlog recovery** — reset `is_done` and bump epochs for sessions where file grew

### P2 — Server Maintenance Jobs
11. **Fix SummarizationJob query** — `ORDER BY created_at ASC` or remove limit
12. **Decide observation pipeline** — `extract_observations()` is dead code, `ObservationReflector`
    feeds on non-existent data. Either re-wire as post-processing on session summaries or delete.
13. **Add job execution logging** — track when jobs run and their results
14. **Fix embedding error messages** — use `repr(e)` instead of `str(e)`

### P3 — Nice to Have
15. **Add `/clear` hook/signal** — no Clear hook event exists in Claude Code, may need heuristic
16. **Add original doc cleanup to SummarizationJob** — soft-delete originals after summarizing
17. **CLI missing `reflect`** — add reflect command to maintenance CLI
18. **Fault-isolate "all" job type** — catch per-job exceptions
