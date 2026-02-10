# Task Plan: PreCompact Handover Hook for CEMS

## Goal
When Claude Code auto-compacts (context window fills up), capture the full session transcript and re-inject CEMS memories after compaction. This prevents amnesia during long sessions.

## Current Problem
- Auto-compaction happens silently mid-session
- Full transcript is lost (replaced with a summary)
- Post-compaction Claude has no CEMS context (no profile, no relevant memories)
- Our `Stop` hook only captures transcript at session END, not at compaction time
- **FOUND BUG:** `cems_session_start.py:112` explicitly skips `compact` events: `if source in ("resume", "compact"): sys.exit(0)` — this means even if SessionStart fires after compaction, we ignore it

## Approach
Three changes:

1. **Add central hook event logging** — all hooks append to one JSONL file so we can observe exactly which hooks fire and when
2. **Create PreCompact[auto] hook** → send full transcript to CEMS before it's lost
3. **Fix SessionStart to handle `compact`** → remove `compact` from the skip list so profile gets re-injected after compaction

---

## Phase 0: Add central hook event logger `[pending]`

**File:** `hooks/utils/hook_logger.py` (shared utility)

A simple function all hooks can call:
```python
def log_hook_event(event_name, session_id, extra=None):
    """Append one line to ~/.claude/hooks/logs/hook_events.jsonl"""
```

Each line: `{"ts": "...", "event": "SessionStart", "session_id": "...", "source": "compact", ...}`

Then add `log_hook_event()` calls to:
- `cems_session_start.py` (log event + source/matcher)
- `stop.py` (log event)
- `pre_tool_use.py` (log event + tool name)
- `user_prompts_submit.py` (log event)
- NEW `pre_compact.py` (log event + trigger)

**Central log file:** `~/.claude/hooks/logs/hook_events.jsonl`

This lets us:
- Open a separate Claude Code instance, do work, run `/compact`
- Then `tail -f ~/.claude/hooks/logs/hook_events.jsonl` to see exactly what fired
- Answer the matcher question empirically

## Phase 1: Create PreCompact hook script `[pending]`

**File:** `hooks/pre_compact.py` (canonical source in CEMS repo)

What it does:
- Reads stdin JSON (gets `transcript_path`, `session_id`, `trigger`)
- Logs the event via `log_hook_event()`
- Reads the full JSONL transcript
- Sends to CEMS `/api/session/analyze` (same as `stop.py:analyze_session()`)
- Fire-and-forget (non-blocking, informational only)

**Only fires on `auto` matcher** — manual `/compact` is user-controlled.

## Phase 2: Fix SessionStart to handle compact events `[pending]`

**File:** `hooks/cems_session_start.py`

Change line 112 from:
```python
if is_background_agent or source in ("resume", "compact"):
    sys.exit(0)
```
To:
```python
if is_background_agent or source == "resume":
    sys.exit(0)
```

This way, after compaction, the profile (prefs, guidelines, gate rules, recent memories) gets re-injected into the fresh context window. Same behavior as a new session — exactly what we want.

We keep skipping `resume` because that's a session restore where context is already intact.

## Phase 3: Wire hooks in settings.json `[pending]`

**File:** `~/.claude/settings.json`

Add PreCompact entry:
```json
"PreCompact": [
  {
    "matcher": "auto",
    "hooks": [
      { "type": "command", "command": "uv run ~/.claude/hooks/pre_compact.py" }
    ]
  }
]
```

No need to add a separate `SessionStart[compact]` entry — the existing empty matcher `""` already matches all events including `compact`. We just need to remove the skip in the Python code (Phase 2).

## Phase 4: Install + update install.sh `[pending]`

1. Copy `pre_compact.py` to `~/.claude/hooks/`
2. Copy updated `hook_logger.py` to `~/.claude/hooks/utils/`
3. Update `hooks/install.sh` in CEMS repo

## Phase 5: Test with logging `[pending]`

1. Deploy all hooks
2. Open a separate Claude Code instance
3. Have a conversation, then run `/compact`
4. Check `~/.claude/hooks/logs/hook_events.jsonl` to verify:
   - PreCompact hook fired (with trigger=manual or auto)
   - SessionStart hook fired after compaction (with source=compact)
   - CEMS profile was re-injected
5. Check CEMS API logs to verify transcript was analyzed

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `hooks/utils/hook_logger.py` | CREATE | Central event logger utility |
| `hooks/pre_compact.py` | CREATE | PreCompact hook script |
| `hooks/cems_session_start.py` | MODIFY | Remove `compact` from skip list |
| `hooks/stop.py` | MODIFY | Add log_hook_event() call |
| `hooks/pre_tool_use.py` | MODIFY | Add log_hook_event() call |
| `hooks/user_prompts_submit.py` | MODIFY | Add log_hook_event() call |
| `hooks/install.sh` | MODIFY | Add new files to install list |
| `~/.claude/settings.json` | MODIFY | Add PreCompact entry |

## Key Decisions
- Central JSONL log file (not per-session) for easy `tail -f` observability
- Reuse `stop.py`'s `analyze_session()` pattern in pre_compact.py (duplicate ~30 lines rather than refactoring shared module)
- Only fire PreCompact on `auto` (not `manual`)
- Keep skipping `resume` in SessionStart (context intact), stop skipping `compact` (context lost)
