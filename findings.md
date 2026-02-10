# Findings: PreCompact Handover Hook

## Research Date: 2026-02-10

## 1. PreCompact Hook EXISTS in Claude Code

**Confirmed.** Claude Code has 14 hook events. `PreCompact` fires before context compaction.

### Matcher values:
- `"auto"` — fires when context window fills up (automatic compaction)
- `"manual"` — fires when user runs `/compact`

### Input data (via stdin JSON):
```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/.../.claude/projects/.../abc123.jsonl",
  "cwd": "/Users/...",
  "permission_mode": "default",
  "hook_event_name": "PreCompact",
  "trigger": "manual|auto",
  "custom_instructions": ""
}
```

**Key insight:** `transcript_path` gives the FULL conversation (JSONL) BEFORE compaction.

## 2. SessionStart Hook Has "compact" Matcher

After compaction completes, `SessionStart` fires with matcher `"compact"`. Stdout from the hook becomes `additionalContext` injected into Claude's context.

## 3. BUG FOUND: cems_session_start.py skips compact!

**Line 112:**
```python
if is_background_agent or source in ("resume", "compact"):
    sys.exit(0)
```

This means even though SessionStart fires after compaction, our hook explicitly ignores it. Post-compaction Claude gets NO CEMS profile. This is a bug — `compact` should be handled like `startup` (profile needs re-injection since context is lost).

`resume` is correctly skipped (context intact on resume).

## 4. Existing Hook Inventory

| Hook | Script | Logging? |
|------|--------|----------|
| SessionStart (all) | `cems_session_start.py` | None |
| UserPromptSubmit | `user_prompts_submit.py` | None |
| PreToolUse | `pre_tool_use.py` | None |
| PostToolUse | `cems_post_tool_use.py` | None |
| Stop | `stop.py` | Per-session JSON |
| SubagentStop | `subagent_stop.py` | None |
| Notification | `notification.py` | None |

**No central event log.** Stop hook writes per-session JSON but there's no single place to see all hook events across the system.

## 5. CEMS Advantage Over Static HANDOVER.md

Unlike the Twitter approach (generating a static HANDOVER.md via `claude -p`), we can:
1. **PreCompact[auto]** → send transcript to CEMS for learning extraction
2. **SessionStart[compact]** → re-inject dynamic profile from all memories

Benefits: searchable, reusable across sessions, no file cleanup, leverages existing infrastructure.
