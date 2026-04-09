---
title: "fix: Maintenance Hardening — Entity Page Protection & Tool Learning Removal"
type: fix
date: 2026-04-09
brainstorm: docs/brainstorms/2026-04-08-entity-aware-maintenance-brainstorm.md
prior_plan: docs/plans/2026-04-08-feat-entity-aware-maintenance-plan.md
---

# Maintenance Hardening — Entity Page Protection & Tool Learning Removal

## Overview

Two critical gaps found in the maintenance pipeline, plus removal of a redundant
memory creation source that accounts for 42% of all memories.

**Prior context**: The entity-aware maintenance plan from Apr 8 was mostly implemented
(relation builder, orphan assigner, simplified summarization, compilation staleness).
But two maintenance jobs were missed, and tool learning wasn't addressed.

## Problem Statement

### 1. Entity Pages Unprotected in Consolidation & Distillation

Entity pages (`category='entity-page'`) are stored as regular `memory_documents` rows.
`PROTECTED_CATEGORIES` at `__init__.py:4` includes `entity-page`, but two jobs don't use it:

- **ConsolidationJob** (`consolidation.py:131`): Only checks `pinned` tag. Two similar
  entity pages can be auto-merged (≥0.98 similarity), soft-deleting one. At Tier 2
  (0.80-0.98), `_metadata_distinct` allows it since both share `category='entity-page'`.

- **DistillationJob** (`distillation.py:87-94`): Processes ALL docs >500 chars. Entity
  page wiki content (2000+ chars) gets condensed to ~500 chars. The original is preserved
  in `content_detailed`, but search/agentic retrieval only sees the condensed version,
  defeating the purpose of comprehensive entity pages.

### 2. Tool Learning Hook Creates Excessive Memories

Production breakdown:
```
general:         1,793 (42%) ← mostly tool learning
session-summary: 1,116 (26%) ← observer daemon
entity-page:        40 ( 1%) ← compilation
(other):         1,327 (31%)
```

The PostToolUse hook (`hooks/cems_post_tool_use.py`) fires on every Edit, Write, Bash
(commit/install/docker), and Task completion. Each creates a new memory via
`POST /api/tool/learning` → `memory.add_async()`. This was designed for "SuperMemory-style
incremental learning" before the observer daemon existed.

**Why it's redundant now**: The observer daemon (`observer/daemon.py`) already captures
session activity holistically — it reads the full transcript, extracts learnings via LLM,
and creates session-summary documents. Tool learning adds noisy, granular duplicates of
what the observer already captures at a higher quality level.

**The tool learning hook only works in Claude Code** (PostToolUse event).
Cursor uses `afterAgentResponse` (different hook, different purpose). Codex and Goose
have no hook system — they rely on MCP only. So this is a Claude Code-only change.

### Comparison: Tool Learning vs Observer Daemon (Same Session)

Tested on session 8e9cb6fe (today's relation builder fix work):

| Source | Memories Created | Quality |
|--------|-----------------|---------|
| **Tool learning** | **19+ memories** | Granular debugging steps. 4 DECISION memories about self-relation approaches we iterated through and abandoned. Multiple WORKING_SOLUTION about the same deploy flow. |
| **Observer daemon** | **~5 session summaries** | Holistic. One covers the self-relation challenge end-to-end. Captures final decisions, not every intermediate step. |

The same fact — "push a git commit with version tag triggers CI pipeline" — exists as
3 separate tool-learning WORKING_SOLUTION memories (scored 0.87, 0.78, 0.71) vs 1
session-summary + 1 entity page. Zero tool-learning memories have `shown_count > 0` —
they never get shown to users because entity pages and session summaries rank higher.

**Verdict**: The observer daemon is strictly better. Tool learning is noise.

## Proposed Solution

### Phase 1: Entity Page Protection (critical, deploy immediately)

#### 1a. `src/cems/maintenance/consolidation.py` — Skip entity pages

Add entity-page check in two places:

**At line 131** (before processing each doc):
```python
# Skip pinned memories — fully untouchable
if "pinned" in (doc.get("tags") or []):
    continue

# Skip entity pages — these are compiled wiki documents
if doc.get("category") == "entity-page":
    continue
```

**At line 169** (before processing each merge candidate):
```python
dup_doc = await doc_store.get_document(chunk_doc_id, user_id=self.config.user_id)
if not dup_doc:
    continue

# Never merge/delete entity pages
if dup_doc.get("category") == "entity-page":
    continue
```

#### 1b. `src/cems/maintenance/distillation.py` — Skip entity pages

**At line 90-94** (in the candidates filter):
```python
candidates = [
    d for d in all_docs
    if len(d.get("content", "")) > DISTILLATION_THRESHOLD
    and "pinned" not in (d.get("tags") or [])
    and d.get("category") != "entity-page"
]
```

#### 1c. Tests

- Add test to `test_consolidation.py`: entity-page docs are never merged or deleted
- Add test to distillation tests: entity-page docs are never condensed
- Verify: 734+ tests pass

### Phase 2: Remove Tool Learning Hook (Claude Code only)

**Harness impact map:**

| Harness | Has PostToolUse? | Action |
|---------|-----------------|--------|
| Claude Code | YES — `~/.claude/settings.json` | Remove from settings.json + migration |
| Cursor | NO — uses `afterAgentResponse` | Nothing to do |
| Codex | NO — MCP only | Nothing to do |
| Goose | NO — MCP only | Nothing to do |

#### 2a. `src/cems/data/claude/settings.json` — Remove PostToolUse block

Remove the entire `PostToolUse` section from the template. This prevents new
installs from getting the hook.

```json
// REMOVE this entire block from settings.json template:
"PostToolUse": [
  {
    "matcher": "",
    "hooks": [
      {
        "type": "command",
        "command": "$HOME/.claude/hooks/run_with_uv.sh $HOME/.claude/hooks/cems_post_tool_use.py"
      }
    ]
  }
]
```

#### 2b. `src/cems/commands/setup.py` — Active migration for existing users

Add a `_migrate_removed_hooks()` function (following the precedent of
`_migrate_old_hook_names()` at line 643) that removes `cems_post_tool_use.py`
entries from the user's existing `settings.json`.

```python
def _migrate_removed_hooks(hooks: dict) -> bool:
    """Remove deprecated CEMS hooks from settings.

    Returns True if any hooks were removed (for messaging).

    Hooks removed:
    - cems_post_tool_use.py: Tool learning hook superseded by observer daemon.
      The daemon captures session learnings holistically, making per-tool-call
      learning redundant and noisy.
    """
    removed_scripts = {"cems_post_tool_use.py"}
    changed = False

    for event_name in list(hooks.keys()):
        original = hooks[event_name]
        hooks[event_name] = [
            entry for entry in original
            if not any(
                any(script in hook.get("command", "")
                    for script in removed_scripts)
                for hook in entry.get("hooks", [])
            )
        ]
        if len(hooks[event_name]) != len(original):
            changed = True
        # Remove empty event arrays
        if not hooks[event_name]:
            del hooks[event_name]

    return changed
```

Call this in `_merge_settings()` right after `_migrate_old_hook_names()`:
```python
_migrate_old_hook_names(existing_hooks)
if _migrate_removed_hooks(existing_hooks):
    console.print("  Removed tool learning hook (superseded by observer daemon)")
```

This runs automatically on both `cems setup` (interactive) and `cems update`
(which calls `cems setup --claude --api-url ... --api-key ...` non-interactively).

#### 2c. `setup.py:528-531` — Stop copying the hook file

Remove `cems_post_tool_use.py` from the `hook_files` list:

```python
# Before:
hook_files = [
    "cems_session_start.py", "cems_user_prompts_submit.py",
    "cems_post_tool_use.py", "cems_stop.py", "cems_pre_tool_use.py", "cems_pre_compact.py",
]

# After:
hook_files = [
    "cems_session_start.py", "cems_user_prompts_submit.py",
    "cems_stop.py", "cems_pre_tool_use.py", "cems_pre_compact.py",
]
```

#### 2d. Make the hook file a no-op (safety net)

Replace `hooks/cems_post_tool_use.py` and `src/cems/data/claude/hooks/cems_post_tool_use.py`
with a minimal no-op script. This covers the edge case where a user's settings.json still
references it (migration didn't run yet) — the hook exits cleanly instead of erroring.

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
CEMS PostToolUse Hook — DISABLED

Tool learning was superseded by the observer daemon, which captures session
learnings holistically. This hook is kept as a no-op for backwards
compatibility (users who haven't run `cems update` yet).

The observer daemon provides better quality learnings with less noise:
- Sees the full session transcript, not just individual tool calls
- Extracts final decisions, not intermediate debugging steps
- Creates 5-20x fewer memories per session
- Works across all harnesses (Claude Code, Cursor, Codex, Goose)
"""
import sys
sys.exit(0)
```

#### 2e. Server-side handler — Keep as-is

Keep `src/cems/api/handlers/tool.py` and the `/api/tool/learning` endpoint.
Harmless dead code — could be useful for future integrations.
Don't delete working code that might be useful later.

### Phase 3: Production Cleanup (manual, after deploy)

#### 3a. Soft-delete tool-learning memories on production

Run via Coolify SSH or direct DB access:

```sql
-- Preview: count tool-learning memories
SELECT COUNT(*) FROM memory_documents
WHERE deleted_at IS NULL
  AND 'tool-learning' = ANY(tags);

-- Soft-delete them
UPDATE memory_documents
SET deleted_at = NOW()
WHERE deleted_at IS NULL
  AND 'tool-learning' = ANY(tags);
```

This removes the ~1,793 noisy tool-learning memories. The observer daemon's
session-summaries (1,116 docs) already capture the same learnings better.

#### 3b. Trigger a maintenance sweep

After the soft-delete, run consolidation + relation builder + compilation
to process the remaining memories through the entity pipeline:

```bash
# Via CEMS API
curl -X POST .../api/memory/maintenance -d '{"job_type":"relations","limit":100}'
curl -X POST .../api/memory/maintenance -d '{"job_type":"compilation"}'
```

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/cems/maintenance/consolidation.py` | Modify | Add entity-page skip at lines 131 and 169 |
| `src/cems/maintenance/distillation.py` | Modify | Add entity-page filter in candidates list |
| `src/cems/data/claude/settings.json` | Modify | Remove PostToolUse block from template |
| `src/cems/commands/setup.py` | Modify | Add `_migrate_removed_hooks()`, remove from `hook_files` list |
| `hooks/cems_post_tool_use.py` | Modify | Replace with no-op (safety net for unmigrated users) |
| `src/cems/data/claude/hooks/cems_post_tool_use.py` | Modify | Same no-op — bundled copy |
| `tests/test_consolidation.py` | Modify | Add entity-page protection test |
| `pyproject.toml` | Modify | Bump version |

## Acceptance Criteria

### Phase 1: Entity Page Protection
- [ ] Consolidation skips entity-page documents (both as source and merge target)
- [ ] Distillation skips entity-page documents
- [ ] New tests verify entity-page protection in both jobs
- [ ] All existing tests pass (734+)

### Phase 2: Tool Learning Removal
- [ ] `src/cems/data/claude/settings.json` template has no PostToolUse block
- [ ] `_migrate_removed_hooks()` removes `cems_post_tool_use.py` from existing settings.json
- [ ] Migration prints `"Removed tool learning hook (superseded by observer daemon)"`
- [ ] `_install_claude_hooks()` no longer copies `cems_post_tool_use.py`
- [ ] Hook file is a no-op (exits 0 immediately) for unmigrated users
- [ ] `cems update` flow triggers migration automatically (via `_redeploy_hooks`)

### Phase 3: Production Cleanup
- [ ] Version bumped, tagged, pushed → CI builds Docker → Coolify deploys
- [ ] Production: tool-learning memories soft-deleted via SQL
- [ ] Production: relation builder + compilation sweep triggered

## Non-Goals

- Don't delete `/api/tool/learning` endpoint (harmless, might be useful later)
- Don't change session-summary behavior (observer daemon is working well)
- Don't make summarization more aggressive yet (wait for entity pipeline to catch up)
- Don't change consolidation/distillation thresholds (just add protection)
- Don't touch Cursor/Codex/Goose setup — they don't have PostToolUse

## References

- Prior brainstorm: `docs/brainstorms/2026-04-08-entity-aware-maintenance-brainstorm.md`
- Prior plan: `docs/plans/2026-04-08-feat-entity-aware-maintenance-plan.md`
- `PROTECTED_CATEGORIES`: `src/cems/maintenance/__init__.py:4`
- Consolidation: `src/cems/maintenance/consolidation.py`
- Distillation: `src/cems/maintenance/distillation.py`
- PostToolUse hook: `hooks/cems_post_tool_use.py`
- Tool learning handler: `src/cems/api/handlers/tool.py`
- Setup command: `src/cems/commands/setup.py`
