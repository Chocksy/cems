# CEMS owns its Claude Code hook entries — strip-and-replace

**Status**: Design — pending implementation plan
**Target version**: v0.13.3
**Owner**: Razvan
**Date**: 2026-04-30

## Problem

`cems setup` deploys a `~/.claude/settings.json` that hardcodes references
to `$HOME/.claude/hooks/run_with_uv.sh`, but CEMS has never shipped that
script. Existing developers (e.g. razvan) have a copy from a manual
install months ago, so they don't see the breakage. New developers
(e.g. Alex) hit `SessionStart:startup hook error: /bin/sh:
/Users/X/.claude/hooks/run_with_uv.sh: No such file or directory` on
their first session.

The deeper problem is that `_merge_settings` is *additive*: it dedups
hook commands by `cems_*.py` script-name and skips re-adding when the
script is already registered, regardless of how the surrounding command
differs. Consequence: even if we fix the template, no existing user
ever picks up the fix on `cems update` — their stale `run_with_uv.sh`
references remain forever. Same applies to any future hook command
change we ever want to ship.

References:
- `src/cems/data/claude/settings.json:9, 20, 31, 42, 53` — the broken
  template entries.
- `src/cems/commands/setup.py:686-690` — the script-name dedup logic
  that masks command-shape changes.
- `docs/maintenance-audit-2026-03-03.md:209, 366` — prior audit flagged
  the duplicate registration but the cleanup never landed.
- Test session 2026-04-30: confirmed Claude Code's hook spawn env on
  macOS inherits the parent shell's full `PATH` (including
  `/opt/homebrew/bin`), so `uv` is resolvable without a wrapper.

## Goals

1. New CEMS installs work out of the box — no missing-script error.
2. Existing CEMS installs heal automatically on `cems update` — stale
   references get rewritten to the current template form.
3. Future hook-command changes (whatever shape we choose tomorrow)
   reach existing users without manual surgery.
4. Non-CEMS hook entries in `~/.claude/settings.json` are never touched.

## Non-goals

- No changes to Cursor, Codex, or Goose hook setup. Cursor already
  uses direct invocation; Codex/Goose don't use shell-launched hooks.
- No changes to MCP server registration in `~/.claude.json`.
- No changes to which hook events CEMS subscribes to.
- No new `cems doctor` subcommand. The fix is automatic via
  `setup`/`update`; users don't have to run anything special.

## Design

### Ownership model

CEMS owns its hook entries authoritatively. On every `cems setup` and
`cems update`, CEMS removes any pre-existing hook entry it recognizes
as its own and re-injects fresh entries from the template. User
customization of CEMS hook *launch lines* is not supported (custom
behavior belongs inside the hook script, not in the launch command).

### Detector

A hook command counts as "owned by CEMS" if **any space-separated token
ends in a `cems_*.py` filename**. Concretely, all of the following
match and are stripped on next setup/update:

- `$HOME/.claude/hooks/run_with_uv.sh $HOME/.claude/hooks/cems_session_start.py`
- `$HOME/.claude/hooks/cems_session_start.py`
- `uv run --script $HOME/.claude/hooks/cems_session_start.py`

The `cems_` prefix is our reliable namespace — no marker field
needed. Existing hook files are listed in `src/cems/commands/setup.py`
already (`cems_session_start.py`, `cems_user_prompts_submit.py`,
`cems_stop.py`, `cems_pre_tool_use.py`, `cems_pre_compact.py`, plus
the deprecated `cems_post_tool_use.py`). All match the prefix rule.

### Strip granularity

Hook-level, not entry-level. The settings.json schema is:

```json
"SessionStart": [
  { "matcher": "", "hooks": [ {cmd1}, {cmd2} ] }
]
```

A user *could* register a custom hook in the same `hooks` array as a
CEMS hook (e.g. `[{cems_session_start}, {my_audit_log}]`). Stripping
the whole entry would clobber the custom hook. We strip just the
matching `hooks[i]` entries, drop the wrapping entry only if its
`hooks` list becomes empty, and drop the event key only if its
entries array becomes empty.

### Updated `_merge_settings` algorithm

```
1. Load existing ~/.claude/settings.json (preserve env, permissions,
   non-hook keys verbatim).
2. existing_hooks = existing.setdefault("hooks", {})
3. _migrate_old_hook_names(existing_hooks)
   — kept; renames pre-prefix legacy names so the detector catches them.
4. NEW: _strip_cems_hook_entries(existing_hooks)
   — for each event, for each entry, drop hooks whose command matches
     the detector. Drop empty entries. Drop empty event keys.
5. For each event in template_hooks:
     - If event is missing from existing_hooks, add the template's
       entries directly.
     - Else, append template entries to the existing array (since
       step 4 has guaranteed no CEMS-owned hooks survive there).
6. Write back to settings.json.
```

`_migrate_removed_hooks` (which currently special-cases the disabled
tool-learning hook) becomes redundant and is deleted — step 4 strips
`cems_post_tool_use.py` references because the detector matches them,
and step 5 doesn't re-add them because the template doesn't include
them. One fewer special case to maintain.

### Template change (W2 — drop the wrapper)

Test results from 2026-04-30 (logged in
`/tmp/cems-hook-probe-py.log`): Claude Code on macOS inherits the
parent shell's full `PATH`, so `uv` is directly resolvable from a hook
spawn. A Python file with `#!/usr/bin/env -S uv run --script` shebang
fires correctly when invoked as a bare command.

`src/cems/data/claude/settings.json` changes from

```json
"command": "$HOME/.claude/hooks/run_with_uv.sh $HOME/.claude/hooks/cems_session_start.py"
```

to

```json
"command": "$HOME/.claude/hooks/cems_session_start.py"
```

across all 5 hook events (`SessionStart`, `UserPromptSubmit`,
`PreToolUse`, `Stop`, `PreCompact`). This matches the pattern Cursor
already uses successfully ([src/cems/data/cursor/hooks.json](../src/cems/data/cursor/hooks.json)).

`run_with_uv.sh` is not added to `src/cems/data/claude/hooks/`. The
Python hook files already carry the `#!/usr/bin/env -S uv run --script`
shebang, which is sufficient.

### `_install_claude_hooks` change

[src/cems/commands/setup.py:464-502](../src/cems/commands/setup.py).
The hook-file copy block already `chmod`s `+x` on each Python hook
file. No new file to copy. The function continues to be the single
caller of `_merge_settings`, so the new strip-and-replace behavior
flows through both `cems setup` and `cems update --hooks`.

### Side effects (intentional)

- Existing users on next `cems update` get their `run_with_uv.sh`
  references rewritten to bare paths. The old script is left on disk
  (orphaned, harmless) — we do not delete user files.
- Idempotent: running `cems setup` twice in a row produces identical
  output.
- The `_merge_settings` comment block at lines 686-690 (which
  references both `uv run` and `run_with_uv.sh` as recognized
  patterns) becomes outdated; updated to describe the new
  strip-and-replace behavior.

### Edge cases

- **User has no `~/.claude/settings.json`**: `cems setup` creates it
  fresh from the template. Same as today.
- **User's `settings.json` has malformed JSON**:
  `_merge_settings` already handles this with
  `existing = {}` fallback (setup.py:660).
  Behavior unchanged.
- **User has a CEMS hook with a `matcher` we never use** (e.g.
  `"matcher": "Bash.*"`): stripped by the detector, replaced with
  the template's `matcher: ""`. Acceptable — CEMS doesn't support
  matcher customization, and template behavior wins.
- **User has manually edited a CEMS hook command** (e.g.
  `cems_session_start.py --debug`): launch-line customization is
  lost. Documented as unsupported — put logic in the hook itself.
- **A future template change introduces a new hook event** (e.g.
  `Notification`): step 5 adds it without disturbing existing event
  arrays.

## Testing strategy (TDD)

Unit tests covering `_strip_cems_hook_entries` and the integrated
`_merge_settings` behavior, extending the existing
[tests/test_setup.py](../tests/test_setup.py):

1. **Strip rewrites stale `run_with_uv.sh` references**.
   Given an existing `settings.json` containing the legacy command
   shape across all 5 events, after `_merge_settings` every CEMS hook
   command equals the new template form (`$HOME/.claude/hooks/cems_*.py`).
   *Failing-test phrasing*: assert that no `run_with_uv.sh` substring
   survives anywhere under `hooks` after merge.

2. **Strip removes deprecated `cems_post_tool_use.py`**.
   Given an existing `settings.json` with a `PostToolUse` event whose
   only hook references `cems_post_tool_use.py`, after merge the
   `PostToolUse` key is absent (since the template doesn't include it
   and no other entries remain).

3. **Non-CEMS hooks are preserved**.
   Given an existing `SessionStart` event with two hooks in one entry
   — one CEMS, one custom (e.g. `~/my_audit.sh`) — after merge the
   custom hook still exists alongside the new CEMS hook. The
   `matcher` and ordering of the user's entry is preserved.

4. **Empty entries are pruned**.
   Given an existing `SessionStart` event whose only entry contains
   only a CEMS hook, after merge that entry is replaced by the
   template entry (no orphaned `{"matcher": "", "hooks": []}` left
   behind).

5. **Idempotency**.
   Running `_merge_settings` twice on the same starting file produces
   byte-identical output (modulo JSON ordering — compare
   semantically).

6. **Legacy un-prefixed names are migrated then stripped**.
   Given an existing `Stop` event referencing the pre-rename
   `~/.claude/hooks/stop.py`, after merge the file path points to
   `cems_stop.py` (existing behavior preserved by step 3 of the
   algorithm running before strip).

End-to-end smoke test (subprocess style, similar to
`tests/test_observer.py::TestObserverStartup`):

7. **`cems setup --claude` against a temp `HOME` produces a valid
   settings.json that fires SessionStart**. We don't actually invoke
   `claude`, but we assert that for every event in the new
   `settings.json`, the referenced script file exists and is
   executable. Catches the original "missing wrapper" bug class.

## Migration / rollout

- Tag `v0.13.3` after merge.
- `.github/workflows/docker-publish.yml` builds and publishes
  `chocksy/cems-server:0.13.3` (no Python-package implications for
  Docker users — they don't use the hook system).
- Python users on `uv tool install "cems @ git+https://github.com/chocksy/cems.git"`
  pick up the fix on next `cems update`. `cems update` already calls
  `_redeploy_hooks` → `cems setup --<target>` → `_install_claude_hooks`
  → updated `_merge_settings`. Stale `run_with_uv.sh` references are
  rewritten in place.
- Send Alex (and any other affected colleague) a one-liner: `cems
  update`. No manual settings-file editing required.

## Out of scope (deferred)

- A `cems doctor` subcommand for explicit diagnostics. Not needed for
  this fix; revisit if we accumulate more drift classes.
- Cleanup of the orphaned `run_with_uv.sh` file on disk for users who
  manually installed it. CEMS shouldn't delete files it didn't create.
- Generalizing the strip-and-replace logic across other config files
  CEMS writes (e.g. `~/.claude.json` MCP entries). Different concern;
  that file already has its own replace-on-write logic at
  [`_register_claude_mcp_server`](../src/cems/commands/setup.py).

## Open questions

None at design time. Implementation plan to be drafted next via the
`writing-plans` skill.
