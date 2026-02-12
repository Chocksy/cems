# Task: Hook Naming Consistency + Versioning System + TTS Separation

## Goal
1. Rename all CEMS hooks to `cems_` prefix for consistency
2. Add a versioning/update system so `cems setup` doesn't go stale
3. Separate TTS announcement into a standalone hook (not part of CEMS core)

## Context
- Committed: package data bundling, `cems setup`, `remote-install.sh`, repo cleanup, TTS stripped from stop.py
- Remaining: hook renames touch ~30 test references, settings.json, setup.py, and the live `~/.claude/settings.json`

## Code Review Findings (codex-investigator)
Issues found in the committed code that should be fixed alongside the rename:

1. **`curl | bash` + interactive prompt conflict** — **FIXED**: Added `--api-url` and `--api-key` flags, TTY detection, non-interactive defaults.

2. **`_merge_settings()` brittleness** — **FIXED**: Added `_migrate_old_hook_names()` that removes old-named entries before merging new ones. Also removes old hook files from `~/.claude/hooks/`.

3. **API key prompt shows input** — **FIXED**: `hide_input=True`.

4. **Dual version source** — **FIXED**: `importlib.metadata.version("cems")` at runtime. `cems --version` works.

5. **Missing file in rename plan** — `tests/test_transcript.py` reference is test data for compaction, not a hook filename reference. Left as-is.

---

## Phase 1: Rename hooks to `cems_` prefix — `status: complete`

### Files renamed (8)
| Old Name | New Name |
|----------|----------|
| `hooks/user_prompts_submit.py` | `hooks/cems_user_prompts_submit.py` |
| `hooks/stop.py` | `hooks/cems_stop.py` |
| `hooks/pre_tool_use.py` | `hooks/cems_pre_tool_use.py` |
| `hooks/pre_compact.py` | `hooks/cems_pre_compact.py` |
| `src/cems/data/claude/hooks/user_prompts_submit.py` | `src/cems/data/claude/hooks/cems_user_prompts_submit.py` |
| `src/cems/data/claude/hooks/stop.py` | `src/cems/data/claude/hooks/cems_stop.py` |
| `src/cems/data/claude/hooks/pre_tool_use.py` | `src/cems/data/claude/hooks/cems_pre_tool_use.py` |
| `src/cems/data/claude/hooks/pre_compact.py` | `src/cems/data/claude/hooks/cems_pre_compact.py` |

### References updated (8 files)
1. `src/cems/commands/setup.py` — `hook_files` list ✓
2. `src/cems/data/claude/settings.json` — all `command` fields ✓
3. `scripts/sync-package-data.sh` — cp commands ✓
4. `hooks/install.sh` — for loop list ✓
5. `tests/test_hooks.py` — all filename references ✓
6. `tests/test_hooks_integration.py` — HOOK_USER_PROMPT, HOOK_PRE_TOOL_USE, HOOK_STOP constants ✓
7. `README.md` — directory structure section ✓
8. `findings.md` — access pattern example ✓

### Migration for existing installs ✓
- `_migrate_old_hook_names()` removes old-named entries from settings.json
- `_install_claude_hooks()` deletes old hook files from `~/.claude/hooks/`

## Phase 2: Fix `cems setup` for non-interactive use — `status: complete`

- Added `--api-url` and `--api-key` CLI options ✓
- Added `_is_interactive()` TTY detection ✓
- Non-interactive mode defaults to `--claude` ✓
- `hide_input=True` for API key prompt ✓
- Updated `remote-install.sh` to pass `CEMS_API_KEY`/`CEMS_API_URL` env vars ✓

## Phase 3: Versioning — `status: complete`

- `importlib.metadata.version("cems")` as single source of truth ✓
- `cems --version` shows version from pyproject.toml ✓
- Remote installer is idempotent (`--force` flag) — re-run to upgrade ✓

## Phase 4: TTS as separate hook — `status: complete`

- TTS already stripped from stop.py (previous commit)
- TTS is personal, NOT part of CEMS package — no code changes needed

## Phase 5: Tests + Verification — `status: complete`

- Full test suite: **417 passed, 7 skipped** ✓
- `cems --version`: verified ✓
- `cems setup --help`: shows new options ✓
- TTY detection: verified with piped stdin ✓

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Test failure after renaming hooks | 1 | Reverted renames, committed clean state, plan properly |
| Test failure (attempt 2) | 2 | Correctly updated all 8 reference files — 417 tests pass |
