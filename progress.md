# Progress: One-Command CEMS Install

## Session: 2026-02-12

### Pre-planning: Observer + Install Fixes — COMPLETE
- Fixed `observer_manager.py` to use `shutil.which("cems-observer")` — no CEMS_PROJECT_ROOT needed
- Added `cems-observer` entry point to `pyproject.toml`
- Rewrote root `install.sh` to use `~/.cems/credentials` + canonical hooks
- Simplified `hooks/install.sh`
- Updated tests — 28 pass
- Full test suite: 417 passed, 7 skipped

### Phase 1: Bundle hooks as package data — COMPLETE
- Created `src/cems/data/` with all hooks, utils, skills, cursor hooks, settings template
- `importlib.resources.files("cems.data")` works correctly
- Hatchling includes it automatically via existing `packages = ["src/cems"]`
- Created `scripts/sync-package-data.sh` for dev convenience

### Phase 2: `cems setup` CLI command — COMPLETE
- Created `src/cems/commands/setup.py` with interactive IDE selection + credential prompts
- Registered in `src/cems/cli.py`
- Settings.json merge preserves existing non-CEMS config
- Tested: `cems setup --help` works, `cems setup --claude` installs correctly

### Phase 3: Remote install script — COMPLETE
- Created `remote-install.sh` — installs uv, then `uv tool install cems from git`, then runs `cems setup`
- URL: `curl -sSf https://raw.githubusercontent.com/chocksy/cems/main/remote-install.sh | bash`

### Phase 4: Repo cleanup — COMPLETE
- Deleted `claude-setup/` (stale, only had 2 of 6 hooks)
- Deleted `skills/` top-level (moved to `src/cems/data/claude/skills/cems/`)
- Deleted `cursor-setup/` (moved to `src/cems/data/cursor/`)
- Deleted `deploy/setup-client.sh` and `deploy/client-config.example.json` (obsolete)
- Simplified `install.sh` → slim dev script that delegates to `cems setup`
- Simplified `hooks/install.sh` → dev-only, points to `cems setup` for non-devs
- Updated `scripts/sync-package-data.sh` (removed cursor-setup/skills references)
- Updated `README.md` with new install flow
- Updated `IMPLEMENTATION_STATUS.md` with new file paths

### Phase 5: Tests — COMPLETE
- Full test suite: **417 passed, 7 skipped** in 31.66s
- Package data accessible via `importlib.resources`: verified
- `cems setup --help`: verified
- `cems-observer` on PATH: verified

## Session: 2026-02-12 (continued)

### Hook Naming + Versioning + Non-interactive Setup — COMPLETE

#### Phase 1: Hook renames to `cems_` prefix
- Renamed 8 hook files (4 in `hooks/`, 4 in `src/cems/data/claude/hooks/`)
- Updated references in 8 files: setup.py, settings.json, sync script, install.sh, test_hooks.py, test_hooks_integration.py, README.md, findings.md
- Added `_migrate_old_hook_names()` to remove old-named entries from settings.json
- Added old hook file cleanup in `_install_claude_hooks()`
- Verified: `cems setup --claude` installs new names, removes old files

#### Phase 2: Non-interactive `cems setup`
- Added `--api-url` and `--api-key` CLI options
- Added `_is_interactive()` TTY detection via `sys.stdin.isatty()`
- Non-interactive mode defaults to `--claude`
- `hide_input=True` for API key prompt
- Updated `remote-install.sh` to pass through `CEMS_API_KEY`/`CEMS_API_URL` env vars

#### Phase 3: Versioning
- `importlib.metadata.version("cems")` — single source of truth from pyproject.toml
- `cems --version` shows version via `@click.version_option()`
- Remote installer is idempotent — re-run to upgrade

#### Phase 4: TTS separation
- TTS stripped from stop.py (previous commit)
- No further code changes needed

#### Tests
- Full test suite: **417 passed, 7 skipped**

## Summary of All Changes

### Created
- `remote-install.sh` — curl|bash remote installer
- `src/cems/data/` — package data (hooks, skills, settings, cursor)
- `src/cems/commands/setup.py` — `cems setup` CLI command
- `scripts/sync-package-data.sh` — dev sync script

### Deleted
- `claude-setup/` — stale hooks + settings
- `skills/` — duplicated in package data
- `cursor-setup/` — duplicated in package data
- `deploy/setup-client.sh` — obsolete MCP setup
- `deploy/client-config.example.json` — obsolete

### Renamed (hooks to `cems_` prefix)
- `hooks/user_prompts_submit.py` → `hooks/cems_user_prompts_submit.py`
- `hooks/stop.py` → `hooks/cems_stop.py`
- `hooks/pre_tool_use.py` → `hooks/cems_pre_tool_use.py`
- `hooks/pre_compact.py` → `hooks/cems_pre_compact.py`
- (Same renames in `src/cems/data/claude/hooks/`)

### Modified
- `install.sh` — slim dev script delegating to `cems setup`
- `hooks/install.sh` — simplified, dev-only
- `src/cems/cli.py` — registered setup command, added `--version`
- `src/cems/__init__.py` — version from importlib.metadata
- `src/cems/commands/setup.py` — non-interactive support, migration logic
- `src/cems/data/claude/settings.json` — new hook names
- `scripts/sync-package-data.sh` — new hook names
- `README.md` — new install instructions + hook names
- `tests/test_hooks.py` — updated hook name references
- `tests/test_hooks_integration.py` — updated hook name constants
