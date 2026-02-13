# Move Observer Runtime from ~/.claude/observer → ~/.cems/observer

## Goal
Move all observer runtime files (PID, lock, state, signals, logs) from `~/.claude/observer/` to `~/.cems/observer/`. The observer watches Claude, Codex, and Cursor — it's a CEMS subsystem, not Claude-specific.

## Change Summary
All references to `Path.home() / ".claude" / "observer"` become `Path.home() / ".cems" / "observer"`.

---

## Phase 1: Core daemon code `status: pending`
**Files:**
- `src/cems/observer/__main__.py` — lines 18-20: OBSERVER_DIR, PID_FILE, LOCK_FILE
- `src/cems/observer/state.py` — line 15: OBSERVER_STATE_DIR
- `src/cems/observer/signals.py` — line 18: SIGNALS_DIR

**Change:** `.claude` → `.cems` in the Path constants

## Phase 2: Hook code (dev copies) `status: pending`
**Files:**
- `hooks/utils/observer_manager.py` — line 21: OBSERVER_DIR + derived paths
- `hooks/cems_stop.py` — line 132: signals_dir
- `hooks/cems_pre_compact.py` — line 126: signals_dir

**Change:** `.claude` → `.cems` in signal/observer paths

## Phase 3: Bundled hook copies `status: pending`
**Files:**
- `src/cems/data/claude/hooks/utils/observer_manager.py` — line 21
- `src/cems/data/claude/hooks/cems_stop.py` — line 132
- `src/cems/data/claude/hooks/cems_pre_compact.py` — line 126

**Change:** Sync from dev copies after Phase 2

## Phase 4: Migration helper in daemon startup `status: pending`
**What:** Add auto-migration in `__main__.py` that moves existing files from `~/.claude/observer/` to `~/.cems/observer/` on first run. This handles existing installs gracefully.
- Move state .json files, signals/, daemon.log
- Don't move .pid or .lock (daemon restarts fresh)

## Phase 5: Tests `status: pending`
**Files:**
- `tests/test_observer_manager.py` — no path changes needed (uses tmp_path + patches)
- `tests/test_observer.py` — no path changes needed (patches OBSERVER_STATE_DIR/SIGNALS_DIR)
- `tests/test_adapters.py` — no path changes needed (patches constants)
- `tests/test_signals.py` — no path changes needed (patches SIGNALS_DIR)

**Action:** Run full test suite to verify. Tests mock the dir constants so they should pass without changes.

## Phase 6: Install to runtime + version bump `status: pending`
- Copy updated hooks to `~/.claude/hooks/` (via `cems setup` or manual cp)
- Bump version 0.4.4 → 0.4.5
- Run integration tests
- Commit & push (triggers production deploy + auto-update)

## Phase 7: Docs update `status: pending`
- `research/observer-v2-multi-tool.md` — update path references
- `research/option-d-observer-plan.md` — update path references
- MEMORY.md — update observer paths

---

## Decisions
- **Migration**: Auto-migrate on daemon startup (Phase 4). Existing state files are valuable, don't want users to lose observation history.
- **No symlink**: Clean move, not symlink. Simpler.
- **PID/lock files**: Don't migrate — daemon writes fresh ones on start.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |
