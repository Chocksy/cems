# Move Observer to ~/.cems — Findings

## Files with `~/.claude/observer` references

### Core daemon (3 files)
| File | Line | Constant | Current Value |
|------|------|----------|---------------|
| `src/cems/observer/__main__.py` | 18 | `OBSERVER_DIR` | `~/.claude/observer` |
| `src/cems/observer/__main__.py` | 19 | `PID_FILE` | derived from OBSERVER_DIR |
| `src/cems/observer/__main__.py` | 20 | `LOCK_FILE` | derived from OBSERVER_DIR |
| `src/cems/observer/state.py` | 15 | `OBSERVER_STATE_DIR` | `~/.claude/observer` |
| `src/cems/observer/signals.py` | 18 | `SIGNALS_DIR` | `~/.claude/observer/signals` |

### Hook code (3 dev + 3 bundled = 6 files)
| File | Line | Reference |
|------|------|-----------|
| `hooks/utils/observer_manager.py` | 21-24 | `OBSERVER_DIR` + PID/COOLDOWN/LOCK |
| `hooks/cems_stop.py` | 132 | `signals_dir` inline |
| `hooks/cems_pre_compact.py` | 126 | `signals_dir` inline |
| (+ bundled copies in `src/cems/data/claude/hooks/`) | | |

### Tests (4 files — all use patches, no hardcoded paths)
- `tests/test_observer_manager.py` — patches OBSERVER_DIR constant
- `tests/test_observer.py` — patches OBSERVER_STATE_DIR, SIGNALS_DIR
- `tests/test_adapters.py` — patches OBSERVER_STATE_DIR, SIGNALS_DIR
- `tests/test_signals.py` — patches SIGNALS_DIR

### Docs (2 files)
- `research/observer-v2-multi-tool.md` line 154
- `research/option-d-observer-plan.md` line 132

## Runtime directory contents (~/.claude/observer/)
- `daemon.pid` — current daemon PID
- `daemon.lock` — singleton flock
- `daemon.log` — 48KB log file
- `.spawn.lock` — spawn race prevention
- `*.json` — ~25 session state files
- `signals/` — signal files (6 files)

## Key insight
All tests use `tmp_path` + mock patches for directory constants. No test has hardcoded `~/.claude/observer`. So tests pass without modification — just need to run the suite to confirm.
