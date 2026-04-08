# Project-Scoped CEMS Activation

**Date:** 2026-04-08
**Status:** Brainstorm complete — ready for planning

## What We're Building

Make CEMS work cleanly in a "global install, project-scoped activation" model. Colleagues install CEMS globally (hooks, MCP, daemon) but credentials only exist in company project repos via `.cems/credentials`. CEMS is silently inactive outside those repos.

Three changes:

### 1. Flip Credential Precedence (project file wins)

**Current order:**
1. Environment variables (CEMS_API_URL + CEMS_API_KEY) — wins
2. Per-project `.cems/credentials` (walk up from CWD)
3. Global `~/.cems/credentials`

**New order:**
1. Per-project `.cems/credentials` (walk up from CWD) — wins
2. Environment variables (CEMS_API_URL + CEMS_API_KEY)
3. Global `~/.cems/credentials`

**Why:** Env vars set in `.zshrc` (e.g., via `eval`) bleed into every project. With the current order, a developer can't override env vars with project-specific credentials. CI still works — CI repos won't have `.cems/credentials`, so env vars naturally apply.

**Files to change:**
- `src/cems/shared/credentials.py` — `resolve_credentials()`
- `hooks/utils/credentials.py` — `resolve_credentials()`
- `src/cems/data/claude/hooks/utils/credentials.py` — bundled copy
- Update docstrings/comments in all three

### 2. MCP Graceful No-Op

When no credentials are resolved, MCP tools should return a friendly message instead of crashing on empty URL.

**Current behavior:** `API_URL = ""` → `_request()` builds URL like `""/api/memory/search` → urllib crashes with a confusing error.

**New behavior:** Each tool checks `if not API_URL:` and returns `"CEMS is not configured for this project. Run: cems setup --project"`.

**File:** `src/cems/mcp_stdio.py`

**Note:** MCP still resolves credentials once at startup. Mid-session `cems setup` requires restarting Claude Code. This is acceptable — hooks already work per-call (each hook is a fresh process).

### 3. Interactive Project Setup

Enhance `cems setup` so `--project` can run interactively (prompt for API URL and key) instead of requiring `--api-url` and `--api-key` flags.

Also: when running `cems setup` interactively without `--project`, add a prompt asking "Where should credentials be stored?" with choices:
- **This project only** (writes to `CWD/.cems/credentials`)
- **Global** (writes to `~/.cems/credentials`)

**File:** `src/cems/commands/setup.py` — `_setup_project_credentials()` and `_setup_credentials()`

## Why This Approach

- **Soft boundary, not hard enforcement.** Colleagues can still add global `~/.cems/credentials` for personal use. The company setup just doesn't create one.
- **Project file wins over env vars** prevents credential bleed across projects when env vars are set in shell profiles.
- **Silent no-op** means zero noise in personal projects — hooks skip, MCP returns a helpful message.
- **YAGNI:** No config flags, no per-call MCP resolve, no hard lockout mode. The simplest change that solves the real problem.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Enforcement model | Soft boundary | Colleagues can opt in to global if they want |
| Credential precedence | Project file > env vars > global file | Prevents env var bleed, CI unaffected |
| MCP no-creds behavior | Return friendly message | Silent no-op, no crashes |
| MCP resolve timing | Keep once-at-startup | Hooks are already per-call; simplicity wins |
| Interactive setup | Add to `--project` flow | Remove friction for onboarding |

## Open Questions

1. **Daemon behavior with no credentials** — the observer daemon also starts globally. Should it also gracefully skip sessions where no credentials resolve? (Currently it tries and fails silently per-session — probably fine.)
2. **Should `cems setup` default to project-scoped when run inside a git repo?** Could detect `.git` and suggest project setup. Nice UX but maybe too opinionated.
3. **Test coverage** — `tests/test_per_project_config.py` exists. Need to add tests for the flipped precedence and MCP no-op behavior.
