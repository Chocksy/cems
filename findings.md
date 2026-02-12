# Findings: One-Command CEMS Install

## Current Repo Structure (Duplicates/Stale)

### Hook Directories — 3 places, should be 1 canonical + 1 package data
| Directory | Files | Status |
|-----------|-------|--------|
| `hooks/` | 6 hooks + 5 utils = 11 files | **Canonical** (dev source) |
| `claude-setup/hooks/` | 2 hooks only | **STALE** — missing 4 hooks, no utils |
| `cursor-setup/hooks/` | 3 hooks | OK (different hooks for Cursor) |

### Skills — duplicated in 2 places
| Directory | Files | Status |
|-----------|-------|--------|
| `skills/` | 5 .md files | Top-level copy |
| `claude-setup/skills/cems/` | Same 5 .md files | Identical duplicate |

### Settings Templates
| File | Hooks Configured | Status |
|------|-----------------|--------|
| `claude-setup/settings.json` | 2 (UserPromptSubmit, Stop) | **STALE** — real setup has 7 hooks |
| `~/.claude/settings.json` (live) | 7 hooks (all events) | Reference for template |

### Install Scripts — 3 scripts doing different things
| Script | What It Does | Credentials Storage |
|--------|-------------|-------------------|
| `install.sh` (root) | CLI + IDE hooks | `~/.cems/credentials` (fixed this session) |
| `hooks/install.sh` | Hooks only | `~/.cems/credentials` |
| `deploy/setup-client.sh` | MCP server config | N/A (env vars) — obsolete |

## Settings.json Hook Template (from live config)
The proper settings.json needs these hook events:
- `SessionStart` → `cems_session_start.py`
- `UserPromptSubmit` → `user_prompts_submit.py`
- `PostToolUse` → `cems_post_tool_use.py`
- `PreToolUse` → `pre_tool_use.py`
- `PreCompact` → `pre_compact.py` (matcher: "auto")
- `Stop` → `stop.py`

Note: the live config also has non-CEMS hooks (post_tool_use.py, notification.py, subagent_stop.py).
The CEMS template should only include CEMS hooks. The `cems setup` command needs to MERGE
into existing settings, not overwrite.

## Package Data Strategy
- Use `importlib.resources` to read package data at runtime
- `src/cems/data/` directory with `__init__.py` makes it a package
- Hatchling includes all files in packages by default
- Access pattern: `importlib.resources.files("cems.data") / "claude" / "hooks" / "cems_stop.py"`

## Hatch Build Config
Current pyproject.toml has:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/cems"]
```
This already includes everything under `src/cems/`, so `src/cems/data/` will be included
automatically. No extra config needed as long as files exist there.
