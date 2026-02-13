# Progress: One-Command CEMS Install

## Session: 2026-02-13 (continued)

### Bug Fixes from Code Review — COMPLETE
- Fixed `_migrate_old_hook_names()` — uses full path pattern matching instead of suffix
- Fixed Cursor hooks — added `_read_credentials()` for `~/.cems/credentials` fallback
- Fixed Cursor hooks — added `chmod +x` after copying hook files
- README — removed all Mem0/Kuzu/Qdrant references

### Deep Architecture Investigation — COMPLETE
- Ran 3 parallel agents (codex-investigator, deploy research, logo research)
- Ground truth captured in `findings.md`
- Key findings: NO Mem0, NO Kuzu, NO Qdrant — just PostgreSQL+pgvector+OpenRouter
- No uninstall command existed

### README Rewrite — COMPLETE
- Completely rewrote README from scratch based on ground truth
- Added shields.io badges (license, python, MCP, Claude Code, Recall@5)
- Correct services: postgres, cems-server, cems-mcp (no Qdrant)
- Correct architecture: pgvector+tsvector, OpenRouter embeddings, 9-stage pipeline
- Correct features: observer daemon, maintenance jobs, MCP integration
- Full API reference in collapsible section
- Documented uninstall command

### Uninstall Command — COMPLETE
- Created `src/cems/commands/uninstall.py`
- Removes: Claude hooks, Cursor hooks, skills, settings.json entries
- Options: `--all` (also remove credentials), `--yes` (skip confirmation)
- Registered in `cli.py`
- All 417 tests pass

### Deploy Research Findings
- Platforms with deploy buttons (Railway, Render, DO) DON'T support docker-compose
- Platforms with docker-compose (Coolify, Portainer) are self-hosted — no buttons
- Best option: keep using Coolify (already have it)
- Railway template possible but high effort

### Logo/Banner Research
- Can generate with Gemini via OpenRouter (already have key)
- SVG banner recommended for dark/light mode support
- `<picture>` element for automatic theme switching

## Session: 2026-02-12

### Pre-planning: Observer + Install Fixes — COMPLETE
- Fixed `observer_manager.py` to use `shutil.which("cems-observer")`
- Added `cems-observer` entry point to `pyproject.toml`
- Rewrote root `install.sh` to use `~/.cems/credentials`
- Updated tests — 417 pass
