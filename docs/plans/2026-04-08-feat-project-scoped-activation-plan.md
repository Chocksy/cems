---
title: "feat: Project-Scoped CEMS Activation"
type: feat
date: 2026-04-08
brainstorm: docs/brainstorms/2026-04-08-project-scoped-activation-brainstorm.md
---

# feat: Project-Scoped CEMS Activation

## Overview

Enable a "global install, project-scoped activation" deployment model where CEMS is installed globally (hooks, MCP, daemon) but only activates when a `.cems/credentials` file exists in the project. Three focused changes: flip credential precedence so project files win over env vars, make MCP tools return a friendly message when unconfigured instead of crashing, and add interactive prompting to `cems setup --project`.

## Problem Statement

1. **Env var bleed**: Developers with `CEMS_API_URL`/`CEMS_API_KEY` in `.zshrc` can't use per-project credentials — env vars always win (short-circuit at `resolve_credentials()` line 72-80).
2. **MCP crashes when unconfigured**: When no credentials exist, `API_URL = ""` causes `_request()` to build invalid URLs like `""/api/memory/search` — urllib throws confusing errors instead of a helpful message.
3. **Project setup requires flags**: `cems setup --project` demands `--api-url` and `--api-key` as CLI flags — no interactive prompting, unlike the global `cems setup` which prompts interactively.

## Proposed Solution

### Change 1: Flip Credential Precedence

**New order** (was: env > project > global):
1. Per-project `.cems/credentials` (walk up from CWD) — **wins**
2. Environment variables (`CEMS_API_URL` + `CEMS_API_KEY`)
3. Global `~/.cems/credentials`

**Files to change:**

| File | Role |
|------|------|
| `src/cems/shared/credentials.py` | Used by MCP, CLI, daemon imports |
| `hooks/utils/credentials.py` | Used by all Claude Code hooks (standalone) |
| `src/cems/data/claude/hooks/utils/credentials.py` | Bundled copy installed by `cems setup` |

**The daemon (`src/cems/observer/daemon.py` CredentialResolver) already has file-over-env precedence — no change needed.**

In each `resolve_credentials()`, swap steps 1 and 2:

```python
def resolve_credentials(cwd: str | None = None) -> dict[str, str]:
    """Resolve CEMS credentials with full precedence chain.

    1. Per-project .cems/credentials (walk up from CWD, stop before $HOME)
    2. Environment variables (both CEMS_API_URL and CEMS_API_KEY must be set)
    3. Global ~/.cems/credentials (fallback)
    """
    # 1. Walk up from CWD looking for project .cems/credentials
    if cwd:
        project_path = find_project_credentials(cwd)  # or _find_project_credentials
        if project_path:
            return parse_credentials_file(project_path)  # or _parse_credentials_file

    # 2. Check env vars — require BOTH URL and key
    env_url = os.environ.get("CEMS_API_URL", "")
    env_key = os.environ.get("CEMS_API_KEY", "")
    if env_url and env_key:
        result = {"CEMS_API_URL": env_url, "CEMS_API_KEY": env_key}
        for k in ("CEMS_TEAM_ID", "CEMS_SEARCH_MODE"):
            v = os.environ.get(k, "")
            if v:
                result[k] = v
        return result

    # 3. Global fallback
    global_path = os.getenv("CEMS_CREDENTIALS_FILE", _DEFAULT_CREDENTIALS_PATH)
    return parse_credentials_file(global_path)  # or _parse_credentials_file
```

**CI is unaffected**: CI repos won't have `.cems/credentials`, so env vars naturally apply as step 2.

### Change 2: MCP Graceful No-Op

Add a guard to each `@mcp.tool()` function in `src/cems/mcp_stdio.py`:

```python
_NOT_CONFIGURED_MSG = (
    "CEMS is not configured for this project. "
    "Run: cems setup --project"
)

@mcp.tool()
def memory_search(...) -> str:
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    # ... existing logic
```

Apply to all 7 tools: `memory_search`, `memory_add`, `memory_forget`, `memory_update`, `memory_maintenance`, `memory_pin`, `memory_get`.

The `instructions` line already handles this pattern: `instructions = _fetch_profile() if API_URL and API_KEY else ""`.

### Change 3: Interactive Project Setup

Modify `_setup_project_credentials()` in `src/cems/commands/setup.py`:

**Current**: requires `--api-url` and `--api-key` flags, aborts if missing.
**New**: if flags not provided and stdin is TTY, prompt interactively (same pattern as `_setup_credentials()`).

```python
def _setup_project_credentials(api_url: str | None, api_key: str | None) -> None:
    cwd = os.getcwd()
    cems_dir = Path(cwd) / ".cems"
    creds_file = cems_dir / "credentials"

    console.print()
    console.print(f"[bold]Project CEMS Setup[/bold] -- {cwd}")
    console.print()

    # Interactive prompting if flags not provided
    if not api_url or not api_key:
        if not _is_interactive():
            console.print("[red]--api-url and --api-key required in non-interactive mode[/red]")
            raise click.Abort()

        if not api_url:
            api_url = click.prompt("CEMS API URL", default="http://localhost:8765", show_default=True)
        if not api_key:
            api_key = click.prompt("CEMS API Key", hide_input=True)
        if not api_key:
            console.print("[red]API key is required.[/red]")
            raise click.Abort()

    # ... rest of existing logic (write file, gitignore, team discovery, version check)
```

**Additionally**, in the main `_setup_credentials()` interactive flow, add a prompt asking where to store credentials:

```python
# After collecting api_url and api_key interactively...
location = click.prompt(
    "Store credentials",
    type=click.Choice(["global", "project"], case_sensitive=False),
    default="global",
    show_default=True,
)
if location == "project":
    _setup_project_credentials(api_url, api_key)
    return True
# ... existing global write logic
```

## Acceptance Criteria

- [x] Project `.cems/credentials` takes precedence over env vars in all 3 credential modules
- [x] Env vars still work when no project credentials exist (CI compatibility)
- [x] Global `~/.cems/credentials` still works as final fallback
- [x] MCP tools return `"CEMS is not configured for this project. Run: cems setup --project"` when unconfigured
- [x] MCP tools do NOT crash with confusing urllib errors when unconfigured
- [x] `cems setup --project` prompts interactively for URL and key when flags omitted
- [x] `cems setup --project` still works non-interactively with `--api-url` and `--api-key` flags
- [x] `cems setup` (global) offers choice: store in project or global
- [x] Non-TTY environments (CI) get clear error messages, not hangs
- [x] All 3 credential files are kept in sync (shared, hooks, bundled copy)
- [x] Existing tests updated, new tests added for flipped precedence and MCP no-op

## Test Plan

### Update existing test: `tests/test_per_project_config.py`

**Flip `test_env_vars_override_project`** — rename to `test_project_overrides_env_vars`:
```python
def test_project_overrides_env_vars(self, tmp_path, monkeypatch):
    """Project .cems/credentials wins over env vars."""
    # Set env vars
    monkeypatch.setenv("CEMS_API_URL", "https://env-server.com")
    monkeypatch.setenv("CEMS_API_KEY", "env-key")
    # Create project credentials
    cems_dir = tmp_path / ".cems"
    cems_dir.mkdir()
    (cems_dir / "credentials").write_text("CEMS_API_URL=https://project.com\nCEMS_API_KEY=proj-key\n")
    creds = resolve_credentials(str(tmp_path))
    assert creds["CEMS_API_URL"] == "https://project.com"
    assert creds["CEMS_API_KEY"] == "proj-key"
```

**Add `test_env_vars_used_without_project_file`**:
```python
def test_env_vars_used_without_project_file(self, tmp_path, monkeypatch):
    """Env vars apply when no project .cems/credentials exists."""
    monkeypatch.setenv("CEMS_API_URL", "https://env-server.com")
    monkeypatch.setenv("CEMS_API_KEY", "env-key")
    monkeypatch.setenv("CEMS_CREDENTIALS_FILE", "/dev/null")
    creds = resolve_credentials(str(tmp_path))
    assert creds["CEMS_API_URL"] == "https://env-server.com"
```

### Add to `tests/test_per_project_config.py`: shared module tests

Mirror the hooks tests for `src/cems/shared/credentials.py` — currently only the hooks copy is tested.

### Add to `tests/test_mcp_stdio.py`: no-op behavior

```python
class TestNoCredentials:
    """MCP tools return friendly message when unconfigured."""

    def test_memory_search_no_creds(self, monkeypatch):
        monkeypatch.setattr(mcp_module, "API_URL", "")
        monkeypatch.setattr(mcp_module, "API_KEY", "")
        result = mcp_module.memory_search("test query")
        assert "not configured" in result.lower()

    # ... same for all 7 tools
```

### Manual test: interactive setup

```bash
# Test interactive project setup (no flags)
cd /tmp/test-project && git init
cems setup --project
# Should prompt for URL and key

# Test global setup with location choice
cd /tmp/test-project
cems setup --claude
# Should ask "Store credentials: global/project"
```

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| Bundled copy divergence | Checklist item: always update all 3 credential files. Consider adding a CI check that diffs them. |
| Breaking existing users with env vars | Env vars still work as step 2 when no project file exists. Document the change in release notes. |
| Cache invalidation after precedence flip | Cache is keyed by resolved file path, not by precedence tier. No cache changes needed. |
| MCP module-level side effects in tests | Follow existing `_real_get_config` pattern in `test_mcp_stdio.py`. Use `monkeypatch.setattr` on module globals. |

## References

- Brainstorm: `docs/brainstorms/2026-04-08-project-scoped-activation-brainstorm.md`
- Original per-project config plan: `docs/plans/2026-03-25-feat-per-project-cems-configuration-plan.md`
- Shared credentials: `src/cems/shared/credentials.py`
- Hooks credentials: `hooks/utils/credentials.py`
- Bundled credentials: `src/cems/data/claude/hooks/utils/credentials.py`
- MCP server: `src/cems/mcp_stdio.py`
- Setup command: `src/cems/commands/setup.py`
- Existing tests: `tests/test_per_project_config.py`, `tests/test_mcp_stdio.py`
