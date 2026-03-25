---
title: "feat: Per-Project CEMS Configuration"
type: feat
date: 2026-03-25
brainstorm: docs/brainstorms/2026-03-25-per-project-config.md
---

# Per-Project CEMS Configuration

## Overview

Enable different projects to point to different CEMS servers via a `.cems/credentials` file in the project root. Hooks walk up from CWD to resolve credentials per-invocation. The observer daemon routes observations per-project using a client pool. Skills and hooks remain global — only credential routing changes.

## Problem Statement / Motivation

A team (e.g., Hubstaff) has deployed their own CEMS instance. When working in `hubstaff-server/`, all memory operations should route to Hubstaff's CEMS server exclusively. Currently, all hooks read from a single `~/.cems/credentials` file — there's no way to use different servers per project without manually swapping env vars.

## Proposed Solution

**Walk-up credential resolution**: `resolve_credentials(cwd)` walks up the directory tree from CWD looking for `.cems/credentials`. First match wins. If nothing found below `$HOME`, falls back to `~/.cems/credentials` (global default).

### Credential Resolution Order
```
1. Environment variables (CEMS_API_URL, CEMS_API_KEY) — highest priority (CI, testing)
2. .cems/credentials found by walking up from CWD — project override
3. ~/.cems/credentials — global fallback
```

### Walk-Up Boundary
Walk from CWD up to (but NOT including) `$HOME`. This prevents `~/.cems/credentials` from being found during walk-up (which would collapse the project/global distinction). After walk-up fails, explicitly fall back to `~/.cems/credentials`.

### Project Credentials File
Same dotenv format as global. **Both `CEMS_API_URL` and `CEMS_API_KEY` are required** — partial overrides (key from project, URL from global) are not supported to avoid silent auth mismatches.

```
# hubstaff-server/.cems/credentials
CEMS_API_URL=https://cems.hubstaff.com
CEMS_API_KEY=cems_ak_hubstaff_...
CEMS_TEAM_ID=abc123
```

## Technical Approach

### Phase 1: Core — Walk-Up Credential Resolution

The foundational piece everything else depends on.

#### 1.1 `hooks/utils/credentials.py` — Add `resolve_credentials(cwd)`

```python
# hooks/utils/credentials.py

def resolve_credentials(cwd: str | None = None) -> dict[str, str]:
    """Resolve credentials by walking up from CWD, falling back to global.

    Walk-up stops BEFORE $HOME to avoid finding ~/.cems/credentials
    during the walk (that's the global fallback, checked separately).

    Precedence: env vars > project .cems/credentials > ~/.cems/credentials
    """
    # 1. Check env vars first (always win)
    env_url = os.environ.get("CEMS_API_URL")
    env_key = os.environ.get("CEMS_API_KEY")
    if env_url and env_key:
        return {"CEMS_API_URL": env_url, "CEMS_API_KEY": env_key, **_env_extras()}

    # 2. Walk up from CWD looking for .cems/credentials
    home = str(Path.home())
    if cwd:
        path = Path(cwd).resolve()
        while str(path) != home and path != path.parent:
            project_creds = path / ".cems" / "credentials"
            if project_creds.is_file():
                return _load_credentials_file(str(project_creds))
            path = path.parent

    # 3. Global fallback
    return _load_credentials_file(_get_credentials_path())
```

**Key changes to existing code:**
- `_cache` becomes a `dict[str, dict]` keyed by resolved file path (not a flat dict)
- Add `get_cems_url(cwd=None)` and `get_cems_key(cwd=None)` overloads that delegate to `resolve_credentials(cwd)`
- Backward compatible: calling `get_cems_url()` without args behaves exactly as today
- `get_credentials_env(cwd=None)` passes CWD through for observer manager

**Files:**
- `hooks/utils/credentials.py` (~30 lines added/changed)
- `src/cems/data/claude/hooks/utils/credentials.py` (mirror, also has `get_search_mode()`)

#### 1.2 Update All Hooks — Defer Credential Resolution to `main()`

**Current anti-pattern** (all 4 API-calling hooks):
```python
# Module level — CWD not yet known!
CEMS_API_URL = get_cems_url()
CEMS_API_KEY = get_cems_key()
```

**New pattern:**
```python
# No module-level credential assignment

def main():
    input_data = json.loads(sys.stdin.read())
    cwd = input_data.get("cwd", "")

    # Resolve credentials with CWD
    creds = resolve_credentials(cwd)
    api_url = creds.get("CEMS_API_URL", "")
    api_key = creds.get("CEMS_API_KEY", "")

    if not api_url or not api_key:
        return  # No credentials configured, exit silently

    # ... rest of hook logic using api_url, api_key ...
```

**Hooks to update (4 files + 4 bundled copies = 8 files):**

| Hook | File | Current Lines | Change |
|------|------|--------------|--------|
| SessionStart | `hooks/cems_session_start.py` | L35-36 | Move to main(), pass creds to helpers |
| UserPromptSubmit | `hooks/cems_user_prompts_submit.py` | L40-41 | Move to main(), pass creds to helpers |
| PostToolUse | `hooks/cems_post_tool_use.py` | L43-44 | Move to main(), pass creds to helpers |
| Stop | `hooks/cems_stop.py` | L38-39 | Move to main(), pass creds to helpers |

**PreToolUse** and **PreCompact** don't make API calls — no credential changes needed.

Each helper function that currently references the module-level `CEMS_API_URL` / `CEMS_API_KEY` globals needs a `(api_url, api_key)` parameter added. This is a mechanical refactor.

#### 1.3 Cache Key Scoping — Prevent Cross-Server Contamination

Gate rules cached at `~/.cems/cache/gate_rules/{project}.json` could collide if two projects share the same git remote but point to different CEMS servers.

**Fix:** Include a URL hash in cache paths.

```python
import hashlib

def _cache_prefix(api_url: str) -> str:
    """Short hash of API URL for cache key scoping."""
    return hashlib.sha256(api_url.encode()).hexdigest()[:8]

# Cache path becomes:
# ~/.cems/cache/gate_rules/{url_hash}_{project}.json
```

**Files:**
- `hooks/cems_pre_tool_use.py` — gate rule cache path (line ~90)
- `hooks/cems_session_start.py` — foundation cache path
- `hooks/cems_user_prompts_submit.py` — gate cache loading
- All bundled copies

### Phase 2: Observer Daemon — Per-Project Routing

#### 2.1 Add `cwd` to Signal Format

**`src/cems/observer/signals.py`:**

```python
@dataclass
class Signal:
    type: SignalType
    ts: float
    tool: ToolName
    cwd: str = ""  # NEW: project directory for credential resolution

def write_signal(session_id: str, signal_type: str, tool: str = "claude", cwd: str = "") -> None:
    data = {
        "type": signal_type,
        "ts": time.time(),
        "tool": tool,
        "cwd": cwd,  # NEW
    }
    # ... write to file ...
```

**Signal writers to update:**
- `hooks/cems_pre_compact.py` L61 — pass `cwd` from stdin
- `hooks/cems_stop.py` — pass `cwd` from stdin
- `cursor-plugin/hooks/cems_stop.py` — pass `cwd`

#### 2.2 Daemon Credential Resolution — Walk-Up in `__main__.py`

The daemon already has a duplicate credentials parser (lines 165-192 of `__main__.py`). Extend it with walk-up logic (same algorithm as hooks, duplicated intentionally since the daemon can't import from `hooks/`).

```python
# src/cems/observer/__main__.py

def _resolve_credentials_for_cwd(cwd: str | None = None) -> tuple[str, str]:
    """Walk-up credential resolution (daemon-side duplicate of hooks/utils/credentials.py).

    Daemon intentionally prefers file over env vars (env may be stale for long-running daemon).
    """
    home = str(Path.home())
    if cwd:
        path = Path(cwd).resolve()
        while str(path) != home and path != path.parent:
            project_creds = path / ".cems" / "credentials"
            if project_creds.is_file():
                creds = _parse_creds_file(str(project_creds))
                return creds.get("CEMS_API_URL", ""), creds.get("CEMS_API_KEY", "")
            path = path.parent

    # Global fallback (existing behavior)
    creds = _parse_creds_file(str(Path.home() / ".cems" / "credentials"))
    return creds.get("CEMS_API_URL", ""), creds.get("CEMS_API_KEY", "")
```

**Note:** Daemon keeps "file over env var" priority (existing intentional choice for long-running daemons where env vars go stale).

#### 2.3 Client Pool in `daemon.py`

Replace the flat `api_url: str, api_key: str` parameters with a resolver pattern.

```python
# src/cems/observer/daemon.py

class ClientPool:
    """Manages API clients keyed by CEMS server URL."""

    def __init__(self, resolve_fn):
        self._resolve = resolve_fn  # (cwd: str) -> (api_url, api_key)
        self._clients: dict[str, tuple[str, str]] = {}  # url -> (url, key)
        self._failures: dict[str, int] = {}  # url -> consecutive failure count

    def get_for_cwd(self, cwd: str) -> tuple[str, str]:
        api_url, api_key = self._resolve(cwd)
        self._clients[api_url] = (api_url, api_key)
        return api_url, api_key

    def record_failure(self, api_url: str):
        self._failures[api_url] = self._failures.get(api_url, 0) + 1

    def record_success(self, api_url: str):
        self._failures[api_url] = 0

    def should_skip(self, api_url: str, max_failures: int = 10) -> bool:
        return self._failures.get(api_url, 0) >= max_failures
```

**Refactor `run_cycle()` and `run_daemon()`:**
- Replace `api_url: str, api_key: str` params with `client_pool: ClientPool`
- In `process_session_growth()`: `api_url, api_key = client_pool.get_for_cwd(session.cwd)`
- In `handle_signal()`: resolve from `signal.cwd` or fall back to `session.cwd`
- In `handle_finalize()`: same pattern
- Failure counting: `client_pool.record_failure(api_url)` per URL, not global

**Functions to update in `daemon.py`:**
- `send_summary()` L55-131 — already takes `api_url, api_key`, just need to pass per-session values
- `run_cycle()` L347 — accept `client_pool` instead of flat creds
- `process_session_growth()` — resolve creds from `session.cwd`
- `handle_signal()` L165 — resolve from `signal.cwd`
- `handle_finalize()` — resolve from session
- `run_daemon()` L444 — create `ClientPool`, pass to `run_cycle()`

### Phase 3: CLI — `cems setup --project`

#### 3.1 Add `--project` Flag to Setup Command

**`src/cems/commands/setup.py`:**

```python
@click.option("--project", is_flag=True, help="Create per-project credentials in CWD")
def setup_cmd(claude, cursor, codex, goose, api_url, api_key, project, ...):
    if project:
        _setup_project_credentials(api_url, api_key)
        return
    # ... existing global setup flow ...
```

**`_setup_project_credentials()` flow:**
1. Validate `--api-url` and `--api-key` are provided (required for `--project`, no interactive mode)
2. Create `.cems/` directory in CWD
3. Write `.cems/credentials` with `chmod 600`
4. Append `.cems/` to `.gitignore` (create if needed, skip if already present)
5. Optionally append to `.dockerignore` if that file exists
6. Call `_discover_team()` against the project server, write `CEMS_TEAM_ID` if found
7. Check server version, warn if behind client
8. Print confirmation: `✓ Project credentials written to ./.cems/credentials`

#### 3.2 Version Skew Check

Add a lightweight version check function:

```python
def _check_server_version(api_url: str, api_key: str) -> str | None:
    """Check server version, return version string or None on failure."""
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(f"{api_url}/api/health", headers={"X-API-Key": api_key}),
            timeout=5
        )
        data = json.loads(resp.read())
        return data.get("version")
    except Exception:
        return None
```

Called during:
- `cems setup --project` — warn if server version < client version
- `cems update` (optional enhancement) — check known project servers after update

#### 3.3 `cems env` — CWD Awareness

**`src/cems/commands/env.py`:**

Update to use `resolve_credentials(os.getcwd())` instead of hardcoded global path. When a per-project file is found, output those credentials and print a comment indicating the source:

```bash
$ cd ~/Development/hubstaff-server
$ cems env
# Credentials from: ./cems/credentials (project)
export CEMS_API_URL="https://cems.hubstaff.com"
export CEMS_API_KEY="cems_ak_hubstaff_..."
```

### Phase 4: Testing

#### 4.1 Unit Tests for `resolve_credentials()`

```python
# tests/test_credentials_resolution.py

class TestResolveCredentials:
    def test_finds_project_credentials(self, tmp_path):
        """Walk-up finds .cems/credentials in project root."""

    def test_stops_before_home(self, tmp_path, monkeypatch):
        """Walk-up does not find ~/.cems/credentials during walk."""

    def test_falls_back_to_global(self, tmp_path):
        """No project creds → uses ~/.cems/credentials."""

    def test_env_vars_override_project(self, monkeypatch, tmp_path):
        """CEMS_API_URL env var wins over project file."""

    def test_nested_project_finds_parent(self, tmp_path):
        """CWD in subdirectory finds .cems/credentials at project root."""

    def test_symlink_resolution(self, tmp_path):
        """Symlinked CWD resolves to real path before walking up."""

    def test_no_cwd_uses_global(self):
        """resolve_credentials(None) returns global credentials."""

    def test_malformed_project_file(self, tmp_path):
        """Missing CEMS_API_URL in project file → skip, fall back to global."""

    def test_backward_compat_no_cwd(self):
        """get_cems_url() without args behaves identically to before."""
```

#### 4.2 Daemon Routing Tests

```python
# tests/test_daemon_routing.py

class TestClientPool:
    def test_routes_different_cwds_to_different_servers(self):
    def test_caches_client_by_url(self):
    def test_per_url_failure_tracking(self):
    def test_one_server_down_doesnt_affect_other(self):
```

#### 4.3 Integration Test

```python
# tests/test_project_credentials_integration.py

class TestPerProjectIntegration:
    def test_hook_uses_project_credentials(self, tmp_path):
        """SessionStart hook with CWD in a project uses project server."""

    def test_signal_carries_cwd(self, tmp_path):
        """PreCompact signal includes CWD from hook stdin."""
```

#### 4.4 CLI Tests

```python
# tests/test_setup_project.py

class TestSetupProject:
    def test_creates_credentials_file(self, tmp_path):
    def test_requires_api_url_and_key(self):
    def test_adds_to_gitignore(self, tmp_path):
    def test_idempotent_gitignore(self, tmp_path):
    def test_sets_file_permissions_600(self, tmp_path):
    def test_skips_dockerignore_if_absent(self, tmp_path):
    def test_adds_to_dockerignore_if_present(self, tmp_path):
```

## Acceptance Criteria

### Functional
- [ ] `resolve_credentials(cwd)` walks up from CWD, stops before `$HOME`, falls back to global
- [ ] Env vars override project file; project file overrides global
- [ ] All 4 API-calling hooks use CWD-based resolution (no module-level creds)
- [ ] Observer daemon routes observations to correct server based on session CWD
- [ ] Per-URL failure isolation in daemon (one server down doesn't affect others)
- [ ] Signal files include `cwd` field
- [ ] `cems setup --project --api-url URL --api-key KEY` creates `.cems/credentials` in CWD
- [ ] `.cems/` added to `.gitignore` during project setup
- [ ] `cems env` shows resolved credentials for current directory
- [ ] Cache keys include URL hash to prevent cross-server contamination
- [ ] Version skew warning during `cems setup --project`
- [ ] Backward compatible: projects without `.cems/credentials` work exactly as before

### Non-Functional
- [ ] `.cems/credentials` created with `chmod 600`
- [ ] Partial project credentials (missing URL or key) logged and skipped, not crash
- [ ] Hook latency increase < 5ms (filesystem walk is fast, ~3-4 stat calls typical)
- [ ] Daemon client pool memory bounded (1-2 entries expected, pool is a simple dict)

## Dependencies & Risks

**Risk: Bundled copy divergence** — 8 hook files + 2 utility files have bundled copies under `src/cems/data/claude/hooks/`. All changes must be mirrored. Mitigation: update bundled copies as part of each phase, test `cems setup` deploys the updated versions.

**Risk: Daemon credential cache invalidation** — If a user rotates their project API key while daemon is running, daemon holds the old key. Mitigation: re-resolve credentials on 401 response (retry once with fresh creds).

**Risk: Walk-up performance on deep directory trees** — A CWD 20 levels deep means 20 `stat()` calls. Mitigation: practically fast (<1ms); most projects are 2-4 levels from home.

**Dependency: `/api/health` must return version** — For skew detection. Check if this already exists; if not, add a `version` field to the health endpoint response.

## Implementation Order

```
Phase 1 (Core) → Phase 2 (Daemon) → Phase 3 (CLI) → Phase 4 (Tests throughout)
```

Phase 1 is the load-bearing piece. Phase 2 and 3 can be done in parallel after Phase 1 ships. Tests accompany each phase.

## File Modification Summary

### Must Change (P0 — Core)

| File | Change |
|------|--------|
| `hooks/utils/credentials.py` | Add `resolve_credentials(cwd)`, CWD-keyed cache |
| `hooks/cems_session_start.py` | Defer creds to `main()`, pass to helpers |
| `hooks/cems_user_prompts_submit.py` | Same pattern |
| `hooks/cems_post_tool_use.py` | Same pattern |
| `hooks/cems_stop.py` | Same pattern + pass CWD in signal |
| `hooks/cems_pre_compact.py` | Pass CWD in signal write |
| `hooks/cems_pre_tool_use.py` | URL hash in gate rule cache key |
| `src/cems/observer/signals.py` | Add `cwd` field to Signal dataclass |
| `src/cems/observer/__main__.py` | Walk-up resolution (daemon-side) |
| `src/cems/observer/daemon.py` | ClientPool, per-session routing, per-URL failures |
| `src/cems/commands/setup.py` | `--project` flag, `.gitignore` handling |

### Must Mirror (bundled copies)

| File |
|------|
| `src/cems/data/claude/hooks/utils/credentials.py` |
| `src/cems/data/claude/hooks/cems_session_start.py` |
| `src/cems/data/claude/hooks/cems_user_prompts_submit.py` |
| `src/cems/data/claude/hooks/cems_post_tool_use.py` |
| `src/cems/data/claude/hooks/cems_stop.py` |
| `src/cems/data/claude/hooks/cems_pre_compact.py` |
| `src/cems/data/claude/hooks/cems_pre_tool_use.py` |

### Optional Enhancement (P2)

| File | Change |
|------|--------|
| `src/cems/commands/env.py` | CWD-aware credential display |
| `src/cems/commands/update.py` | Version skew check after update |
| `cursor-plugin/hooks/cems_stop.py` | Pass CWD in signal |
| `src/cems/observer/state.py` | Store `api_url` for daemon restart resilience |

### No Changes Needed

| File | Why |
|------|-----|
| `hooks/utils/project.py` | Project identity is orthogonal to credential routing |
| `hooks/utils/observer_manager.py` | Singleton pattern stays, env stripping already correct |
| `src/cems/commands/update.py` | Stays global-only per design decision |
| `src/cems/observer/adapters/*` | Adapters discover sessions, not credentials |

## References

- Brainstorm: `docs/brainstorms/2026-03-25-per-project-config.md`
- Current credentials: `hooks/utils/credentials.py`
- Daemon entry: `src/cems/observer/__main__.py:165-192` (inline credential parser)
- Signal format: `src/cems/observer/signals.py:25-58`
- Setup CLI: `src/cems/commands/setup.py:272-343`
- Similar pattern: `.nvmrc` / `.node-version` walk-up resolution
