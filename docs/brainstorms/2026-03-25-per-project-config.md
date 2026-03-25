# Per-Project CEMS Configuration

**Date:** 2026-03-25
**Status:** Brainstorm complete, ready for planning

## What We're Building

Multi-instance CEMS support: different projects can point to different CEMS servers with separate URLs, API keys, and settings. When working in a project with its own `.cems/credentials`, all hooks, recall, observations, and daemon routing go to that project's CEMS instance exclusively.

**Use case:** User's company (e.g., Hubstaff) deploys their own CEMS instance. When working in `hubstaff-server/`, all CEMS traffic routes to Hubstaff's server. Personal projects continue using the global `~/.cems/credentials`.

## Why This Approach

### Walk-up Config Resolution

Hooks gain a `resolve_credentials(cwd)` function that walks up directory tree from CWD looking for `.cems/credentials`. First one found wins. If none found, falls back to `~/.cems/credentials` (global default).

**Why walk-up:**
- Familiar pattern (like `.env`, `.nvmrc`, `.node-version`)
- Works across all IDEs (not tied to Claude Code's settings system)
- Project config is discoverable — teammates can see `.cems/` exists
- Gitignore-able — add `.cems/` to `.gitignore`
- Same file format as global credentials — no new formats to learn

**Why not alternatives:**
- **Central registry (`~/.cems/projects.toml`)**: Less discoverable, new format, team members can't see config in repo
- **Claude project settings (env vars)**: Tied to Claude Code, doesn't work for daemon, security concern with API keys in settings.json

## Key Decisions

1. **Fully separate servers** — each company runs their own CEMS deployment. No shared-server team scoping.

2. **Clean separation** — when in a project with its own CEMS, everything routes there exclusively (observations, recall, /remember, gate rules). No merging of personal + project memories.

3. **`.cems/credentials` in project root** — same dotenv format as `~/.cems/credentials`. Walk up from CWD to find it. Fall back to global.

4. **Single daemon, route by signal** — keep one observer daemon. Each signal file carries CWD. Daemon resolves credentials from CWD at processing time. Caches API clients per unique URL (dict of URL → client).

5. **`cems setup --project`** — explicit flag creates `.cems/credentials` in CWD. Prompts for URL + API key. Regular `cems setup` continues to do global config only.

6. **`cems update` is global-only** — updates hooks/skills globally (which includes the walk-up resolution logic). Never touches per-project `.cems/credentials` files. Project credentials are user data.

7. **Hooks are agnostic** — hooks remain globally installed in `~/.claude/hooks/`. The walk-up logic in `credentials.py` is what makes them project-aware. No per-project hook installation needed.

## How It Works

### Credential Resolution Order
```
1. Environment variables (CEMS_API_URL, CEMS_API_KEY) — highest priority
2. .cems/credentials in CWD or ancestor directory — walk up
3. ~/.cems/credentials — global fallback
```

### Hook Flow
```
Hook fires → get CWD → resolve_credentials(cwd) → walk up for .cems/credentials
  → Found in /Users/razvan/Development/hubstaff-server/.cems/credentials
  → Use Hubstaff's URL + API key for this request
```

### Daemon Flow
```
Signal written with CWD → Daemon reads signal → resolve_credentials(signal.cwd)
  → Get or create API client for that URL (client pool)
  → Route observation to correct CEMS instance
```

### Setup Flow
```bash
# Global (existing)
cems setup --claude --api-url https://cems.personal.com --api-key KEY

# Per-project (new)
cd ~/Development/hubstaff-server
cems setup --project --api-url https://cems.hubstaff.com --api-key HUBSTAFF_KEY
# Creates ./cems/credentials with URL + key
# Adds .cems/ to .gitignore if not already there
```

## Implementation Sketch

### credentials.py changes
```python
def resolve_credentials(cwd: str | None = None) -> dict:
    """Walk up from CWD looking for .cems/credentials, fall back to global."""
    if cwd:
        path = Path(cwd)
        while path != path.parent:
            project_creds = path / ".cems" / "credentials"
            if project_creds.exists():
                return _parse_credentials(project_creds)
            path = path.parent
    return _parse_credentials(GLOBAL_CREDENTIALS_PATH)
```

### Daemon client pool
```python
# In observer daemon
_client_pool: dict[str, CEMSClient] = {}

def get_client_for_cwd(cwd: str) -> CEMSClient:
    creds = resolve_credentials(cwd)
    url = creds["CEMS_API_URL"]
    if url not in _client_pool:
        _client_pool[url] = CEMSClient(url, creds["CEMS_API_KEY"])
    return _client_pool[url]
```

## Version Skew & Update Safety

**Problem:** Skills and hooks are global (one copy, updated via `cems update`). But different CEMS servers may run different versions. If you update your client to v0.9.35 but Hubstaff's server is still on v0.9.31, hooks might call API endpoints that don't exist on the older server.

**Design principles:**
- **Skills and hooks stay global** — no duplication per project. One `cems update` updates everything.
- **Graceful degradation at runtime** — if a hook gets a 404 from an older server, log it and continue. Don't crash.
- **Version warnings at key moments:**
  - `cems setup --project`: check server version, warn if behind client
  - `cems update`: after updating, check all known project CEMS versions (can discover from `.cems/credentials` files via recent CWDs or `~/.cems/observer/` state), warn about skew
  - Example: `⚠ hubstaff-server CEMS (v0.9.31) is behind your client (v0.9.35). Some features may not work until the server is updated.`

**What we don't do:**
- No per-project skill/hook copies (maintenance nightmare)
- No blocking on version mismatch (just warn)
- No auto-updating remote servers (that's their admin's job)

## Open Questions

1. **Team ID per project?** — Should `.cems/credentials` also support `CEMS_TEAM_ID`? Probably yes, same format.
2. **MCP server routing** — The MCP wrapper (port 8766) currently serves one instance. Should it be project-aware too, or is hook-level routing sufficient?
3. **`cems env` command** — Currently outputs global credentials for shell eval. Should it be CWD-aware? Probably yes.
4. **Cache isolation** — Gate rule cache (`~/.cems/cache/`) is currently global. Should per-project CEMS have separate cache dirs?
5. **Observer state isolation** — Signal files and session state are in `~/.cems/observer/`. Should project-specific observations use subdirectories?
6. **Version check endpoint** — Server needs a lightweight `/api/version` endpoint (or use existing `/api/health`) that returns the server version for skew detection.

## Scale

Expected: 1-2 projects with their own CEMS instance. Most projects continue using global config. Design is simple and doesn't over-engineer for scale.
