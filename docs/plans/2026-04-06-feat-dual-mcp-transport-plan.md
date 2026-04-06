---
title: "feat: Dual MCP transport — stdio + HTTP for non-technical users"
type: feat
date: 2026-04-06
---

# Dual MCP Transport — stdio + HTTP

## Overview

Bring back HTTP MCP transport as an alternative to stdio so non-technical users can connect to CEMS without installing the Python package. `cems setup` offers both modes; stdio remains the default for developers.

## Problem Statement

Currently `cems setup` only registers stdio transport (`"command": "cems-mcp"`), which requires:
- Python package installed (`uv tool install cems`)
- `cems-mcp` binary on PATH
- Per-project credential resolution from filesystem

Non-technical team members (designers, PMs) who just need memory access in Cursor/Claude can't set this up. The old HTTP transport was simpler: just a URL + API key in the MCP config.

## Proposed Solution

Support both transports in `cems setup`:

| Mode | Config | Who it's for |
|------|--------|-------------|
| **stdio** (default) | `{"command": "/path/to/cems-mcp"}` | Developers with cems installed |
| **HTTP** | `{"type": "http", "url": "https://...", "headers": {"Authorization": "Bearer <key>"}}` | Non-technical users, remote setups |

The Express MCP wrapper at `mcp-cems.chocksy.com` already handles HTTP transport — no server changes needed.

## Implementation

### Phase 1: Restore HTTP registration in `cems setup`

#### `src/cems/commands/setup.py`

- [ ] Restore `_discover_mcp_url(api_url)` function (deleted in commit `5eaa843`)
  - Priority: credentials CEMS_MCP_URL > server discovery `/api/config/setup` > fallback derive from api_url
  - Old code: `git show 5eaa843^:src/cems/commands/setup.py` lines 492-528

- [ ] Add `--transport` flag to `cems setup` (choices: `stdio`, `http`, default: `stdio`)
  - When interactive (no flags): ask user which transport after IDE selection
  - When non-interactive (`--claude --api-url X --api-key Y`): default to stdio unless `--transport http`

- [ ] Update `_register_claude_mcp_server()` to support both modes:
  ```python
  if transport == "http":
      mcp_url = _discover_mcp_url(api_url)
      mcp_servers["cems"] = {
          "type": "http",
          "url": mcp_url,
          "headers": {
              "Authorization": f"Bearer {api_key}",
              **({"X-Team-Id": team_id} if team_id else {}),
          },
      }
  else:  # stdio (default)
      cems_mcp_cmd = _resolve_cems_mcp_path()
      mcp_servers["cems"] = {
          "command": cems_mcp_cmd,
          "args": [],
      }
  ```

- [ ] Update `_register_cursor_mcp()` same pattern — HTTP uses `type: http` with URL + headers

- [ ] Update `_register_codex_mcp()` — Codex used HTTP with bearer token env var (keep as-is, it already works)

### Phase 2: Verify Express MCP wrapper compatibility

The wrapper at `mcp-cems.chocksy.com` uses StreamableHTTP transport. Need to verify:

- [ ] Claude Code supports `"type": "http"` with StreamableHTTP (not SSE)
- [ ] Cursor supports same format
- [ ] Auth headers (`Authorization`, `X-Team-Id`) are forwarded correctly through the wrapper

#### `mcp-wrapper/src/index.ts`

- [ ] Verify auth header extraction passes through to Python API
- [ ] Test with: `curl -X POST https://mcp-cems.chocksy.com/mcp -H "Authorization: Bearer <key>" -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'`

### Phase 3: Credential fallback in MCP stdio server

The `mcp_stdio.py` server currently resolves credentials from filesystem only. When running behind the HTTP wrapper, auth comes from request headers instead.

- [ ] No changes needed to `mcp_stdio.py` — the Express wrapper handles auth extraction from HTTP headers and forwards to the Python REST API. The stdio server only runs in stdio mode.

### Phase 4: Update `cems setup` interactive flow

Current interactive flow:
```
1. Select IDEs (Claude, Cursor, Codex, Goose)
2. Enter API URL
3. Enter API key
4. Install hooks/skills/config
```

New flow adds one question:
```
1. Select IDEs
2. Enter API URL
3. Enter API key
4. **MCP transport: stdio (recommended) or HTTP (no install needed)?**
5. Install hooks/skills/config
```

- [ ] Add transport selection to interactive flow
- [ ] Show explanation: "stdio requires cems package installed. HTTP connects directly to server — simpler but requires server URL."

### Phase 5: Hubstaff-specific — per-project HTTP config

For hubstaff-server, the team uses a different CEMS instance (`cems.ai.hbstf.co`). With HTTP transport, each project can have its own MCP config pointing to a different server.

- [ ] `cems setup --project --transport http` should write to per-project MCP config:
  - Claude: `.claude/settings.local.json` (project-level MCP)
  - Cursor: `.cursor/mcp.json` in project root (if Cursor supports project-level)

- [ ] Or: for HTTP mode, the API key and URL are embedded in the config, so per-project credentials aren't needed — the MCP config IS the credential.

## Acceptance Criteria

- [ ] `cems setup --claude` defaults to stdio (no change for existing users)
- [ ] `cems setup --claude --transport http` registers HTTP with discovered MCP URL
- [ ] `cems setup --cursor --transport http` registers HTTP in Cursor mcp.json
- [ ] Interactive mode asks transport choice after API key entry
- [ ] HTTP config includes `Authorization: Bearer <key>` and optional `X-Team-Id`
- [ ] Non-technical user can set up CEMS in Cursor with just URL + key (no Python install)
- [ ] Existing stdio setups are not broken
- [ ] Tests cover both registration paths

## Technical Details

### Old HTTP config format (from commit `5eaa843^`)

```json
{
  "mcpServers": {
    "cems": {
      "type": "http",
      "url": "https://mcp-cems.chocksy.com/mcp",
      "headers": {
        "Authorization": "Bearer ${CEMS_API_KEY}",
        "X-Team-Id": "1d2971be-..."
      }
    }
  }
}
```

### Current stdio config format

```json
{
  "mcpServers": {
    "cems": {
      "command": "/Users/razvan/.local/bin/cems-mcp",
      "args": []
    }
  }
}
```

### MCP URL discovery logic (to restore)

```
1. CEMS_MCP_URL from ~/.cems/credentials (explicit override)
2. Server discovery: GET /api/config/setup → {"mcp_url": "..."}
3. Fallback: localhost → http://localhost:8766/mcp
           remote → https://mcp-{hostname}/mcp
```

### Files to modify

| File | Changes |
|------|---------|
| `src/cems/commands/setup.py` | Add `--transport` flag, restore `_discover_mcp_url`, update registration functions |
| `tests/test_setup.py` | Add tests for HTTP registration path |

### Files NOT modified

| File | Why |
|------|-----|
| `src/cems/mcp_stdio.py` | Only runs in stdio mode — HTTP is handled by Express wrapper |
| `mcp-wrapper/` | Already working, no changes needed |
| `docker-compose.coolify.yml` | MCP wrapper already deployed |

## References

- Old HTTP code: `git show 5eaa843^:src/cems/commands/setup.py`
- Switch commit: `5eaa843` ("refactor: switch MCP registration from HTTP to stdio transport")
- Express wrapper: `mcp-wrapper/src/index.ts`
- MCP wrapper health: `https://mcp-cems.chocksy.com/health`
- Claude MCP docs: StreamableHTTP transport uses `"type": "http"`
