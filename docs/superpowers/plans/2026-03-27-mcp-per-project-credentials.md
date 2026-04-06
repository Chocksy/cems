# MCP Per-Project Credentials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CEMS MCP server resolve per-project credentials so it routes to the correct CEMS API instance based on which project Claude is running in.

**Architecture:** Switch from HTTP MCP (hardcoded remote URL) to stdio MCP (local proxy). The stdio server (`cems-mcp`) is spawned by Claude Code with the project's CWD. It uses `resolve_credentials(cwd)` to find per-project `.cems/credentials`, falling back to `~/.cems/credentials`. This gives us per-project routing with a single MCP entry.

**Tech Stack:** Python (stdlib + FastMCP), Claude Code MCP config (`~/.claude.json`)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/cems/mcp_stdio.py` | Modify | Replace `_read_credentials()` + `_get_config()` with `resolve_credentials(os.getcwd())` |
| `src/cems/shared/credentials.py` | No change | Already has `resolve_credentials()` with CWD walk-up |
| `src/cems/commands/setup.py` | Modify | Change `_register_claude_mcp_server()` from HTTP to stdio type |
| `tests/test_mcp_stdio.py` | Modify | Update tests for new credential resolution + add per-project test |
| `~/.claude.json` | Auto-updated | Will be updated by running `cems setup --claude` after code changes |

---

### Task 1: Update mcp_stdio.py to use shared credential resolver

**Files:**
- Modify: `src/cems/mcp_stdio.py:26-55`
- Test: `tests/test_mcp_stdio.py`

- [ ] **Step 1: Write the failing test for per-project credential resolution**

Add to `tests/test_mcp_stdio.py`:

```python
class TestGetConfigWithResolver:
    """Tests for _get_config using resolve_credentials."""

    def test_uses_project_credentials_from_cwd(self, tmp_path):
        """When CWD has .cems/credentials, uses those instead of global."""
        from cems.mcp_stdio import _get_config

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        creds_dir = project_dir / ".cems"
        creds_dir.mkdir()
        (creds_dir / "credentials").write_text(
            "CEMS_API_URL=https://project.example.com\n"
            "CEMS_API_KEY=project-key\n"
        )

        with patch.dict("os.environ", {}, clear=True):
            with patch("os.getcwd", return_value=str(project_dir)):
                url, key = _get_config()

        assert url == "https://project.example.com"
        assert key == "project-key"

    def test_falls_back_to_global_credentials(self, tmp_path):
        """When CWD has no .cems/credentials, falls back to global."""
        global_creds = tmp_path / ".cems" / "credentials"
        global_creds.parent.mkdir(parents=True)
        global_creds.write_text(
            "CEMS_API_URL=https://global.example.com\n"
            "CEMS_API_KEY=global-key\n"
        )

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.dict("os.environ", {"CEMS_CREDENTIALS_FILE": str(global_creds)}, clear=True):
            with patch("os.getcwd", return_value=str(project_dir)):
                # Patch HOME so walk-up stops before finding project creds
                with patch("cems.shared.credentials._HOME", str(tmp_path)):
                    url, key = _get_config()

        assert url == "https://global.example.com"
        assert key == "global-key"

    def test_env_vars_still_win(self):
        """Env vars take priority over any credentials file."""
        from cems.mcp_stdio import _get_config

        with patch.dict("os.environ", {
            "CEMS_API_URL": "https://env.example.com",
            "CEMS_API_KEY": "env-key",
        }):
            url, key = _get_config()

        assert url == "https://env.example.com"
        assert key == "env-key"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_mcp_stdio.py::TestGetConfigWithResolver -v`
Expected: FAIL — `_get_config()` doesn't call `resolve_credentials()` yet.

- [ ] **Step 3: Update `_get_config()` in mcp_stdio.py**

Replace the credentials section (lines 26-55) in `src/cems/mcp_stdio.py`:

```python
# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

from cems.shared.credentials import resolve_credentials


def _get_config() -> tuple[str, str]:
    """Return (api_url, api_key) using the shared credential resolver.

    Precedence: env vars > per-project .cems/credentials > global ~/.cems/credentials.
    """
    creds = resolve_credentials(os.getcwd())
    api_url = creds.get("CEMS_API_URL", "")
    api_key = creds.get("CEMS_API_KEY", "")
    return api_url.rstrip("/"), api_key
```

Remove the now-unused `_read_credentials()` function entirely.

Also update the `memory_search` tool's search mode resolution (line 122) to use the resolved creds instead of calling the deleted function:

```python
    # Pass search mode from env or resolved credentials
    search_mode = os.getenv("CEMS_SEARCH_MODE") or resolve_credentials(os.getcwd()).get("CEMS_SEARCH_MODE", "")
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_mcp_stdio.py::TestGetConfigWithResolver -v`
Expected: PASS

- [ ] **Step 5: Run the full test file to verify nothing broke**

Run: `.venv/bin/python3 -m pytest tests/test_mcp_stdio.py -v`
Expected: All pass. The `TestReadCredentials` class tests a function that no longer exists — these tests should be removed in the next step.

- [ ] **Step 6: Remove stale `TestReadCredentials` tests**

Delete the entire `TestReadCredentials` class and the now-redundant `TestGetConfig` class from `tests/test_mcp_stdio.py`. The credential parsing logic is tested in `tests/test_shared_credentials.py` (or should be — verify it exists).

Run: `.venv/bin/python3 -m pytest tests/test_mcp_stdio.py -v`
Expected: All remaining tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/cems/mcp_stdio.py tests/test_mcp_stdio.py
git commit -m "feat: MCP stdio server uses per-project credential resolution

Replace _read_credentials() with shared resolve_credentials(os.getcwd()).
Precedence: env vars > project .cems/credentials > global ~/.cems/credentials.
Fixes MCP tools failing with 'Invalid API key' in per-project CEMS setups."
```

---

### Task 2: Switch MCP registration from HTTP to stdio

**Files:**
- Modify: `src/cems/commands/setup.py:535-565`

- [ ] **Step 1: Write the failing test**

Check if there's a test for `_register_claude_mcp_server`. Search:

```bash
grep -rn "register_claude_mcp" tests/
```

If no test exists, add to the appropriate test file (likely `tests/test_commands_setup.py` or create a focused test):

```python
class TestRegisterClaudeMcpServer:
    def test_registers_stdio_server(self, tmp_path):
        from cems.commands.setup import _register_claude_mcp_server

        claude_json = tmp_path / ".claude.json"

        with patch("cems.commands.setup.Path.home", return_value=tmp_path):
            _register_claude_mcp_server("https://example.com")

        config = json.loads(claude_json.read_text())
        mcp = config["mcpServers"]["cems"]
        assert mcp["command"] == "cems-mcp"
        assert mcp.get("type") is None or mcp["type"] == "stdio"  # stdio is default
        assert "url" not in mcp  # No hardcoded URL
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: FAIL — current code writes `type: "http"` with a `url`.

- [ ] **Step 3: Update `_register_claude_mcp_server()` in setup.py**

Replace the function body at `src/cems/commands/setup.py:535-565`:

```python
def _register_claude_mcp_server(api_url: str, team_id: str | None = None) -> None:
    """Register CEMS MCP server in Claude Code config (~/.claude.json).

    Uses stdio transport — Claude Code spawns `cems-mcp` locally.
    The server resolves credentials from CWD, so per-project
    .cems/credentials files are picked up automatically.
    """
    claude_json = Path.home() / ".claude.json"

    existing: dict = {}
    if claude_json.exists():
        try:
            existing = json.loads(claude_json.read_text())
        except json.JSONDecodeError:
            existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})

    mcp_servers["cems"] = {
        "command": "cems-mcp",
        "args": [],
    }

    claude_json.write_text(json.dumps(existing, indent=2) + "\n")
    console.print("  MCP server registered: cems-mcp (stdio)")
```

- [ ] **Step 4: Run the test to verify it passes**

Expected: PASS

- [ ] **Step 5: Clean up dead code**

The `_discover_mcp_url()` function (lines 495-532) is now unused. Delete it entirely.

Run: `.venv/bin/python3 -m pytest tests/ -x -q --timeout=60`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cems/commands/setup.py
git commit -m "feat: switch MCP registration from HTTP to stdio

Claude Code now spawns cems-mcp locally instead of connecting to a
remote HTTP MCP server. The local process resolves per-project
credentials from CWD, enabling multi-instance CEMS setups."
```

---

### Task 3: Apply the new config and verify end-to-end

- [ ] **Step 1: Re-register the MCP server**

```bash
cd /Users/razvan/Development/cems && cems setup --claude
```

This should update `~/.claude.json` with the stdio config.

- [ ] **Step 2: Verify the new config**

```bash
cat ~/.claude.json | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['mcpServers']['cems'], indent=2))"
```

Expected output:
```json
{
  "command": "cems-mcp",
  "args": []
}
```

- [ ] **Step 3: Verify MCP works from cems project (global credentials)**

Open a new Claude Code session in `/Users/razvan/Development/cems` and run:
```
Use mcp__cems__memory_search with query "test"
```
Expected: Results from `cems.chocksy.com`

- [ ] **Step 4: Verify MCP works from hubstaff-server project (per-project credentials)**

Open a new Claude Code session in `/Users/razvan/Development/hubstaff-server` and run:
```
Use mcp__cems__memory_add with content "MCP per-project test" category "test"
```
Expected: Success from `cems.ai.hbstf.co` (no "Invalid API key" error)

- [ ] **Step 5: Clean up test memory**

Delete the test memory created in step 4.
