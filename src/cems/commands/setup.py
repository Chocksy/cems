"""Setup command for CEMS CLI.

Installs hooks, skills, and credentials for Claude Code, Cursor, Codex, and/or Goose.
All files are bundled inside the cems package — no clone needed.

Usage:
    cems setup              # Interactive (prompts for IDE + credentials)
    cems setup --claude     # Claude Code only, no prompts for IDE choice
    cems setup --cursor     # Cursor only
    cems setup --codex      # Codex only
    cems setup --goose      # Goose only
    cems setup --claude --api-url URL --api-key KEY   # Non-interactive
"""

import json
import shutil
import stat
import sys
from importlib import resources
from pathlib import Path

import click
from rich.table import Table

from cems.cli_utils import console


# ---------------------------------------------------------------------------
# Interactive multiselect (checkbox-style, like skills.sh)
# ---------------------------------------------------------------------------

def _multiselect(
    title: str,
    options: list[tuple[str, str, bool]],
) -> list[str]:
    """Interactive checkbox selector using raw terminal input.

    Renders a list with ● / ○ markers. Arrow keys to move, space to toggle,
    enter to confirm. Inspired by @clack/prompts and Vercel skills.sh.

    Args:
        title: Prompt title
        options: List of (key, label, default_selected) tuples

    Returns:
        List of selected keys
    """
    if not sys.stdin.isatty():
        # Non-interactive fallback: return defaults
        return [key for key, _, default in options if default]

    import tty
    import termios

    selected = [default for _, _, default in options]
    cursor = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # In raw mode, \n only moves down — need \r\n to also return to column 0
    NL = "\r\n"

    def render(final: bool = False) -> None:
        # Clear previous render
        if not final:
            total_lines = len(options) + 3
            sys.stdout.write(f"\033[{total_lines}A\033[J")

        icon = "\033[32m◇\033[0m" if final else "\033[32m◆\033[0m"
        sys.stdout.write(f"{icon}  \033[1m{title}\033[0m{NL}")

        if not final:
            for i, (key, label, _) in enumerate(options):
                check = "\033[32m●\033[0m" if selected[i] else "\033[2m○\033[0m"
                prefix = "\033[36m❯\033[0m" if i == cursor else " "
                label_fmt = f"\033[4m{label}\033[0m" if i == cursor else label
                sys.stdout.write(f"\033[2m│\033[0m {prefix} {check} {label_fmt}{NL}")

            sys.stdout.write(f"\033[2m│  ↑↓ move, space toggle, enter confirm\033[0m{NL}")

            names = [label for (_, label, _), sel in zip(options, selected) if sel]
            summary = ", ".join(names) if names else "(none)"
            sys.stdout.write(f"\033[2m└\033[0m  \033[32mSelected:\033[0m {summary}{NL}")
        else:
            names = [label for (_, label, _), sel in zip(options, selected) if sel]
            sys.stdout.write(f"\033[2m│\033[0m  \033[2m{', '.join(names)}\033[0m{NL}")

        sys.stdout.flush()

    try:
        tty.setraw(fd)

        # Initial render (with padding for first clear)
        sys.stdout.write(NL * (len(options) + 3))
        sys.stdout.flush()
        render()

        while True:
            ch = sys.stdin.read(1)

            if ch == "\r" or ch == "\n":  # Enter
                if not any(selected):
                    continue  # Require at least one
                render(final=True)
                break
            elif ch == "\x03":  # Ctrl+C
                render(final=True)
                raise KeyboardInterrupt()
            elif ch == " ":  # Space — toggle
                selected[cursor] = not selected[cursor]
                render()
            elif ch == "\x1b":  # Escape sequence (arrow keys)
                seq = sys.stdin.read(2)
                if seq == "[A":  # Up
                    cursor = max(0, cursor - 1)
                    render()
                elif seq == "[B":  # Down
                    cursor = min(len(options) - 1, cursor + 1)
                    render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return [key for (key, _, _), sel in zip(options, selected) if sel]


def _single_select(
    title: str,
    options: list[tuple[str, str]],
    default: int = 0,
) -> str:
    """Interactive single-select using raw terminal input.

    Args:
        title: Prompt title
        options: List of (key, label) tuples
        default: Default selected index

    Returns:
        Selected key
    """
    if not sys.stdin.isatty():
        return options[default][0]

    import tty
    import termios

    cursor = default
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    NL = "\r\n"

    def render(final: bool = False) -> None:
        total_lines = len(options) + 2
        sys.stdout.write(f"\033[{total_lines}A\033[J")

        icon = "\033[32m◇\033[0m" if final else "\033[32m◆\033[0m"
        sys.stdout.write(f"{icon}  \033[1m{title}\033[0m{NL}")

        if not final:
            for i, (key, label) in enumerate(options):
                radio = "\033[32m●\033[0m" if i == cursor else "\033[2m○\033[0m"
                prefix = "\033[36m❯\033[0m" if i == cursor else " "
                label_fmt = f"\033[4m{label}\033[0m" if i == cursor else label
                sys.stdout.write(f"\033[2m│\033[0m {prefix} {radio} {label_fmt}{NL}")
            sys.stdout.write(f"\033[2m└  ↑↓ move, enter confirm\033[0m{NL}")
        else:
            sys.stdout.write(f"\033[2m│\033[0m  \033[2m{options[cursor][1]}\033[0m{NL}")

        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write(NL * (len(options) + 2))
        sys.stdout.flush()
        render()

        while True:
            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":
                render(final=True)
                break
            elif ch == "\x03":
                render(final=True)
                raise KeyboardInterrupt()
            elif ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    cursor = max(0, cursor - 1)
                    render()
                elif seq == "[B":
                    cursor = min(len(options) - 1, cursor + 1)
                    render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return options[cursor][0]


INSTALL_PREFS_FILE = Path.home() / ".cems" / "install.conf"


def _get_data_path() -> Path:
    """Get the path to bundled package data."""
    return Path(str(resources.files("cems.data")))


def _save_install_preferences(*, claude: bool, cursor: bool, codex: bool, goose: bool) -> None:
    """Save which IDEs were installed to ~/.cems/install.conf.

    Used by `cems update` to non-interactively redeploy the right targets.
    """
    targets = []
    if claude:
        targets.append("claude")
    if cursor:
        targets.append("cursor")
    if codex:
        targets.append("codex")
    if goose:
        targets.append("goose")

    INSTALL_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    INSTALL_PREFS_FILE.write_text(
        f"# CEMS install preferences — generated by cems setup\n"
        f"# Used by `cems update` to know which IDEs to redeploy to.\n"
        f"TARGETS={','.join(targets)}\n"
    )


def get_install_preferences() -> list[str]:
    """Read saved IDE targets from ~/.cems/install.conf.

    Returns list like ["claude", "cursor"] or empty list if no prefs saved.
    Public API used by update command.
    """
    if not INSTALL_PREFS_FILE.exists():
        return []
    try:
        for line in INSTALL_PREFS_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("TARGETS="):
                return [t.strip() for t in line.partition("=")[2].split(",") if t.strip()]
    except OSError:
        pass
    return []


def _read_credentials() -> dict[str, str]:
    """Read ~/.cems/credentials as key=value pairs."""
    creds_file = Path.home() / ".cems" / "credentials"
    result = {}
    try:
        if creds_file.exists():
            for line in creds_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    result[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return result


def _is_interactive() -> bool:
    """Check if stdin is a terminal (not piped)."""
    return sys.stdin.isatty()


def _setup_credentials(api_url: str | None = None, api_key: str | None = None) -> bool:
    """Ensure ~/.cems/credentials exists.

    If api_url/api_key are provided, use them directly (non-interactive).
    Otherwise prompt interactively if stdin is a TTY.

    Returns True if credentials are available.
    """
    cems_dir = Path.home() / ".cems"
    creds_file = cems_dir / "credentials"

    # Already configured
    if creds_file.exists() and not api_key:
        console.print(f"[green]Credentials:[/green] {creds_file}")
        return True

    # Non-interactive with explicit values — preserve existing custom keys
    if api_key:
        url = api_url or "http://localhost:8765"
        cems_dir.mkdir(parents=True, exist_ok=True)

        # Preserve existing keys like CEMS_SEARCH_MODE, CEMS_TEAM_ID
        existing_creds = _read_credentials() if creds_file.exists() else {}
        existing_creds["CEMS_API_URL"] = url
        existing_creds["CEMS_API_KEY"] = api_key

        lines = ["# CEMS credentials — generated by cems setup"]
        for k, v in existing_creds.items():
            lines.append(f"{k}={v}")

        creds_file.write_text("\n".join(lines) + "\n")
        creds_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        console.print(f"[green]Saved credentials to {creds_file}[/green]")
        return True

    # Non-interactive without values — can't prompt
    if not _is_interactive():
        console.print("[yellow]No credentials found and stdin is not a terminal.[/yellow]")
        console.print()
        console.print("Configure credentials with one of:")
        console.print("  cems setup --claude --api-url URL --api-key KEY")
        console.print("  Or create ~/.cems/credentials manually:")
        console.print("    CEMS_API_URL=http://localhost:8765")
        console.print("    CEMS_API_KEY=your-key-here")
        return False

    # Interactive prompt
    console.print()
    console.print("[bold]Credentials Setup[/bold]")
    console.print("No credentials found. Get these from your CEMS admin.")
    console.print()

    api_url = click.prompt(
        "CEMS API URL",
        default="http://localhost:8765",
        show_default=True,
    )
    api_key = click.prompt("CEMS API Key", hide_input=True)

    if not api_key:
        console.print("[red]API key is required.[/red]")
        return False

    cems_dir.mkdir(parents=True, exist_ok=True)
    creds_file.write_text(
        f"# CEMS credentials — generated by cems setup\n"
        f"CEMS_API_URL={api_url}\n"
        f"CEMS_API_KEY={api_key}\n"
    )
    creds_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
    console.print(f"[green]Saved credentials to {creds_file} (mode 600)[/green]")
    return True


def _discover_team(api_url: str, api_key: str) -> str | None:
    """Query the CEMS server for the user's team memberships.

    If exactly 1 team → auto-select it.
    If multiple teams → prompt interactively, or skip in non-interactive mode.
    """
    import urllib.error
    import urllib.request

    url = api_url.rstrip("/") + "/api/me/teams"
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("User-Agent", "CEMS-CLI/1.0")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            teams = data.get("teams", [])
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        console.print(f"  [yellow]Could not discover teams: {e}[/yellow]")
        return None

    if not teams:
        console.print("  [dim]No team membership found (personal memories only)[/dim]")
        return None

    if len(teams) == 1:
        team = teams[0]
        console.print(f"  [green]Auto-selected team:[/green] {team['name']} ({team['id'][:8]}…)")
        _append_credential("CEMS_TEAM_ID", team["id"])
        return team["id"]

    # Multiple teams — prompt if interactive
    if _is_interactive():
        console.print()
        console.print("[bold]Multiple teams found:[/bold]")
        for i, team in enumerate(teams, 1):
            console.print(f"  {i}) {team['name']}")
        choice = click.prompt("Select default team", type=click.IntRange(1, len(teams)))
        selected = teams[choice - 1]
        console.print(f"  [green]Selected team:[/green] {selected['name']}")
        _append_credential("CEMS_TEAM_ID", selected["id"])
        return selected["id"]
    else:
        console.print(f"  [yellow]Multiple teams found ({len(teams)}). Set CEMS_TEAM_ID manually.[/yellow]")
        return None


def _remove_credential(key: str, creds_path: Path | None = None) -> None:
    """Remove a key from a credentials file (defaults to ~/.cems/credentials)."""
    creds_file = creds_path or (Path.home() / ".cems" / "credentials")
    if not creds_file.exists():
        return
    lines = [
        line for line in creds_file.read_text().splitlines()
        if not line.strip().startswith(f"{key}=")
    ]
    creds_file.write_text("\n".join(lines) + "\n")
    creds_file.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _append_credential(key: str, value: str, creds_path: Path | None = None) -> None:
    """Add or update a key in a credentials file (defaults to ~/.cems/credentials)."""
    creds_file = creds_path or (Path.home() / ".cems" / "credentials")
    lines = []
    replaced = False

    if creds_file.exists():
        for line in creds_file.read_text().splitlines():
            if line.strip().startswith(f"{key}="):
                lines.append(f"{key}={value}")
                replaced = True
            else:
                lines.append(line)

    if not replaced:
        lines.append(f"{key}={value}")

    creds_file.write_text("\n".join(lines) + "\n")
    creds_file.chmod(stat.S_IRUSR | stat.S_IWUSR)


_CEMS_INSTRUCTIONS = """\
## CEMS — Long-Term Memory System

You have CEMS (Contextual Episodic Memory System) connected as an MCP server. It provides persistent memory across all projects and sessions.

### MCP Tools (use ONLY these for memory operations)

| Tool | Purpose |
|------|---------|
| `mcp__cems__memory_search` | Search memories (primary recall tool) |
| `mcp__cems__memory_add` | Store a new memory |
| `mcp__cems__memory_forget` | Delete/archive a memory |
| `mcp__cems__memory_update` | Update existing memory content |
| `mcp__cems__memory_get` | Get full document by ID (for truncated results) |
| `mcp__cems__memory_pin` | Pin/unpin a memory (protect from pruning) |
| `mcp__cems__memory_maintenance` | Run maintenance jobs |

### Important Rules

- **When user says "remember"** -> use `mcp__cems__memory_add`, NOT built-in markdown memory
- **When searching memories** -> use `mcp__cems__memory_search`, NOT any `Context` tool from other MCP servers
- **NEVER use `mcp__apigcp__context` or similar tools for memory** — those are different systems
- **Always detect project** for source_ref: `git remote get-url origin` -> `project:org/repo`
- **Memories are auto-injected** by hooks on each prompt — manual search only when user explicitly asks
"""


def _append_cems_instructions(config_file: Path, marker: str = "## CEMS") -> None:
    """Append CEMS instructions to an IDE config file if not already present.

    Idempotent — checks for the marker before appending.
    """
    existing = ""
    if config_file.exists():
        existing = config_file.read_text()

    if marker in existing:
        console.print(f"  CEMS instructions already in {config_file.name}")
        return

    if existing and not existing.endswith("\n"):
        existing += "\n"

    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(existing + "\n" + _CEMS_INSTRUCTIONS)
    console.print(f"  CEMS instructions added to {config_file}")


def _discover_mcp_url(api_url: str) -> str:
    """Discover the MCP HTTP endpoint URL.

    Priority:
    1. CEMS_MCP_URL from ~/.cems/credentials (user override)
    2. Server discovery endpoint (/api/config/setup)
    3. Fallback: derive from api_url (localhost → :8766, remote → mcp- prefix)
    """
    import urllib.error
    import urllib.request

    creds = _read_credentials()
    creds_mcp = creds.get("CEMS_MCP_URL", "")
    if creds_mcp:
        return creds_mcp

    discovery_url = api_url.rstrip("/") + "/api/config/setup"
    try:
        req = urllib.request.Request(discovery_url, method="GET")
        req.add_header("User-Agent", "CEMS-CLI/1.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            mcp_url = data.get("mcp_url")
            if mcp_url:
                return mcp_url
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        pass

    from urllib.parse import urlparse
    parsed = urlparse(api_url.rstrip("/"))
    if parsed.hostname in ("localhost", "127.0.0.1"):
        return f"{parsed.scheme}://{parsed.hostname}:8766/mcp"
    return f"{parsed.scheme}://mcp-{parsed.hostname}/mcp"


def _install_claude_hooks(data_path: Path, api_url: str, team_id: str | None = None, transport: str = "stdio", api_key: str | None = None) -> None:
    """Install Claude Code hooks, skills, settings, and MCP config."""
    claude_dir = Path.home() / ".claude"
    hooks_dir = claude_dir / "hooks"
    skills_dir = claude_dir / "skills" / "cems"
    src_hooks = data_path / "claude" / "hooks"
    src_skills = data_path / "claude" / "skills" / "cems"

    # Copy hooks
    hooks_dir.mkdir(parents=True, exist_ok=True)
    (hooks_dir / "utils").mkdir(exist_ok=True)

    # Remove old-named hook files from previous installs
    for old_name in ["user_prompts_submit.py", "stop.py", "pre_tool_use.py", "pre_compact.py"]:
        old_file = hooks_dir / old_name
        if old_file.exists():
            old_file.unlink()

    hook_files = [
        "cems_session_start.py", "cems_user_prompts_submit.py",
        "cems_stop.py", "cems_pre_tool_use.py", "cems_pre_compact.py",
    ]
    for f in hook_files:
        src = src_hooks / f
        if src.exists():
            dst = hooks_dir / f
            shutil.copy2(src, dst)
            dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    util_files = [
        "constants.py", "credentials.py", "hook_logger.py",
        "observer_manager.py", "project.py", "transcript.py",
    ]
    for f in util_files:
        src = src_hooks / "utils" / f
        if src.exists():
            shutil.copy2(src, hooks_dir / "utils" / f)

    console.print(f"  Hooks installed to {hooks_dir}")

    # Copy skills
    skills_dir.mkdir(parents=True, exist_ok=True)
    for f in src_skills.iterdir():
        if f.suffix == ".md":
            shutil.copy2(f, skills_dir / f.name)
    console.print(f"  Skills installed to {skills_dir}")

    # Copy commands (slash commands like /recall, /store)
    src_commands = data_path / "claude" / "commands"
    if src_commands.exists():
        commands_dir = claude_dir / "commands"
        commands_dir.mkdir(parents=True, exist_ok=True)
        # Remove old-named command files from previous installs
        old_cmd = commands_dir / "remember.md"
        if old_cmd.exists():
            old_cmd.unlink()
        for f in src_commands.iterdir():
            if f.suffix == ".md":
                shutil.copy2(f, commands_dir / f.name)
        console.print(f"  Commands installed to {commands_dir}")

    # Merge settings.json
    _merge_settings(claude_dir, data_path / "claude" / "settings.json")

    # Register MCP server
    _register_claude_mcp_server(api_url, team_id=team_id, transport=transport, api_key=api_key)

    # Add CEMS instructions to CLAUDE.md
    _append_cems_instructions(claude_dir / "CLAUDE.md")


def _register_claude_mcp_server(api_url: str, team_id: str | None = None, transport: str = "stdio", api_key: str | None = None) -> None:
    """Register CEMS MCP server in Claude Code config (~/.claude.json).

    Supports two transports:
    - stdio (default): Claude Code spawns `cems-mcp` locally. Credentials resolved from CWD.
    - http: Connects to remote MCP wrapper. API key embedded in config. No install needed.
    """
    claude_json = Path.home() / ".claude.json"

    existing: dict = {}
    if claude_json.exists():
        try:
            existing = json.loads(claude_json.read_text())
        except json.JSONDecodeError:
            existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})

    # Show what was there before
    old_config = mcp_servers.get("cems")
    if old_config:
        old_cmd = old_config.get("command", old_config.get("url", "unknown"))
        old_type = old_config.get("type", "stdio")
        console.print(f"  Replacing MCP config: {old_type} ({old_cmd}) → {transport}")

    if transport == "http":
        # HTTP mode — connect to remote MCP wrapper (no local install needed)
        mcp_url = _discover_mcp_url(api_url)
        resolved_key = api_key or _read_credentials().get("CEMS_API_KEY", "")
        headers: dict[str, str] = {"Authorization": f"Bearer {resolved_key}"}
        if team_id:
            headers["X-Team-Id"] = team_id
        mcp_servers["cems"] = {
            "type": "http",
            "url": mcp_url,
            "headers": headers,
        }
        claude_json.write_text(json.dumps(existing, indent=2) + "\n")
        console.print(f"  MCP server registered: {mcp_url} (http) ✓")
    else:
        # stdio mode (default) — resolve full path to cems-mcp binary
        cems_mcp_cmd = shutil.which("cems-mcp")
        if not cems_mcp_cmd:
            for candidate in [
                Path(sys.prefix) / "bin" / "cems-mcp",
                Path.home() / ".local" / "bin" / "cems-mcp",
            ]:
                if candidate.exists():
                    cems_mcp_cmd = str(candidate)
                    break
        if not cems_mcp_cmd:
            cems_mcp_cmd = "cems-mcp"
            console.print("  [yellow]⚠ cems-mcp not found — run: uv tool install cems[/yellow]")

        mcp_servers["cems"] = {
            "command": cems_mcp_cmd,
            "args": [],
        }
        claude_json.write_text(json.dumps(existing, indent=2) + "\n")
        console.print(f"  MCP server registered: {cems_mcp_cmd} (stdio) ✓")


def _migrate_old_hook_names(hooks: dict) -> None:
    """Remove old-named CEMS hooks from settings so they get replaced by new cems_ prefixed ones."""
    # Match only hooks under ~/.claude/hooks/ with old names (won't touch user hooks elsewhere)
    old_patterns = {
        ".claude/hooks/user_prompts_submit.py",
        ".claude/hooks/stop.py",
        ".claude/hooks/pre_tool_use.py",
        ".claude/hooks/pre_compact.py",
    }
    for event_name, entries in hooks.items():
        hooks[event_name] = [
            entry for entry in entries
            if not any(
                any(pattern in hook.get("command", "") for pattern in old_patterns)
                for hook in entry.get("hooks", [])
            )
        ]


def _migrate_removed_hooks(hooks: dict) -> bool:
    """Remove deprecated CEMS hooks from settings.

    Returns True if any hooks were removed (for messaging).

    Removed hooks:
    - cems_post_tool_use.py: Tool learning superseded by observer daemon.
      The daemon captures learnings holistically from the full session
      transcript, producing fewer, higher-quality memories.
    """
    removed_scripts = {"cems_post_tool_use.py"}
    changed = False

    for event_name in list(hooks.keys()):
        original = hooks[event_name]
        hooks[event_name] = [
            entry for entry in original
            if not any(
                any(script in hook.get("command", "")
                    for script in removed_scripts)
                for hook in entry.get("hooks", [])
            )
        ]
        if len(hooks[event_name]) != len(original):
            changed = True
        # Remove empty event arrays
        if not hooks[event_name]:
            del hooks[event_name]

    return changed


def _merge_settings(claude_dir: Path, template_path: Path) -> None:
    """Merge CEMS hook config into existing ~/.claude/settings.json.

    Preserves all existing settings (env, permissions, non-CEMS hooks).
    Appends CEMS hook entries that are not already registered (idempotent).
    """
    settings_file = claude_dir / "settings.json"

    # Load existing settings or start fresh
    existing: dict = {}
    if settings_file.exists():
        try:
            existing = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            existing = {}

    # Load CEMS hook template
    if not template_path.exists():
        console.print(f"  [red]Template not found: {template_path}[/red]")
        return
    template = json.loads(template_path.read_text())
    cems_hooks = template.get("hooks", {})

    # Migrate old hook names → new cems_ prefix names
    existing_hooks = existing.setdefault("hooks", {})
    _migrate_old_hook_names(existing_hooks)

    # Remove deprecated hooks (tool learning superseded by observer daemon)
    if _migrate_removed_hooks(existing_hooks):
        console.print("  Removed tool learning hook (superseded by observer daemon)")

    # Merge hooks: for each hook event, add CEMS hooks if not already present
    for event_name, cems_entries in cems_hooks.items():
        if event_name not in existing_hooks:
            # No existing config for this event — add CEMS entries directly
            existing_hooks[event_name] = cems_entries
        else:
            # Event exists — check if CEMS hooks are already registered
            # Match on script filename (not full command) to catch different runners
            # e.g. "uv run ~/.claude/hooks/cems_stop.py" and
            #      "$HOME/.claude/hooks/run_with_uv.sh $HOME/.claude/hooks/cems_stop.py"
            # both resolve to cems_stop.py
            def _script_names(cmd: str) -> set[str]:
                """Extract cems_*.py script names from a hook command."""
                return {part.rsplit("/", 1)[-1] for part in cmd.split() if part.endswith(".py") and "cems_" in part}

            existing_scripts: set[str] = set()
            for entry in existing_hooks[event_name]:
                for hook in entry.get("hooks", []):
                    existing_scripts |= _script_names(hook.get("command", ""))

            for cems_entry in cems_entries:
                entry_scripts: set[str] = set()
                for h in cems_entry.get("hooks", []):
                    entry_scripts |= _script_names(h.get("command", ""))
                if not entry_scripts & existing_scripts:
                    existing_hooks[event_name].append(cems_entry)

    settings_file.write_text(json.dumps(existing, indent=2) + "\n")
    console.print(f"  Settings merged into {settings_file}")


def _install_goose_config(api_url: str) -> None:
    """Install CEMS extension config for Goose."""
    goose_config_dir = Path.home() / ".config" / "goose"
    goose_config_file = goose_config_dir / "config.yaml"

    # Read existing config or start fresh
    existing_content = ""
    if goose_config_file.exists():
        existing_content = goose_config_file.read_text()

    # Check if CEMS extension is already configured
    if "cems-mcp" in existing_content or "CEMS Memory" in existing_content:
        console.print("  CEMS extension already configured in Goose config")
        return

    # Build the extension config block
    cems_block = f"""\
  cems:
    name: CEMS Memory
    description: Long-term memory system for search, store, and manage memories
    cmd: cems-mcp
    args: []
    enabled: true
    type: stdio
    timeout: 300
    envs:
      CEMS_API_URL: "{api_url}"
"""

    goose_config_dir.mkdir(parents=True, exist_ok=True)

    if "extensions:" in existing_content:
        # Append CEMS config under existing extensions section
        lines = existing_content.split("\n")
        new_lines = []
        inserted = False
        for line in lines:
            new_lines.append(line)
            if line.strip() == "extensions:" and not inserted:
                new_lines.append(cems_block.rstrip())
                inserted = True

        if not inserted:
            # extensions: wasn't on its own line, just append
            new_lines.append(cems_block)

        goose_config_file.write_text("\n".join(new_lines))
    else:
        # No extensions section — append the full block
        extension_block = f"""
# CEMS Memory extension — added by cems setup
extensions:
{cems_block}"""
        if existing_content and not existing_content.endswith("\n"):
            existing_content += "\n"
        goose_config_file.write_text(existing_content + extension_block)

    console.print(f"  Goose config updated: {goose_config_file}")


def _install_codex_config(data_path: Path, api_url: str) -> None:
    """Install Codex commands, skills, and MCP config."""
    codex_dir = Path.home() / ".codex"
    src_commands = data_path / "codex" / "commands"
    src_skills = data_path / "codex" / "skills"

    # Copy commands
    commands_dir = codex_dir / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    for f in src_commands.iterdir():
        if f.suffix == ".md":
            shutil.copy2(f, commands_dir / f.name)
    console.print(f"  Commands installed to {commands_dir}")

    # Copy skills (each skill is a directory with SKILL.md)
    skills_dir = codex_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for skill in src_skills.iterdir():
        if skill.is_dir():
            dst_skill = skills_dir / skill.name
            dst_skill.mkdir(exist_ok=True)
            for f in skill.iterdir():
                if f.suffix == ".md":
                    shutil.copy2(f, dst_skill / f.name)
    console.print(f"  Skills installed to {skills_dir}")

    # Register MCP server in config.toml
    _register_codex_mcp(codex_dir, api_url)

    # Add CEMS instructions to AGENTS.md
    _append_cems_instructions(codex_dir / "AGENTS.md")


def _register_codex_mcp(codex_dir: Path, api_url: str) -> None:
    """Register CEMS MCP server in Codex config.toml."""
    config_file = codex_dir / "config.toml"

    mcp_url = _discover_mcp_url(api_url)

    # Read existing config
    existing = ""
    if config_file.exists():
        existing = config_file.read_text()

    # Check if already configured
    if "mcp_servers.cems" in existing:
        console.print("  MCP server already registered in config.toml")
        return

    # Append CEMS MCP config
    cems_block = f"""
[mcp_servers.cems]
url = "{mcp_url}"
bearer_token_env_var = "CEMS_API_KEY"
"""
    codex_dir.mkdir(parents=True, exist_ok=True)
    if existing and not existing.endswith("\n"):
        existing += "\n"
    config_file.write_text(existing + cems_block)
    console.print(f"  MCP server registered: {mcp_url}")


def _install_cursor_hooks(data_path: Path, api_url: str, team_id: str | None = None, transport: str = "stdio", api_key: str | None = None) -> None:
    """Install Cursor hooks, skills, and MCP config."""
    cursor_dir = Path.home() / ".cursor"
    hooks_dir = cursor_dir / "hooks"
    src_hooks = data_path / "cursor" / "hooks"
    src_hooks_json = data_path / "cursor" / "hooks.json"

    cursor_dir.mkdir(parents=True, exist_ok=True)
    hooks_dir.mkdir(exist_ok=True)

    # Copy hook scripts
    for f in src_hooks.iterdir():
        if f.suffix == ".py":
            dst = hooks_dir / f.name
            shutil.copy2(f, dst)
            dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    # Handle hooks.json
    hooks_json = cursor_dir / "hooks.json"
    if hooks_json.exists():
        console.print(f"  [yellow]Existing hooks.json found — you may need to merge manually[/yellow]")
    else:
        shutil.copy2(src_hooks_json, hooks_json)

    console.print(f"  Hooks installed to {hooks_dir}")

    # Copy skills (each skill is a directory with SKILL.md)
    src_skills = data_path / "cursor" / "skills"
    if src_skills.exists():
        skills_dir = cursor_dir / "skills"
        skills_dir.mkdir(exist_ok=True)
        for skill in src_skills.iterdir():
            if skill.is_dir():
                dst_skill = skills_dir / skill.name
                dst_skill.mkdir(exist_ok=True)
                for f in skill.iterdir():
                    if f.suffix == ".md":
                        shutil.copy2(f, dst_skill / f.name)
        console.print(f"  Skills installed to {skills_dir}")

    # Register MCP server in mcp.json
    _register_cursor_mcp(cursor_dir, api_url, team_id=team_id, transport=transport, api_key=api_key)

    # Add CEMS instructions to Cursor rules
    rules_dir = cursor_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    _append_cems_instructions(rules_dir / "cems.md")


def _register_cursor_mcp(cursor_dir: Path, api_url: str, team_id: str | None = None, transport: str = "stdio", api_key: str | None = None) -> None:
    """Register CEMS MCP server in Cursor mcp.json."""
    mcp_file = cursor_dir / "mcp.json"

    existing: dict = {}
    if mcp_file.exists():
        try:
            existing = json.loads(mcp_file.read_text())
        except json.JSONDecodeError:
            existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})

    if transport == "http":
        mcp_url = _discover_mcp_url(api_url)
        resolved_key = api_key or _read_credentials().get("CEMS_API_KEY", "")
        headers: dict[str, str] = {"Authorization": f"Bearer {resolved_key}"}
        if team_id:
            headers["X-Team-Id"] = team_id
        mcp_servers["cems"] = {
            "type": "http",
            "url": mcp_url,
            "headers": headers,
        }
        mcp_file.write_text(json.dumps(existing, indent=2) + "\n")
        console.print(f"  MCP server registered: {mcp_url} (http) ✓")
    else:
        cems_mcp_cmd = shutil.which("cems-mcp")
        if not cems_mcp_cmd:
            for candidate in [
                Path(sys.prefix) / "bin" / "cems-mcp",
                Path.home() / ".local" / "bin" / "cems-mcp",
            ]:
                if candidate.exists():
                    cems_mcp_cmd = str(candidate)
                    break
        if not cems_mcp_cmd:
            cems_mcp_cmd = "cems-mcp"
        mcp_servers["cems"] = {
            "command": cems_mcp_cmd,
            "args": [],
        }
        mcp_file.write_text(json.dumps(existing, indent=2) + "\n")
        console.print(f"  MCP server registered: {cems_mcp_cmd} (stdio) ✓")


def _setup_project_credentials(api_url: str | None, api_key: str | None) -> None:
    """Create per-project .cems/credentials in the current directory.

    Prompts interactively for API URL and key when flags are omitted.
    Also adds .cems/ to .gitignore and .dockerignore if present.
    """
    import os

    if not api_url or not api_key:
        if not _is_interactive():
            console.print("[red]--api-url and --api-key are required in non-interactive mode[/red]")
            console.print()
            console.print("Usage: cems setup --project --api-url URL --api-key KEY")
            raise click.Abort()

        # Use existing global credentials as defaults for project setup
        existing_creds = _read_credentials() if (Path.home() / ".cems" / "credentials").exists() else {}
        default_url = existing_creds.get("CEMS_API_URL", "http://localhost:8765")
        default_key = existing_creds.get("CEMS_API_KEY", "")

        console.print()
        console.print("[bold]Project Credentials Setup[/bold]")
        if default_key:
            console.print("Using global credentials as defaults. Press enter to keep, or type new values.")
        else:
            console.print("Get these from your CEMS admin.")
        console.print()

        if not api_url:
            api_url = click.prompt(
                "CEMS API URL",
                default=default_url,
                show_default=True,
            )
        if not api_key:
            if default_key:
                use_same = click.confirm(
                    f"Use same API key as global ({default_key[:8]}...)?",
                    default=True,
                )
                if use_same:
                    api_key = default_key
                else:
                    api_key = click.prompt("CEMS API Key", hide_input=True)
            else:
                api_key = click.prompt("CEMS API Key", hide_input=True)

        if not api_key:
            console.print("[red]API key is required.[/red]")
            raise click.Abort()

    cwd = os.getcwd()
    cems_dir = Path(cwd) / ".cems"
    creds_file = cems_dir / "credentials"

    console.print()
    console.print(f"[bold]Project CEMS Setup[/bold] — {cwd}")
    console.print()

    # Write credentials
    cems_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CEMS project credentials — generated by cems setup --project",
        f"CEMS_API_URL={api_url}",
        f"CEMS_API_KEY={api_key}",
    ]

    # Team discovery against the project server
    team_id = _discover_team(api_url, api_key)
    if team_id:
        lines.append(f"CEMS_TEAM_ID={team_id}")

    creds_file.write_text("\n".join(lines) + "\n")
    creds_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
    console.print(f"  [green]✓ Credentials:[/green] {creds_file}")

    # Add .cems/ to .gitignore
    gitignore = Path(cwd) / ".gitignore"
    _ensure_in_ignorefile(gitignore, ".cems/")

    # Add .cems/ to .dockerignore if it exists
    dockerignore = Path(cwd) / ".dockerignore"
    if dockerignore.exists():
        _ensure_in_ignorefile(dockerignore, ".cems/")

    # Version skew check
    _check_server_version(api_url, api_key)

    console.print()
    console.print("[bold green]Project setup complete![/bold green]")
    console.print()
    console.print(f"  Hooks will now route to [cyan]{api_url}[/cyan] when working in this project.")
    console.print()


def _ensure_in_ignorefile(path: Path, pattern: str) -> None:
    """Add a pattern to an ignore file (.gitignore, .dockerignore) if not present."""
    if path.exists():
        content = path.read_text()
        if pattern in content.splitlines():
            return  # Already present
        # Append with newline separator
        if content and not content.endswith("\n"):
            content += "\n"
        content += f"{pattern}\n"
        path.write_text(content)
        console.print(f"  [green]✓ Updated:[/green] {path.name} (added {pattern})")
    else:
        path.write_text(f"{pattern}\n")
        console.print(f"  [green]✓ Created:[/green] {path.name} (added {pattern})")


def _check_server_version(api_url: str, api_key: str) -> None:
    """Check server version and warn about skew with client."""
    import urllib.error
    import urllib.request

    try:
        from importlib.metadata import version as get_version
        client_version = get_version("cems")
    except Exception:
        return

    try:
        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/api/health",
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "CEMS-CLI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            server_version = data.get("version", "")
            if server_version and server_version != client_version:
                console.print(
                    f"  [yellow]⚠ Server version ({server_version}) differs from "
                    f"client ({client_version}). Some features may not work until "
                    f"the server is updated.[/yellow]"
                )
    except Exception:
        pass  # Version check is best-effort


@click.command()
@click.option("--claude", "install_claude", is_flag=True, help="Install Claude Code hooks only")
@click.option("--cursor", "install_cursor", is_flag=True, help="Install Cursor hooks only")
@click.option("--codex", "install_codex", is_flag=True, help="Install Codex commands/skills only")
@click.option("--goose", "install_goose", is_flag=True, help="Install Goose extension config only")
@click.option("--project", "install_project", is_flag=True, help="Create per-project .cems/credentials in CWD")
@click.option("--api-url", help="CEMS server URL (non-interactive)")
@click.option("--api-key", help="CEMS API key (non-interactive)")
@click.option("--transport", type=click.Choice(["stdio", "http"]), default=None, help="MCP transport: stdio (local, default) or http (remote, no install needed)")
def setup(install_claude: bool, install_cursor: bool, install_codex: bool, install_goose: bool, install_project: bool, api_url: str | None, api_key: str | None, transport: str | None) -> None:
    """Set up CEMS hooks and credentials for your IDE.

    Installs hooks, skills, and settings for Claude Code, Cursor, Codex, and/or Goose.
    Prompts for API credentials if not already configured.

    \b
    Examples:
        cems setup              # Interactive
        cems setup --claude     # Claude Code only
        cems setup --cursor     # Cursor only
        cems setup --codex      # Codex only
        cems setup --goose      # Goose only
        cems setup --claude --api-url URL --api-key KEY  # Non-interactive
        cems setup --project --api-url URL --api-key KEY  # Per-project config
    """
    # Non-interactive --project flag — separate flow
    if install_project and not _is_interactive():
        _setup_project_credentials(api_url, api_key)
        return

    console.print()
    console.print("[bold]CEMS Setup[/bold]")
    console.print()

    data_path = _get_data_path()

    # Step 1: Select IDEs
    if not install_claude and not install_cursor and not install_codex and not install_goose:
        if _is_interactive():
            selected = _multiselect(
                "Select IDEs to configure",
                [
                    ("claude", "Claude Code", True),
                    ("cursor", "Cursor", False),
                    ("codex", "Codex", False),
                    ("goose", "Goose", False),
                ],
            )
            install_claude = "claude" in selected
            install_cursor = "cursor" in selected
            install_codex = "codex" in selected
            install_goose = "goose" in selected
        else:
            # Non-interactive — default to Claude Code
            console.print("[yellow]Non-interactive mode: installing Claude Code hooks (use --cursor/--codex/--goose for others)[/yellow]")
            install_claude = True

    # Step 2: Credential scope
    credential_scope = "global"
    if _is_interactive() and not install_project:
        console.print()
        credential_scope = _single_select(
            "Credential scope",
            [
                ("global", "Global — CEMS accesses all your projects"),
                ("project", "Project — CEMS accesses only this project's data"),
            ],
            default=0,
        )
        install_project = credential_scope == "project"
    elif install_project:
        credential_scope = "project"

    # Step 3: Credentials (adapts based on scope + existing state)
    import os as _os
    global_creds_file = Path.home() / ".cems" / "credentials"
    project_creds_file = Path(_os.getcwd()) / ".cems" / "credentials"

    if credential_scope == "project":
        # Project scope — write to CWD/.cems/credentials
        if project_creds_file.exists() and _is_interactive() and not api_key:
            console.print()
            action = _single_select(
                f"Project credentials found ({project_creds_file})",
                [
                    ("keep", "Keep existing"),
                    ("overwrite", "Overwrite with new values"),
                ],
                default=0,
            )
            if action == "overwrite":
                _setup_project_credentials(api_url, api_key)
        elif not project_creds_file.exists():
            _setup_project_credentials(api_url, api_key)
        else:
            console.print(f"[green]Project credentials:[/green] {project_creds_file}")

        # Also ensure global creds exist for hooks/MCP that need them
        if not global_creds_file.exists() and not api_key:
            # Read the project creds we just wrote to populate global as well
            pass  # Global is optional in project-only mode
    else:
        # Global scope — write to ~/.cems/credentials
        if global_creds_file.exists() and _is_interactive() and not api_key:
            console.print()
            action = _single_select(
                f"Global credentials found ({global_creds_file})",
                [
                    ("keep", "Keep existing"),
                    ("overwrite", "Overwrite with new values"),
                ],
                default=0,
            )
            if action == "overwrite":
                if not _setup_credentials(api_url=None, api_key=None):
                    raise click.Abort()
            else:
                console.print(f"[green]Credentials:[/green] {global_creds_file}")
        elif not _setup_credentials(api_url=api_url, api_key=api_key):
            raise click.Abort()

    console.print()

    # Resolve API URL and key for team discovery
    # Read from whichever credential file we just wrote/kept
    if credential_scope == "project" and project_creds_file.exists():
        from cems.shared.credentials import parse_credentials_file
        creds = parse_credentials_file(str(project_creds_file))
        # Merge with global if project creds are sparse
        if global_creds_file.exists():
            global_creds = _read_credentials()
            for k, v in global_creds.items():
                creds.setdefault(k, v)
    else:
        creds = _read_credentials()
    resolved_url = api_url or creds.get("CEMS_API_URL", "http://localhost:8765")
    resolved_key = api_key or creds.get("CEMS_API_KEY", "")

    # Search mode selection
    # Write to project creds when scope is project, global otherwise
    active_creds_file = project_creds_file if credential_scope == "project" and project_creds_file.exists() else None
    existing_mode = creds.get("CEMS_SEARCH_MODE", "")
    if _is_interactive():
        console.print()
        default_idx = 1 if existing_mode == "agentic" else 0
        search_mode = _single_select(
            "Search mode",
            [
                ("auto", "Auto — Embedding-based (fast, ~100ms)"),
                ("agentic", "Agentic — LLM agents reason over memories (smarter, ~3s)"),
            ],
            default=default_idx,
        )
        if search_mode == "agentic":
            _append_credential("CEMS_SEARCH_MODE", "agentic", creds_path=active_creds_file)
        elif existing_mode:
            _remove_credential("CEMS_SEARCH_MODE", creds_path=active_creds_file)

    # Transport selection (interactive or from flag)
    resolved_transport = transport or "stdio"
    if not transport and _is_interactive():
        console.print()
        resolved_transport = _single_select(
            "MCP transport",
            [
                ("stdio", "stdio — local cems-mcp process (recommended, requires cems installed)"),
                ("http", "HTTP — remote server, no install needed (simpler for non-devs)"),
            ],
            default=0,
        )

    # Discover user's team membership
    team_id = creds.get("CEMS_TEAM_ID")  # Preserve existing if set
    if not team_id and resolved_key:
        team_id = _discover_team(resolved_url, resolved_key)

    # Install
    if install_claude:
        console.print("[bold blue]Claude Code[/bold blue]")
        _install_claude_hooks(data_path, resolved_url, team_id=team_id, transport=resolved_transport, api_key=resolved_key)
        console.print()

    if install_cursor:
        console.print("[bold blue]Cursor[/bold blue]")
        _install_cursor_hooks(data_path, resolved_url, team_id=team_id, transport=resolved_transport, api_key=resolved_key)
        console.print()

    if install_codex:
        console.print("[bold blue]Codex[/bold blue]")
        _install_codex_config(data_path, resolved_url)
        console.print()

    if install_goose:
        console.print("[bold blue]Goose[/bold blue]")
        _install_goose_config(resolved_url)
        console.print()

    # Save install preferences for `cems update` to use
    _save_install_preferences(
        claude=install_claude, cursor=install_cursor,
        codex=install_codex, goose=install_goose,
    )

    # Summary
    console.print("[bold green]Setup complete![/bold green]")
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="white")
    if credential_scope == "project" and project_creds_file.exists():
        table.add_row("Project creds", str(project_creds_file))
    if global_creds_file.exists():
        table.add_row("Global creds", str(global_creds_file))
    if install_claude:
        table.add_row("Claude hooks", str(Path.home() / ".claude" / "hooks"))
        table.add_row("Claude skills", str(Path.home() / ".claude" / "skills" / "cems"))
        table.add_row("MCP server", str(Path.home() / ".claude.json"))
    if install_cursor:
        table.add_row("Cursor hooks", str(Path.home() / ".cursor" / "hooks"))
        table.add_row("Cursor skills", str(Path.home() / ".cursor" / "skills"))
        table.add_row("Cursor MCP config", str(Path.home() / ".cursor" / "mcp.json"))
    if install_codex:
        table.add_row("Codex commands", str(Path.home() / ".codex" / "commands"))
        table.add_row("Codex skills", str(Path.home() / ".codex" / "skills"))
        table.add_row("Codex MCP config", str(Path.home() / ".codex" / "config.toml"))
    if install_goose:
        table.add_row("Goose config", str(Path.home() / ".config" / "goose" / "config.yaml"))
    console.print(table)

    console.print()
    console.print("[bold]Add to your shell profile (~/.zshrc or ~/.bashrc):[/bold]")
    console.print()
    console.print('  [cyan]eval "$(cems env)"[/cyan]')
    console.print()
    console.print("This exports CEMS_API_KEY so the MCP server can authenticate.")
    console.print("Then restart your shell and IDE.")
    console.print()
    console.print("Test: [cyan]cems status[/cyan]")
