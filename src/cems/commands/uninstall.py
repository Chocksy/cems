"""Uninstall command for CEMS CLI.

Removes hooks, skills, settings entries, and optionally credentials.

Usage:
    cems uninstall              # Remove hooks, skills, settings (keep credentials)
    cems uninstall --all        # Also remove credentials
"""

import json
from pathlib import Path

import click

from cems.cli_utils import console

# CEMS hook files (must match setup.py)
CLAUDE_HOOK_FILES = [
    "cems_session_start.py", "cems_user_prompts_submit.py",
    "cems_post_tool_use.py", "cems_stop.py", "cems_pre_tool_use.py", "cems_pre_compact.py",
]

CLAUDE_UTIL_FILES = [
    "constants.py", "credentials.py", "hook_logger.py",
    "observer_manager.py", "project.py", "transcript.py",
]

CLAUDE_COMMAND_FILES = [
    "recall.md", "remember.md",
]

CURSOR_HOOK_FILES = [
    "cems_session_start.py", "cems_agent_response.py", "cems_stop.py",
]

CURSOR_SKILL_DIRS = [
    "cems-recall", "cems-remember", "cems-forget", "cems-share", "cems-context",
]

CODEX_COMMAND_FILES = [
    "recall.md", "remember.md", "foundation.md",
]

CODEX_SKILL_DIRS = [
    "recall", "remember",
]


def _remove_claude_hooks() -> int:
    """Remove CEMS hooks and skills from ~/.claude/. Returns count of removed files."""
    claude_dir = Path.home() / ".claude"
    hooks_dir = claude_dir / "hooks"
    skills_dir = claude_dir / "skills" / "cems"
    removed = 0

    # Remove hook files
    for f in CLAUDE_HOOK_FILES:
        path = hooks_dir / f
        if path.exists():
            path.unlink()
            removed += 1

    # Remove util files
    for f in CLAUDE_UTIL_FILES:
        path = hooks_dir / "utils" / f
        if path.exists():
            path.unlink()
            removed += 1

    # Remove utils dir if empty
    utils_dir = hooks_dir / "utils"
    if utils_dir.exists() and not any(utils_dir.iterdir()):
        utils_dir.rmdir()

    # Remove skills
    if skills_dir.exists():
        for f in skills_dir.iterdir():
            f.unlink()
            removed += 1
        skills_dir.rmdir()

    # Remove commands
    commands_dir = claude_dir / "commands"
    for f in CLAUDE_COMMAND_FILES:
        path = commands_dir / f
        if path.exists():
            path.unlink()
            removed += 1

    # Clean CEMS hooks from settings.json
    settings_file = claude_dir / "settings.json"
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
            hooks = settings.get("hooks", {})
            changed = False
            for event_name in list(hooks.keys()):
                original_len = len(hooks[event_name])
                hooks[event_name] = [
                    entry for entry in hooks[event_name]
                    if not any(
                        "cems_" in hook.get("command", "")
                        for hook in entry.get("hooks", [])
                    )
                ]
                if len(hooks[event_name]) != original_len:
                    changed = True
                # Remove empty event arrays
                if not hooks[event_name]:
                    del hooks[event_name]
                    changed = True
            if changed:
                settings_file.write_text(json.dumps(settings, indent=2) + "\n")
                console.print("  Cleaned CEMS entries from settings.json")
        except (json.JSONDecodeError, OSError):
            pass

    # Clean MCP entry from ~/.claude.json
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            config = json.loads(claude_json.read_text())
            if "cems" in config.get("mcpServers", {}):
                del config["mcpServers"]["cems"]
                if not config["mcpServers"]:
                    del config["mcpServers"]
                claude_json.write_text(json.dumps(config, indent=2) + "\n")
                console.print("  Cleaned CEMS MCP entry from ~/.claude.json")
        except (json.JSONDecodeError, OSError):
            pass

    return removed


def _remove_cursor_hooks() -> int:
    """Remove CEMS hooks, skills, and MCP config from ~/.cursor/. Returns count of removed files."""
    cursor_dir = Path.home() / ".cursor"
    hooks_dir = cursor_dir / "hooks"
    removed = 0

    # Remove hook files
    for f in CURSOR_HOOK_FILES:
        path = hooks_dir / f
        if path.exists():
            path.unlink()
            removed += 1

    # Remove skill directories
    for d in CURSOR_SKILL_DIRS:
        skill_dir = cursor_dir / "skills" / d
        if skill_dir.exists():
            for f in skill_dir.iterdir():
                f.unlink()
                removed += 1
            skill_dir.rmdir()

    # Clean CEMS MCP entry from mcp.json
    mcp_file = cursor_dir / "mcp.json"
    if mcp_file.exists():
        try:
            config = json.loads(mcp_file.read_text())
            if "cems" in config.get("mcpServers", {}):
                del config["mcpServers"]["cems"]
                if not config["mcpServers"]:
                    del config["mcpServers"]
                mcp_file.write_text(json.dumps(config, indent=2) + "\n")
                console.print("  Cleaned CEMS MCP entry from mcp.json")
        except (json.JSONDecodeError, OSError):
            pass

    return removed


def _remove_codex() -> int:
    """Remove CEMS commands, skills, and MCP config from ~/.codex/. Returns count of removed files."""
    codex_dir = Path.home() / ".codex"
    removed = 0

    # Remove commands
    for f in CODEX_COMMAND_FILES:
        path = codex_dir / "commands" / f
        if path.exists():
            path.unlink()
            removed += 1

    # Remove skill directories
    for d in CODEX_SKILL_DIRS:
        skill_dir = codex_dir / "skills" / d
        if skill_dir.exists():
            for f in skill_dir.iterdir():
                f.unlink()
                removed += 1
            skill_dir.rmdir()

    # Clean CEMS MCP entry from config.toml
    config_file = codex_dir / "config.toml"
    if config_file.exists():
        try:
            content = config_file.read_text()
            if "mcp_servers.cems" in content:
                # Remove the [mcp_servers.cems] block
                lines = content.split("\n")
                new_lines = []
                skip = False
                for line in lines:
                    if line.strip() == "[mcp_servers.cems]":
                        skip = True
                        continue
                    # Stop skipping at next section header
                    if skip and line.strip().startswith("["):
                        skip = False
                    if skip:
                        continue
                    new_lines.append(line)
                # Remove trailing blank lines
                while new_lines and not new_lines[-1].strip():
                    new_lines.pop()
                config_file.write_text("\n".join(new_lines) + "\n")
                console.print("  Cleaned CEMS MCP entry from config.toml")
        except OSError:
            pass

    return removed


def _remove_goose_config() -> int:
    """Remove CEMS extension block from ~/.config/goose/config.yaml.

    Returns:
        Count of removed config blocks (0 or 1) for consistency with other _remove_* functions.
    """
    goose_config = Path.home() / ".config" / "goose" / "config.yaml"
    if not goose_config.exists():
        return 0

    try:
        content = goose_config.read_text()
        if "cems-mcp" not in content and "CEMS Memory" not in content:
            return 0

        # Remove the CEMS extension block (from "  cems:" to next extension or end of section)
        lines = content.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            # Start skipping at "  cems:" under extensions
            if line.strip().startswith("cems:") and not skip:
                skip = True
                continue
            # Stop skipping at next top-level extension key (2-space indent + word + colon)
            if skip and line and not line.startswith("    ") and not line.startswith("\t\t"):
                if line.startswith("  ") and ":" in line and not line.strip().startswith("#"):
                    skip = False
                elif not line.startswith("  "):
                    skip = False
            if skip:
                continue
            new_lines.append(line)

        # Remove the comment line if it exists
        new_lines = [l for l in new_lines if "CEMS Memory extension" not in l]

        goose_config.write_text("\n".join(new_lines))
        return 1
    except OSError:
        return 0


def _remove_credentials() -> bool:
    """Remove ~/.cems/credentials. Returns True if removed."""
    creds_file = Path.home() / ".cems" / "credentials"
    if creds_file.exists():
        creds_file.unlink()
        # Remove dir if empty
        cems_dir = creds_file.parent
        if cems_dir.exists() and not any(cems_dir.iterdir()):
            cems_dir.rmdir()
        return True
    return False


@click.command()
@click.option("--all", "remove_all", is_flag=True, help="Also remove credentials (~/.cems/)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def uninstall(remove_all: bool, yes: bool) -> None:
    """Remove CEMS hooks, skills, and settings from your IDE.

    By default keeps credentials so you can re-install easily.
    Use --all to also remove credentials.

    \b
    Examples:
        cems uninstall          # Remove hooks/skills, keep credentials
        cems uninstall --all    # Remove everything
        cems uninstall -y       # Skip confirmation
    """
    console.print()
    console.print("[bold]CEMS Uninstall[/bold]")
    console.print()

    what = "hooks, skills, and settings entries"
    if remove_all:
        what += " + credentials"

    if not yes:
        console.print(f"This will remove: {what}")
        console.print()
        if not click.confirm("Continue?", default=False):
            raise click.Abort()

    console.print()

    # Claude Code
    claude_removed = _remove_claude_hooks()
    if claude_removed:
        console.print(f"  [red]Removed {claude_removed} Claude Code files[/red]")
    else:
        console.print("  No Claude Code hooks found")

    # Cursor
    cursor_removed = _remove_cursor_hooks()
    if cursor_removed:
        console.print(f"  [red]Removed {cursor_removed} Cursor files[/red]")
    else:
        console.print("  No Cursor hooks found")

    # Codex
    codex_removed = _remove_codex()
    if codex_removed:
        console.print(f"  [red]Removed {codex_removed} Codex files[/red]")
    else:
        console.print("  No Codex files found")

    # Goose
    if _remove_goose_config():
        console.print("  [red]Removed CEMS extension from Goose config[/red]")
    else:
        console.print("  No Goose config found")

    # Credentials
    if remove_all:
        if _remove_credentials():
            console.print("  [red]Removed credentials[/red]")
        else:
            console.print("  No credentials found")
    else:
        console.print("  [green]Credentials kept[/green] (use --all to remove)")

    console.print()
    console.print("[bold]Uninstall complete.[/bold] Restart your IDE.")
    console.print("Re-install anytime: [cyan]cems setup[/cyan]")
