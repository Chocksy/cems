"""Update command for CEMS CLI.

Pulls the latest version from GitHub and re-installs hooks/skills.

Usage:
    cems update             # Upgrade package + re-deploy hooks/skills
    cems update --hooks     # Only re-deploy hooks/skills (no package upgrade)
"""

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import click

from cems.cli_utils import console

PACKAGE_SOURCE = "cems @ git+https://github.com/chocksy/cems.git"
UPDATE_LOG = Path.home() / ".cems" / "update.log"


def _get_current_version() -> str:
    """Get currently installed CEMS version."""
    try:
        return pkg_version("cems")
    except Exception:
        return "unknown"


def _get_installed_version_fresh() -> str:
    """Get version from a fresh process (bypasses importlib cache)."""
    cems_bin = shutil.which("cems")
    if not cems_bin:
        return "unknown"
    try:
        result = subprocess.run(
            [cems_bin, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        # Output is like "cems, version 0.2.0"
        if result.returncode == 0:
            parts = result.stdout.strip().rsplit(" ", 1)
            if len(parts) == 2:
                return parts[1]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _log_update(message: str) -> None:
    """Append a line to ~/.cems/update.log."""
    try:
        UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with UPDATE_LOG.open("a") as f:
            f.write(f"[{ts}] {message}\n")
    except OSError:
        pass


def _upgrade_package() -> bool:
    """Run uv tool install to pull latest from GitHub.

    Returns True if upgrade succeeded.
    """
    uv = shutil.which("uv")
    if not uv:
        console.print("[red]uv not found on PATH. Install it: https://docs.astral.sh/uv/[/red]")
        return False

    console.print("Pulling latest from GitHub...")
    result = subprocess.run(
        [uv, "tool", "install", PACKAGE_SOURCE, "--force"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        msg = result.stderr.strip()
        console.print(f"[red]Upgrade failed:[/red] {msg}")
        _log_update(f"FAILED: {msg[:200]}")
        return False

    return True


def _redeploy_hooks() -> None:
    """Re-run setup to copy latest hooks/skills."""
    cems_bin = shutil.which("cems")
    if not cems_bin:
        console.print("[yellow]cems not found on PATH — skipping hook re-deploy[/yellow]")
        console.print("Run manually: [cyan]cems setup[/cyan]")
        return

    has_claude = (Path.home() / ".claude" / "hooks" / "cems_session_start.py").exists()
    has_cursor = (Path.home() / ".cursor" / "hooks" / "cems_session_start.py").exists()

    if not has_claude and not has_cursor:
        console.print("[yellow]No existing hooks found — run [cyan]cems setup[/cyan] to configure[/yellow]")
        return

    # Use the NEW cems binary (just installed) to redeploy
    args = [cems_bin, "setup"]
    if has_claude and not has_cursor:
        args.append("--claude")
    elif has_cursor and not has_claude:
        args.append("--cursor")
    else:
        args.extend(["--claude", "--cursor"])

    # Pass existing credentials so setup doesn't prompt
    creds_file = Path.home() / ".cems" / "credentials"
    if creds_file.exists():
        for line in creds_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("CEMS_API_URL="):
                args.extend(["--api-url", line.partition("=")[2].strip().strip("'\"")])
            elif line.startswith("CEMS_API_KEY="):
                args.extend(["--api-key", line.partition("=")[2].strip().strip("'\"")])

    result = subprocess.run(args, capture_output=True, text=True)
    if result.stdout:
        console.print(result.stdout.rstrip())
    if result.returncode != 0 and result.stderr:
        console.print(f"[yellow]{result.stderr.strip()}[/yellow]")


@click.command("update")
@click.option("--hooks", "hooks_only", is_flag=True, help="Only re-deploy hooks/skills (skip package upgrade)")
def update_cmd(hooks_only: bool) -> None:
    """Update CEMS to the latest version.

    Pulls the latest code from GitHub and re-deploys hooks/skills
    to pick up any changes.

    \b
    Examples:
        cems update             # Full update (package + hooks)
        cems update --hooks     # Re-deploy hooks only (no package upgrade)
    """
    console.print()
    console.print("[bold]CEMS Update[/bold]")
    console.print()

    old_version = _get_current_version()
    version_changed = False

    if not hooks_only:
        console.print(f"Current version: [cyan]{old_version}[/cyan]")
        console.print()

        if not _upgrade_package():
            raise click.Abort()

        # Get version from fresh process (importlib cache won't see the new version)
        new_version = _get_installed_version_fresh()
        if new_version != old_version:
            version_changed = True
            console.print(f"[green]Updated: {old_version} -> {new_version}[/green]")
            _log_update(f"Updated {old_version} -> {new_version}")
        else:
            console.print(f"[green]Already on latest ({new_version})[/green]")
            _log_update(f"Already on latest ({new_version})")
        console.print()

    # Re-deploy hooks/skills only if version changed or --hooks flag used
    if hooks_only or version_changed:
        console.print("Re-deploying hooks and skills...")
        _redeploy_hooks()
    else:
        console.print("[dim]Hooks up to date, skipping re-deploy[/dim]")

    console.print()
    console.print("[bold green]Update complete![/bold green] Restart your IDE to pick up changes.")
