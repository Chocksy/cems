"""Update command for CEMS CLI.

Pulls the latest version from GitHub and re-installs hooks/skills.

Usage:
    cems update             # Upgrade package + re-deploy hooks/skills
    cems update --hooks     # Only re-deploy hooks/skills (no package upgrade)
"""

import shutil
import subprocess
import sys
from importlib.metadata import version as pkg_version

import click

from cems.cli_utils import console

PACKAGE_SOURCE = "cems @ git+https://github.com/chocksy/cems.git"


def _get_current_version() -> str:
    """Get currently installed CEMS version."""
    try:
        return pkg_version("cems")
    except Exception:
        return "unknown"


def _upgrade_package() -> bool:
    """Run uv tool upgrade to pull latest from GitHub.

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
        console.print(f"[red]Upgrade failed:[/red] {result.stderr.strip()}")
        return False

    return True


def _redeploy_hooks() -> None:
    """Re-run setup to copy latest hooks/skills.

    Imports setup internals to avoid spawning another subprocess.
    """
    # Re-import after upgrade so we get the NEW code
    # But since we're already running the old binary, we call setup via subprocess
    cems_bin = shutil.which("cems")
    if not cems_bin:
        console.print("[yellow]cems not found on PATH — skipping hook re-deploy[/yellow]")
        console.print("Run manually: [cyan]cems setup[/cyan]")
        return

    # Detect which IDEs are currently configured
    from pathlib import Path

    has_claude = (Path.home() / ".claude" / "hooks" / "cems_session_start.py").exists()
    has_cursor = (Path.home() / ".cursor" / "hooks" / "cems_session_start.py").exists()

    if not has_claude and not has_cursor:
        console.print("[yellow]No existing hooks found — run [cyan]cems setup[/cyan] to configure[/yellow]")
        return

    args = [sys.executable, "-m", "cems.commands._redeploy"]
    if has_claude:
        args.append("--claude")
    if has_cursor:
        args.append("--cursor")

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

    if not hooks_only:
        console.print(f"Current version: [cyan]{old_version}[/cyan]")
        console.print()

        if not _upgrade_package():
            raise click.Abort()

        new_version = _get_current_version()
        if new_version != old_version:
            console.print(f"[green]Updated: {old_version} → {new_version}[/green]")
        else:
            console.print(f"[green]Already on latest ({new_version})[/green]")
        console.print()

    # Re-deploy hooks/skills
    console.print("Re-deploying hooks and skills...")
    _redeploy_hooks()

    console.print()
    console.print("[bold green]Update complete![/bold green] Restart your IDE to pick up changes.")
