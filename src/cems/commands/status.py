"""Status command for CEMS CLI.

Shows a unified view of client, server, and observer daemon state.
"""

import os
from importlib.metadata import version as pkg_version
from pathlib import Path

import click
from rich.table import Table

from cems.cli_utils import console, get_client, handle_error
from cems.client import CEMSClientError

DAEMON_PID_FILE = Path.home() / ".cems" / "observer" / "daemon.pid"


def _get_client_version() -> str:
    """Get installed CEMS package version."""
    try:
        return pkg_version("cems")
    except Exception:
        return "unknown"


def _get_daemon_info() -> dict:
    """Get observer daemon status."""
    info: dict = {"status": "stopped", "pid": None}
    try:
        if DAEMON_PID_FILE.exists():
            pid = int(DAEMON_PID_FILE.read_text().strip())
            # Check if process is alive
            os.kill(pid, 0)
            info["status"] = "running"
            info["pid"] = pid
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pass
    return info


@click.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show CEMS status — client, server, and daemon."""
    client_version = _get_client_version()

    # --- Client ---
    api_url = ctx.obj.get("api_url") or "(not configured)"
    console.print()
    console.print("[bold]CEMS Status[/bold]")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("", "")
    table.add_row("[cyan bold]Client[/cyan bold]", "")
    table.add_row("  Version", client_version)
    table.add_row("  API URL", str(api_url))

    # --- Server ---
    table.add_row("", "")
    table.add_row("[cyan bold]Server[/cyan bold]", "")

    try:
        client = get_client(ctx)
        health = client.health()
        server_status = health.get("status", "unknown")
        server_version = health.get("version", "unknown")
        db_status = health.get("database", "unknown")

        color = "green" if server_status == "healthy" else "red"
        table.add_row("  Status", f"[{color}]{server_status}[/{color}]")
        table.add_row("  Version", server_version)

        db_color = "green" if db_status == "healthy" else "yellow"
        table.add_row("  Database", f"[{db_color}]{db_status}[/{db_color}]")

    except CEMSClientError:
        table.add_row("  Status", "[red]unreachable[/red]")

    # --- Observer Daemon ---
    table.add_row("", "")
    table.add_row("[cyan bold]Observer Daemon[/cyan bold]", "")

    daemon = _get_daemon_info()
    if daemon["status"] == "running":
        table.add_row("  Status", f"[green]running[/green] (PID {daemon['pid']})")
    else:
        table.add_row("  Status", "[yellow]stopped[/yellow]")

    console.print(table)
    console.print()
