"""Status and health commands for CEMS CLI."""

import click
from rich.table import Table

from cems.cli_utils import console, get_client, handle_error
from cems.client import CEMSClientError


@click.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show CEMS status and configuration."""
    try:
        client = get_client(ctx)
        data = client.status()

        table = Table(title="CEMS Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Server", client.api_url)
        table.add_row("User ID", data.get("user_id", "?"))
        table.add_row("Team ID", data.get("team_id") or "(not set)")
        table.add_row("Status", data.get("status", "unknown"))
        table.add_row("Backend", data.get("backend", "?"))
        table.add_row("Vector Store", data.get("vector_store", "?"))
        table.add_row("Graph Store", data.get("graph_store") or "disabled")
        table.add_row("Query Synthesis", str(data.get("query_synthesis", False)))

        console.print(table)

        # Show graph stats if available
        if data.get("graph_stats"):
            console.print("\n[bold]Graph Statistics:[/bold]")
            for key, value in data["graph_stats"].items():
                console.print(f"  {key}: {value}")

    except CEMSClientError as e:
        handle_error(e)


@click.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check server health."""
    try:
        client = get_client(ctx)

        with console.status("Checking health..."):
            result = client.health()

        status_text = result.get("status", "unknown")
        if status_text == "healthy":
            console.print(f"[green]Server: {status_text}[/green]")
        else:
            console.print(f"[yellow]Server: {status_text}[/yellow]")

        console.print(f"Service: {result.get('service', '?')}")
        console.print(f"Mode: {result.get('mode', '?')}")
        console.print(f"Auth: {result.get('auth', '?')}")
        console.print(f"Database: {result.get('database', '?')}")

    except CEMSClientError as e:
        handle_error(e)
