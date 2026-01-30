"""Maintenance commands for CEMS CLI."""

import json

import click

from cems.cli_utils import console, get_client, handle_error
from cems.client import CEMSClientError


@click.group()
def maintenance() -> None:
    """Maintenance commands for memory system."""
    pass


@maintenance.command("run")
@click.argument("job_type", type=click.Choice(["consolidation", "summarization", "reindex", "all"]))
@click.pass_context
def run_maintenance(ctx: click.Context, job_type: str) -> None:
    """Run a maintenance job immediately.

    JOB_TYPE: consolidation, summarization, reindex, or all
    """
    try:
        client = get_client(ctx)

        console.print(f"[cyan]Running {job_type}...[/cyan]")
        with console.status("Running maintenance..."):
            result = client.maintenance(job_type)

        if result.get("success"):
            console.print(f"[green]{job_type} completed[/green]")
            if ctx.obj["verbose"]:
                console.print(json.dumps(result, indent=2, default=str))
        else:
            console.print(f"[yellow]{job_type} may have failed[/yellow]")
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)
