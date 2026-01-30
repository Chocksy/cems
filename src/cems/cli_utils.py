"""Shared utilities for CEMS CLI commands."""

import logging
import sys

import click
from rich.console import Console

from cems.client import (
    CEMSAdminClient,
    CEMSAuthError,
    CEMSClient,
    CEMSClientError,
    CEMSConnectionError,
)

console = Console()


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def handle_error(e: Exception) -> None:
    """Handle and display errors nicely."""
    if isinstance(e, CEMSConnectionError):
        console.print(f"[red]Connection Error:[/red] {e}")
        console.print("[dim]Is the CEMS server running? Check CEMS_API_URL.[/dim]")
    elif isinstance(e, CEMSAuthError):
        console.print(f"[red]Authentication Error:[/red] {e}")
        console.print("[dim]Check your CEMS_API_KEY or admin key.[/dim]")
    elif isinstance(e, CEMSClientError):
        console.print(f"[red]Error:[/red] {e}")
    else:
        console.print(f"[red]Unexpected Error:[/red] {e}")
    sys.exit(1)


def get_client(ctx: click.Context) -> CEMSClient:
    """Get a CEMS client from context."""
    return CEMSClient(
        api_url=ctx.obj.get("api_url"),
        api_key=ctx.obj.get("api_key"),
    )


def get_admin_client(ctx: click.Context, admin_key: str | None = None) -> CEMSAdminClient:
    """Get an admin client from context."""
    return CEMSAdminClient(
        api_url=ctx.obj.get("api_url"),
        admin_key=admin_key,  # Falls back to env vars in client
    )
