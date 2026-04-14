"""CLI interface for CEMS.

This CLI communicates with a CEMS server via HTTP API.
Reads credentials from env vars, CLI flags, or ~/.cems/credentials.
"""

import os
from importlib.metadata import version

import click

from cems.cli_utils import setup_logging
from cems.commands.admin import admin
from cems.commands.debug import debug
from cems.commands.env import env
from cems.commands.index import index
from cems.commands.maintenance import maintenance
from cems.commands.memory import add, delete, list_memories, search
from cems.commands.memory import update as update_memory
from cems.commands.rule import rule
from cems.commands.setup import setup
from cems.commands.status import status
from cems.commands.uninstall import uninstall
from cems.commands.update import update_cmd
from cems.shared.credentials import resolve_credentials


@click.group()
@click.version_option(version=version("cems"), prog_name="cems")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--api-url", help="CEMS server URL")
@click.option("--api-key", help="API key for authentication")
@click.pass_context
def main(ctx: click.Context, verbose: bool, api_url: str | None, api_key: str | None) -> None:
    """CEMS - Continuous Evolving Memory System.

    A memory system for AI assistants. Requires a CEMS server.

    Configuration (checked in order):
      1. CLI flags: --api-url, --api-key
      2. Per-project: .cems/credentials (walk up from CWD)
      3. Environment: CEMS_API_URL, CEMS_API_KEY
      4. Global: ~/.cems/credentials
    """
    # resolve_credentials handles the full chain:
    # project .cems/credentials (walk-up) → env vars → global ~/.cems/credentials
    # CLI flags (api_url/api_key) override everything when explicitly passed.
    creds = resolve_credentials(cwd=os.getcwd())
    if not api_url:
        api_url = creds.get("CEMS_API_URL")
    if not api_key:
        api_key = creds.get("CEMS_API_KEY")
    team_id = creds.get("CEMS_TEAM_ID")

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["api_url"] = api_url
    ctx.obj["api_key"] = api_key
    ctx.obj["team_id"] = team_id
    setup_logging(verbose)


# Register all commands
main.add_command(status)
main.add_command(add)
main.add_command(search)
main.add_command(list_memories, name="list")
main.add_command(delete)
main.add_command(update_memory, name="edit")
main.add_command(update_cmd, name="update")
main.add_command(index)
main.add_command(maintenance)
main.add_command(admin)
main.add_command(rule)
main.add_command(env)
main.add_command(setup)
main.add_command(uninstall)
main.add_command(debug)


if __name__ == "__main__":
    main()
