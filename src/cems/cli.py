"""CLI interface for CEMS.

This CLI communicates with a CEMS server via HTTP API.
All operations require CEMS_API_URL and CEMS_API_KEY to be set.
"""

import click

from cems.cli_utils import setup_logging
from cems.commands.status import health, status
from cems.commands.memory import add, delete, list_memories, search, update
from cems.commands.index import index
from cems.commands.maintenance import maintenance
from cems.commands.admin import admin
from cems.commands.setup import setup


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--api-url", envvar="CEMS_API_URL", help="CEMS server URL")
@click.option("--api-key", envvar="CEMS_API_KEY", help="API key for authentication")
@click.pass_context
def main(ctx: click.Context, verbose: bool, api_url: str | None, api_key: str | None) -> None:
    """CEMS - Continuous Evolving Memory System.

    A memory system for AI assistants. Requires a CEMS server.

    Configuration:
      CEMS_API_URL  - Server URL (e.g., https://cems.example.com)
      CEMS_API_KEY  - Your API key
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["api_url"] = api_url
    ctx.obj["api_key"] = api_key
    setup_logging(verbose)


# Register all commands
main.add_command(status)
main.add_command(health)
main.add_command(add)
main.add_command(search)
main.add_command(list_memories, name="list")
main.add_command(delete)
main.add_command(update)
main.add_command(index)
main.add_command(maintenance)
main.add_command(admin)
main.add_command(setup)


if __name__ == "__main__":
    main()
