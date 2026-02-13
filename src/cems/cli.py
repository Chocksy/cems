"""CLI interface for CEMS.

This CLI communicates with a CEMS server via HTTP API.
Reads credentials from env vars, CLI flags, or ~/.cems/credentials.
"""

from importlib.metadata import version
from pathlib import Path

import click

from cems.cli_utils import setup_logging
from cems.commands.status import health, status
from cems.commands.memory import add, delete, list_memories, search, update
from cems.commands.index import index
from cems.commands.maintenance import maintenance
from cems.commands.admin import admin
from cems.commands.setup import setup
from cems.commands.uninstall import uninstall


def _read_credentials_file() -> dict[str, str]:
    """Read ~/.cems/credentials as fallback for env vars."""
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


@click.group()
@click.version_option(version=version("cems"), prog_name="cems")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--api-url", envvar="CEMS_API_URL", help="CEMS server URL")
@click.option("--api-key", envvar="CEMS_API_KEY", help="API key for authentication")
@click.pass_context
def main(ctx: click.Context, verbose: bool, api_url: str | None, api_key: str | None) -> None:
    """CEMS - Continuous Evolving Memory System.

    A memory system for AI assistants. Requires a CEMS server.

    Configuration (checked in order):
      1. CLI flags: --api-url, --api-key
      2. Environment: CEMS_API_URL, CEMS_API_KEY
      3. Credentials file: ~/.cems/credentials
    """
    # Fall back to ~/.cems/credentials if no env vars or flags
    if not api_url or not api_key:
        creds = _read_credentials_file()
        if not api_url:
            api_url = creds.get("CEMS_API_URL")
        if not api_key:
            api_key = creds.get("CEMS_API_KEY")

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
main.add_command(uninstall)


if __name__ == "__main__":
    main()
