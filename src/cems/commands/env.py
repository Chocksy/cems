"""Env command for CEMS CLI.

Reads ~/.cems/credentials and outputs shell export statements.
Designed for shell profile integration:

    eval "$(cems env)"

This keeps ~/.cems/credentials as the single source of truth.
"""

from pathlib import Path

import click


@click.command()
def env() -> None:
    """Output shell exports for CEMS credentials.

    Reads ~/.cems/credentials and prints export statements.
    Add to your shell profile:

    \b
        eval "$(cems env)"
    """
    creds_file = Path.home() / ".cems" / "credentials"

    if not creds_file.exists():
        raise click.ClickException(
            f"{creds_file} not found. Run: cems setup"
        )

    for line in creds_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value:
                click.echo(f"export {key}={value}")
