"""Env command for CEMS CLI.

Reads credentials (per-project or global) and outputs shell export statements.
Designed for shell profile integration:

    eval "$(cems env)"

Supports per-project .cems/credentials — walks up from CWD.
"""

import os
from pathlib import Path

import click

from cems.shared.credentials import find_project_credentials, parse_credentials_file


@click.command()
def env() -> None:
    """Output shell exports for CEMS credentials.

    Resolves credentials from per-project .cems/credentials (walk-up from CWD)
    or global ~/.cems/credentials. Add to your shell profile:

    \b
        eval "$(cems env)"
    """
    # Try per-project first
    project_path = find_project_credentials(os.getcwd())
    if project_path:
        creds_file = Path(project_path)
        click.echo(f"# Credentials from: {creds_file} (project)")
    else:
        creds_file = Path.home() / ".cems" / "credentials"
        if not creds_file.exists():
            raise click.ClickException(
                f"{creds_file} not found. Run: cems setup"
            )
        click.echo(f"# Credentials from: {creds_file} (global)")

    creds = parse_credentials_file(str(creds_file))
    for key, value in creds.items():
        click.echo(f"export {key}={value}")
