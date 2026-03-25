"""Env command for CEMS CLI.

Reads credentials (per-project or global) and outputs shell export statements.
Designed for shell profile integration:

    eval "$(cems env)"

Supports per-project .cems/credentials — walks up from CWD.
"""

import os
from pathlib import Path

import click


def _find_project_credentials() -> Path | None:
    """Walk up from CWD looking for .cems/credentials. Stops before $HOME."""
    home = str(Path.home())
    try:
        path = Path(os.getcwd()).resolve()
    except (OSError, ValueError):
        return None
    while str(path) != home and path != path.parent:
        project_creds = path / ".cems" / "credentials"
        if project_creds.is_file():
            return project_creds
        path = path.parent
    return None


@click.command()
def env() -> None:
    """Output shell exports for CEMS credentials.

    Resolves credentials from per-project .cems/credentials (walk-up from CWD)
    or global ~/.cems/credentials. Add to your shell profile:

    \b
        eval "$(cems env)"
    """
    # Try per-project first
    project_creds = _find_project_credentials()
    if project_creds:
        creds_file = project_creds
        click.echo(f"# Credentials from: {creds_file} (project)")
    else:
        creds_file = Path.home() / ".cems" / "credentials"
        if not creds_file.exists():
            raise click.ClickException(
                f"{creds_file} not found. Run: cems setup"
            )
        click.echo(f"# Credentials from: {creds_file} (global)")

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
