"""Index commands for CEMS CLI."""

import json

import click
from rich.table import Table

from cems.cli_utils import console, get_client, handle_error
from cems.client import CEMSClientError


@click.group()
def index() -> None:
    """Index repositories to extract knowledge into CEMS."""
    pass


@index.command("repo")
@click.argument("repo_url")
@click.option("--branch", "-b", default="main", help="Branch to index")
@click.option("--scope", "-s", default="shared", type=click.Choice(["personal", "shared"]))
@click.option("--patterns", "-p", multiple=True, help="Specific patterns to use (default: all)")
@click.pass_context
def index_repo(
    ctx: click.Context,
    repo_url: str,
    branch: str,
    scope: str,
    patterns: tuple,
) -> None:
    """Index a git repository.

    Clones the repo, extracts knowledge (docs, conventions, schemas, etc.),
    and stores as pinned memories.

    Examples:
        cems index repo https://github.com/org/repo
        cems index repo https://github.com/org/repo -b develop
        cems index repo https://github.com/org/repo -p readme_docs -p rspec_conventions
    """
    try:
        client = get_client(ctx)

        console.print(f"[cyan]Indexing {repo_url} ({branch})...[/cyan]")
        with console.status("Cloning and extracting knowledge..."):
            result = client.index_repo(
                repo_url,
                branch=branch,
                scope=scope,
                patterns=list(patterns) if patterns else None,
            )

        if result.get("success"):
            r = result.get("result", {})
            console.print(f"[green]Indexing complete![/green]")
            console.print(f"  Files scanned: {r.get('files_scanned', 0)}")
            console.print(f"  Knowledge extracted: {r.get('knowledge_extracted', 0)}")
            console.print(f"  Memories created: {r.get('memories_created', 0)}")
            console.print(f"  Patterns used: {', '.join(r.get('patterns_used', []))}")

            errors = r.get("errors", [])
            if errors:
                console.print(f"\n[yellow]Warnings ({len(errors)}):[/yellow]")
                for err in errors[:5]:
                    console.print(f"  [dim]{err}[/dim]")

            if ctx.obj["verbose"]:
                console.print(json.dumps(result, indent=2, default=str))
        else:
            console.print("[yellow]Indexing may have failed[/yellow]")
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)


@index.command("path")
@click.argument("path")
@click.option("--scope", "-s", default="shared", type=click.Choice(["personal", "shared"]))
@click.option("--patterns", "-p", multiple=True, help="Specific patterns to use (default: all)")
@click.pass_context
def index_path(
    ctx: click.Context,
    path: str,
    scope: str,
    patterns: tuple,
) -> None:
    """Index a local directory path (server-side).

    The path must be accessible from the CEMS server.

    Examples:
        cems index path /home/user/projects/myapp
        cems index path . -p python_conventions
    """
    try:
        client = get_client(ctx)

        console.print(f"[cyan]Indexing {path}...[/cyan]")
        with console.status("Extracting knowledge..."):
            result = client.index_path(
                path,
                scope=scope,
                patterns=list(patterns) if patterns else None,
            )

        if result.get("success"):
            r = result.get("result", {})
            console.print(f"[green]Indexing complete![/green]")
            console.print(f"  Files scanned: {r.get('files_scanned', 0)}")
            console.print(f"  Knowledge extracted: {r.get('knowledge_extracted', 0)}")
            console.print(f"  Memories created: {r.get('memories_created', 0)}")
            console.print(f"  Patterns used: {', '.join(r.get('patterns_used', []))}")

            errors = r.get("errors", [])
            if errors:
                console.print(f"\n[yellow]Warnings ({len(errors)}):[/yellow]")
                for err in errors[:5]:
                    console.print(f"  [dim]{err}[/dim]")
        else:
            console.print("[yellow]Indexing may have failed[/yellow]")
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)


@index.command("patterns")
@click.pass_context
def list_patterns(ctx: click.Context) -> None:
    """List available index patterns.

    Shows all patterns that can be used to extract knowledge from repositories.

    Example:
        cems index patterns
    """
    try:
        client = get_client(ctx)

        with console.status("Loading patterns..."):
            result = client.list_index_patterns()

        patterns = result.get("patterns", [])
        if patterns:
            table = Table(title="Available Index Patterns")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("File Patterns", style="dim")
            table.add_column("Category", style="yellow")

            for p in patterns:
                file_patterns = ", ".join(p.get("file_patterns", [])[:3])
                if len(p.get("file_patterns", [])) > 3:
                    file_patterns += ", ..."
                table.add_row(
                    p["name"],
                    p.get("description", ""),
                    file_patterns,
                    p.get("pin_category", ""),
                )

            console.print(table)
        else:
            console.print("[yellow]No patterns available[/yellow]")

    except CEMSClientError as e:
        handle_error(e)
