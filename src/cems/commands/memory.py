"""Memory commands for CEMS CLI."""

import json

import click
from rich.table import Table

from cems.cli_utils import console, get_client, handle_error
from cems.client import CEMSClientError


@click.command()
@click.argument("content")
@click.option("--scope", "-s", default="personal", type=click.Choice(["personal", "shared"]))
@click.option("--category", "-c", default="general", help="Memory category")
@click.option("--tags", "-t", multiple=True, help="Tags for the memory")
@click.option("--source-ref", help="Project reference (e.g., project:org/repo)")
@click.option("--pin", is_flag=True, help="Pin memory (never auto-pruned)")
@click.option("--pin-reason", help="Reason for pinning")
@click.pass_context
def add(
    ctx: click.Context,
    content: str,
    scope: str,
    category: str,
    tags: tuple,
    source_ref: str | None,
    pin: bool,
    pin_reason: str | None,
) -> None:
    """Add a memory.

    Examples:
        cems add "Always use TypeScript for new projects" -c preferences

        # Add a gate rule (for PreToolUse hook blocking)
        cems add "Bash: coolify deploy — Never use CLI for production" \\
            -c gate-rules --pin --source-ref "project:EpicCoders/pxls"
    """
    try:
        client = get_client(ctx)

        with console.status("Adding memory..."):
            result = client.add(
                content,
                category=category,
                scope=scope,
                tags=list(tags),
                source_ref=source_ref,
                pinned=pin,
                pin_reason=pin_reason,
            )

        if result.get("success"):
            console.print("[green]Memory added successfully[/green]")
            if pin:
                console.print("[dim]Memory is pinned (will not be auto-pruned)[/dim]")
            if ctx.obj["verbose"]:
                console.print(json.dumps(result, indent=2, default=str))
        else:
            console.print("[yellow]Memory may not have been added[/yellow]")
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)


@click.command()
@click.argument("query")
@click.option("--scope", "-s", default="both", type=click.Choice(["personal", "shared", "both"]))
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--max-tokens", "-t", default=4000, help="Token budget for results")
@click.option("--no-graph", is_flag=True, help="Disable graph traversal")
@click.option("--no-synthesis", is_flag=True, help="Disable LLM query expansion")
@click.option("--raw", is_flag=True, help="Debug mode: bypass filtering to see all results")
@click.option("--verbose", "-v", is_flag=True, help="Show full content without truncation")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    scope: str,
    limit: int,
    max_tokens: int,
    no_graph: bool,
    no_synthesis: bool,
    raw: bool,
    verbose: bool,
) -> None:
    """Search memories using unified retrieval pipeline.

    Uses 5-stage retrieval: query synthesis, vector+graph search,
    relevance filtering, temporal ranking, and token budgeting.

    Example:
        cems search "TypeScript preferences"
        cems search "coding style" --raw  # Debug mode
    """
    try:
        client = get_client(ctx)

        with console.status("Searching..."):
            result = client.search(
                query,
                scope=scope,
                limit=limit,
                max_tokens=max_tokens,
                enable_graph=not no_graph,
                enable_query_synthesis=not no_synthesis,
                raw=raw,
            )

        results = result.get("results", [])
        mode = result.get("mode", "unified")

        if results:
            title = f"Search Results for: {query}"
            if mode == "raw":
                title += " [RAW MODE]"
            table = Table(title=title)
            table.add_column("ID", style="dim", max_width=12)
            table.add_column("Content", style="white")
            table.add_column("Score", style="cyan")
            table.add_column("Scope", style="yellow")

            for r in results:
                content = r.get("content", r.get("memory", ""))
                # Show full content if verbose, otherwise truncate to 120 chars (up from 80)
                if verbose:
                    display_content = content
                else:
                    display_content = content[:120] + "..." if len(content) > 120 else content
                table.add_row(
                    (r.get("memory_id") or r.get("id", "?"))[:12],
                    display_content,
                    f"{r.get('score', 0):.3f}",
                    r.get("scope", "?"),
                )

            console.print(table)

            # Show pipeline stats in unified mode
            if mode == "unified":
                console.print(
                    f"[dim]Pipeline: {result.get('total_candidates', '?')} candidates → "
                    f"{result.get('filtered_count', '?')} after filtering → "
                    f"{len(results)} returned | "
                    f"Tokens: {result.get('tokens_used', '?')} | "
                    f"Queries: {len(result.get('queries_used', []))}[/dim]"
                )
        else:
            console.print("[yellow]No relevant results found[/yellow]")
            if mode == "unified" and not raw:
                console.print("[dim]Tip: Use --raw to see unfiltered results[/dim]")

    except CEMSClientError as e:
        handle_error(e)


@click.command("list")
@click.option("--scope", "-s", default="personal", type=click.Choice(["personal", "shared"]))
@click.pass_context
def list_memories(ctx: click.Context, scope: str) -> None:
    """List memories (via summary).

    Shows a summary of memories by category.
    """
    try:
        client = get_client(ctx)

        with console.status("Loading summary..."):
            result = client.summary(scope=scope)

        if result.get("success"):
            table = Table(title=f"Memory Summary ({scope})")
            table.add_column("Category", style="cyan")
            table.add_column("Count", style="green")

            categories = result.get("categories", {})
            for cat, count in sorted(categories.items()):
                table.add_row(cat, str(count))

            console.print(table)
            console.print(f"\n[bold]Total:[/bold] {result.get('total', 0)} memories")
        else:
            console.print("[yellow]Could not load summary[/yellow]")

    except CEMSClientError as e:
        handle_error(e)


@click.command()
@click.argument("memory_id")
@click.option("--hard", is_flag=True, help="Permanently delete instead of archive")
@click.pass_context
def delete(ctx: click.Context, memory_id: str, hard: bool) -> None:
    """Delete or archive a memory.

    Example:
        cems delete abc123 --hard
    """
    try:
        client = get_client(ctx)

        action = "Deleting" if hard else "Archiving"
        with console.status(f"{action} memory..."):
            result = client.delete(memory_id, hard=hard)

        console.print(f"[green]Memory {'deleted' if hard else 'archived'} successfully[/green]")
        if ctx.obj["verbose"]:
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)


@click.command()
@click.argument("memory_id")
@click.argument("content")
@click.pass_context
def update(ctx: click.Context, memory_id: str, content: str) -> None:
    """Update a memory's content.

    Example:
        cems edit abc123 "Updated content here"
    """
    try:
        client = get_client(ctx)

        with console.status("Updating memory..."):
            result = client.update(memory_id, content)

        console.print("[green]Memory updated successfully[/green]")
        if ctx.obj["verbose"]:
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)
