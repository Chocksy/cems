"""CLI interface for CEMS."""

import json
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from cems.config import CEMSConfig
from cems.memory import CEMSMemory
from cems.scheduler import CEMSScheduler

console = Console()


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """CEMS - Continuous Evolving Memory System.

    A dual-layer memory system with scheduled maintenance.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def status(user: str, team: str | None) -> None:
    """Show CEMS status and configuration."""
    config = CEMSConfig(user_id=user, team_id=team)

    table = Table(title="CEMS Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("User ID", config.user_id)
    table.add_row("Team ID", config.team_id or "(not set)")
    table.add_row("Storage Directory", str(config.storage_dir))
    table.add_row("Memory Backend", config.memory_backend)
    table.add_row("Vector Store", config.vector_store)
    table.add_row("Qdrant URL", config.qdrant_url or "(embedded)")
    table.add_row("Mem0 Model", config.get_mem0_model())
    table.add_row("Embedding Model", config.embedding_model)
    table.add_row("Maintenance Model", config.llm_model)
    table.add_row("Graph Store", config.graph_store if config.enable_graph else "disabled")
    table.add_row("Scheduler Enabled", str(config.enable_scheduler))

    console.print(table)


@main.command()
@click.argument("content")
@click.option("--scope", "-s", default="personal", type=click.Choice(["personal", "shared"]))
@click.option("--category", "-c", default="general", help="Memory category")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def add(content: str, scope: str, category: str, user: str, team: str | None) -> None:
    """Add a memory."""
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)

    with console.status("Adding memory..."):
        result = memory.add(content, scope=scope, category=category)

    if result:
        console.print("[green]Memory added successfully[/green]")
        console.print(json.dumps(result, indent=2, default=str))
    else:
        console.print("[red]Failed to add memory[/red]")


@main.command()
@click.argument("query")
@click.option("--scope", "-s", default="both", type=click.Choice(["personal", "shared", "both"]))
@click.option("--limit", "-l", default=5, help="Maximum results")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def search(query: str, scope: str, limit: int, user: str, team: str | None) -> None:
    """Search memories."""
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)

    with console.status("Searching..."):
        results = memory.search(query, scope=scope, limit=limit)

    if results:
        table = Table(title=f"Search Results for: {query}")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Content", style="white")
        table.add_column("Score", style="cyan")
        table.add_column("Scope", style="yellow")

        for r in results:
            table.add_row(
                r.memory_id[:12] if r.memory_id else "?",
                r.content[:80] + "..." if len(r.content) > 80 else r.content,
                f"{r.score:.3f}",
                r.scope.value,
            )

        console.print(table)
    else:
        console.print("[yellow]No results found[/yellow]")


@main.command("list")
@click.option("--scope", "-s", default="both", type=click.Choice(["personal", "shared", "both"]))
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def list_memories(scope: str, user: str, team: str | None) -> None:
    """List all memories."""
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)

    with console.status("Loading memories..."):
        memories = memory.get_all(scope=scope)

    if memories:
        table = Table(title=f"All Memories ({scope})")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Content", style="white")
        table.add_column("Scope", style="yellow")

        for m in memories:
            content = m.get("memory", "")
            table.add_row(
                m.get("id", "?")[:12],
                content[:80] + "..." if len(content) > 80 else content,
                m.get("scope", "?"),
            )

        console.print(table)
        console.print(f"\nTotal: {len(memories)} memories")
    else:
        console.print("[yellow]No memories found[/yellow]")


@main.command()
@click.argument("memory_id")
@click.option("--hard", is_flag=True, help="Permanently delete instead of archive")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def delete(memory_id: str, hard: bool, user: str, team: str | None) -> None:
    """Delete or archive a memory."""
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)

    action = "Deleting" if hard else "Archiving"
    with console.status(f"{action} memory..."):
        result = memory.delete(memory_id, hard=hard)

    console.print(f"[green]Memory {'deleted' if hard else 'archived'} successfully[/green]")
    console.print(json.dumps(result, indent=2, default=str))


# Maintenance commands
@main.group()
def maintenance() -> None:
    """Maintenance commands for memory system."""
    pass


@maintenance.command("run")
@click.argument("job_type", type=click.Choice(["consolidation", "summarization", "reindex", "all"]))
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def run_maintenance(job_type: str, user: str, team: str | None) -> None:
    """Run a maintenance job immediately.

    JOB_TYPE: consolidation, summarization, reindex, or all
    """
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)
    scheduler = CEMSScheduler(memory)

    jobs = (
        ["consolidation", "summarization", "reindex"]
        if job_type == "all"
        else [job_type]
    )

    for job in jobs:
        console.print(f"\n[cyan]Running {job}...[/cyan]")
        try:
            result = scheduler.run_now(job)
            console.print(f"[green]{job} completed:[/green]")
            console.print(json.dumps(result, indent=2))
        except Exception as e:
            console.print(f"[red]{job} failed: {e}[/red]")


@maintenance.command("schedule")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def show_schedule(user: str, team: str | None) -> None:
    """Show the maintenance schedule."""
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)
    scheduler = CEMSScheduler(memory)

    # Start scheduler to get job info
    scheduler.start()
    jobs = scheduler.get_jobs()
    scheduler.stop()

    table = Table(title="Maintenance Schedule")
    table.add_column("Job", style="cyan")
    table.add_column("Next Run", style="green")

    for job in jobs:
        table.add_row(job["name"], job["next_run"] or "Not scheduled")

    console.print(table)

    # Also show config
    console.print("\n[bold]Schedule Configuration:[/bold]")
    console.print(f"  Nightly consolidation: {config.nightly_hour}:00")
    console.print(f"  Weekly summarization: {config.weekly_day} at {config.weekly_hour}:00")
    console.print(f"  Monthly re-index: Day {config.monthly_day} at {config.monthly_hour}:00")


@maintenance.command("history")
@click.option("--limit", "-l", default=10, help="Number of entries to show")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def maintenance_history(limit: int, user: str, team: str | None) -> None:
    """Show maintenance job history."""
    import sqlite3

    config = CEMSConfig(user_id=user, team_id=team)

    conn = sqlite3.connect(config.metadata_db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT * FROM maintenance_log
            WHERE user_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (user, limit),
        ).fetchall()

        if rows:
            table = Table(title="Maintenance History")
            table.add_column("Job", style="cyan")
            table.add_column("Started", style="white")
            table.add_column("Status", style="green")
            table.add_column("Details", style="dim", max_width=40)

            for row in rows:
                status_color = "green" if row["status"] == "completed" else "red"
                table.add_row(
                    row["job_type"],
                    row["started_at"],
                    f"[{status_color}]{row['status']}[/{status_color}]",
                    (row["details"] or "")[:40],
                )

            console.print(table)
        else:
            console.print("[yellow]No maintenance history found[/yellow]")
    finally:
        conn.close()


@main.command("generate-config")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def generate_config(user: str, team: str | None, output: str | None) -> None:
    """Generate MCP configuration for Claude Code."""
    from pathlib import Path

    mcp_config = {
        "mcpServers": {
            "cems": {
                "command": "cems-server",
                "args": [],
                "env": {
                    "CEMS_USER_ID": user,
                },
            }
        }
    }

    if team:
        mcp_config["mcpServers"]["cems"]["env"]["CEMS_TEAM_ID"] = team

    config_json = json.dumps(mcp_config, indent=2)

    if output:
        Path(output).write_text(config_json)
        console.print(f"[green]Configuration written to {output}[/green]")
    else:
        console.print("\n[bold]Add this to ~/.claude/mcp_config.json:[/bold]\n")
        console.print(config_json)


# Indexer commands
@main.group()
def index() -> None:
    """Repository indexing commands."""
    pass


@index.command("repo")
@click.argument("path", type=click.Path(exists=True))
@click.option("--scope", "-s", default="shared", type=click.Choice(["personal", "shared"]))
@click.option("--patterns", "-p", multiple=True, help="Specific patterns to use")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def index_repo(path: str, scope: str, patterns: tuple, user: str, team: str | None) -> None:
    """Index a local repository.

    Scans the repository for patterns like RSpec conventions, architecture
    decisions, documentation, etc. Extracted knowledge is stored as
    PINNED memories that won't be auto-pruned.

    PATH: Path to the repository to index
    """
    from cems.indexer import RepositoryIndexer

    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)
    indexer = RepositoryIndexer(memory)

    pattern_list = list(patterns) if patterns else None

    with console.status(f"Indexing {path}..."):
        results = indexer.index_local_path(path, scope=scope, patterns=pattern_list)

    console.print(f"\n[bold green]Indexing complete![/bold green]\n")

    table = Table(title="Indexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", results["repo_path"])
    table.add_row("Files scanned", str(results["files_scanned"]))
    table.add_row("Knowledge extracted", str(results["knowledge_extracted"]))
    table.add_row("Memories created", str(results["memories_created"]))
    table.add_row("Patterns used", ", ".join(results["patterns_used"]))

    console.print(table)

    if results["errors"]:
        console.print(f"\n[yellow]Warnings: {len(results['errors'])} errors[/yellow]")


@index.command("git")
@click.argument("repo_url")
@click.option("--branch", "-b", default="main", help="Branch to clone")
@click.option("--scope", "-s", default="shared", type=click.Choice(["personal", "shared"]))
@click.option("--patterns", "-p", multiple=True, help="Specific patterns to use")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def index_git(
    repo_url: str, branch: str, scope: str, patterns: tuple, user: str, team: str | None
) -> None:
    """Index a git repository.

    Clones and indexes a git repository. Useful for indexing external
    repositories or remote team repos.

    REPO_URL: Git repository URL (https or ssh)
    """
    from cems.indexer import RepositoryIndexer

    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)
    indexer = RepositoryIndexer(memory)

    pattern_list = list(patterns) if patterns else None

    with console.status(f"Cloning and indexing {repo_url}..."):
        results = indexer.index_git_repo(repo_url, branch=branch, scope=scope, patterns=pattern_list)

    console.print(f"\n[bold green]Indexing complete![/bold green]\n")

    table = Table(title="Indexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", results.get("repo_url", ""))
    table.add_row("Branch", results.get("branch", ""))
    table.add_row("Files scanned", str(results["files_scanned"]))
    table.add_row("Knowledge extracted", str(results["knowledge_extracted"]))
    table.add_row("Memories created", str(results["memories_created"]))
    table.add_row("Patterns used", ", ".join(results["patterns_used"]))

    console.print(table)


@index.command("patterns")
def list_patterns() -> None:
    """List available index patterns."""
    from cems.indexer import get_default_patterns

    patterns = get_default_patterns()

    table = Table(title="Available Index Patterns")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Pin Category", style="yellow")
    table.add_column("File Patterns", style="dim", max_width=40)

    for p in patterns:
        table.add_row(
            p.name,
            p.description,
            p.pin_category,
            ", ".join(p.file_patterns[:2]) + ("..." if len(p.file_patterns) > 2 else ""),
        )

    console.print(table)


# Pin/unpin commands
@main.command("pin")
@click.argument("memory_id")
@click.option("--reason", "-r", required=True, help="Reason for pinning")
@click.option(
    "--category",
    "-c",
    default="guideline",
    type=click.Choice(["guideline", "convention", "architecture", "standard", "documentation"]),
)
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def pin_memory(memory_id: str, reason: str, category: str, user: str, team: str | None) -> None:
    """Pin a memory to prevent automatic decay.

    Pinned memories are never auto-pruned by maintenance jobs.
    Use this for important guidelines, conventions, or architecture decisions.
    """
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)

    memory.metadata_store.pin_memory(memory_id, reason=reason, pin_category=category)
    console.print(f"[green]Memory {memory_id} pinned as {category}[/green]")
    console.print(f"Reason: {reason}")


@main.command("unpin")
@click.argument("memory_id")
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def unpin_memory(memory_id: str, user: str, team: str | None) -> None:
    """Unpin a memory, allowing normal decay."""
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)

    memory.metadata_store.unpin_memory(memory_id)
    console.print(f"[green]Memory {memory_id} unpinned[/green]")


@main.command("pinned")
@click.option(
    "--category",
    "-c",
    default=None,
    type=click.Choice(["guideline", "convention", "architecture", "standard", "documentation"]),
)
@click.option("--user", "-u", envvar="CEMS_USER_ID", default="default", help="User ID")
@click.option("--team", "-t", envvar="CEMS_TEAM_ID", default=None, help="Team ID")
def list_pinned(category: str | None, user: str, team: str | None) -> None:
    """List pinned memories."""
    config = CEMSConfig(user_id=user, team_id=team)
    memory = CEMSMemory(config)

    pinned_ids = memory.metadata_store.get_pinned_memories(user, pin_category=category)

    if pinned_ids:
        table = Table(title="Pinned Memories")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Content", style="white", max_width=50)
        table.add_column("Category", style="yellow")
        table.add_column("Reason", style="cyan")

        for mem_id in pinned_ids:
            metadata = memory.get_metadata(mem_id)
            mem = memory.get(mem_id)
            content = mem.get("memory", "") if mem else ""

            if metadata:
                table.add_row(
                    mem_id[:12],
                    content[:50] + "..." if len(content) > 50 else content,
                    metadata.pin_category or "",
                    (metadata.pin_reason or "")[:30],
                )

        console.print(table)
        console.print(f"\nTotal pinned: {len(pinned_ids)}")
    else:
        console.print("[yellow]No pinned memories found[/yellow]")


if __name__ == "__main__":
    main()
