"""CLI interface for CEMS.

This CLI communicates with a CEMS server via HTTP API.
All operations require CEMS_API_URL and CEMS_API_KEY to be set.
"""

import json
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

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


# =============================================================================
# Status Command
# =============================================================================


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show CEMS status and configuration."""
    try:
        client = get_client(ctx)
        data = client.status()

        table = Table(title="CEMS Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Server", client.api_url)
        table.add_row("User ID", data.get("user_id", "?"))
        table.add_row("Team ID", data.get("team_id") or "(not set)")
        table.add_row("Status", data.get("status", "unknown"))
        table.add_row("Backend", data.get("backend", "?"))
        table.add_row("Vector Store", data.get("vector_store", "?"))
        table.add_row("Graph Store", data.get("graph_store") or "disabled")
        table.add_row("Query Synthesis", str(data.get("query_synthesis", False)))

        console.print(table)

        # Show graph stats if available
        if data.get("graph_stats"):
            console.print("\n[bold]Graph Statistics:[/bold]")
            for key, value in data["graph_stats"].items():
                console.print(f"  {key}: {value}")

    except CEMSClientError as e:
        handle_error(e)


# =============================================================================
# Memory Commands
# =============================================================================


@main.command()
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


@main.command()
@click.argument("query")
@click.option("--scope", "-s", default="both", type=click.Choice(["personal", "shared", "both"]))
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--max-tokens", "-t", default=2000, help="Token budget for results")
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


@main.command("list")
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


@main.command()
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


@main.command()
@click.argument("memory_id")
@click.argument("content")
@click.pass_context
def update(ctx: click.Context, memory_id: str, content: str) -> None:
    """Update a memory's content.

    Example:
        cems update abc123 "Updated content here"
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


# =============================================================================
# Maintenance Commands
# =============================================================================


@main.group()
def maintenance() -> None:
    """Maintenance commands for memory system."""
    pass


@maintenance.command("run")
@click.argument("job_type", type=click.Choice(["consolidation", "summarization", "reindex", "all"]))
@click.pass_context
def run_maintenance(ctx: click.Context, job_type: str) -> None:
    """Run a maintenance job immediately.

    JOB_TYPE: consolidation, summarization, reindex, or all
    """
    try:
        client = get_client(ctx)

        console.print(f"[cyan]Running {job_type}...[/cyan]")
        with console.status("Running maintenance..."):
            result = client.maintenance(job_type)

        if result.get("success"):
            console.print(f"[green]{job_type} completed[/green]")
            if ctx.obj["verbose"]:
                console.print(json.dumps(result, indent=2, default=str))
        else:
            console.print(f"[yellow]{job_type} may have failed[/yellow]")
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)


# =============================================================================
# Admin Commands
# =============================================================================


@main.group()
@click.option("--admin-key", envvar="CEMS_ADMIN_KEY", help="Admin API key")
@click.pass_context
def admin(ctx: click.Context, admin_key: str | None) -> None:
    """Admin commands for user and team management.

    Requires admin privileges (CEMS_ADMIN_KEY or admin user API key).
    """
    ctx.obj["admin_key"] = admin_key


# -----------------------------------------------------------------------------
# User Management
# -----------------------------------------------------------------------------


@admin.group("users")
def admin_users() -> None:
    """User management commands."""
    pass


@admin_users.command("list")
@click.option("--include-inactive", is_flag=True, help="Include inactive users")
@click.option("--limit", default=100, help="Maximum users to return")
@click.pass_context
def users_list(ctx: click.Context, include_inactive: bool, limit: int) -> None:
    """List all users."""
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Loading users..."):
            result = client.list_users(include_inactive=include_inactive, limit=limit)

        users = result.get("users", [])
        if users:
            table = Table(title="Users")
            table.add_column("Username", style="cyan")
            table.add_column("Email", style="white")
            table.add_column("Admin", style="yellow")
            table.add_column("Active", style="green")
            table.add_column("API Key Prefix", style="dim")

            for u in users:
                table.add_row(
                    u.get("username", "?"),
                    u.get("email") or "-",
                    "Yes" if u.get("is_admin") else "No",
                    "Yes" if u.get("is_active") else "No",
                    u.get("api_key_prefix", "?"),
                )

            console.print(table)
            console.print(f"\nTotal: {len(users)} users")
        else:
            console.print("[yellow]No users found[/yellow]")

    except CEMSClientError as e:
        handle_error(e)


@admin_users.command("create")
@click.argument("username")
@click.option("--email", "-e", help="User email")
@click.option("--admin", "is_admin", is_flag=True, help="Make user an admin")
@click.pass_context
def users_create(ctx: click.Context, username: str, email: str | None, is_admin: bool) -> None:
    """Create a new user.

    The API key will be displayed ONCE - save it!

    Example:
        cems admin users create john --email john@example.com
    """
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Creating user..."):
            result = client.create_user(username, email=email, is_admin=is_admin)

        console.print("[green]User created successfully![/green]\n")

        user = result.get("user", {})
        table = Table(title="New User")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Username", user.get("username", "?"))
        table.add_row("Email", user.get("email") or "-")
        table.add_row("Admin", "Yes" if user.get("is_admin") else "No")
        table.add_row("API Key Prefix", user.get("api_key_prefix", "?"))

        console.print(table)

        # Show API key prominently
        api_key = result.get("api_key")
        if api_key:
            console.print("\n[bold red]IMPORTANT: Save this API key - it will NOT be shown again![/bold red]")
            console.print(f"\n[bold]API Key:[/bold] {api_key}\n")

    except CEMSClientError as e:
        handle_error(e)


@admin_users.command("get")
@click.argument("user")
@click.pass_context
def users_get(ctx: click.Context, user: str) -> None:
    """Get user details.

    USER can be a username or UUID.
    """
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Loading user..."):
            result = client.get_user(user)

        table = Table(title=f"User: {result.get('username', '?')}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("ID", result.get("id", "?"))
        table.add_row("Username", result.get("username", "?"))
        table.add_row("Email", result.get("email") or "-")
        table.add_row("Admin", "Yes" if result.get("is_admin") else "No")
        table.add_row("Active", "Yes" if result.get("is_active") else "No")
        table.add_row("API Key Prefix", result.get("api_key_prefix", "?"))
        table.add_row("Created", result.get("created_at", "?"))
        table.add_row("Last Active", result.get("last_active", "?"))

        console.print(table)

    except CEMSClientError as e:
        handle_error(e)


@admin_users.command("delete")
@click.argument("user_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def users_delete(ctx: click.Context, user_id: str, yes: bool) -> None:
    """Delete a user.

    USER_ID must be the UUID.
    """
    if not yes:
        if not click.confirm(f"Delete user {user_id}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Deleting user..."):
            client.delete_user(user_id)

        console.print("[green]User deleted[/green]")

    except CEMSClientError as e:
        handle_error(e)


@admin_users.command("reset-key")
@click.argument("user_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def users_reset_key(ctx: click.Context, user_id: str, yes: bool) -> None:
    """Reset a user's API key.

    The new API key will be displayed ONCE - save it!
    """
    if not yes:
        if not click.confirm(f"Reset API key for user {user_id}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Resetting API key..."):
            result = client.reset_api_key(user_id)

        console.print("[green]API key reset successfully![/green]\n")

        user = result.get("user", {})
        console.print(f"[bold]Username:[/bold] {user.get('username', '?')}")
        console.print(f"[bold]New Prefix:[/bold] {user.get('api_key_prefix', '?')}")

        api_key = result.get("api_key")
        if api_key:
            console.print("\n[bold red]IMPORTANT: Save this API key - it will NOT be shown again![/bold red]")
            console.print(f"\n[bold]API Key:[/bold] {api_key}\n")

    except CEMSClientError as e:
        handle_error(e)


# -----------------------------------------------------------------------------
# Team Management
# -----------------------------------------------------------------------------


@admin.group("teams")
def admin_teams() -> None:
    """Team management commands."""
    pass


@admin_teams.command("list")
@click.option("--limit", default=100, help="Maximum teams to return")
@click.pass_context
def teams_list(ctx: click.Context, limit: int) -> None:
    """List all teams."""
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Loading teams..."):
            result = client.list_teams(limit=limit)

        teams = result.get("teams", [])
        if teams:
            table = Table(title="Teams")
            table.add_column("Name", style="cyan")
            table.add_column("Company ID", style="white")
            table.add_column("Created", style="dim")
            table.add_column("ID", style="dim", max_width=12)

            for t in teams:
                table.add_row(
                    t.get("name", "?"),
                    t.get("company_id", "?"),
                    t.get("created_at", "?")[:10] if t.get("created_at") else "?",
                    t.get("id", "?")[:12],
                )

            console.print(table)
            console.print(f"\nTotal: {len(teams)} teams")
        else:
            console.print("[yellow]No teams found[/yellow]")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("create")
@click.argument("name")
@click.option("--company-id", "-c", required=True, help="Company identifier")
@click.pass_context
def teams_create(ctx: click.Context, name: str, company_id: str) -> None:
    """Create a new team.

    Example:
        cems admin teams create engineering --company-id acme
    """
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Creating team..."):
            result = client.create_team(name, company_id)

        console.print("[green]Team created successfully![/green]\n")

        team = result.get("team", {})
        console.print(f"[bold]Name:[/bold] {team.get('name', '?')}")
        console.print(f"[bold]Company:[/bold] {team.get('company_id', '?')}")
        console.print(f"[bold]ID:[/bold] {team.get('id', '?')}")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("get")
@click.argument("team")
@click.pass_context
def teams_get(ctx: click.Context, team: str) -> None:
    """Get team details with members.

    TEAM can be a team name or UUID.
    """
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Loading team..."):
            result = client.get_team(team)

        console.print(f"\n[bold]Team: {result.get('name', '?')}[/bold]\n")
        console.print(f"ID: {result.get('id', '?')}")
        console.print(f"Company: {result.get('company_id', '?')}")
        console.print(f"Created: {result.get('created_at', '?')}")

        members = result.get("members", [])
        if members:
            console.print("\n[bold]Members:[/bold]")
            table = Table()
            table.add_column("User ID", style="dim", max_width=12)
            table.add_column("Role", style="cyan")
            table.add_column("Joined", style="white")

            for m in members:
                table.add_row(
                    m.get("user_id", "?")[:12],
                    m.get("role", "?"),
                    m.get("joined_at", "?")[:10] if m.get("joined_at") else "?",
                )

            console.print(table)
        else:
            console.print("\n[yellow]No members[/yellow]")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("delete")
@click.argument("team_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def teams_delete(ctx: click.Context, team_id: str, yes: bool) -> None:
    """Delete a team.

    TEAM_ID must be the UUID.
    """
    if not yes:
        if not click.confirm(f"Delete team {team_id}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Deleting team..."):
            client.delete_team(team_id)

        console.print("[green]Team deleted[/green]")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("add-member")
@click.argument("team_id")
@click.argument("user_id")
@click.option("--role", "-r", default="member", type=click.Choice(["admin", "member", "viewer"]))
@click.pass_context
def teams_add_member(ctx: click.Context, team_id: str, user_id: str, role: str) -> None:
    """Add a user to a team.

    Example:
        cems admin teams add-member <team-uuid> <user-uuid> --role member
    """
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Adding member..."):
            result = client.add_team_member(team_id, user_id, role)

        console.print("[green]Member added successfully![/green]")
        member = result.get("member", {})
        console.print(f"Role: {member.get('role', '?')}")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("remove-member")
@click.argument("team_id")
@click.argument("user_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def teams_remove_member(ctx: click.Context, team_id: str, user_id: str, yes: bool) -> None:
    """Remove a user from a team."""
    if not yes:
        if not click.confirm(f"Remove user {user_id} from team {team_id}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Removing member..."):
            client.remove_team_member(team_id, user_id)

        console.print("[green]Member removed[/green]")

    except CEMSClientError as e:
        handle_error(e)


# =============================================================================
# Health Check
# =============================================================================


@main.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check server health."""
    try:
        client = get_client(ctx)

        with console.status("Checking health..."):
            result = client.health()

        status_text = result.get("status", "unknown")
        if status_text == "healthy":
            console.print(f"[green]Server: {status_text}[/green]")
        else:
            console.print(f"[yellow]Server: {status_text}[/yellow]")

        console.print(f"Service: {result.get('service', '?')}")
        console.print(f"Mode: {result.get('mode', '?')}")
        console.print(f"Auth: {result.get('auth', '?')}")
        console.print(f"Database: {result.get('database', '?')}")

    except CEMSClientError as e:
        handle_error(e)


if __name__ == "__main__":
    main()
