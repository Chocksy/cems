"""Team management commands for CEMS CLI."""

import click
from rich.table import Table

from cems.cli_utils import console, get_admin_client, handle_error
from cems.client import CEMSClientError


@click.group("teams")
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
            table.add_column("Username", style="cyan")
            table.add_column("Role", style="white")
            table.add_column("Joined", style="dim")

            for m in members:
                table.add_row(
                    m.get("username", m.get("user_id", "?")[:12]),
                    m.get("role", "?"),
                    m.get("joined_at", "?")[:10] if m.get("joined_at") else "?",
                )

            console.print(table)
        else:
            console.print("\n[yellow]No members[/yellow]")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("delete")
@click.argument("team")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def teams_delete(ctx: click.Context, team: str, yes: bool) -> None:
    """Delete a team.

    TEAM can be a team name or UUID.
    """
    if not yes:
        if not click.confirm(f"Delete team {team}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Deleting team..."):
            client.delete_team(team)

        console.print("[green]Team deleted[/green]")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("add-member")
@click.argument("team")
@click.argument("user")
@click.option("--role", "-r", default="member", type=click.Choice(["admin", "member", "viewer"]))
@click.pass_context
def teams_add_member(ctx: click.Context, team: str, user: str, role: str) -> None:
    """Add a user to a team.

    TEAM can be a team name or UUID.
    USER can be a username or UUID.

    Example:
        cems admin teams add-member tr hubble --role member
    """
    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Adding member..."):
            result = client.add_team_member(team, user, role)

        console.print("[green]Member added successfully![/green]")
        member = result.get("member", {})
        console.print(f"Role: {member.get('role', '?')}")

    except CEMSClientError as e:
        handle_error(e)


@admin_teams.command("remove-member")
@click.argument("team")
@click.argument("user")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def teams_remove_member(ctx: click.Context, team: str, user: str, yes: bool) -> None:
    """Remove a user from a team.

    TEAM can be a team name or UUID.
    USER can be a username or UUID.
    """
    if not yes:
        if not click.confirm(f"Remove user {user} from team {team}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        client = get_admin_client(ctx, ctx.obj.get("admin_key"))

        with console.status("Removing member..."):
            client.remove_team_member(team, user)

        console.print("[green]Member removed[/green]")

    except CEMSClientError as e:
        handle_error(e)
