"""User management commands for CEMS CLI."""

import click
from rich.table import Table

from cems.cli_utils import console, get_admin_client, handle_error
from cems.client import CEMSClientError


@click.group("users")
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
