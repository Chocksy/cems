"""Admin commands for CEMS CLI."""

import click

from cems.commands.admin.users import admin_users
from cems.commands.admin.teams import admin_teams


@click.group()
@click.option("--admin-key", envvar="CEMS_ADMIN_KEY", help="Admin API key")
@click.pass_context
def admin(ctx: click.Context, admin_key: str | None) -> None:
    """Admin commands for user and team management.

    Requires admin privileges (CEMS_ADMIN_KEY or admin user API key).
    """
    ctx.obj["admin_key"] = admin_key


# Register subcommands
admin.add_command(admin_users, name="users")
admin.add_command(admin_teams, name="teams")
