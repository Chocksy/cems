"""CEMS CLI command modules."""

from cems.commands.status import health, status
from cems.commands.memory import add, delete, list_memories, search, update
from cems.commands.maintenance import maintenance
from cems.commands.admin import admin

__all__ = [
    "status",
    "health",
    "add",
    "search",
    "list_memories",
    "delete",
    "update",
    "maintenance",
    "admin",
]
