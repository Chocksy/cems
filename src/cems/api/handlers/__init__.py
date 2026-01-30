"""API handlers package.

Exports all handler functions for route registration in server.py.
"""

from cems.api.handlers.health import health_check, ping
from cems.api.handlers.memory import (
    api_memory_add,
    api_memory_add_batch,
    api_memory_forget,
    api_memory_gate_rules,
    api_memory_maintenance,
    api_memory_profile,
    api_memory_search,
    api_memory_status,
    api_memory_summary_personal,
    api_memory_summary_shared,
    api_memory_update,
)
from cems.api.handlers.session import api_session_analyze
from cems.api.handlers.tool import api_tool_learning

__all__ = [
    # Health
    "ping",
    "health_check",
    # Memory
    "api_memory_add",
    "api_memory_add_batch",
    "api_memory_search",
    "api_memory_gate_rules",
    "api_memory_profile",
    "api_memory_forget",
    "api_memory_update",
    "api_memory_maintenance",
    "api_memory_status",
    "api_memory_summary_personal",
    "api_memory_summary_shared",
    # Session
    "api_session_analyze",
    # Tool
    "api_tool_learning",
]
