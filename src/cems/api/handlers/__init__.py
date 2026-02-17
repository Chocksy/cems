"""API handlers package.

Exports all handler functions for route registration in server.py.
"""

from cems.api.handlers.health import config_setup, health_check, ping
from cems.api.handlers.memory import (
    api_memory_add,
    api_memory_add_batch,
    api_memory_forget,
    api_memory_foundation,
    api_memory_gate_rules,
    api_memory_list,
    api_memory_log_shown,
    api_memory_maintenance,
    api_memory_profile,
    api_memory_search,
    api_memory_status,
    api_memory_summary_personal,
    api_memory_summary_shared,
    api_memory_update,
)
from cems.api.handlers.index import api_index_patterns, api_index_path, api_index_repo
from cems.api.handlers.session import api_session_summarize
from cems.api.handlers.tool import api_tool_learning

__all__ = [
    # Health / Config
    "ping",
    "health_check",
    "config_setup",
    # Memory
    "api_memory_add",
    "api_memory_add_batch",
    "api_memory_search",
    "api_memory_foundation",
    "api_memory_gate_rules",
    "api_memory_profile",
    "api_memory_forget",
    "api_memory_list",
    "api_memory_log_shown",
    "api_memory_update",
    "api_memory_maintenance",
    "api_memory_status",
    "api_memory_summary_personal",
    "api_memory_summary_shared",
    # Index
    "api_index_repo",
    "api_index_path",
    "api_index_patterns",
    # Session
    "api_session_summarize",
    # Tool
    "api_tool_learning",
]
