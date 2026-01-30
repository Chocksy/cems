"""Shared dependencies for API handlers.

This module provides:
- Context variables for per-request user identification
- Memory instance caching per user
- Scheduler instance management
"""

import logging
from contextvars import ContextVar

from cems.config import CEMSConfig
from cems.memory import CEMSMemory
from cems.scheduler import CEMSScheduler

logger = logging.getLogger(__name__)

# Context variables for per-request user identification (HTTP mode)
request_user_id: ContextVar[str | None] = ContextVar("request_user_id", default=None)
request_team_id: ContextVar[str | None] = ContextVar("request_team_id", default=None)

# Global memory instances per user (for HTTP mode)
_memory_cache: dict[str, CEMSMemory] = {}
_scheduler_cache: dict[str, CEMSScheduler] = {}

# Base config (loaded once at startup with API keys)
_base_config: CEMSConfig | None = None


def get_base_config() -> CEMSConfig:
    """Get the base config with API keys (loaded from env)."""
    global _base_config
    if _base_config is None:
        _base_config = CEMSConfig()
    return _base_config


def get_memory() -> CEMSMemory:
    """Get or create the memory instance for the current request.

    In HTTP mode: Uses user_id/team_id from request headers (contextvars)
    In stdio mode: Uses user_id/team_id from environment variables
    """
    # Check for request-scoped user context (HTTP mode)
    user_id = request_user_id.get()
    team_id = request_team_id.get()

    if user_id:
        # HTTP mode: Create per-user memory instance
        cache_key = f"{user_id}:{team_id or 'none'}"
        if cache_key not in _memory_cache:
            # Create config with header values but inherit settings from base
            base = get_base_config()
            # Override user/team via environment for this instance
            config = CEMSConfig(
                user_id=user_id,
                team_id=team_id,
                # Inherit all other settings from base config
                storage_dir=base.storage_dir,
                memory_backend=base.memory_backend,
                mem0_model=base.mem0_model,
                embedding_model=base.embedding_model,
                llm_model=base.llm_model,
                enable_graph=base.enable_graph,
                enable_scheduler=False,  # Scheduler runs separately
                enable_query_synthesis=base.enable_query_synthesis,
                relevance_threshold=base.relevance_threshold,
                default_max_tokens=base.default_max_tokens,
            )
            _memory_cache[cache_key] = CEMSMemory(config)
            logger.info(f"Created memory instance for user: {user_id}, team: {team_id}")
        return _memory_cache[cache_key]
    else:
        # stdio mode: Use default config from environment
        cache_key = "default"
        if cache_key not in _memory_cache:
            config = CEMSConfig()
            _memory_cache[cache_key] = CEMSMemory(config)
            logger.info(f"Initialized CEMS memory for user: {config.user_id}")
        return _memory_cache[cache_key]


def get_scheduler() -> CEMSScheduler:
    """Get or create the scheduler instance."""
    cache_key = "default"  # Scheduler is shared
    if cache_key not in _scheduler_cache:
        # Use base config for scheduler
        config = get_base_config()
        memory = CEMSMemory(config)
        _scheduler_cache[cache_key] = CEMSScheduler(memory)
    return _scheduler_cache[cache_key]
