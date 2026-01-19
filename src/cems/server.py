"""MCP server for CEMS using FastMCP.

Supports two modes:
1. stdio (default) - For local Claude Code usage, uses env vars for user/team
2. http - For server deployment, uses headers for user/team identification

Provides a simplified API with 5 essential tools:
1. memory_add - Store memories (personal or shared)
2. memory_search - Unified intelligent search with 5-stage pipeline
3. memory_forget - Delete or archive memories
4. memory_maintenance - Run background maintenance jobs
5. memory_update - Update existing memory content

And 3 resources:
1. memory://status - System status
2. memory://personal/summary - Personal memories overview
3. memory://shared/summary - Shared memories overview
"""

import logging
import os
from contextvars import ContextVar
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from cems.config import CEMSConfig
from cems.memory import CEMSMemory
from cems.scheduler import CEMSScheduler

logger = logging.getLogger(__name__)

# Context variables for per-request user identification (HTTP mode)
_request_user_id: ContextVar[str | None] = ContextVar("request_user_id", default=None)
_request_team_id: ContextVar[str | None] = ContextVar("request_team_id", default=None)

# Initialize FastMCP server
# Disable DNS rebinding protection for production deployment (auth handled by Bearer token)
mcp = FastMCP(
    "CEMS Memory Server",
    host="0.0.0.0",  # Bind to all interfaces
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

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
    user_id = _request_user_id.get()
    team_id = _request_team_id.get()

    if user_id:
        # HTTP mode: Create per-user memory instance
        cache_key = f"{user_id}:{team_id or 'none'}"
        if cache_key not in _memory_cache:
            # Create config with header values but inherit API keys from base
            base = get_base_config()
            # Override user/team via environment for this instance
            config = CEMSConfig(
                user_id=user_id,
                team_id=team_id,
                # Inherit all other settings from base config
                storage_dir=base.storage_dir,
                memory_backend=base.memory_backend,
                mem0_llm_provider=base.mem0_llm_provider,
                mem0_model=base.mem0_model,
                embedding_model=base.embedding_model,
                llm_model=base.llm_model,
                vector_store=base.vector_store,
                qdrant_url=base.qdrant_url,
                enable_graph=base.enable_graph,
                graph_store=base.graph_store,
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


# =============================================================================
# MCP Tools (5 Essential Tools)
# =============================================================================


@mcp.tool()
def memory_add(
    content: str,
    scope: str = "personal",
    category: str = "general",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Store a memory in your personal or shared namespace.

    Use this to remember important information, preferences, decisions,
    or any context you want to persist across sessions.

    Args:
        content: What to remember (fact, preference, decision, etc.)
        scope: "personal" for private memory, "shared" for team memory
        category: Category for organization (preferences, decisions, patterns, etc.)
        tags: Optional tags for additional organization

    Returns:
        Result dict with success status and memory IDs
    """
    memory = get_memory()

    try:
        result = memory.add(
            content=content,
            scope=scope,  # type: ignore
            category=category,
            tags=tags or [],
        )

        # Extract memory IDs from result
        memory_ids = []
        if result and "results" in result:
            for r in result["results"]:
                if r.get("id"):
                    memory_ids.append(r["id"])

        return {
            "success": True,
            "message": f"Memory added to {scope} namespace",
            "memory_ids": memory_ids,
        }
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return {
            "success": False,
            "message": str(e),
            "memory_ids": [],
        }


@mcp.tool()
def memory_search(
    query: str,
    scope: str = "both",
    max_results: int = 10,
) -> dict[str, Any]:
    """Search your memories for relevant information.

    This is the primary search tool. It automatically:
    - Expands your query for better matching
    - Searches both vector similarity and knowledge graph
    - Prioritizes recent and frequently-accessed memories
    - Returns results within a reasonable token budget

    Args:
        query: What to search for (natural language)
        scope: "personal", "shared", or "both" namespaces
        max_results: Maximum memories to return (1-20)

    Returns:
        Dict with results, token count, and formatted context
    """
    memory = get_memory()

    try:
        # Use the 5-stage inference retrieval pipeline
        result = memory.retrieve_for_inference(
            query=query,
            scope=scope,  # type: ignore
            max_tokens=max(1, min(max_results, 20)) * 200,  # ~200 tokens per memory
            enable_query_synthesis=True,
            enable_graph=True,
        )

        return {
            "success": True,
            "query": query,
            **result,
        }
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        return {
            "success": False,
            "results": [],
            "query": query,
            "error": str(e),
        }


@mcp.tool()
def memory_forget(
    memory_id: str,
    hard_delete: bool = False,
) -> dict[str, Any]:
    """Forget (delete or archive) a memory.

    By default, memories are archived (soft delete) so they can be recovered.
    Use hard_delete=True for permanent deletion.

    Args:
        memory_id: ID of memory to forget
        hard_delete: If True, permanently delete. If False, archive (can recover).

    Returns:
        Result dict with success status
    """
    memory = get_memory()

    try:
        memory.delete(memory_id, hard=hard_delete)
        action = "deleted" if hard_delete else "archived"
        return {
            "success": True,
            "message": f"Memory {memory_id} {action}",
            "memory_id": memory_id,
        }
    except Exception as e:
        logger.error(f"Failed to forget memory: {e}")
        return {
            "success": False,
            "message": str(e),
            "memory_id": memory_id,
        }


@mcp.tool()
def memory_update(
    memory_id: str,
    content: str,
) -> dict[str, Any]:
    """Update an existing memory's content.

    Args:
        memory_id: ID of the memory to update
        content: New content for the memory

    Returns:
        Result dict with success status
    """
    memory = get_memory()

    try:
        memory.update(memory_id, content)
        return {
            "success": True,
            "message": f"Memory {memory_id} updated",
            "memory_id": memory_id,
        }
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")
        return {
            "success": False,
            "message": str(e),
            "memory_id": memory_id,
        }


@mcp.tool()
def memory_maintenance(
    job_type: str = "consolidation",
) -> dict[str, Any]:
    """Run a memory maintenance job.

    Maintenance operations keep your memory system healthy:
    - consolidation: Merge duplicate memories, promote frequently used ones
    - summarization: Create category summaries, compress old memories
    - reindex: Rebuild embeddings, archive dead memories
    - all: Run all maintenance jobs

    Args:
        job_type: Type of maintenance to run

    Returns:
        Maintenance job results
    """
    scheduler = get_scheduler()

    try:
        if job_type == "all":
            results = {}
            for jt in ["consolidation", "summarization", "reindex"]:
                results[jt] = scheduler.run_now(jt)
            return {
                "success": True,
                "job_type": "all",
                "results": results,
            }
        else:
            result = scheduler.run_now(job_type)
            return {
                "success": True,
                "job_type": job_type,
                "results": result,
            }
    except Exception as e:
        logger.error(f"Failed to run maintenance: {e}")
        return {
            "success": False,
            "job_type": job_type,
            "message": str(e),
            "results": {},
        }


# =============================================================================
# MCP Resources (3 Essential Resources)
# =============================================================================


@mcp.resource("memory://status")
def memory_status() -> str:
    """Get the current status of the memory system."""
    memory = get_memory()
    config = memory.config

    status = "CEMS Memory System Status\n"
    status += "=" * 40 + "\n\n"
    status += f"User ID: {config.user_id}\n"
    status += f"Team ID: {config.team_id or '(not set)'}\n"
    status += f"Storage: {config.storage_dir}\n"
    status += f"Backend: {config.memory_backend}\n"
    status += f"Vector Store: {config.vector_store}\n"
    status += f"Graph Store: {config.graph_store if config.enable_graph else 'disabled'}\n"
    status += f"Scheduler: {'enabled' if config.enable_scheduler else 'disabled'}\n"
    status += f"\nRetrieval Settings:\n"
    status += f"  Query Synthesis: {'enabled' if config.enable_query_synthesis else 'disabled'}\n"
    status += f"  Relevance Threshold: {config.relevance_threshold}\n"
    status += f"  Max Tokens: {config.default_max_tokens}\n"
    status += f"\nLLM Config:\n"
    status += f"  Mem0: {config.get_mem0_provider()}/{config.get_mem0_model()}\n"
    status += f"  Maintenance: OpenRouter/{config.llm_model}\n"

    # Add graph stats if enabled
    if memory.graph_store:
        stats = memory.get_graph_stats()
        status += f"\nGraph Stats:\n"
        for key, value in stats.items():
            status += f"  {key}: {value}\n"

    return status


@mcp.resource("memory://personal/summary")
def personal_summary() -> str:
    """Get a summary of personal memories."""
    memory = get_memory()
    memories = memory.get_all(scope="personal")

    if not memories:
        return "No personal memories stored yet."

    summary = "Personal Memory Summary:\n"
    summary += f"- Total memories: {len(memories)}\n"

    # Group by category (from metadata)
    categories: dict[str, int] = {}
    for m in memories:
        mem_id = m.get("id")
        if mem_id:
            metadata = memory.get_metadata(mem_id)
            cat = metadata.category if metadata else "general"
            categories[cat] = categories.get(cat, 0) + 1

    summary += "\nBy category:\n"
    for cat, count in sorted(categories.items()):
        summary += f"  - {cat}: {count}\n"

    return summary


@mcp.resource("memory://shared/summary")
def shared_summary() -> str:
    """Get a summary of shared team memories."""
    memory = get_memory()

    if not memory.config.team_id:
        return "No team configured. Set CEMS_TEAM_ID to enable shared memory."

    memories = memory.get_all(scope="shared")

    if not memories:
        return f"No shared memories for team '{memory.config.team_id}' yet."

    summary = f"Shared Memory Summary (Team: {memory.config.team_id}):\n"
    summary += f"- Total memories: {len(memories)}\n"

    # Group by category
    categories: dict[str, int] = {}
    for m in memories:
        mem_id = m.get("id")
        if mem_id:
            metadata = memory.get_metadata(mem_id)
            cat = metadata.category if metadata else "general"
            categories[cat] = categories.get(cat, 0) + 1

    if categories:
        summary += "\nBy category:\n"
        for cat, count in sorted(categories.items()):
            summary += f"  - {cat}: {count}\n"

    return summary


# =============================================================================
# Server Entry Points
# =============================================================================


def create_http_app():
    """Create the HTTP server app with user context middleware.

    Returns a Starlette app that:
    1. Provides /health endpoint for Docker health checks
    2. Validates Authorization: Bearer <token> header
    3. Extracts user context from headers or API key lookup
    4. Sets them in contextvars for the request
    5. Routes to the MCP server
    6. Provides /admin/* endpoints for user management (if DATABASE_URL set)
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    # Get config
    config = get_base_config()
    expected_api_key = config.api_key
    use_database = config.database_url is not None

    class UserContextMiddleware(BaseHTTPMiddleware):
        """Middleware to extract user context and validate API key from headers."""

        async def dispatch(self, request: Request, call_next):
            # Skip auth for health check
            if request.url.path == "/health":
                return await call_next(request)

            # Skip middleware for admin routes (they handle their own auth)
            if request.url.path.startswith("/admin"):
                return await call_next(request)

            # Get authorization header
            auth_header = request.headers.get("authorization", "")

            if use_database:
                # Database mode: Validate user API key from database
                if not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        {"error": "Authorization: Bearer <api_key> header required"},
                        status_code=401,
                    )
                provided_key = auth_header[7:]

                # Look up user by API key
                from cems.admin.services import UserService
                from cems.db.database import get_database, is_database_initialized

                if not is_database_initialized():
                    return JSONResponse(
                        {"error": "Database not initialized"},
                        status_code=503,
                    )

                db = get_database()
                with db.session() as session:
                    service = UserService(session)
                    user = service.get_user_by_api_key(provided_key)

                    if not user:
                        return JSONResponse(
                            {"error": "Invalid API key"},
                            status_code=401,
                        )

                    if not user.is_active:
                        return JSONResponse(
                            {"error": "User account is deactivated"},
                            status_code=403,
                        )

                    # Set user context from database
                    user_id = user.username
                    # Get team from header or user's first team
                    team_id = request.headers.get("x-team-id")

            else:
                # Simple mode: Validate against static API key
                if expected_api_key:
                    if not auth_header.startswith("Bearer "):
                        return JSONResponse(
                            {"error": "Authorization: Bearer <token> header required"},
                            status_code=401,
                        )
                    provided_token = auth_header[7:]
                    if provided_token != expected_api_key:
                        return JSONResponse(
                            {"error": "Invalid bearer token"},
                            status_code=401,
                        )

                # Extract user context from headers
                user_id = request.headers.get("x-user-id")
                team_id = request.headers.get("x-team-id")

                if not user_id:
                    return JSONResponse(
                        {"error": "X-User-ID header required"},
                        status_code=400,
                    )

            # Set context variables
            user_token = _request_user_id.set(user_id)
            team_token = _request_team_id.set(team_id)

            try:
                response = await call_next(request)
                return response
            finally:
                # Reset context variables
                _request_user_id.reset(user_token)
                _request_team_id.reset(team_token)

    async def health_check(request: Request):
        """Health check endpoint for Docker/Kubernetes."""
        from cems.db.database import get_database, is_database_initialized

        db_status = "not_configured"
        if is_database_initialized():
            db = get_database()
            db_status = "healthy" if db.health_check() else "unhealthy"

        return JSONResponse({
            "status": "healthy",
            "service": "cems-mcp-server",
            "mode": "http",
            "auth": "database" if use_database else ("bearer" if expected_api_key else "none"),
            "database": db_status,
        })

    # Get the base MCP app
    app = mcp.streamable_http_app()

    # Remove FastMCP's restrictive TrustedHostMiddleware (allows only localhost)
    # We'll add our own with wildcard hosts since auth is handled by Bearer token
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    app.user_middleware = [
        m for m in app.user_middleware
        if not (hasattr(m, 'cls') and getattr(m.cls, '__name__', '') == 'TrustedHostMiddleware')
    ]

    # Add health check route
    app.routes.insert(0, Route("/health", health_check, methods=["GET"]))

    # Add admin routes if database is configured
    if use_database:
        from cems.admin.routes import admin_routes
        for route in admin_routes:
            app.routes.insert(0, route)
        logger.info("Admin API routes enabled (/admin/*)")

    # Add our middlewares (order: last added = first executed)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(UserContextMiddleware)

    return app


def wait_for_qdrant(url: str, max_retries: int = 30, delay: float = 2.0) -> bool:
    """Wait for Qdrant to be available.

    Args:
        url: Qdrant URL (e.g., http://cems-qdrant:6333)
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds

    Returns:
        True if Qdrant is available, False otherwise
    """
    import time
    import urllib.request
    import urllib.error

    health_url = f"{url.rstrip('/')}/healthz"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(health_url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info(f"Qdrant is available at {url}")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning(f"Waiting for Qdrant (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(delay)

    logger.error(f"Qdrant not available at {url} after {max_retries} attempts")
    return False


def run_http_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run the CEMS MCP server in HTTP mode.

    Args:
        host: Host to bind to (default: 0.0.0.0 for Docker)
        port: Port to listen on (default: 8765)
    """
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize base config (loads API keys from env)
    config = get_base_config()
    logger.info(f"Starting CEMS HTTP server on {host}:{port}")
    logger.info(f"Vector store: {config.vector_store}")
    logger.info(f"Qdrant URL: {config.qdrant_url}")
    logger.info(f"API key auth: {'enabled' if config.api_key else 'disabled'}")

    # Initialize PostgreSQL database if configured
    if config.database_url:
        from cems.db.database import init_database, run_migrations

        logger.info("Initializing PostgreSQL database...")
        db = init_database(config.database_url)
        db.create_tables()
        run_migrations()
        logger.info("PostgreSQL database initialized")
        logger.info(f"Admin API: {'enabled' if config.admin_key else 'WARNING: CEMS_ADMIN_KEY not set'}")

    # Wait for Qdrant if URL is configured
    if config.qdrant_url:
        if not wait_for_qdrant(config.qdrant_url):
            logger.error("Cannot start server: Qdrant is not available")
            raise RuntimeError(f"Qdrant not available at {config.qdrant_url}")

    # Start scheduler if enabled (runs for all users)
    if config.enable_scheduler:
        scheduler = get_scheduler()
        scheduler.start()
        logger.info("Background scheduler started")

    # Create and run HTTP app
    app = create_http_app()
    uvicorn.run(app, host=host, port=port)


def run_server() -> None:
    """Run the CEMS MCP server (stdio or http based on CEMS_MODE)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    mode = os.environ.get("CEMS_MODE", "stdio").lower()

    if mode == "http" or mode == "server":
        # HTTP mode - for Docker deployment
        host = os.environ.get("CEMS_SERVER_HOST", "0.0.0.0")
        port = int(os.environ.get("CEMS_SERVER_PORT", "8765"))
        run_http_server(host=host, port=port)
    else:
        # stdio mode - for local Claude Code usage
        # Initialize memory on startup
        memory = get_memory()
        config = memory.config

        logger.info(f"Starting CEMS MCP server (stdio) for user: {config.user_id}")

        # Start scheduler if enabled
        if config.enable_scheduler:
            scheduler = get_scheduler()
            scheduler.start()
            logger.info("Background scheduler started")

        # Run the MCP server in stdio mode
        mcp.run()


if __name__ == "__main__":
    run_server()
