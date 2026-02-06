"""CEMS REST API Server.

Provides REST API endpoints for the CEMS memory system.
MCP protocol is handled by the Express wrapper (mcp-wrapper/).

REST API endpoints:
- POST /api/memory/add - Store memories
- POST /api/memory/search - Search memories
- POST /api/memory/forget - Delete/archive memories
- POST /api/memory/update - Update memory content
- POST /api/memory/maintenance - Run maintenance jobs
- GET /api/memory/status - System status
- GET /api/memory/summary/personal - Personal summary
- GET /api/memory/summary/shared - Shared summary
- POST /api/index/repo - Index a git repository
- POST /api/index/path - Index a local directory
- GET /api/index/patterns - List available index patterns
- POST /api/session/analyze - Analyze session transcripts
"""

import logging
import os

# Import shared dependencies from api/deps module
from cems.api.deps import (
    get_base_config,
    get_scheduler,
    request_team_id,
    request_user_id,
)

# Import all handlers from api/handlers package
from cems.api.handlers import (
    api_index_patterns,
    api_index_path,
    api_index_repo,
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
    api_session_analyze,
    api_tool_learning,
    health_check,
    ping,
)

logger = logging.getLogger(__name__)


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
    from starlette.applications import Starlette
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

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

            # Validate user API key from database
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
            try:
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

                    # Set user context from database (using UUID for pgvector)
                    user_id = str(user.id)
                    # Get team from header (optional, to select team context)
                    team_id = request.headers.get("x-team-id")
            except Exception as e:
                return JSONResponse(
                    {"error": f"Database error: {e}"},
                    status_code=503,
                )

            # Set context variables
            user_token = request_user_id.set(user_id)
            team_token = request_team_id.set(team_id)

            try:
                response = await call_next(request)
                return response
            except Exception as e:
                raise
            finally:
                # Reset context variables
                request_user_id.reset(user_token)
                request_team_id.reset(team_token)

    # Define routes using imported handlers
    routes = [
        # Health check endpoints
        Route("/ping", ping, methods=["GET"]),
        Route("/health", health_check, methods=["GET"]),
        # REST API routes - Memory
        Route("/api/memory/add", api_memory_add, methods=["POST"]),
        Route("/api/memory/add_batch", api_memory_add_batch, methods=["POST"]),
        Route("/api/memory/search", api_memory_search, methods=["POST"]),
        Route("/api/memory/forget", api_memory_forget, methods=["POST"]),
        Route("/api/memory/update", api_memory_update, methods=["POST"]),
        Route("/api/memory/maintenance", api_memory_maintenance, methods=["POST"]),
        Route("/api/memory/status", api_memory_status, methods=["GET"]),
        Route("/api/memory/gate-rules", api_memory_gate_rules, methods=["GET"]),
        Route("/api/memory/profile", api_memory_profile, methods=["GET"]),
        Route("/api/memory/summary/personal", api_memory_summary_personal, methods=["GET"]),
        Route("/api/memory/summary/shared", api_memory_summary_shared, methods=["GET"]),
        # REST API routes - Index
        Route("/api/index/repo", api_index_repo, methods=["POST"]),
        Route("/api/index/path", api_index_path, methods=["POST"]),
        Route("/api/index/patterns", api_index_patterns, methods=["GET"]),
        # REST API routes - Session
        Route("/api/session/analyze", api_session_analyze, methods=["POST"]),
        # REST API routes - Tool
        Route("/api/tool/learning", api_tool_learning, methods=["POST"]),
    ]
    logger.info("REST API routes enabled (/api/memory/*, /api/index/*, /api/session/*, /api/tool/*)")

    # Add admin routes (always available in HTTP mode with database)
    from cems.admin.routes import admin_routes
    routes.extend(admin_routes)
    logger.info("Admin API routes enabled (/admin/*)")

    # Create Starlette app
    app = Starlette(routes=routes)

    # Add middlewares (order: last added = first executed)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(UserContextMiddleware)

    return app


def run_http_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run the CEMS MCP server in HTTP mode.

    HTTP mode requires:
    - CEMS_DATABASE_URL: PostgreSQL for user management
    - OPENROUTER_API_KEY: For LLM and embedding operations

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

    # Validate required configuration for HTTP mode
    errors = []
    if not config.database_url:
        errors.append("CEMS_DATABASE_URL is required for HTTP mode")
    if not os.environ.get("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY is required")
    if errors:
        for error in errors:
            logger.error(error)
        raise RuntimeError("Missing required configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    logger.info(f"Starting CEMS HTTP server on {host}:{port}")
    logger.info("Vector store: pgvector (unified PostgreSQL)")
    logger.info("Authentication: per-user API keys via PostgreSQL")

    # Initialize PostgreSQL database (required for HTTP mode)
    from cems.db.database import init_database, run_migrations

    logger.info("Initializing PostgreSQL database...")
    db = init_database(config.database_url)
    db.create_tables()
    run_migrations()
    logger.info("PostgreSQL database initialized")
    if not config.admin_key:
        logger.warning("CEMS_ADMIN_KEY not set - admin API will be inaccessible")

    # Start scheduler if enabled (runs for all users)
    if config.enable_scheduler:
        scheduler = get_scheduler()
        scheduler.start()
        logger.info("Background scheduler started")

    # Create and run HTTP app
    app = create_http_app()
    # Configure uvicorn with extended timeouts for remote MCP connections
    # This helps prevent Cursor's MCP client from marking server as "red" due to timeouts
    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=120,  # Keep connections alive longer (default: 5s)
    )


def run_server() -> None:
    """Run the CEMS REST API server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    host = os.environ.get("CEMS_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("CEMS_SERVER_PORT", "8765"))
    run_http_server(host=host, port=port)


if __name__ == "__main__":
    run_server()
