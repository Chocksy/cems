"""Health check handlers."""

from starlette.requests import Request
from starlette.responses import JSONResponse


async def ping(request: Request):
    """Ultra-lightweight ping endpoint for MCP heartbeat checks.

    Returns immediately with no database or service checks.
    This helps prevent Cursor's MCP client timeout issues.
    """
    return JSONResponse({"status": "ok"})


async def health_check(request: Request):
    """Health check endpoint for Docker/Kubernetes."""
    from cems.db.database import get_database, is_database_initialized

    db_status = "not_configured"
    try:
        if is_database_initialized():
            db = get_database()
            db_status = "healthy" if db.health_check() else "unhealthy"
    except Exception as e:
        db_status = f"error: {e}"

    overall = "healthy" if db_status in ("healthy", "not_configured") else "unhealthy"
    status_code = 200 if overall == "healthy" else 503

    return JSONResponse({
        "status": overall,
        "service": "cems-mcp-server",
        "mode": "http",
        "auth": "database",
        "database": db_status,
    }, status_code=status_code)


async def config_discovery(request: Request):
    """Client setup discovery endpoint (unauthenticated).

    GET /api/config/setup

    Returns server URLs for client auto-configuration.
    Used by `cems setup` to discover the MCP wrapper URL.
    """
    from cems.api.deps import get_base_config

    config = get_base_config()

    result = {}
    if config.mcp_public_url:
        result["mcp_url"] = config.mcp_public_url

    return JSONResponse(result)
