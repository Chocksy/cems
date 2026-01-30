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

    return JSONResponse({
        "status": "healthy",
        "service": "cems-mcp-server",
        "mode": "http",
        "auth": "database",
        "database": db_status,
    })
