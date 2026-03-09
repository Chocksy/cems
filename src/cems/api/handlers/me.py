"""User self-service API handlers.

Endpoints that let an authenticated user query their own profile/teams.
These use the standard user auth middleware (not admin auth).
"""

import logging
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import request_user_id

logger = logging.getLogger(__name__)


async def api_me_teams(request: Request) -> JSONResponse:
    """GET /api/me/teams — Return the current user's team memberships."""
    try:
        user_id = request_user_id.get()
        if not user_id:
            return JSONResponse({"error": "Not authenticated"}, status_code=401)

        from cems.admin.services import TeamService
        from cems.db.database import get_database

        db = get_database()
        with db.session() as session:
            service = TeamService(session)
            teams = service.get_user_teams(UUID(user_id))

            return JSONResponse({
                "teams": [
                    {"id": str(t.id), "name": t.name, "company_id": t.company_id}
                    for t in teams
                ],
            })
    except Exception as e:
        logger.error(f"API me/teams error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)
