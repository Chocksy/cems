"""Admin API routes for CEMS.

These routes require admin authentication via CEMS_ADMIN_KEY.
"""

import logging
import os
import uuid

from sqlalchemy import text
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from cems.admin.services import TeamService, UserService
from cems.db.database import get_database, is_database_initialized

logger = logging.getLogger(__name__)


def get_admin_key() -> str | None:
    """Get the admin API key from environment."""
    return os.environ.get("CEMS_ADMIN_KEY")


def require_admin_auth(request: Request) -> JSONResponse | None:
    """Check admin authentication.

    Returns None if authenticated, JSONResponse with error otherwise.
    """
    admin_key = get_admin_key()
    if not admin_key:
        return JSONResponse(
            {"error": "Admin API not configured (CEMS_ADMIN_KEY not set)"},
            status_code=503,
        )

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            {"error": "Authorization: Bearer <admin_key> header required"},
            status_code=401,
        )

    provided_key = auth_header[7:]  # Strip "Bearer "
    if provided_key != admin_key:
        return JSONResponse({"error": "Invalid admin key"}, status_code=403)

    return None  # Authenticated


def require_database(request: Request) -> JSONResponse | None:
    """Check if database is initialized.

    Returns None if initialized, JSONResponse with error otherwise.
    """
    if not is_database_initialized():
        return JSONResponse(
            {"error": "Database not configured (CEMS_DATABASE_URL not set)"},
            status_code=503,
        )
    return None


# =============================================================================
# User Routes
# =============================================================================


async def list_users(request: Request) -> JSONResponse:
    """List all users."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    include_inactive = request.query_params.get("include_inactive", "false") == "true"
    limit = int(request.query_params.get("limit", "100"))
    offset = int(request.query_params.get("offset", "0"))

    db = get_database()
    with db.session() as session:
        service = UserService(session)
        users = service.list_users(
            include_inactive=include_inactive, limit=limit, offset=offset
        )

        return JSONResponse(
            {
                "users": [
                    {
                        "id": str(u.id),
                        "username": u.username,
                        "email": u.email,
                        "is_admin": u.is_admin,
                        "is_active": u.is_active,
                        "api_key_prefix": u.api_key_prefix,
                        "created_at": u.created_at.isoformat(),
                        "last_active": u.last_active.isoformat(),
                    }
                    for u in users
                ],
                "count": len(users),
            }
        )


async def create_user(request: Request) -> JSONResponse:
    """Create a new user."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    username = data.get("username")
    if not username:
        return JSONResponse({"error": "username is required"}, status_code=400)

    email = data.get("email")
    is_admin = data.get("is_admin", False)
    settings = data.get("settings", {})

    db = get_database()
    try:
        with db.session() as session:
            service = UserService(session)
            try:
                result = service.create_user(
                    username=username,
                    email=email,
                    is_admin=is_admin,
                    settings=settings,
                )
                return JSONResponse(
                    {
                        "user": {
                            "id": str(result.user.id),
                            "username": result.user.username,
                            "email": result.user.email,
                            "is_admin": result.user.is_admin,
                            "api_key_prefix": result.user.api_key_prefix,
                        },
                        "api_key": result.api_key,  # Show only once!
                        "message": "User created. Save the API key - it will not be shown again.",
                    },
                    status_code=201,
                )
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("Failed to create user")
        return JSONResponse({"error": f"Database error: {e}"}, status_code=500)


async def get_user(request: Request) -> JSONResponse:
    """Get a user by ID or username."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    user_id_or_name = request.path_params["user_id"]

    db = get_database()
    with db.session() as session:
        service = UserService(session)

        # Try UUID first, then username
        try:
            user_uuid = uuid.UUID(user_id_or_name)
            user = service.get_user_by_id(user_uuid)
        except ValueError:
            user = service.get_user_by_username(user_id_or_name)

        if not user:
            return JSONResponse({"error": "User not found"}, status_code=404)

        return JSONResponse(
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_admin,
                "is_active": user.is_active,
                "api_key_prefix": user.api_key_prefix,
                "created_at": user.created_at.isoformat(),
                "last_active": user.last_active.isoformat(),
                "settings": user.settings,
            }
        )


async def update_user(request: Request) -> JSONResponse:
    """Update a user."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    user_id_str = request.path_params["user_id"]
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return JSONResponse({"error": "Invalid user ID"}, status_code=400)

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    db = get_database()
    with db.session() as session:
        service = UserService(session)
        user = service.update_user(
            user_id=user_id,
            email=data.get("email"),
            is_active=data.get("is_active"),
            is_admin=data.get("is_admin"),
            settings=data.get("settings"),
        )

        if not user:
            return JSONResponse({"error": "User not found"}, status_code=404)

        return JSONResponse(
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_admin,
                "is_active": user.is_active,
            }
        )


async def delete_user(request: Request) -> JSONResponse:
    """Delete a user."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    user_id_str = request.path_params["user_id"]
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return JSONResponse({"error": "Invalid user ID"}, status_code=400)

    db = get_database()
    with db.session() as session:
        service = UserService(session)
        if service.delete_user(user_id):
            return JSONResponse({"message": "User deleted"})
        return JSONResponse({"error": "User not found"}, status_code=404)


async def reset_user_api_key(request: Request) -> JSONResponse:
    """Reset a user's API key."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    user_id_str = request.path_params["user_id"]
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return JSONResponse({"error": "Invalid user ID"}, status_code=400)

    db = get_database()
    with db.session() as session:
        service = UserService(session)
        result = service.reset_api_key(user_id)

        if not result:
            return JSONResponse({"error": "User not found"}, status_code=404)

        return JSONResponse(
            {
                "user": {
                    "id": str(result.user.id),
                    "username": result.user.username,
                    "api_key_prefix": result.user.api_key_prefix,
                },
                "api_key": result.api_key,  # Show only once!
                "message": "API key reset. Save the new key - it will not be shown again.",
            }
        )


# =============================================================================
# Team Routes
# =============================================================================


async def list_teams(request: Request) -> JSONResponse:
    """List all teams."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    limit = int(request.query_params.get("limit", "100"))
    offset = int(request.query_params.get("offset", "0"))

    db = get_database()
    with db.session() as session:
        service = TeamService(session)
        teams = service.list_teams(limit=limit, offset=offset)

        return JSONResponse(
            {
                "teams": [
                    {
                        "id": str(t.id),
                        "name": t.name,
                        "company_id": t.company_id,
                        "created_at": t.created_at.isoformat(),
                    }
                    for t in teams
                ],
                "count": len(teams),
            }
        )


async def create_team(request: Request) -> JSONResponse:
    """Create a new team."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    name = data.get("name")
    company_id = data.get("company_id")
    if not name or not company_id:
        return JSONResponse(
            {"error": "name and company_id are required"}, status_code=400
        )

    settings = data.get("settings", {})

    db = get_database()
    with db.session() as session:
        service = TeamService(session)
        try:
            team = service.create_team(
                name=name,
                company_id=company_id,
                settings=settings,
            )
            return JSONResponse(
                {
                    "team": {
                        "id": str(team.id),
                        "name": team.name,
                        "company_id": team.company_id,
                    },
                    "message": "Team created",
                },
                status_code=201,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)


async def get_team(request: Request) -> JSONResponse:
    """Get a team by ID or name."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    team_id_or_name = request.path_params["team_id"]

    db = get_database()
    with db.session() as session:
        service = TeamService(session)

        # Try UUID first, then name
        try:
            team_uuid = uuid.UUID(team_id_or_name)
            team = service.get_team_by_id(team_uuid)
        except ValueError:
            team = service.get_team_by_name(team_id_or_name)

        if not team:
            return JSONResponse({"error": "Team not found"}, status_code=404)

        # Get members
        members = service.list_members(team.id)

        return JSONResponse(
            {
                "id": str(team.id),
                "name": team.name,
                "company_id": team.company_id,
                "created_at": team.created_at.isoformat(),
                "settings": team.settings,
                "members": [
                    {
                        "user_id": str(m.user_id),
                        "role": m.role,
                        "joined_at": m.joined_at.isoformat(),
                    }
                    for m in members
                ],
            }
        )


async def delete_team(request: Request) -> JSONResponse:
    """Delete a team."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    team_id_str = request.path_params["team_id"]
    try:
        team_id = uuid.UUID(team_id_str)
    except ValueError:
        return JSONResponse({"error": "Invalid team ID"}, status_code=400)

    db = get_database()
    with db.session() as session:
        service = TeamService(session)
        if service.delete_team(team_id):
            return JSONResponse({"message": "Team deleted"})
        return JSONResponse({"error": "Team not found"}, status_code=404)


async def add_team_member(request: Request) -> JSONResponse:
    """Add a user to a team."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    team_id_str = request.path_params["team_id"]
    try:
        team_id = uuid.UUID(team_id_str)
    except ValueError:
        return JSONResponse({"error": "Invalid team ID"}, status_code=400)

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    user_id_str = data.get("user_id")
    if not user_id_str:
        return JSONResponse({"error": "user_id is required"}, status_code=400)

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return JSONResponse({"error": "Invalid user_id"}, status_code=400)

    role = data.get("role", "member")

    db = get_database()
    with db.session() as session:
        service = TeamService(session)
        try:
            member = service.add_member(team_id=team_id, user_id=user_id, role=role)
            if member:
                return JSONResponse(
                    {
                        "member": {
                            "team_id": str(member.team_id),
                            "user_id": str(member.user_id),
                            "role": member.role,
                        },
                        "message": "Member added",
                    },
                    status_code=201,
                )
            return JSONResponse({"error": "Team or user not found"}, status_code=404)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)


async def remove_team_member(request: Request) -> JSONResponse:
    """Remove a user from a team."""
    if err := require_admin_auth(request):
        return err
    if err := require_database(request):
        return err

    team_id_str = request.path_params["team_id"]
    user_id_str = request.path_params["user_id"]

    try:
        team_id = uuid.UUID(team_id_str)
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        return JSONResponse({"error": "Invalid ID"}, status_code=400)

    db = get_database()
    with db.session() as session:
        service = TeamService(session)
        if service.remove_member(team_id, user_id):
            return JSONResponse({"message": "Member removed"})
        return JSONResponse({"error": "Member not found"}, status_code=404)


# =============================================================================
# Admin Info Route
# =============================================================================


async def admin_info(request: Request) -> JSONResponse:
    """Get admin API status."""
    if err := require_admin_auth(request):
        return err

    db_status = "not_configured"
    if is_database_initialized():
        db = get_database()
        db_status = "healthy" if db.health_check() else "unhealthy"

    return JSONResponse(
        {
            "status": "ok",
            "database": db_status,
            "version": "1.0.0",
        }
    )


async def debug_config(request: Request) -> JSONResponse:
    """Debug endpoint to check configuration (admin only)."""
    if err := require_admin_auth(request):
        return err

    import os

    # Check which LLM-related env vars are set (values masked for security)
    # Note: Only OPENROUTER_API_KEY is required - it handles both LLM and embeddings
    env_check = {
        "OPENROUTER_API_KEY": "set" if os.environ.get("OPENROUTER_API_KEY") else "NOT SET (required)",
        "CEMS_MEM0_MODEL": os.environ.get("CEMS_MEM0_MODEL", "openai/gpt-4o-mini (default)"),
        "CEMS_EMBEDDING_MODEL": os.environ.get("CEMS_EMBEDDING_MODEL", "openai/text-embedding-3-small (default)"),
        "CEMS_LLM_MODEL": os.environ.get("CEMS_LLM_MODEL", "anthropic/claude-3-haiku (default)"),
        "VECTOR_STORE": "pgvector (unified PostgreSQL)",
        # Legacy env vars (no longer required)
        "OPENAI_API_KEY": "set (legacy)" if os.environ.get("OPENAI_API_KEY") else "not set (not required)",
    }

    return JSONResponse({"config": env_check})


async def debug_llm_test(request: Request) -> JSONResponse:
    """Test LLM connectivity (admin only)."""
    if err := require_admin_auth(request):
        return err

    import os
    from openai import OpenAI

    results = {}

    # Test OpenRouter LLM
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'OpenRouter OK' in 3 words"}],
                max_tokens=20,
            )
            results["openrouter_llm"] = {
                "status": "ok",
                "response": response.choices[0].message.content,
            }
        except Exception as e:
            results["openrouter_llm"] = {"status": "error", "error": str(e)}

        # Test OpenRouter Embeddings
        try:
            response = client.embeddings.create(
                model="openai/text-embedding-3-small",
                input="test embedding via openrouter",
            )
            results["openrouter_embeddings"] = {
                "status": "ok",
                "dimensions": len(response.data[0].embedding),
            }
        except Exception as e:
            results["openrouter_embeddings"] = {"status": "error", "error": str(e)}
    else:
        results["openrouter_llm"] = {"status": "NOT CONFIGURED (required)"}
        results["openrouter_embeddings"] = {"status": "NOT CONFIGURED (required)"}

    return JSONResponse({"llm_tests": results})


async def cleanup_eval_data(request: Request) -> JSONResponse:
    """Delete all eval data from the database (admin only).

    DELETE /admin/eval/cleanup?source_prefix=project:longmemeval

    This is used by the eval script to clean up stale data before runs.
    """
    if err := require_admin_auth(request):
        return err

    source_prefix = request.query_params.get("source_prefix", "project:longmemeval")

    try:
        db = get_database()
        if not db:
            return JSONResponse({"error": "Database not initialized"}, status_code=500)

        async with db.async_session() as session:
            # Delete chunks first (foreign key constraint)
            chunk_result = await session.execute(
                text("""
                    DELETE FROM memory_chunks
                    WHERE document_id IN (
                        SELECT id FROM memory_documents
                        WHERE source_ref LIKE :prefix
                    )
                """),
                {"prefix": f"{source_prefix}%"},
            )
            chunks_deleted = chunk_result.rowcount

            # Delete documents
            doc_result = await session.execute(
                text("""
                    DELETE FROM memory_documents
                    WHERE source_ref LIKE :prefix
                """),
                {"prefix": f"{source_prefix}%"},
            )
            docs_deleted = doc_result.rowcount

            await session.commit()

        return JSONResponse({
            "success": True,
            "chunks_deleted": chunks_deleted,
            "documents_deleted": docs_deleted,
            "source_prefix": source_prefix,
        })

    except Exception as e:
        logger.error(f"Eval cleanup failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def database_stats(request: Request) -> JSONResponse:
    """Get database statistics (admin only).

    GET /admin/db/stats

    Returns counts and embedding dimensions for all tables.
    """
    if err := require_admin_auth(request):
        return err

    try:
        db = get_database()
        if not db:
            return JSONResponse({"error": "Database not initialized"}, status_code=500)

        stats = {}

        async with db.async_session() as session:
            # Count memories
            result = await session.execute(
                text("SELECT COUNT(*) FROM memories WHERE archived = FALSE")
            )
            stats["memories_active"] = result.scalar()

            result = await session.execute(text("SELECT COUNT(*) FROM memories"))
            stats["memories_total"] = result.scalar()

            # Check embedding dimension (sample first memory)
            result = await session.execute(
                text("SELECT array_length(embedding::real[], 1) FROM memories LIMIT 1")
            )
            dim = result.scalar()
            stats["embedding_dimension"] = dim

            # Count documents and chunks (new model)
            result = await session.execute(text("SELECT COUNT(*) FROM memory_documents"))
            stats["documents_total"] = result.scalar()

            result = await session.execute(text("SELECT COUNT(*) FROM memory_chunks"))
            stats["chunks_total"] = result.scalar()

            # Check chunk embedding dimension
            result = await session.execute(
                text("SELECT array_length(embedding::real[], 1) FROM memory_chunks LIMIT 1")
            )
            chunk_dim = result.scalar()
            stats["chunk_embedding_dimension"] = chunk_dim

            # Count users
            result = await session.execute(text("SELECT COUNT(*) FROM users"))
            stats["users_total"] = result.scalar()

        return JSONResponse({"stats": stats})

    except Exception as e:
        logger.error(f"Database stats failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Route Definitions
# =============================================================================

admin_routes = [
    # Admin info
    Route("/admin", admin_info, methods=["GET"]),
    Route("/admin/debug", debug_config, methods=["GET"]),
    Route("/admin/debug/llm", debug_llm_test, methods=["GET"]),
    Route("/admin/db/stats", database_stats, methods=["GET"]),
    # Eval cleanup
    Route("/admin/eval/cleanup", cleanup_eval_data, methods=["DELETE"]),
    # Users
    Route("/admin/users", list_users, methods=["GET"]),
    Route("/admin/users", create_user, methods=["POST"]),
    Route("/admin/users/{user_id}", get_user, methods=["GET"]),
    Route("/admin/users/{user_id}", update_user, methods=["PATCH"]),
    Route("/admin/users/{user_id}", delete_user, methods=["DELETE"]),
    Route("/admin/users/{user_id}/reset-key", reset_user_api_key, methods=["POST"]),
    # Teams
    Route("/admin/teams", list_teams, methods=["GET"]),
    Route("/admin/teams", create_team, methods=["POST"]),
    Route("/admin/teams/{team_id}", get_team, methods=["GET"]),
    Route("/admin/teams/{team_id}", delete_team, methods=["DELETE"]),
    Route("/admin/teams/{team_id}/members", add_team_member, methods=["POST"]),
    Route(
        "/admin/teams/{team_id}/members/{user_id}",
        remove_team_member,
        methods=["DELETE"],
    ),
]
