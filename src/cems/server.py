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
- POST /api/session/analyze - Analyze session transcripts
"""

import json
import logging
import os
from contextvars import ContextVar
from typing import Any

from cems.config import CEMSConfig
from cems.memory import CEMSMemory
from cems.scheduler import CEMSScheduler

logger = logging.getLogger(__name__)

# Context variables for per-request user identification (HTTP mode)
_request_user_id: ContextVar[str | None] = ContextVar("request_user_id", default=None)
_request_team_id: ContextVar[str | None] = ContextVar("request_team_id", default=None)

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

    # Get config - HTTP mode requires database for user management
    config = get_base_config()

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

                    # Set user context from database
                    user_id = user.username
                    # Get team from header (optional, to select team context)
                    team_id = request.headers.get("x-team-id")
            except Exception as e:
                return JSONResponse(
                    {"error": f"Database error: {e}"},
                    status_code=503,
                )

            # Set context variables
            user_token = _request_user_id.set(user_id)
            team_token = _request_team_id.set(team_id)

            try:
                response = await call_next(request)
                return response
            except Exception as e:
                raise
            finally:
                # Reset context variables
                _request_user_id.reset(user_token)
                _request_team_id.reset(team_token)

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

    async def api_memory_add(request: Request):
        """REST API endpoint to add a memory.

        POST /api/memory/add
        Body: {
            "content": "...",
            "category": "...",
            "scope": "personal|shared",
            "infer": true/false  (optional, default true),
            "source_ref": "project:org/repo"  (optional, for project-scoped recall),
            "ttl_hours": 24  (optional, memory expires after this many hours),
            "pinned": true/false  (optional, default false - pinned memories never auto-prune),
            "pin_reason": "..."  (optional, reason for pinning)
        }

        Note: Set infer=false for bulk imports (100-200ms vs 1-10s per memory).
        With infer=false, content is stored raw without LLM fact extraction.

        Use ttl_hours for short-term session memories that should auto-expire.
        Use pinned=true for important memories like gate rules or guidelines.
        """
        try:
            body = await request.json()
            content = body.get("content")
            if not content:
                return JSONResponse({"error": "content is required"}, status_code=400)

            category = body.get("category", "general")
            scope = body.get("scope", "personal")
            tags = body.get("tags", [])  # Optional: tags for organization
            # Gate rules must preserve exact pattern format, so disable LLM inference
            default_infer = category != "gate-rules"
            infer = body.get("infer", default_infer)
            source_ref = body.get("source_ref")  # e.g., "project:org/repo"
            ttl_hours = body.get("ttl_hours")  # Optional: memory expires after N hours
            pinned = body.get("pinned", False)  # Optional: pin memory (never auto-prune)
            pin_reason = body.get("pin_reason")  # Optional: reason for pinning

            memory = get_memory()
            result = memory.add(
                content,
                scope=scope,
                category=category,
                tags=tags,
                infer=infer,
                source_ref=source_ref,
                ttl_hours=ttl_hours,
                pinned=pinned,
                pin_reason=pin_reason,
            )

            return JSONResponse({
                "success": True,
                "result": result,
            })
        except Exception as e:
            logger.error(f"API memory_add error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory_search(request: Request):
        """REST API endpoint to search memories using enhanced retrieval pipeline.

        POST /api/memory/search
        Body: {
            "query": "...",
            "limit": 10,
            "scope": "personal|shared|both",
            "max_tokens": 2000,              # Token budget for results
            "enable_graph": true,            # Include graph traversal
            "enable_query_synthesis": true,  # LLM query expansion
            "raw": false,                    # Bypass filtering (debug mode)
            "project": "org/repo",           # Project ID for scoped boost
            "mode": "auto|vector|hybrid",    # NEW: Retrieval mode
            "enable_hyde": true,             # NEW: Use HyDE for better matching
            "enable_rerank": true            # NEW: Use LLM re-ranking
        }

        The enhanced pipeline (retrieve_for_inference) implements 9 stages:
        1. Query understanding (intent, domains, entities) - NEW
        2. Query synthesis (LLM expands query for better retrieval)
        3. HyDE (hypothetical document generation) - NEW
        4. Candidate retrieval (vector + graph search)
        5. RRF fusion (combine multi-query results) - NEW
        6. LLM re-ranking (smarter relevance) - NEW
        7. Relevance filtering (threshold-based)
        8. Unified scoring (time decay + priority + project)
        9. Token-budgeted assembly

        Modes:
        - "auto": Smart routing based on query analysis (default)
        - "vector": Fast path, minimal LLM calls
        - "hybrid": Full pipeline with HyDE + RRF + re-ranking

        Use raw=true for debugging to see all results without filtering.
        """
        try:
            body = await request.json()
            query = body.get("query")
            if not query:
                return JSONResponse({"error": "query is required"}, status_code=400)

            limit = body.get("limit", 10)
            scope = body.get("scope", "both")
            max_tokens = body.get("max_tokens", 2000)
            enable_graph = body.get("enable_graph", True)
            enable_query_synthesis = body.get("enable_query_synthesis", True)
            raw_mode = body.get("raw", False)
            project = body.get("project")  # e.g., "org/repo"
            # NEW parameters
            mode = body.get("mode", "auto")  # auto, vector, hybrid
            enable_hyde = body.get("enable_hyde", True)
            enable_rerank = body.get("enable_rerank", True)

            logger.info(f"[API] Search request: query='{query[:50]}...', mode={mode}, raw={raw_mode}")

            memory = get_memory()

            if raw_mode:
                # Debug mode: use raw search without filtering
                results = memory.search(query, scope=scope, limit=limit)
                serialized_results = [r.model_dump(mode="json") for r in results]
                logger.info(f"[API] Raw search: {len(serialized_results)} results")
                return JSONResponse({
                    "success": True,
                    "results": serialized_results,
                    "count": len(serialized_results),
                    "mode": "raw",
                })

            # Production mode: use enhanced retrieval pipeline
            result = memory.retrieve_for_inference(
                query=query,
                scope=scope,
                max_tokens=max_tokens,
                enable_query_synthesis=enable_query_synthesis,
                enable_graph=enable_graph,
                project=project,
                mode=mode,
                enable_hyde=enable_hyde,
                enable_rerank=enable_rerank,
            )

            # Apply limit to results
            limited_results = result["results"][:limit]

            logger.info(f"[API] Search complete: {len(limited_results)} results, mode={result.get('mode')}")

            return JSONResponse({
                "success": True,
                "results": limited_results,
                "count": len(limited_results),
                "mode": result.get("mode", "unified"),
                "tokens_used": result["tokens_used"],
                "queries_used": result["queries_used"],
                "total_candidates": result["total_candidates"],
                "filtered_count": result["filtered_count"],
                "intent": result.get("intent"),  # NEW: Return query intent for debugging
            })
        except Exception as e:
            logger.error(f"API memory_search error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory_gate_rules(request: Request):
        """REST API endpoint to fetch gate rules by category.

        GET /api/memory/gate-rules?project=org/repo

        Returns all memories with category='gate-rules', optionally filtered by project.
        This bypasses semantic search and queries the metadata store directly.

        Response: {
            "success": true,
            "rules": [
                {
                    "memory_id": "...",
                    "content": "Bash: coolify deploy — reason",
                    "category": "gate-rules",
                    "source_ref": "project:org/repo",
                    "pinned": true,
                    "tags": ["block", "coolify"]
                }
            ],
            "count": 1
        }
        """
        try:
            project = request.query_params.get("project")  # e.g., "org/repo"

            memory = get_memory()

            # Get gate-rules memory IDs from metadata store
            memory_ids = memory._metadata.get_memories_by_category(
                memory.config.user_id, "gate-rules"
            )

            if not memory_ids:
                return JSONResponse({
                    "success": True,
                    "rules": [],
                    "count": 0,
                })

            # Get metadata for all gate rules
            metadata_map = memory._metadata.get_metadata_batch(memory_ids)

            gate_rules = []
            for memory_id in memory_ids:
                meta = metadata_map.get(memory_id)
                if not meta:
                    continue

                # Filter by project if specified
                if project:
                    source_ref = meta.source_ref or ""
                    # Skip if source_ref is for a different project
                    if source_ref and source_ref.startswith("project:"):
                        if source_ref != f"project:{project}":
                            continue
                    # Global rules (no source_ref) are included for all projects

                # Get the memory content from Mem0
                try:
                    mem_data = memory._memory.get(memory_id)
                    content = mem_data.get("memory", "") if mem_data else ""
                except Exception:
                    content = ""

                gate_rules.append({
                    "memory_id": memory_id,
                    "content": content,
                    "category": meta.category,
                    "source_ref": meta.source_ref,
                    "pinned": meta.pinned,
                    "pin_reason": meta.pin_reason,
                    "tags": meta.tags or [],
                })

            logger.info(f"[API] Gate rules: found {len(gate_rules)} rules for project={project}")

            return JSONResponse({
                "success": True,
                "rules": gate_rules,
                "count": len(gate_rules),
            })
        except Exception as e:
            logger.error(f"API memory_gate_rules error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_session_analyze(request: Request):
        """REST API endpoint to analyze a session transcript and extract learnings.

        POST /api/session/analyze
        Body: {
            "transcript": "..." | [...],  # String or array of messages
            "session_id": "...",          # Session identifier
            "working_dir": "...",         # Optional: project context
            "tool_summary": {...}         # Optional: tools used, files changed
        }

        Response: {
            "success": true,
            "learnings_stored": 3,
            "learnings": [...],
            "skipped_reason": null
        }
        """
        from cems.llm import extract_session_learnings

        try:
            body = await request.json()
            transcript = body.get("transcript")
            if not transcript:
                return JSONResponse({"error": "transcript is required"}, status_code=400)

            session_id = body.get("session_id", "unknown")
            working_dir = body.get("working_dir")
            tool_summary = body.get("tool_summary")

            # Debug logging to help diagnose transcript issues
            transcript_preview = transcript[:500] if isinstance(transcript, str) else str(transcript)[:500]
            transcript_len = len(transcript) if isinstance(transcript, str) else len(str(transcript))
            logger.info(f"session_analyze called: session_id={session_id}, transcript_len={transcript_len}")
            logger.debug(f"session_analyze transcript preview: {transcript_preview}...")

            # Run analysis (synchronous since haiku is fast)
            learnings = extract_session_learnings(
                transcript=transcript,
                working_dir=working_dir,
                tool_summary=tool_summary,
            )

            if not learnings:
                return JSONResponse({
                    "success": True,
                    "learnings_stored": 0,
                    "learnings": [],
                    "skipped_reason": "no_significant_learnings",
                })

            # Store each learning as a memory
            memory = get_memory()
            stored_learnings = []

            for learning in learnings:
                try:
                    # Build content with type prefix for searchability
                    learning_type = learning.get("type", "GENERAL")
                    content = learning.get("content", "")
                    category = learning.get("category", "learnings")

                    # Format: [TYPE] Content (from session X)
                    formatted_content = f"[{learning_type}] {content}"
                    if session_id != "unknown":
                        formatted_content += f" (session: {session_id[:8]})"

                    # Use infer=False since learnings are already extracted by LLM
                    # This bypasses Mem0's fact extraction (much faster, no extra LLM calls)
                    result = memory.add(
                        content=formatted_content,
                        scope="personal",
                        category=category,
                        tags=["session-learning", learning_type.lower()],
                        infer=False,
                    )

                    # Extract memory ID
                    memory_id = None
                    if result and "results" in result:
                        for r in result["results"]:
                            if r.get("id"):
                                memory_id = r["id"]
                                break

                    stored_learnings.append({
                        "type": learning_type,
                        "content": content,
                        "memory_id": memory_id,
                        "category": category,
                    })

                except Exception as e:
                    logger.error(f"Failed to store learning: {e}")
                    continue

            return JSONResponse({
                "success": True,
                "learnings_stored": len(stored_learnings),
                "learnings": stored_learnings,
                "skipped_reason": None,
            })

        except Exception as e:
            logger.error(f"API session_analyze error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory_forget(request: Request):
        """REST API endpoint to forget (delete/archive) a memory.

        POST /api/memory/forget
        Body: {"memory_id": "...", "hard_delete": false}
        """
        try:
            body = await request.json()
            memory_id = body.get("memory_id")
            if not memory_id:
                return JSONResponse({"error": "memory_id is required"}, status_code=400)

            hard_delete = body.get("hard_delete", False)

            memory = get_memory()
            memory.delete(memory_id, hard=hard_delete)

            action = "deleted" if hard_delete else "archived"
            return JSONResponse({
                "success": True,
                "message": f"Memory {memory_id} {action}",
                "memory_id": memory_id,
            })
        except Exception as e:
            logger.error(f"API memory_forget error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory_update(request: Request):
        """REST API endpoint to update a memory.

        POST /api/memory/update
        Body: {"memory_id": "...", "content": "..."}
        """
        try:
            body = await request.json()
            memory_id = body.get("memory_id")
            content = body.get("content")

            if not memory_id:
                return JSONResponse({"error": "memory_id is required"}, status_code=400)
            if not content:
                return JSONResponse({"error": "content is required"}, status_code=400)

            memory = get_memory()
            memory.update(memory_id, content)

            return JSONResponse({
                "success": True,
                "message": f"Memory {memory_id} updated",
                "memory_id": memory_id,
            })
        except Exception as e:
            logger.error(f"API memory_update error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory_maintenance(request: Request):
        """REST API endpoint to run maintenance jobs.

        POST /api/memory/maintenance
        Body: {"job_type": "consolidation|summarization|reindex|all"}
        """
        try:
            body = await request.json()
            job_type = body.get("job_type", "consolidation")

            scheduler = get_scheduler()

            if job_type == "all":
                results = {}
                for jt in ["consolidation", "summarization", "reindex"]:
                    results[jt] = scheduler.run_now(jt)
                return JSONResponse({
                    "success": True,
                    "job_type": "all",
                    "results": results,
                })
            else:
                result = scheduler.run_now(job_type)
                return JSONResponse({
                    "success": True,
                    "job_type": job_type,
                    "results": result,
                })
        except Exception as e:
            logger.error(f"API memory_maintenance error: {e}")
            return JSONResponse({
                "success": False,
                "job_type": job_type,
                "error": str(e),
            }, status_code=500)

    async def api_memory_status(request: Request):
        """REST API endpoint to get system status.

        GET /api/memory/status
        """
        try:
            memory = get_memory()
            config = memory.config
            
            # Get actual scheduler state from global scheduler (not per-user config)
            scheduler_running = False
            scheduler_jobs = []
            if "default" in _scheduler_cache:
                scheduler = _scheduler_cache["default"]
                scheduler_running = scheduler.is_running
                if scheduler_running:
                    scheduler_jobs = scheduler.get_jobs()

            status = {
                "status": "healthy",
                "user_id": config.user_id,
                "team_id": config.team_id,
                "storage_dir": str(config.storage_dir),  # Convert Path to string for JSON
                "backend": config.memory_backend,
                "vector_store": config.vector_store,
                "graph_store": config.graph_store if config.enable_graph else None,
                "scheduler": scheduler_running,
                "scheduler_jobs": scheduler_jobs,
                "query_synthesis": config.enable_query_synthesis,
                "relevance_threshold": config.relevance_threshold,
                "max_tokens": config.default_max_tokens,
            }

            # Add graph stats if enabled
            if memory.graph_store:
                stats = memory.get_graph_stats()
                status["graph_stats"] = stats

            return JSONResponse(status)
        except Exception as e:
            logger.error(f"API memory_status error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory_summary_personal(request: Request):
        """REST API endpoint to get personal memory summary.

        GET /api/memory/summary/personal
        """
        try:
            memory = get_memory()
            # Use efficient GROUP BY query instead of N+1 queries
            categories = memory.get_category_counts(scope="personal")
            total = sum(categories.values())

            return JSONResponse({
                "success": True,
                "total": total,
                "categories": categories,
            })
        except Exception as e:
            logger.error(f"API memory_summary_personal error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory_summary_shared(request: Request):
        """REST API endpoint to get shared memory summary.

        GET /api/memory/summary/shared
        """
        try:
            memory = get_memory()

            if not memory.config.team_id:
                return JSONResponse({
                    "success": True,
                    "total": 0,
                    "categories": {},
                    "message": "No team configured",
                })

            # Use efficient GROUP BY query instead of N+1 queries
            categories = memory.get_category_counts(scope="shared")
            total = sum(categories.values())

            return JSONResponse({
                "success": True,
                "total": total,
                "categories": categories,
                "team_id": memory.config.team_id,
            })
        except Exception as e:
            logger.error(f"API memory_summary_shared error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    # Create pure REST API app (MCP protocol handled by Express wrapper)
    from starlette.applications import Starlette
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    routes = [
        # Health check endpoints
        Route("/ping", ping, methods=["GET"]),
        Route("/health", health_check, methods=["GET"]),
        # REST API routes
        Route("/api/memory/add", api_memory_add, methods=["POST"]),
        Route("/api/memory/search", api_memory_search, methods=["POST"]),
        Route("/api/memory/forget", api_memory_forget, methods=["POST"]),
        Route("/api/memory/update", api_memory_update, methods=["POST"]),
        Route("/api/memory/maintenance", api_memory_maintenance, methods=["POST"]),
        Route("/api/memory/status", api_memory_status, methods=["GET"]),
        Route("/api/memory/gate-rules", api_memory_gate_rules, methods=["GET"]),
        Route("/api/memory/summary/personal", api_memory_summary_personal, methods=["GET"]),
        Route("/api/memory/summary/shared", api_memory_summary_shared, methods=["GET"]),
        Route("/api/session/analyze", api_session_analyze, methods=["POST"]),
    ]
    logger.info("REST API routes enabled (/api/memory/*, /api/session/*)")

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
    logger.info(f"Vector store: {config.vector_store}")
    logger.info(f"Qdrant URL: {config.qdrant_url}")
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
