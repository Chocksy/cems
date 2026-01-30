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
            "pin_reason": "..."  (optional, reason for pinning),
            "timestamp": "2023-04-10T17:50:00Z"  (optional, ISO format for historical imports)
        }

        Note: Set infer=false for bulk imports (100-200ms vs 1-10s per memory).
        With infer=false, content is stored raw without LLM fact extraction.

        Use ttl_hours for short-term session memories that should auto-expire.
        Use pinned=true for important memories like gate rules or guidelines.
        Use timestamp for historical imports (e.g., eval benchmarks with event dates).
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
            timestamp_str = body.get("timestamp")  # Optional: historical timestamp (ISO format)

            # Parse timestamp if provided
            timestamp = None
            if timestamp_str:
                from datetime import datetime
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except ValueError:
                    return JSONResponse({"error": "Invalid timestamp format. Use ISO format."}, status_code=400)

            memory = get_memory()
            result = await memory.add_async(
                content,
                scope=scope,
                category=category,
                tags=tags,
                infer=infer,
                source_ref=source_ref,
                ttl_hours=ttl_hours,
                pinned=pinned,
                pin_reason=pin_reason,
                timestamp=timestamp,
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
            max_tokens = body.get("max_tokens", 4000)
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
                results = await memory.search_async(query, scope=scope, limit=limit)
                serialized_results = [r.model_dump(mode="json") for r in results]
                logger.info(f"[API] Raw search: {len(serialized_results)} results")
                return JSONResponse({
                    "success": True,
                    "results": serialized_results,
                    "count": len(serialized_results),
                    "mode": "raw",
                })

            # Production mode: use enhanced retrieval pipeline
            result = await memory.retrieve_for_inference_async(
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

    async def api_memory_profile(request: Request):
        """REST API endpoint to get session profile context.

        GET /api/memory/profile?project=org/repo&token_budget=2500

        Returns pre-formatted context string for session start injection:
        - User preferences and guidelines
        - Recent relevant memories (last 24h)
        - Gate rules summary
        - Project-specific context if applicable

        Response: {
            "success": true,
            "context": "Pre-formatted context string...",
            "components": {
                "preferences": 3,
                "guidelines": 2,
                "recent_memories": 5,
                "gate_rules_count": 4
            },
            "token_estimate": 1850
        }
        """
        try:
            project = request.query_params.get("project")
            token_budget = int(request.query_params.get("token_budget", "2500"))

            memory = get_memory()
            await memory._ensure_initialized_async()

            user_id = memory.config.user_id
            components = {
                "preferences": [],
                "guidelines": [],
                "recent_memories": [],
                "gate_rules_count": 0,
                "project_context": [],
            }

            # 1. Fetch preferences (category: preferences)
            prefs = await memory._vectorstore.search_by_category(
                user_id=user_id,
                category="preferences",
                limit=10,
            )
            components["preferences"] = prefs

            # 2. Fetch guidelines (category: guidelines)
            guidelines = await memory._vectorstore.search_by_category(
                user_id=user_id,
                category="guidelines",
                limit=10,
            )
            components["guidelines"] = guidelines

            # 3. Fetch recent memories (last 24h, any category)
            recent = await memory._vectorstore.get_recent(
                user_id=user_id,
                hours=24,
                limit=15,
            )
            # Filter out preferences/guidelines already included
            recent = [
                m for m in recent
                if m.get("category") not in ("preferences", "guidelines", "gate-rules")
            ]
            components["recent_memories"] = recent

            # 4. Gate rules count (for awareness, not full rules)
            gate_rules = await memory._vectorstore.search_by_category(
                user_id=user_id,
                category="gate-rules",
                limit=50,
            )
            components["gate_rules_count"] = len(gate_rules)

            # 5. Project-specific memories if applicable
            if project:
                # Search for memories with matching source_ref
                project_memories = await memory._vectorstore.search_by_category(
                    user_id=user_id,
                    category="project",  # Project-specific category
                    limit=10,
                )
                # Filter by source_ref prefix (handle None values)
                project_memories = [
                    m for m in project_memories
                    if (m.get("source_ref") or "").startswith(f"project:{project}")
                ]
                components["project_context"] = project_memories

            # Build formatted context string
            context_parts = []

            if components["preferences"]:
                pref_list = [f"- {m['content']}" for m in components["preferences"][:5]]
                context_parts.append("## Your Preferences\n" + "\n".join(pref_list))

            if components["guidelines"]:
                guide_list = [f"- {m['content']}" for m in components["guidelines"][:5]]
                context_parts.append("## Guidelines\n" + "\n".join(guide_list))

            if components["gate_rules_count"] > 0:
                context_parts.append(
                    f"## Gate Rules\n{components['gate_rules_count']} gate rules active "
                    "(checked automatically on tool use)"
                )

            if components["recent_memories"]:
                recent_list = [
                    f"- [{m.get('category', 'general')}] {m['content'][:100]}..."
                    if len(m.get('content', '')) > 100 else f"- [{m.get('category', 'general')}] {m['content']}"
                    for m in components["recent_memories"][:5]
                ]
                context_parts.append("## Recent Context (last 24h)\n" + "\n".join(recent_list))

            if components["project_context"]:
                proj_list = [f"- {m['content'][:100]}..." for m in components["project_context"][:3]]
                context_parts.append(f"## Project Context ({project})\n" + "\n".join(proj_list))

            context = "\n\n".join(context_parts) if context_parts else ""

            # Estimate tokens (rough: 4 chars per token)
            token_estimate = len(context) // 4

            # Truncate if over budget
            if token_estimate > token_budget and context:
                # Simple truncation at roughly token_budget * 4 chars
                max_chars = token_budget * 4
                context = context[:max_chars] + "\n\n[...truncated]"
                token_estimate = token_budget

            logger.info(
                f"[API] Profile: {len(components['preferences'])} prefs, "
                f"{len(components['guidelines'])} guidelines, "
                f"{len(components['recent_memories'])} recent, "
                f"{components['gate_rules_count']} rules, "
                f"~{token_estimate} tokens"
            )

            return JSONResponse({
                "success": True,
                "context": context,
                "components": {
                    "preferences": len(components["preferences"]),
                    "guidelines": len(components["guidelines"]),
                    "recent_memories": len(components["recent_memories"]),
                    "gate_rules_count": components["gate_rules_count"],
                    "project_context": len(components["project_context"]),
                },
                "token_estimate": token_estimate,
            })

        except Exception as e:
            logger.error(f"API memory_profile error: {e}", exc_info=True)
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
                    result = await memory.add_async(
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

    async def api_tool_learning(request: Request):
        """REST API endpoint for incremental tool-based learning.

        POST /api/tool/learning
        Body: {
            "tool_name": "Edit",           # Tool that was used
            "tool_input": {...},           # Tool input (file_path, etc.)
            "tool_output": "...",          # Tool output summary
            "session_id": "...",           # Session identifier
            "context_snippet": "...",      # Recent conversation context
            "working_dir": "..."           # Optional: project context
        }

        This endpoint is designed for SuperMemory-style incremental learning:
        - Called after significant tool completions (Edit, Write, Bash with commits)
        - Extracts quick learnings without full session analysis
        - Uses fast LLM calls (haiku) for immediate processing

        Response: {
            "success": true,
            "stored": true | false,
            "memory_id": "...",
            "reason": "stored" | "skipped_not_learnable" | "skipped_too_brief"
        }
        """
        from cems.llm import extract_tool_learning

        try:
            body = await request.json()
            tool_name = body.get("tool_name")
            tool_input = body.get("tool_input", {})
            tool_output = body.get("tool_output", "")
            session_id = body.get("session_id", "unknown")
            context_snippet = body.get("context_snippet", "")
            working_dir = body.get("working_dir")

            if not tool_name:
                return JSONResponse({"error": "tool_name is required"}, status_code=400)

            # Skip non-significant tools (reads, searches don't produce learnings)
            non_learnable_tools = {"Read", "Glob", "Grep", "LS", "WebFetch", "WebSearch"}
            if tool_name in non_learnable_tools:
                return JSONResponse({
                    "success": True,
                    "stored": False,
                    "memory_id": None,
                    "reason": "skipped_non_learnable_tool",
                })

            # Skip if no meaningful context
            if not context_snippet and not tool_output:
                return JSONResponse({
                    "success": True,
                    "stored": False,
                    "memory_id": None,
                    "reason": "skipped_no_context",
                })

            # Build tool context for learning extraction
            tool_context = f"Tool: {tool_name}\n"
            if tool_input:
                if tool_name == "Edit" and "file_path" in tool_input:
                    tool_context += f"File: {tool_input['file_path']}\n"
                elif tool_name == "Write" and "file_path" in tool_input:
                    tool_context += f"Created: {tool_input['file_path']}\n"
                elif tool_name == "Bash" and "command" in tool_input:
                    cmd = tool_input.get("command", "")[:200]
                    desc = tool_input.get("description", "")
                    tool_context += f"Command: {desc or cmd}\n"
            if tool_output:
                tool_context += f"Result: {tool_output[:500]}\n"

            # Extract learning from tool usage
            learning = extract_tool_learning(
                tool_context=tool_context,
                conversation_snippet=context_snippet,
                working_dir=working_dir,
            )

            if not learning:
                return JSONResponse({
                    "success": True,
                    "stored": False,
                    "memory_id": None,
                    "reason": "skipped_no_learning_extracted",
                })

            # Store the learning
            memory = get_memory()
            content = learning.get("content", "")
            category = learning.get("category", "learnings")
            learning_type = learning.get("type", "TOOL")

            # Format with type prefix
            formatted_content = f"[{learning_type}] {content}"
            if session_id != "unknown":
                formatted_content += f" (session: {session_id[:8]})"

            result = await memory.add_async(
                content=formatted_content,
                scope="personal",
                category=category,
                tags=["tool-learning", tool_name.lower(), learning_type.lower()],
                infer=False,
            )

            # Extract memory ID
            memory_id = None
            if result and "results" in result:
                for r in result["results"]:
                    if r.get("id"):
                        memory_id = r["id"]
                        break

            logger.info(f"Tool learning stored: {tool_name} -> {content[:50]}...")

            return JSONResponse({
                "success": True,
                "stored": True,
                "memory_id": memory_id,
                "reason": "stored",
                "learning": {
                    "type": learning_type,
                    "content": content,
                    "category": category,
                },
            })

        except Exception as e:
            logger.error(f"API tool_learning error: {e}", exc_info=True)
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
            await memory.delete_async(memory_id, hard=hard_delete)

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
            await memory.update_async(memory_id, content)

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
                "vector_store": "pgvector",  # Using native pgvector
                "graph_store": "postgresql" if config.enable_graph else None,
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
            categories = await memory.get_category_counts_async(scope="personal")
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
            categories = await memory.get_category_counts_async(scope="shared")
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
        Route("/api/memory/profile", api_memory_profile, methods=["GET"]),
        Route("/api/memory/summary/personal", api_memory_summary_personal, methods=["GET"]),
        Route("/api/memory/summary/shared", api_memory_summary_shared, methods=["GET"]),
        Route("/api/session/analyze", api_session_analyze, methods=["POST"]),
        Route("/api/tool/learning", api_tool_learning, methods=["POST"]),
    ]
    logger.info("REST API routes enabled (/api/memory/*, /api/session/*, /api/tool/*)")

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
