"""Memory API handlers.

All memory-related REST API endpoints:
- add, add_batch, search, gate_rules, profile, forget, update
- maintenance, status, summary_personal, summary_shared
"""

import logging
from datetime import datetime
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import _scheduler_cache, get_memory
from cems.chunking import Chunk, chunk_document

logger = logging.getLogger(__name__)

FOUNDATION_GUIDELINE_TAGS = {"foundation", "constitution"}
FOUNDATION_SOURCE_REF_PREFIX = "foundation:constitution"


def _normalize_tags(raw_tags: Any) -> set[str]:
    """Normalize tags to lowercase strings for reliable matching."""
    if not isinstance(raw_tags, list):
        return set()
    return {
        str(tag).strip().lower()
        for tag in raw_tags
        if isinstance(tag, str) and str(tag).strip()
    }


def _is_foundation_guideline(doc: dict[str, Any]) -> bool:
    """Return True when a guideline should be treated as foundational."""
    tags = _normalize_tags(doc.get("tags"))
    if tags & FOUNDATION_GUIDELINE_TAGS:
        return True

    source_ref = str(doc.get("source_ref") or "").strip().lower()
    return source_ref.startswith(FOUNDATION_SOURCE_REF_PREFIX)


async def api_memory_add(request: Request):
    """REST API endpoint to add a memory.

    POST /api/memory/add
    Body: {
        "content": "...",
        "category": "...",
        "scope": "personal|shared",
        "source_ref": "project:org/repo"  (optional, for project-scoped recall),
        "tags": ["tag1", "tag2"]  (optional),
        "timestamp": "2023-04-10T17:50:00Z"  (optional, ISO format for historical imports)
    }

    Content is stored as-is using the document+chunk model (no LLM fact extraction).
    The "infer" parameter is accepted but ignored (kept for API backwards compatibility).
    Use timestamp for historical imports (e.g., eval benchmarks with event dates).
    """
    try:
        body = await request.json()
        content = body.get("content")
        if not content:
            return JSONResponse({"error": "content is required"}, status_code=400)

        from cems.llm import normalize_category

        raw_category = body.get("category", "general")
        # Functional categories bypass normalization (exact format required)
        functional = {"gate-rules", "guidelines", "preferences", "category-summary"}
        category = raw_category if raw_category in functional else normalize_category(raw_category)
        scope = body.get("scope", "personal")
        tags = body.get("tags", [])  # Optional: tags for organization
        # Gate rules must preserve exact pattern format, so disable LLM inference
        default_infer = category != "gate-rules"
        infer = body.get("infer", default_infer)
        source_ref = body.get("source_ref")
        timestamp_str = body.get("timestamp")

        # Parse timestamp if provided
        timestamp = None
        if timestamp_str:
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
            timestamp=timestamp,
        )

        return JSONResponse({
            "success": True,
            "result": result,
        })
    except Exception as e:
        logger.error(f"API memory_add error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_memory_add_batch(request: Request):
    """REST API endpoint to add multiple memories in a single request.

    POST /api/memory/add_batch
    Body: {
        "memories": [
            {
                "content": "...",
                "category": "...",
                "source_ref": "project:org/repo:id",
                "tags": ["tag1", "tag2"],
                "timestamp": "2023-04-10T17:50:00Z"  (optional)
            },
            ...
        ]
    }

    Scope is determined by the first memory's "scope" field (default: "personal")
    and applied uniformly to the entire batch.

    This endpoint is optimized for bulk ingestion (e.g., eval benchmarks):
    - Single HTTP request for many memories
    - Batched embedding (100 texts per API call)
    - Single database transaction for all documents
    - Returns document IDs for all memories

    Response: {
        "success": true,
        "document_ids": ["id1", "id2", ...],
        "count": 100,
        "new_count": 95,
        "duplicate_count": 5,
        "total_chunks": 150
    }
    """
    try:
        body = await request.json()
        memories = body.get("memories", [])

        if not memories:
            return JSONResponse({"error": "No memories provided"}, status_code=400)

        if not isinstance(memories, list):
            return JSONResponse({"error": "memories must be an array"}, status_code=400)

        # Validate all memories have content
        for i, mem in enumerate(memories):
            if not mem.get("content"):
                return JSONResponse(
                    {"error": f"Memory at index {i} is missing 'content'"},
                    status_code=400,
                )

        logger.info(f"[API] Batch add request: {len(memories)} memories")

        memory = get_memory()
        await memory._ensure_initialized_async()

        # Get the document store and embedder
        from cems.db.document_store import DocumentStore

        doc_store: DocumentStore = await memory._ensure_document_store()
        embedder = memory._async_embedder

        if not embedder:
            return JSONResponse(
                {"error": "Embedding client not initialized"},
                status_code=500,
            )

        # Step 1: Chunk all documents
        all_chunks: list[list[Chunk]] = []
        all_texts: list[str] = []  # Flat list of all chunk texts for batch embedding
        chunk_counts: list[int] = []  # Track how many chunks per document

        for mem in memories:
            chunks = chunk_document(mem["content"])
            if not chunks:
                # Empty content - create a single empty chunk placeholder
                chunks = [Chunk(seq=0, pos=0, content="", tokens=0, bytes=0)]
            all_chunks.append(chunks)
            chunk_counts.append(len(chunks))
            for chunk in chunks:
                all_texts.append(chunk.content)

        logger.info(f"[API] Chunked {len(memories)} memories into {len(all_texts)} total chunks")

        # Step 2: Batch embed all chunks in one call (internally batches at 100)
        all_embeddings_flat = await embedder.embed_batch(all_texts)

        if len(all_embeddings_flat) != len(all_texts):
            return JSONResponse(
                {
                    "error": f"Embedding count mismatch: expected {len(all_texts)}, got {len(all_embeddings_flat)}"
                },
                status_code=500,
            )

        # Step 3: Reconstruct embeddings per document
        all_embeddings: list[list[list[float]]] = []
        idx = 0
        for count in chunk_counts:
            all_embeddings.append(all_embeddings_flat[idx : idx + count])
            idx += count

        # Step 4: Prepare documents for batch insert
        documents: list[dict] = []
        for mem in memories:
            doc = {
                "content": mem["content"],
                "category": mem.get("category", "general"),
                "source_ref": mem.get("source_ref"),
                "tags": mem.get("tags"),
                "title": mem.get("title"),
                "source": mem.get("source"),
            }
            documents.append(doc)

        # Determine scope (use first memory's scope, default to personal)
        scope = memories[0].get("scope", "personal") if memories else "personal"

        # Step 5: Batch insert into database
        user_id = memory.config.user_id
        team_id = memory.config.team_id if scope == "shared" else None

        if not user_id:
            return JSONResponse({"error": "No user_id configured"}, status_code=500)

        results = await doc_store.add_documents_batch(
            documents=documents,
            all_chunks=all_chunks,
            all_embeddings=all_embeddings,
            user_id=user_id,
            team_id=team_id,
            scope=scope,
        )

        # Collect stats
        document_ids = [doc_id for doc_id, _ in results]
        new_count = sum(1 for _, is_new in results if is_new)
        duplicate_count = len(results) - new_count
        total_chunks = sum(len(c) for c in all_chunks)

        logger.info(
            f"[API] Batch add complete: {new_count} new, {duplicate_count} duplicates, "
            f"{total_chunks} chunks"
        )

        return JSONResponse({
            "success": True,
            "document_ids": document_ids,
            "count": len(document_ids),
            "new_count": new_count,
            "duplicate_count": duplicate_count,
            "total_chunks": total_chunks,
        })

    except Exception as e:
        logger.error(f"API memory_add_batch error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


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

        # Server-side query cap: embedding models work best with focused text.
        # text-embedding-3-small has 8191 token limit (~32K chars), but
        # effectiveness degrades well before that. Cap at 2000 chars.
        if len(query) > 2000:
            query = query[:2000]

        limit = body.get("limit", 10)
        scope = body.get("scope", "both")
        max_tokens = body.get("max_tokens", 4000)
        enable_graph = body.get("enable_graph", True)
        enable_query_synthesis = body.get("enable_query_synthesis", False)
        raw_mode = body.get("raw", False)
        project = body.get("project")
        mode = body.get("mode", "vector")
        enable_hyde = body.get("enable_hyde", False)
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

        # Log intent analysis if present
        intent = result.get("intent")
        if intent:
            logger.info(
                f"[API] Search intent: type={intent.get('primary_intent')}, "
                f"complexity={intent.get('complexity')}, mode_selected={result.get('mode')}"
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
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_memory_gate_rules(request: Request):
    """REST API endpoint to fetch gate rules by category.

    GET /api/memory/gate-rules?project=org/repo

    Returns all documents with category='gate-rules', optionally filtered by project.
    Queries the document store directly (no semantic search).

    Response: {
        "success": true,
        "rules": [
            {
                "memory_id": "...",
                "content": "Bash: coolify deploy — reason",
                "category": "gate-rules",
                "source_ref": "project:org/repo",
                "tags": ["block", "coolify"]
            }
        ],
        "count": 1
    }
    """
    try:
        project = request.query_params.get("project")  # e.g., "org/repo"

        memory = get_memory()
        await memory._ensure_initialized_async()
        doc_store = await memory._ensure_document_store()

        user_id = memory.config.user_id

        # Query document store for gate-rules category
        source_ref_prefix = f"project:{project}" if project else None
        docs = await doc_store.get_documents_by_category(
            user_id=user_id,
            category="gate-rules",
            limit=100,
            source_ref_prefix=source_ref_prefix,
        )

        # If project filter was used, also include global rules (no source_ref)
        if project:
            all_docs = await doc_store.get_documents_by_category(
                user_id=user_id,
                category="gate-rules",
                limit=100,
            )
            # Add global rules (those without project-specific source_ref)
            seen_ids = {d["id"] for d in docs}
            for doc in all_docs:
                if doc["id"] not in seen_ids:
                    source_ref = doc.get("source_ref") or ""
                    # Include if global (no source_ref or not project-scoped)
                    if not source_ref or not source_ref.startswith("project:"):
                        docs.append(doc)

        gate_rules = []
        for doc in docs:
            gate_rules.append({
                "memory_id": doc["id"],
                "content": doc["content"],
                "category": doc["category"],
                "source_ref": doc.get("source_ref"),
                "tags": doc.get("tags", []),
            })

        logger.info(f"[API] Gate rules: found {len(gate_rules)} rules for project={project}")

        return JSONResponse({
            "success": True,
            "rules": gate_rules,
            "count": len(gate_rules),
        })
    except Exception as e:
        logger.error(f"API memory_gate_rules error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_memory_foundation(request: Request):
    """REST API endpoint to fetch foundation guidelines.

    GET /api/memory/foundation?project=org/repo

    Returns all guidelines tagged as foundational (via 'foundation' or
    'constitution' tags, or 'foundation:constitution' source_ref prefix).
    Queries the document store directly (no semantic search).

    Response: {
        "success": true,
        "guidelines": [
            {
                "memory_id": "...",
                "content": "...",
                "category": "guidelines",
                "source_ref": "...",
                "tags": ["foundation"]
            }
        ],
        "count": 1
    }
    """
    try:
        project = request.query_params.get("project")

        memory = get_memory()
        await memory._ensure_initialized_async()
        doc_store = await memory._ensure_document_store()

        user_id = memory.config.user_id

        # Fetch all guidelines (foundation lives under category="guidelines")
        docs = await doc_store.get_documents_by_category(
            user_id=user_id,
            category="guidelines",
            limit=50,
        )

        # Also fetch project-scoped guidelines if project is specified
        if project:
            project_docs = await doc_store.get_documents_by_category(
                user_id=user_id,
                category="guidelines",
                limit=50,
                source_ref_prefix=f"project:{project}",
            )
            seen_ids = {d["id"] for d in docs}
            for doc in project_docs:
                if doc["id"] not in seen_ids:
                    docs.append(doc)

        # Filter to foundation guidelines only
        foundation = []
        for doc in docs:
            if _is_foundation_guideline(doc):
                foundation.append({
                    "memory_id": doc["id"],
                    "content": doc["content"],
                    "category": doc["category"],
                    "source_ref": doc.get("source_ref"),
                    "tags": doc.get("tags", []),
                })

        logger.info(f"[API] Foundation: found {len(foundation)} guidelines for project={project}")

        return JSONResponse({
            "success": True,
            "guidelines": foundation,
            "count": len(foundation),
        })
    except Exception as e:
        logger.error(f"API memory_foundation error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


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
            "foundation_guidelines": 2,
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
        doc_store = await memory._ensure_document_store()

        user_id = memory.config.user_id
        components = {
            "preferences": [],
            "foundation_guidelines": [],
            "all_guidelines": [],  # Foundation + non-foundation combined
            "recent_memories": [],
            "gate_rules_count": 0,
            "project_context": [],
        }

        # 1. Fetch preferences (category: preferences)
        prefs = await doc_store.get_documents_by_category(
            user_id=user_id,
            category="preferences",
            limit=10,
        )
        components["preferences"] = prefs

        # 2. Fetch guidelines (category: guidelines)
        guidelines = await doc_store.get_documents_by_category(
            user_id=user_id,
            category="guidelines",
            limit=25,
        )
        foundation_guidelines = [g for g in guidelines if _is_foundation_guideline(g)]
        non_foundation_guidelines = [g for g in guidelines if not _is_foundation_guideline(g)]
        components["foundation_guidelines"] = foundation_guidelines[:10]
        components["all_guidelines"] = (foundation_guidelines + non_foundation_guidelines)[:10]

        # 3. Fetch recent memories (last 24h, excluding certain categories)
        recent = await doc_store.get_recent_documents(
            user_id=user_id,
            hours=24,
            limit=15,
            exclude_categories=["preferences", "guidelines", "gate-rules"],
        )
        components["recent_memories"] = recent

        # 4. Gate rules count (for awareness, not full rules)
        gate_rules = await doc_store.get_documents_by_category(
            user_id=user_id,
            category="gate-rules",
            limit=50,
        )
        components["gate_rules_count"] = len(gate_rules)

        # 5. Project-specific memories if applicable
        if project:
            project_memories = await doc_store.get_documents_by_category(
                user_id=user_id,
                category="project",
                limit=10,
                source_ref_prefix=f"project:{project}",
            )
            components["project_context"] = project_memories

        # Build formatted context string
        context_parts = []

        if components["preferences"]:
            pref_list = [f"- {m['content']}" for m in components["preferences"][:5]]
            context_parts.append("## Your Preferences\n" + "\n".join(pref_list))

        if components["foundation_guidelines"]:
            foundation_list = [f"- {m['content']}" for m in components["foundation_guidelines"][:5]]
            context_parts.append("## Foundational Principles\n" + "\n".join(foundation_list))

        if components["all_guidelines"]:
            regular_guidelines = [
                m for m in components["all_guidelines"] if not _is_foundation_guideline(m)
            ]
            if regular_guidelines:
                guide_list = [f"- {m['content']}" for m in regular_guidelines[:5]]
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

        # 6. Memory conflicts (unresolved)
        try:
            conflicts = await doc_store.get_open_conflicts(user_id, limit=3)
        except Exception:
            conflicts = []  # Table may not exist yet
        if conflicts:
            conflict_lines = []
            for c in conflicts:
                conflict_lines.append(
                    f"- **Conflict** (id: {c['id'][:8]}): {c['explanation']}"
                )
            context_parts.append(
                "## Memory Conflicts Detected\n"
                + "\n".join(conflict_lines)
                + "\n\nResolve via: POST /api/memory/conflict/resolve"
            )

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
            f"{len(components['foundation_guidelines'])} foundation guidelines, "
            f"{len(components['all_guidelines'])} guidelines, "
            f"{len(components['recent_memories'])} recent, "
            f"{components['gate_rules_count']} rules, "
            f"~{token_estimate} tokens"
        )

        return JSONResponse({
            "success": True,
            "context": context,
            "components": {
                "preferences": len(components["preferences"]),
                "foundation_guidelines": len(components["foundation_guidelines"]),
                "guidelines": len(components["all_guidelines"]),
                "recent_memories": len(components["recent_memories"]),
                "gate_rules_count": components["gate_rules_count"],
                "project_context": len(components["project_context"]),
            },
            "token_estimate": token_estimate,
        })

    except Exception as e:
        logger.error(f"API memory_profile error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


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

        action = "deleted" if hard_delete else "soft-deleted"
        return JSONResponse({
            "success": True,
            "message": f"Memory {memory_id} {action}",
            "memory_id": memory_id,
        })
    except Exception as e:
        logger.error(f"API memory_forget error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_memory_get(request: Request):
    """REST API endpoint to retrieve a full document by ID.

    GET /api/memory/get?id=<memory_id>

    Returns the full document content, category, tags, and metadata.
    Used by LLMs to fetch full context after seeing a snippet in search results.
    """
    try:
        memory_id = request.query_params.get("id")
        if not memory_id:
            return JSONResponse({"error": "id query param is required"}, status_code=400)

        memory = get_memory()
        doc_store = await memory._ensure_document_store()
        doc = await doc_store.get_document(memory_id)

        if not doc:
            return JSONResponse({"error": "Document not found"}, status_code=404)

        return JSONResponse({
            "success": True,
            "document": {
                "id": doc["id"],
                "content": doc.get("content", ""),
                "category": doc.get("category", "general"),
                "source_ref": doc.get("source_ref"),
                "tags": doc.get("tags", []),
                "created_at": str(doc.get("created_at", "")),
                "updated_at": str(doc.get("updated_at", "")),
            },
        })
    except Exception as e:
        logger.error(f"API memory_get error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


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
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_memory_maintenance(request: Request):
    """REST API endpoint to run maintenance jobs.

    POST /api/memory/maintenance
    Body: {"job_type": "consolidation|summarization|reindex|reflect|all"}

    All jobs run with the requesting user's memory context (from API key).
    """
    try:
        body = await request.json()
        job_type = body.get("job_type", "consolidation")
        memory = get_memory()

        if job_type == "reflect":
            from cems.maintenance.observation_reflector import ObservationReflector

            reflector = ObservationReflector(memory)
            result = await reflector.run_async()
            return JSONResponse({
                "success": True,
                "job_type": "reflect",
                "results": result,
            })

        # Sweep params (for consolidation full_sweep)
        full_sweep = body.get("full_sweep", False)
        sweep_limit = body.get("limit", 0)
        sweep_offset = body.get("offset", 0)

        if job_type == "all":
            from cems.maintenance.consolidation import ConsolidationJob
            from cems.maintenance.observation_reflector import ObservationReflector
            from cems.maintenance.reindex import ReindexJob
            from cems.maintenance.summarization import SummarizationJob

            results = {}
            results["consolidation"] = await ConsolidationJob(memory).run_async(
                full_sweep=full_sweep, limit=sweep_limit, offset=sweep_offset
            )
            results["summarization"] = await SummarizationJob(memory).run_async()
            results["reindex"] = await ReindexJob(memory).run_async()
            results["reflect"] = await ObservationReflector(memory).run_async()

            return JSONResponse({
                "success": True,
                "job_type": "all",
                "results": results,
            })

        # Single job type — run directly with user's memory
        from cems.maintenance.consolidation import ConsolidationJob
        from cems.maintenance.reindex import ReindexJob
        from cems.maintenance.summarization import SummarizationJob

        if job_type == "consolidation":
            result = await ConsolidationJob(memory).run_async(
                full_sweep=full_sweep, limit=sweep_limit, offset=sweep_offset
            )
            return JSONResponse({
                "success": True,
                "job_type": job_type,
                "results": result,
            })

        jobs = {
            "summarization": SummarizationJob(memory).run_async,
            "reindex": ReindexJob(memory).run_async,
        }

        if job_type not in jobs:
            return JSONResponse({
                "success": False,
                "error": f"Unknown job type: {job_type}. Use: consolidation, summarization, reindex, reflect, all",
            }, status_code=400)

        result = await jobs[job_type]()
        return JSONResponse({
            "success": True,
            "job_type": job_type,
            "results": result,
        })
    except Exception as e:
        logger.error(f"API memory_maintenance error: {e}")
        return JSONResponse({
            "success": False,
            "job_type": "unknown",
            "error": "Internal server error",
        }, status_code=500)


async def api_memory_conflict_resolve(request: Request):
    """REST API endpoint to resolve a memory conflict.

    POST /api/memory/conflict/resolve
    Body: {"conflict_id": "uuid", "resolution": "keep_a|keep_b|merge|dismiss"}

    Actions:
        keep_a: Soft-delete doc_b, resolve conflict
        keep_b: Soft-delete doc_a, resolve conflict
        merge: LLM merge both docs, update doc_a, soft-delete doc_b, resolve
        dismiss: Mark conflict as dismissed (both docs stay)
    """
    try:
        body = await request.json()
        conflict_id = body.get("conflict_id")
        resolution = body.get("resolution", "dismiss")

        if not conflict_id:
            return JSONResponse(
                {"success": False, "error": "conflict_id required"}, status_code=400
            )

        valid_resolutions = {"keep_a", "keep_b", "merge", "dismiss"}
        if resolution not in valid_resolutions:
            return JSONResponse(
                {"success": False, "error": f"Invalid resolution. Use: {sorted(valid_resolutions)}"},
                status_code=400,
            )

        memory = get_memory()
        doc_store = await memory._ensure_document_store()

        # Direct lookup by ID with user authorization
        conflict = await doc_store.get_conflict(conflict_id, memory.config.user_id)

        if not conflict:
            return JSONResponse(
                {"success": False, "error": "Conflict not found or already resolved"},
                status_code=404,
            )

        doc_a_id = conflict["doc_a_id"]
        doc_b_id = conflict["doc_b_id"]

        if resolution == "keep_a":
            await doc_store.delete_document(doc_b_id, hard=False)
            await doc_store.resolve_conflict(conflict_id, "resolved")

        elif resolution == "keep_b":
            await doc_store.delete_document(doc_a_id, hard=False)
            await doc_store.resolve_conflict(conflict_id, "resolved")

        elif resolution == "merge":
            from cems.llm import merge_memory_contents

            content_a = conflict.get("doc_a_content", "")
            content_b = conflict.get("doc_b_content", "")

            if content_a and content_b:
                merged = merge_memory_contents(
                    memories=[{"memory": content_a}, {"memory": content_b}],
                    model=memory.config.llm_model,
                )
                if merged:
                    await memory.update_async(doc_a_id, merged)
                    await doc_store.delete_document(doc_b_id, hard=False)
                    await doc_store.resolve_conflict(conflict_id, "resolved")
                else:
                    return JSONResponse(
                        {"success": False, "error": "Merge failed — LLM returned empty result"},
                        status_code=500,
                    )
            else:
                # One or both documents already deleted — resolve as no-op
                await doc_store.resolve_conflict(conflict_id, "resolved")

        elif resolution == "dismiss":
            await doc_store.resolve_conflict(conflict_id, "dismissed")

        return JSONResponse({
            "success": True,
            "conflict_id": conflict_id,
            "resolution": resolution,
        })

    except Exception as e:
        logger.error(f"API conflict_resolve error: {e}")
        return JSONResponse({"success": False, "error": "Internal server error"}, status_code=500)


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
            "backend": "documentstore",
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
        return JSONResponse({"error": "Internal server error"}, status_code=500)


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
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_memory_log_shown(request: Request):
    """REST API endpoint to log that memories were shown to the user.

    POST /api/memory/log-shown
    Body: {"memory_ids": ["id1", "id2", ...]}

    Increments shown_count and updates last_shown_at for each memory.
    Called by the UserPromptSubmit hook after injecting search results.
    """
    try:
        body = await request.json()
        memory_ids = body.get("memory_ids", [])
        if not memory_ids:
            return JSONResponse({"success": True, "updated": 0})

        if not isinstance(memory_ids, list):
            return JSONResponse({"error": "memory_ids must be an array"}, status_code=400)

        memory = get_memory()
        doc_store = await memory._ensure_document_store()

        updated = await doc_store.increment_shown_count(memory_ids)

        logger.info(f"[API] Log shown: {updated}/{len(memory_ids)} memories updated")

        return JSONResponse({
            "success": True,
            "updated": updated,
        })
    except Exception as e:
        logger.error(f"API memory_log_shown error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_memory_list(request: Request):
    """REST API endpoint to list memories with pagination and filtering.

    GET /api/memory/list?limit=50&offset=0&category=testing&scope=personal&q=search

    Returns paginated results for browsing. If `q` is provided, uses semantic search
    instead of listing.
    """
    try:
        limit = min(int(request.query_params.get("limit", "50")), 200)
        offset = int(request.query_params.get("offset", "0"))
        category = request.query_params.get("category")
        scope = request.query_params.get("scope")
        q = request.query_params.get("q", "").strip()

        memory = get_memory()
        await memory._ensure_initialized_async()
        doc_store = await memory._ensure_document_store()
        user_id = memory.config.user_id

        if q:
            # Semantic search mode
            results = await memory.search_async(q, scope=scope or "both", limit=limit)
            serialized = []
            for r in results:
                d = r.model_dump(mode="json")
                serialized.append({
                    "id": d.get("document_id") or d.get("id", ""),
                    "content": d.get("content", ""),
                    "category": d.get("category", ""),
                    "tags": d.get("tags", []),
                    "scope": d.get("scope", ""),
                    "source_ref": d.get("source_ref"),
                    "created_at": d.get("created_at"),
                    "shown_count": d.get("shown_count", 0),
                    "score": d.get("score"),
                })

            return JSONResponse({
                "success": True,
                "results": serialized,
                "total": len(serialized),
                "offset": 0,
                "limit": limit,
                "mode": "search",
            })

        # Browse mode — paginated list
        team_id = memory.config.team_id
        docs = await doc_store.get_all_documents(
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            limit=limit,
            offset=offset,
            category=category,
        )
        total = await doc_store.count_documents(
            user_id=user_id,
            team_id=team_id,
            scope=scope,
            category=category,
        )

        results = []
        for doc in docs:
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "category": doc["category"],
                "tags": doc["tags"],
                "scope": doc["scope"],
                "source_ref": doc.get("source_ref"),
                "created_at": str(doc["created_at"]) if doc.get("created_at") else None,
                "shown_count": doc.get("shown_count", 0),
            })

        return JSONResponse({
            "success": True,
            "results": results,
            "total": total,
            "offset": offset,
            "limit": limit,
            "mode": "browse",
        })
    except Exception as e:
        logger.error(f"API memory_list error: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


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
        return JSONResponse({"error": "Internal server error"}, status_code=500)
