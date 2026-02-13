"""Observation API handlers.

Handles session observation extraction — producing high-level observations
from session transcripts (what the user is doing, deciding, preferring).
"""

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory
from cems.llm.observation_extraction import extract_observations

logger = logging.getLogger(__name__)


async def api_session_observe(request: Request):
    """REST API endpoint to extract observations from session content.

    POST /api/session/observe
    Body: {
        "content": "...",              # Session transcript content (text)
        "session_id": "...",           # Session identifier
        "source_ref": "project:org/repo",  # Project context for tagging
        "project_context": "org/repo (main)",  # Human-readable for LLM prompt
    }

    Response: {
        "success": true,
        "observations_stored": 2,
        "observations": [
            {"content": "...", "priority": "high", "category": "observation", "memory_id": "..."}
        ]
    }
    """

    try:
        body = await request.json()
        content = body.get("content")
        if not content:
            return JSONResponse({"error": "content is required"}, status_code=400)

        session_id = body.get("session_id", "unknown")
        source_ref = body.get("source_ref")
        project_context = body.get("project_context") or None

        logger.info(
            f"session_observe called: session_id={session_id}, "
            f"content_len={len(content)}, project={project_context}"
        )

        # Extract observations using LLM
        observations = extract_observations(
            content=content,
            project_context=project_context,
        )

        if not observations:
            return JSONResponse({
                "success": True,
                "observations_stored": 0,
                "observations": [],
            })

        # Store each observation as a memory
        memory = get_memory()
        stored = []

        for obs in observations:
            try:
                result = await memory.add_async(
                    content=obs["content"],
                    scope="personal",
                    category="observation",
                    tags=["observation", obs.get("priority", "medium")],
                    infer=False,
                    source_ref=source_ref,
                )

                memory_id = None
                if result and "results" in result:
                    for r in result["results"]:
                        if r.get("id"):
                            memory_id = r["id"]
                            break

                stored.append({
                    "content": obs["content"],
                    "priority": obs.get("priority", "medium"),
                    "category": "observation",
                    "memory_id": memory_id,
                })

            except Exception as e:
                logger.error(f"Failed to store observation: {e}")
                continue

        logger.info(f"Stored {len(stored)} observations for session {session_id}")

        return JSONResponse({
            "success": True,
            "observations_stored": len(stored),
            "observations": stored,
        })

    except Exception as e:
        logger.error(f"API session_observe error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
