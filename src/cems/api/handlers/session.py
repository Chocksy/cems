"""Session API handlers.

Handles session transcript analysis, learning extraction, and session summarization.
"""

import logging
from datetime import date

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory

logger = logging.getLogger(__name__)


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
        source_ref = body.get("source_ref")  # e.g., "project:org/repo"

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
                    source_ref=source_ref,
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


async def api_session_summarize(request: Request):
    """REST API endpoint to produce and store a session summary.

    POST /api/session/summarize
    Body: {
        "content": "...",              # Session transcript text
        "session_id": "...",           # Session identifier
        "source_ref": "project:org/repo",  # Project reference for tagging
        "project_context": "org/repo (main)",  # Human-readable for LLM prompt
        "segment_number": 1,           # Daemon cycle count (optional)
    }

    Response: {
        "success": true,
        "document_id": "...",
        "title": "...",
    }
    """
    from cems.llm.session_summary_extraction import extract_session_summary

    try:
        body = await request.json()
        content = body.get("content")
        if not content:
            return JSONResponse({"error": "content is required"}, status_code=400)

        session_id = body.get("session_id", "unknown")
        source_ref = body.get("source_ref")
        project_context = body.get("project_context") or None
        segment_number = body.get("segment_number", 0)

        logger.info(
            f"session_summarize called: session_id={session_id}, "
            f"content_len={len(content)}, project={project_context}, segment={segment_number}"
        )

        summary = extract_session_summary(
            content=content,
            project_context=project_context,
        )

        if not summary:
            return JSONResponse({
                "success": True,
                "document_id": None,
                "title": None,
                "skipped_reason": "extraction_produced_no_summary",
            })

        # Build a descriptive title
        project_name = project_context.split(" ")[0] if project_context else "session"
        title = summary.get("title") or f"Session {date.today().isoformat()}"
        full_title = f"Session: {date.today().isoformat()} - {project_name} - {title}"

        # Build tags
        tags = ["session-summary", summary.get("priority", "medium")]
        tags.append(f"session:{session_id[:8]}")
        if summary.get("tags"):
            tags.extend(summary["tags"])

        # Store as a document memory
        memory = get_memory()
        result = await memory.add_async(
            content=summary["content"],
            scope="personal",
            category="session-summary",
            tags=tags,
            infer=False,
            source_ref=source_ref,
            title=full_title,
        )

        document_id = None
        if result and "results" in result:
            for r in result["results"]:
                if r.get("id"):
                    document_id = r["id"]
                    break

        logger.info(f"Stored session summary for {session_id[:8]}: {full_title}")

        return JSONResponse({
            "success": True,
            "document_id": document_id,
            "title": full_title,
        })

    except Exception as e:
        logger.error(f"API session_summarize error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
