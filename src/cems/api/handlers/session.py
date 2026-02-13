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

    Upsert model: ONE document per session. If the daemon already created
    an incremental summary, subsequent calls UPDATE it rather than creating
    duplicates. The stop hook sends mode="finalize" to replace with a
    comprehensive final version.

    POST /api/session/summarize
    Body: {
        "content": "...",              # Session transcript text
        "session_id": "...",           # Session identifier
        "source_ref": "project:org/repo",  # Project reference for tagging
        "project_context": "org/repo (main)",  # Human-readable for LLM prompt
        "mode": "incremental|finalize",  # incremental=daemon, finalize=stop hook
    }

    Response: {
        "success": true,
        "document_id": "...",
        "title": "...",
        "action": "created|updated",
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
        mode = body.get("mode", "incremental")
        epoch = body.get("epoch", 0)

        # Epoch-aware session tag: session:{id[:8]} for epoch 0, session:{id[:8]}:e{N} for N>0
        tag = body.get("session_tag") or f"session:{session_id[:8]}"
        if epoch > 0 and ":e" not in tag:
            tag = f"session:{session_id[:8]}:e{epoch}"

        logger.info(
            f"session_summarize called: session_id={session_id}, "
            f"content_len={len(content)}, project={project_context}, "
            f"mode={mode}, epoch={epoch}, tag={tag}"
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
                "action": None,
                "skipped_reason": "extraction_produced_no_summary",
            })

        # Build a descriptive title
        project_name = project_context.split(" ")[0] if project_context else "session"
        title = summary.get("title") or f"Session {date.today().isoformat()}"
        epoch_suffix = f" (epoch {epoch})" if epoch > 0 else ""
        full_title = f"Session: {date.today().isoformat()} - {project_name} - {title}{epoch_suffix}"

        # Build tags
        tags = ["session-summary", summary.get("priority", "medium")]
        tags.append(tag)
        if summary.get("tags"):
            tags.extend(summary["tags"])

        memory = get_memory()
        await memory._ensure_initialized_async()
        doc_store = await memory._ensure_document_store()
        user_id = memory.config.user_id

        # For append mode, we need the existing content to merge before chunking.
        # Read it OUTSIDE the atomic upsert to avoid holding the row lock during
        # the slow LLM embedding call.
        upsert_content = summary["content"]
        upsert_mode = "replace"

        if mode == "incremental":
            # Check if there's existing content to append to
            existing = await doc_store.find_document_by_tag(
                tag=tag, user_id=user_id, category="session-summary",
            )
            if existing:
                existing_content = existing.get("content", "")
                if existing_content:
                    upsert_content = f"{existing_content}\n\n---\n\n{summary['content']}"
                upsert_mode = "append"
        elif mode == "finalize":
            upsert_mode = "finalize"

        # Chunk and embed the final content (slow, done outside the DB lock)
        from cems.chunking import chunk_document

        chunks = chunk_document(upsert_content)
        if not chunks:
            return JSONResponse({
                "success": True,
                "document_id": None,
                "title": full_title,
                "action": None,
                "skipped_reason": "chunking_produced_no_output",
            })

        chunk_texts = [c.content for c in chunks]
        embeddings = await memory._async_embedder.embed_batch(chunk_texts)

        # Atomic upsert: SELECT FOR UPDATE prevents duplicate creation
        document_id, action = await doc_store.upsert_document_by_tag(
            tag=tag,
            user_id=user_id,
            content=upsert_content,
            chunks=chunks,
            embeddings=embeddings,
            category="session-summary",
            mode=upsert_mode,
            scope="personal",
            title=full_title,
            source_ref=source_ref,
            tags=tags,
        )
        logger.info(f"{action.title()} session summary for {session_id[:8]}: {full_title}")

        return JSONResponse({
            "success": True,
            "document_id": document_id,
            "title": full_title,
            "action": action,
        })

    except Exception as e:
        logger.error(f"API session_summarize error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
