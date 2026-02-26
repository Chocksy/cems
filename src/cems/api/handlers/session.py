"""Session API handlers.

Handles session summarization via the observer daemon.
"""

import logging
from datetime import date

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory

logger = logging.getLogger(__name__)


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

        # Cap content size to prevent OOM
        MAX_SUMMARY_CHARS = 50_000
        original_len = len(content)
        if len(content) > MAX_SUMMARY_CHARS:
            half = MAX_SUMMARY_CHARS // 2
            content = content[:half] + "\n\n[...truncated...]\n\n" + content[-half:]
            logger.warning(
                f"Summary content truncated from {original_len} to ~{MAX_SUMMARY_CHARS} chars "
                f"(session: {session_id})"
            )

        # Epoch-aware session tag: session:{id[:12]} for epoch 0, session:{id[:12]}:e{N} for N>0
        tag = body.get("session_tag") or f"session:{session_id[:12]}"
        if epoch > 0 and ":e" not in tag:
            tag = f"session:{session_id[:12]}:e{epoch}"

        logger.info(
            f"session_summarize called: session_id={session_id}, "
            f"content_len={original_len} (capped to {len(content)}), "
            f"project={project_context}, mode={mode}, epoch={epoch}, tag={tag}"
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

        # Pre-merge: read existing content OUTSIDE the DB lock (the slow
        # embedding call happens between read and upsert).  We pass the merged
        # blob with mode="replace" so upsert_document_by_tag overwrites rather
        # than appending again (which previously caused exponential doubling).
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
                upsert_mode = "replace"  # handler pre-merges; upsert just stores
        elif mode == "finalize":
            upsert_mode = "finalize"

        # Safety cap: prevent stored summaries from growing unboundedly.
        # Each incremental LLM summary is ~1.5-2K chars.
        # 10K chars ≈ 5-7 incremental segments — enough context without bloat.
        MAX_STORED_SUMMARY_CHARS = 10_000
        if len(upsert_content) > MAX_STORED_SUMMARY_CHARS:
            # Keep only the tail (most recent summaries)
            upsert_content = upsert_content[-MAX_STORED_SUMMARY_CHARS:]
            # Clean up: don't start mid-paragraph
            first_sep = upsert_content.find("\n\n---\n\n")
            if first_sep > 0:
                upsert_content = upsert_content[first_sep + 7:]
            logger.warning(
                f"Stored summary capped to ~{MAX_STORED_SUMMARY_CHARS} chars "
                f"(session: {session_id})"
            )

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
        return JSONResponse({"error": "Internal server error"}, status_code=500)
