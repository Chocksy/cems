"""Tool learning API handlers.

Handles incremental tool-based learning extraction.
"""

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory

logger = logging.getLogger(__name__)


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
        source_ref = body.get("source_ref")  # e.g., "project:org/repo"

        if not tool_name:
            return JSONResponse({"error": "tool_name is required"}, status_code=400)

        non_learnable_tools = {"Read", "Glob", "Grep", "LS", "WebFetch", "WebSearch"}
        if tool_name in non_learnable_tools:
            return JSONResponse({
                "success": True,
                "stored": False,
                "memory_id": None,
                "reason": "skipped_non_learnable_tool",
            })

        if not context_snippet and not tool_output:
            return JSONResponse({
                "success": True,
                "stored": False,
                "memory_id": None,
                "reason": "skipped_no_context",
            })

        # Each tool type extracts different relevant fields for the LLM prompt
        tool_context = f"Tool: {tool_name}\n"
        if tool_input:
            if tool_name in ("Edit", "MultiEdit") and "file_path" in tool_input:
                tool_context += f"File: {tool_input['file_path']}\n"
                if "old_string" in tool_input:
                    tool_context += f"Changed from: {str(tool_input['old_string'])[:300]}\n"
                if "new_string" in tool_input:
                    tool_context += f"Changed to: {str(tool_input['new_string'])[:300]}\n"
            elif tool_name == "Write" and "file_path" in tool_input:
                tool_context += f"Created: {tool_input['file_path']}\n"
                if "content" in tool_input:
                    tool_context += f"Content preview: {str(tool_input['content'])[:300]}\n"
            elif tool_name == "Bash" and "command" in tool_input:
                cmd = tool_input.get("command", "")[:200]
                desc = tool_input.get("description", "")
                tool_context += f"Command: {desc or cmd}\n"
            elif tool_name == "Task":
                if "prompt" in tool_input:
                    tool_context += f"Task: {str(tool_input['prompt'])[:300]}\n"
        if tool_output:
            tool_context += f"Result: {tool_output[:500]}\n"

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

        memory = get_memory()
        content = learning.get("content", "")
        category = learning.get("category", "learnings")
        learning_type = learning.get("type", "TOOL")

        formatted_content = f"[{learning_type}] {content}"
        if session_id != "unknown":
            formatted_content += f" (session: {session_id[:8]})"

        result = await memory.add_async(
            content=formatted_content,
            scope="personal",
            category=category,
            tags=["tool-learning", tool_name.lower(), learning_type.lower()],
            source_ref=source_ref,
            infer=False,
        )

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
        return JSONResponse({"error": "Internal server error"}, status_code=500)
