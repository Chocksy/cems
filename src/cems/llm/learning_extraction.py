"""Tool learning extraction and category utilities.

Provides:
- normalize_category(): Clean category strings for consistent display
- extract_tool_learning(): Extract a single learning from tool usage context
"""

import logging

from cems.lib.json_parsing import extract_json_from_response
from cems.llm.client import get_client

logger = logging.getLogger(__name__)

# Valid learning types for tool learning extraction
_TOOL_LEARNING_TYPES = {
    "WORKING_SOLUTION",
    "FAILED_APPROACH",
    "USER_PREFERENCE",
    "ERROR_FIX",
    "DECISION",
    "GUIDELINE",
}

# Minimum confidence threshold for storing learnings
_MIN_CONFIDENCE = 0.6


def normalize_category(raw: str) -> str:
    """Normalize a category string to a clean lowercase-hyphenated form.

    Functional categories (gate-rules, preferences, etc.) are preserved as-is.
    All other categories are free-text from LLM output — just cleaned up for
    consistent display. No alias mapping or canonicalization.

    Args:
        raw: Raw category string from LLM output

    Returns:
        Cleaned category string (lowercase, hyphenated)
    """
    return raw.lower().strip().replace(" ", "-").replace("_", "-").replace("/", "-") or "general"


def extract_tool_learning(
    tool_context: str,
    conversation_snippet: str = "",
    working_dir: str | None = None,
    model: str | None = None,
) -> dict | None:
    """Extract a single learning from tool usage context.

    Lightweight extraction designed for incremental tool-based learning
    (SuperMemory-style). Called by the PostToolUse hook via /api/tool/learning.

    Args:
        tool_context: Description of the tool that was used (name, input, output)
        conversation_snippet: Recent conversation context (last few messages)
        working_dir: Optional working directory for project context
        model: Optional model override (defaults to grok-4.1-fast for speed)

    Returns:
        Learning dict with keys (type, content, category, confidence) or None if
        nothing worth storing.
    """
    import json

    # Skip if context is too brief
    total_context = tool_context + conversation_snippet
    if len(total_context) < 100:
        logger.debug("Tool context too brief for learning extraction")
        return None

    # Build context
    context_parts = []
    if working_dir:
        # Extract project name from path
        project_name = working_dir.split("/")[-1] if "/" in working_dir else working_dir
        context_parts.append(f"Project: {project_name}")

    context_section = "\n".join(context_parts) if context_parts else ""

    system_prompt = """You are a Tool Learning Extractor. Given a tool usage and conversation context, extract ONE meaningful learning if present.

## Learning Types
- WORKING_SOLUTION - A pattern, command, or approach that worked
- ERROR_FIX - How an error was diagnosed and fixed
- DECISION - A design/architecture decision with reasoning
- USER_PREFERENCE - A user preference about tools, styles, or workflows

## Output Format
Return a single JSON object OR null if nothing worth remembering.

{
  "type": "WORKING_SOLUTION",
  "content": "Clear, actionable learning (1-2 sentences)",
  "confidence": 0.7,
  "category": "topic"
}

Be selective - only extract if there's a genuine learning, not routine operations.
Return null for routine reads, searches, or trivial edits."""

    prompt = f"""Analyze this tool usage and extract ONE learning if significant.

{f"## Context\\n{context_section}\\n" if context_section else ""}
## Tool Usage
{tool_context}

{f"## Recent Conversation\\n{conversation_snippet[:1000]}\\n" if conversation_snippet else ""}
## Instructions
If this represents a meaningful pattern, fix, decision, or preference, extract it.
If it's routine (just reading files, simple searches), return null.

Return ONLY valid JSON (object or null)."""

    try:
        client = get_client()
        response = client.complete(
            prompt=prompt,
            system=system_prompt,
            temperature=0.2,  # Lower temperature for consistency
            model=model or "x-ai/grok-4.1-fast",
        )

        # Parse response
        response = response.strip()

        # Handle null response
        if response.lower() in ("null", "none", "{}"):
            return None

        # Use shared utility for extraction
        cleaned = extract_json_from_response(response)

        learning = json.loads(cleaned)

        if not learning or not isinstance(learning, dict):
            return None

        # Validate learning
        if learning.get("type") not in _TOOL_LEARNING_TYPES:
            return None
        if not learning.get("content"):
            return None
        if learning.get("confidence", 0) < _MIN_CONFIDENCE:
            return None

        return {
            "type": learning["type"],
            "content": str(learning["content"])[:500],
            "confidence": float(learning.get("confidence", 0.7)),
            "category": str(learning.get("category", "general"))[:50],
        }

    except Exception as e:
        logger.warning(f"Failed to extract tool learning: {e}")
        return None
