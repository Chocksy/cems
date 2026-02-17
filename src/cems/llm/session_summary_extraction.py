"""Session summary extraction using LLM.

Replaces atomic one-liner observations with rich 2-3 paragraph session
summaries. Each summary captures the full context of a session segment
(what was worked on, decisions made, outcomes) as a single document.

Target: 200-400 words (400-600 tokens) — fits in a single chunk (800-token
threshold), so search returns the full summary without splitting.

Uses Gemini 2.5 Flash via OpenRouter for fast, cheap extraction.
"""

import logging

from cems.lib.json_parsing import parse_json_dict
from cems.llm.client import get_client

logger = logging.getLogger(__name__)

SUMMARY_MODEL = "google/gemini-2.5-flash"

SUMMARY_SYSTEM_PROMPT = """You are the memory consciousness of a coding assistant. You produce session summaries that will be stored as long-term memories and recalled across FUTURE sessions — possibly weeks or months later. These summaries are the ONLY way the assistant remembers what happened.

You are summarizing a coding session{project_label}.

## YOUR TASK

Write a **2-3 paragraph narrative summary** of this session segment. This is NOT a list of bullet points — it's a cohesive story of what happened.

## STRUCTURE

**Paragraph 1**: What was the user working on? What was the goal or problem? Include the project name if known.

**Paragraph 2**: What happened during the session? Key decisions, approaches tried, tools used, architecture choices. Preserve specific names, numbers, versions, and dates.

**Paragraph 3** (if needed): What was the outcome? What changed? Any unresolved issues or next steps the user mentioned?

## RULES

- **Narrative voice**: "The user worked on X. They decided Y because Z. The implementation involved..."
- **Preserve specifics**: Names, numbers, dates, tool names, service names, file paths that represent architecture decisions
- **Temporal flow**: Capture what happened in order, not just topics
- **State changes**: Explicitly note what changed: "migrated from X to Y", "switched to Z", "removed Q"
- **User assertions are authoritative**: When the user states facts about their project, preserve them exactly
- **Target length**: 200-400 words (2-3 paragraphs)

## WHAT TO INCLUDE
- Project goals and high-level context
- Decisions and architecture choices (with reasoning when stated)
- User preferences discovered (tools, styles, workflows)
- Key facts: names, dates, deadlines, services, people mentioned
- What worked well and what didn't
- State changes and updates to previous decisions

## WHAT NOT TO INCLUDE
- Specific CLI commands or exact file contents
- Transient debugging steps (unless they reveal an important pattern)
- Raw tool output, error messages, build logs
- Routine operations (reading files, running tests, git status)

## OUTPUT FORMAT

Return a single JSON object:
{{
  "title": "Brief descriptive title (5-10 words)",
  "content": "The 2-3 paragraph summary...",
  "tags": ["tag1", "tag2"],
  "priority": "high|medium|low"
}}

Priority: "high" (architecture decisions, major changes, user preferences), "medium" (feature work, bug fixes), "low" (routine maintenance, minor updates)

Tags: 2-5 relevant tags (e.g., "deployment", "refactoring", "authentication", "performance")

Only return the JSON object. No other text."""


def extract_session_summary(
    content: str,
    project_context: str | None = None,
    model: str | None = None,
) -> dict | None:
    """Extract a session summary from transcript content.

    Args:
        content: Session transcript content (text, not message array).
        project_context: Human-readable project context (e.g., "Chocksy/cems (main)").
        model: Optional model override (defaults to Gemini 2.5 Flash).

    Returns:
        Summary dict with keys: title, content, tags, priority.
        None if content is too short or extraction fails.
    """
    if not content or len(content) < 200:
        logger.debug("Content too short for session summary extraction")
        return None

    # Truncate to prevent OOM — 50K chars is plenty for summary extraction
    max_chars = 50_000
    if len(content) > max_chars:
        half = max_chars // 2
        content = content[:half] + "\n\n[...truncated...]\n\n" + content[-half:]

    client = get_client()
    use_model = model or SUMMARY_MODEL

    project_label = f" for: {project_context}" if project_context else ""

    system = SUMMARY_SYSTEM_PROMPT.format(project_label=project_label)

    try:
        response = client.complete(
            prompt=f"Session content to summarize:\n\n{content}",
            system=system,
            model=use_model,
            temperature=0.3,
            max_tokens=2000,
            fast_route=False,  # Gemini not on Cerebras/Groq/SambaNova
        )
    except Exception as e:
        logger.error(f"Session summary extraction LLM call failed: {e}")
        return None

    if not response:
        logger.warning("Empty response from session summary extraction LLM")
        return None

    return _parse_summary(response)


def _parse_summary(response: str) -> dict | None:
    """Parse LLM response into a validated summary dict.

    Args:
        response: Raw LLM response text (should be a JSON object).

    Returns:
        Validated summary dict, or None on failure.
    """
    summary = parse_json_dict(response, log_errors=True)

    if not summary:
        logger.warning(f"Could not parse session summary from response: {response[:200]}")
        return None

    content = summary.get("content", "").strip()
    if not content or len(content) < 50:
        logger.debug(f"Summary content too short ({len(content)} chars)")
        return None

    title = summary.get("title", "").strip()
    if not title:
        # Generate a basic title from first sentence
        first_sentence = content.split(".")[0][:80]
        title = first_sentence

    priority = summary.get("priority", "medium").lower()
    if priority not in ("high", "medium", "low"):
        priority = "medium"

    tags = summary.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower() for t in tags if t][:5]

    return {
        "title": title,
        "content": content,
        "tags": tags,
        "priority": priority,
    }
