"""Session summary extraction using LLM.

Extracts structured atomic facts + brief context from session transcripts.
Each fact is a standalone, independently searchable sentence. The context
paragraph provides semantic embedding signal for the full session.

Inspired by top LongMemEval systems: Hindsight (91.4%, structured 5W facts),
Mem0 (atomic fact extraction), and the LongMemEval reference implementation
("extract all personal information... in simple sentences").

Uses Gemini 2.5 Flash via OpenRouter for fast, cheap extraction.
"""

import logging

from cems.lib.json_parsing import parse_json_dict
from cems.llm.client import get_client

logger = logging.getLogger(__name__)

SUMMARY_MODEL = "google/gemini-2.5-flash"

SUMMARY_SYSTEM_PROMPT = """You are the memory consciousness of a coding assistant. You extract structured facts from sessions that will be stored as long-term memories and recalled across FUTURE sessions — possibly weeks or months later. These facts are the ONLY way the assistant remembers what happened.

You are extracting facts from a coding session{project_label}.

## YOUR TASK

Extract **5-15 atomic facts** from this session as standalone sentences. Each fact must be independently meaningful — searchable and understandable without any other context.

Then write a **1-2 sentence context overview** that captures the session's overall purpose (for semantic search).

## FACT EXTRACTION RULES

- **Each fact is a complete, standalone sentence**: "User's dog is named Rex" not just "Rex"
- **Preserve exact values**: prices ("$5 coupon"), percentages ("43.7%% faster"), dates, names, versions, quantities
- **Frame state changes explicitly**: "User switched from SQLite to PostgreSQL" not "User uses PostgreSQL"
- **User assertions are authoritative**: "User stated they are a staunch atheist" not "User discussed religion"
- **Distinguish assertions from questions**: "User asked about deployment options" vs "User decided to deploy on Railway"
- **Use precise action verbs**: "subscribed to", "purchased", "enrolled in" — not vague "got" or "getting"
- **Include project name** in relevant facts for searchability
- **Preserve distinguishing details**: "Assistant recommended Docker Compose (simple), K8s (scalable), Railway (managed)" not "Assistant recommended 3 options"
- **Non-standard terminology**: quote user's exact words: "User did a 'movement session' (their term for exercise)"

## WHAT TO EXTRACT AS FACTS
- User preferences, opinions, and personal information
- Decisions and architecture choices (with reasoning when stated)
- Key names: people, services, tools, versions, file paths that represent architecture
- Numbers: prices, measurements, deadlines, quantities, performance metrics
- State changes: what was replaced and what replaced it
- Project goals, outcomes, and unresolved issues

## WHAT NOT TO EXTRACT
- Specific CLI commands or exact file contents
- Transient debugging steps (unless they reveal an important pattern)
- Raw tool output, error messages, build logs
- Routine operations (reading files, running tests, git status)

{project_name_instruction}

## OUTPUT FORMAT

Return a single JSON object:
{{
  "title": "Brief descriptive title (5-10 words)",
  "facts": [
    "Fact 1 as a standalone sentence",
    "Fact 2 as a standalone sentence",
    "..."
  ],
  "context": "1-2 sentence overview of what the session was about, for semantic search.",
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
    """Extract structured facts from a session transcript.

    Args:
        content: Session transcript content (text, not message array).
        project_context: Human-readable project context (e.g., "Chocksy/cems (main)").
        model: Optional model override (defaults to Gemini 2.5 Flash).

    Returns:
        Summary dict with keys: title, content (assembled facts), tags, priority.
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

    # Dynamic instruction: weave project name into facts for searchability
    if project_context:
        project_name_instruction = (
            "## PROJECT NAME\n\n"
            f"Include the project/repo name in relevant facts so they're searchable without metadata:\n"
            f'- BAD: "User overhauled the memory quality system"\n'
            f'- GOOD: "User overhauled the memory quality system in {project_context}"\n\n'
            f'The project context "{project_context}" should appear naturally in facts and context.'
        )
    else:
        project_name_instruction = ""

    system = SUMMARY_SYSTEM_PROMPT.format(
        project_label=project_label,
        project_name_instruction=project_name_instruction,
    )

    # Enable fast_route (Cerebras/Groq) when using a non-Gemini model override
    use_fast_route = use_model != SUMMARY_MODEL

    try:
        response = client.complete(
            prompt=f"Session content to summarize:\n\n{content}",
            system=system,
            model=use_model,
            temperature=0.3,
            max_tokens=2000,
            fast_route=use_fast_route,
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

    Handles both the new facts+context format and the legacy narrative format
    (for backward compatibility during rollout).

    Args:
        response: Raw LLM response text (should be a JSON object).

    Returns:
        Validated summary dict with assembled 'content' field, or None on failure.
    """
    summary = parse_json_dict(response, log_errors=True)

    if not summary:
        logger.warning(f"Could not parse session summary from response: {response[:200]}")
        return None

    # New format: facts list + context string → assemble into content
    facts = summary.get("facts")
    context = summary.get("context", "").strip()

    if isinstance(facts, list) and facts:
        # Filter and validate facts
        valid_facts = [str(f).strip() for f in facts if f and str(f).strip()]
        if not valid_facts:
            logger.debug("All facts were empty after filtering")
            return None

        # Assemble: one fact per line, then context paragraph
        parts = valid_facts
        if context:
            parts = valid_facts + ["", f"Context: {context}"]
        content = "\n".join(parts)
    else:
        # Legacy fallback: plain "content" field (narrative format)
        content = summary.get("content", "").strip()

    if not content or len(content) < 50:
        logger.debug(f"Summary content too short ({len(content)} chars)")
        return None

    title = summary.get("title", "").strip()
    if not title:
        # Generate a basic title from first fact or sentence
        first_line = content.split("\n")[0].split(".")[0][:80]
        title = first_line

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
