"""Observation extraction using LLM.

Extracts high-level observations from session transcripts, inspired by
Mastra's Observational Memory approach. Observations capture CONTEXT
(what the user is doing, deciding, preferring) rather than implementation
details (specific commands, file paths, error messages).

Uses Gemini 2.5 Flash via OpenRouter for fast, cheap extraction.
"""

import logging

from cems.lib.json_parsing import parse_json_list
from cems.llm.client import get_client
from cems.llm.learning_extraction import normalize_category

logger = logging.getLogger(__name__)

# Default model for observation extraction (same as Mastra uses)
OBSERVER_MODEL = "google/gemini-2.5-flash"

# Maximum observations per extraction call
MAX_OBSERVATIONS = 5

OBSERVER_SYSTEM_PROMPT = """You are the memory consciousness of a coding assistant. Your observations will be stored as long-term memories and recalled across FUTURE sessions — possibly weeks or months from now. These observations are the ONLY way the assistant can remember what happened in previous sessions.

You are observing a coding session for: {project_context}

Extract 1-{max_obs} high-level observations. Use terse language to save tokens.

## CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS

When the user TELLS something about themselves or their project, mark it as HIGH priority:
- "I have two services running" → HIGH: "User stated they have two services running"
- "I work at Acme Corp" → HIGH: "User stated they work at Acme Corp"
- "The deadline is next Friday" → HIGH: "User stated deadline is next Friday (meaning DATE)"

When the user ASKS about something, mark it as MEDIUM priority:
- "Can you help me with X?" → MEDIUM: "User asked for help with X"
- "What's the best way to do Y?" → MEDIUM: "User asked about best approach for Y"

Distinguish QUESTIONS from STATEMENTS OF INTENT:
- "Can you recommend..." → Question (MEDIUM)
- "I need to deploy by Friday" → Statement of intent (HIGH: "User needs to deploy by Friday")
- "I'm switching to PostgreSQL" → Statement of intent (HIGH: "User is switching to PostgreSQL")

USER ASSERTIONS ARE AUTHORITATIVE. The user is the source of truth about their own project and life.

## STATE CHANGES AND UPDATES

When the user indicates they are changing something, frame it as a state change:
- "I'm going to use X instead of Y" → "User will use X (replacing Y)"
- "I'm switching from A to B" → "User is switching from A to B"

If the new state contradicts previous information, make that explicit:
- BAD: "User plans to use the new method"
- GOOD: "User will use the new method (replacing the old approach)"

## TEMPORAL ANCHORING

Each observation has TWO potential timestamps:
1. BEGINNING: When the statement was made — ALWAYS include
2. END: Time being REFERENCED, if different — only when there's a concrete date reference

ONLY add "(meaning DATE)" at the END for concrete time references:
- Past: "last week", "yesterday", "in March" → add estimated date
- Future: "next Friday", "tomorrow", "this weekend" → add calculated date
- DO NOT add dates for vague references: "recently", "a while ago", "soon"

## PRESERVE DETAILS

- Names, handles, @usernames — always preserve
- Numbers, quantities, measurements, prices — always preserve
- When user uses non-standard terminology, quote their exact words:
  BAD: "User exercised"
  GOOD: "User did a 'movement session' (their term for exercise)"

## USE PRECISE ACTION VERBS

Replace vague "getting"/"got" with specific verbs:
- "getting X regularly" → "subscribed to X"
- "getting X once" → "purchased X"
- "got something" → "purchased / received / was given"
- "signed up" → "enrolled in / registered for / subscribed to"

## PRESERVE DISTINGUISHING DETAILS

When assistant provides lists, recommendations, or content the user requested, preserve the KEY ATTRIBUTE that distinguishes each item:
- BAD: "Assistant recommended 3 deployment options"
- GOOD: "Assistant recommended: Docker Compose (simple), K8s (scalable), Railway (managed)"

For technical/numerical results, preserve specific values:
- BAD: "Performance improved after optimization"
- GOOD: "Optimization achieved 43.7% faster load times, memory usage dropped from 2.8GB to 940MB"

## WHAT TO OBSERVE
- Project goals and high-level context
- Decisions and architecture choices
- User preferences (tools, styles, workflows)
- Key facts: names, dates, deadlines, services, people
- Workflow patterns
- What worked well / what user is satisfied with
- State changes and updates to previous decisions

## WHAT NOT TO OBSERVE
- Specific CLI commands or file paths (no "docker compose build" or "src/foo.py:42")
- Transient debugging steps (unless recurring pattern)
- Tool output, error messages, build logs
- Routine operations (reading files, running tests, git status)
- Temporary state (current branch, current file being edited)

## IMPORTANT: INCLUDE PROJECT NAME

Always mention the project/repo name in each observation so it's searchable without metadata:
- BAD: "User is adding relevance scoring to the memory system"
- GOOD: "User is adding relevance scoring to CEMS memory system (Chocksy/cems)"

The project context "{project_context}" should appear naturally in each observation.

## OUTPUT FORMAT

Return a JSON array:
[
  {{
    "content": "In Chocksy/cems: User is overhauling memory quality, reducing 2499 memories to 584 active",
    "priority": "high",
    "category": "observation"
  }}
]

Priority: "high" (decisions, preferences, goals, deadlines), "medium" (patterns, architecture), "low" (informational)

Each observation: 80-250 characters, self-contained, readable without other context.
Only return the JSON array. No other text."""


def extract_observations(
    content: str,
    project_context: str = "unknown project",
    model: str | None = None,
) -> list[dict]:
    """Extract high-level observations from session content.

    Args:
        content: Session transcript content (text, not message array)
        project_context: Human-readable project context (e.g., "chocksy/cems (main)")
        model: Optional model override (defaults to Gemini 2.5 Flash)

    Returns:
        List of observation dicts with keys: content, priority, category
    """
    if not content or len(content) < 200:
        logger.debug("Content too short for observation extraction")
        return []

    # Truncate to ~25k tokens worth of content
    max_chars = 100_000
    if len(content) > max_chars:
        content = content[:max_chars]

    client = get_client()
    use_model = model or OBSERVER_MODEL

    system = OBSERVER_SYSTEM_PROMPT.format(
        project_context=project_context,
        max_obs=MAX_OBSERVATIONS,
    )

    try:
        response = client.complete(
            prompt=f"Session content to observe:\n\n{content}",
            system=system,
            model=use_model,
            temperature=0.3,
            max_tokens=2000,
            fast_route=False,  # Gemini not on Cerebras/Groq/SambaNova
        )
    except Exception as e:
        logger.error(f"Observation extraction LLM call failed: {e}")
        return []

    if not response:
        logger.warning("Empty response from observation extraction LLM")
        return []

    return _parse_observations(response)


def _parse_observations(response: str) -> list[dict]:
    """Parse LLM response into validated observation dicts.

    Args:
        response: Raw LLM response text (should be JSON array)

    Returns:
        List of validated observation dicts
    """
    observations_raw = parse_json_list(response, log_errors=True)

    if not observations_raw:
        logger.warning(f"Could not parse observations from response: {response[:200]}")
        return []

    observations = []
    for item in observations_raw:
        if not isinstance(item, dict):
            continue

        content = item.get("content", "").strip()
        if not content:
            continue

        # Validate content length (80-200 chars target, allow some flexibility)
        if len(content) < 30:
            logger.debug(f"Observation too short ({len(content)} chars), skipping: {content}")
            continue

        # Cap content at 300 chars
        if len(content) > 300:
            content = content[:297] + "..."

        priority = item.get("priority", "medium").lower()
        if priority not in ("high", "medium", "low"):
            priority = "medium"

        # Normalize category
        raw_category = item.get("category", "observation").lower().strip()
        if raw_category == "observation":
            category = "observation"
        else:
            category = normalize_category(raw_category)

        observations.append({
            "content": content,
            "priority": priority,
            "category": category,
        })

    # Cap at MAX_OBSERVATIONS
    return observations[:MAX_OBSERVATIONS]
