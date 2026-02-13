"""Observation reflection — consolidate overlapping observations via LLM.

Inspired by Mastra's Reflector Agent. Takes a set of accumulated observations
for a project and produces a condensed, non-redundant replacement set.

Uses Gemini 2.5 Flash via OpenRouter (same as observer).
"""

import logging

from cems.lib.json_parsing import parse_json_list
from cems.llm.client import get_client

logger = logging.getLogger(__name__)

REFLECTOR_MODEL = "google/gemini-2.5-flash"

REFLECTOR_SYSTEM_PROMPT = """You are the memory consciousness of a coding assistant. You are consolidating accumulated observations into a compressed, non-redundant set.

These observations are the ONLY way the assistant remembers past sessions. Any information you drop will be permanently forgotten.

## YOUR TASK

You will receive a list of observations accumulated over multiple sessions for project: {project_context}

Produce a CONSOLIDATED set that:
1. **Merges overlapping observations** — combine related facts into single, richer observations
2. **Removes redundancy** — if two observations say the same thing, keep the more specific/recent one
3. **Preserves ALL unique information** — every distinct fact must appear in the output
4. **Resolves contradictions** — newer observations supersede older ones (observations are ordered oldest-first)
5. **Keeps a concise but complete style** — each observation should be as long as needed to preserve all detail (typically 1-3 sentences)

## RULES

- Output FEWER observations than input (target: 40-60% reduction)
- Each output observation must be self-contained (readable without other context)
- Preserve: names, numbers, dates, handles, specific tool/service names
- When merging, prefer the MORE SPECIFIC version
- When observations contradict, keep the LATER one and note the change
- Priority: preserve "high" priority info, compress "low" priority more aggressively

## EXAMPLE INPUT
1. [high] User is deploying CEMS to production
2. [medium] User uses Coolify for deployments
3. [high] User deployed CEMS to Coolify on Hetzner via Tailscale
4. [medium] User prefers Docker Compose for local development
5. [low] User ran integration tests against Docker

## EXAMPLE OUTPUT
1. [high] User deployed CEMS to Coolify on Hetzner (accessed via Tailscale), uses Docker Compose locally
2. [low] User ran integration tests against Docker

(3 observations merged into 1, test observation kept as-is since it's distinct)

## OUTPUT FORMAT

Return a JSON array:
[
  {{"content": "...", "priority": "high|medium|low", "category": "observation"}}
]

Only return the JSON array. No other text."""


def reflect_observations(
    observations: list[dict],
    project_context: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Consolidate a list of observations into a compressed set.

    Args:
        observations: List of observation dicts with 'content' and optionally
                     'priority', 'category' keys. Ordered oldest-first.
        project_context: Human-readable project context
        model: Optional model override

    Returns:
        List of consolidated observation dicts (fewer than input).
        Returns empty list on failure.
    """
    if len(observations) < 3:
        return observations

    # Format observations for LLM
    lines = []
    for i, obs in enumerate(observations, 1):
        priority = obs.get("priority", "medium")
        content = obs.get("content", "")
        lines.append(f"{i}. [{priority}] {content}")

    observations_text = "\n".join(lines)

    client = get_client()
    use_model = model or REFLECTOR_MODEL

    system = REFLECTOR_SYSTEM_PROMPT.format(
        project_context=project_context or "various projects"
    )

    try:
        response = client.complete(
            prompt=f"Observations to consolidate ({len(observations)} total):\n\n{observations_text}",
            system=system,
            model=use_model,
            temperature=0.1,
            max_tokens=8000,
            fast_route=False,
        )
    except Exception as e:
        logger.error(f"Observation reflection LLM call failed: {e}")
        return []

    if not response:
        logger.warning("Empty response from observation reflection LLM")
        return []

    return _parse_reflected(response, max_count=len(observations))


def _parse_reflected(response: str, max_count: int) -> list[dict]:
    """Parse and validate LLM reflection response.

    Args:
        response: Raw LLM response (should be JSON array)
        max_count: Original observation count (output should be fewer)

    Returns:
        List of validated observation dicts
    """
    items = parse_json_list(response, log_errors=True)

    if not items:
        logger.warning(f"Could not parse reflected observations: {response[:200]}")
        return []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue

        content = item.get("content", "").strip()
        if not content or len(content) < 30:
            continue

        priority = item.get("priority", "medium").lower()
        if priority not in ("high", "medium", "low"):
            priority = "medium"

        results.append({
            "content": content,
            "priority": priority,
            "category": "observation",
        })

    # Sanity check: output should be fewer than input
    if len(results) >= max_count:
        logger.warning(
            f"Reflection produced {len(results)} observations (>= input {max_count}), "
            "returning as-is but this indicates poor compression"
        )

    return results
