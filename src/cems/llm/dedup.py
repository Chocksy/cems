"""LLM-powered memory pair classification for smart deduplication.

Classifies pairs of memories as: duplicate, related, conflicting, or distinct.
Used by ConsolidationJob for three-tier deduplication (Phase 4).

Uses Gemini 2.5 Flash via OpenRouter for cheap, fast classification.
"""

import logging

from cems.lib.json_parsing import parse_json_dict
from cems.llm.client import get_client

logger = logging.getLogger(__name__)

# Default model — matches observation_extraction.py and observation_reflection.py
CLASSIFIER_MODEL = "google/gemini-2.5-flash"

VALID_CLASSIFICATIONS = {"duplicate", "related", "conflicting", "distinct"}

CLASSIFICATION_SYSTEM_PROMPT = """You classify the relationship between two memories stored in a personal knowledge system.

## Classifications

- **duplicate**: Same fact or preference, possibly different wording. Safe to merge into one.
- **related**: Same topic but distinct information. Both should be kept separately.
- **conflicting**: Contradictory information about the same subject. Needs human resolution.
- **distinct**: Unrelated memories that happen to use similar language. Keep both.

## Guidelines

- Two memories about the same tool/preference ARE duplicates if they say the same thing.
- Two memories about the same topic with DIFFERENT details are related, not duplicates.
- "User works at Google" vs "User works at OpenAI" is conflicting (can't be both simultaneously).
- "User prefers Python" vs "User uses pytest" is related (same ecosystem, different facts).
- Prefer "related" over "duplicate" when in doubt — merging destroys information.
- Prefer "conflicting" when one memory would invalidate the other if both were true.

## Output

Return ONLY a JSON object:
{"classification": "duplicate|related|conflicting|distinct", "explanation": "brief reason", "confidence": 0.0-1.0}

No markdown, no other text. Just the JSON."""


def classify_memory_pair(
    content_a: str,
    content_b: str,
    model: str | None = None,
) -> dict:
    """Classify the relationship between two memories.

    Args:
        content_a: Content of first memory
        content_b: Content of second memory
        model: Optional model override (defaults to Gemini 2.5 Flash)

    Returns:
        Dict with keys:
            classification: "duplicate" | "related" | "conflicting" | "distinct"
            explanation: Brief reason for the classification
            confidence: Float 0.0-1.0
    """
    fallback = {
        "classification": "distinct",
        "explanation": "Classification failed — treating as distinct for safety",
        "confidence": 0.0,
    }

    try:
        client = get_client()
        use_model = model or CLASSIFIER_MODEL

        response = client.complete(
            prompt=f"Memory A:\n{content_a}\n\nMemory B:\n{content_b}",
            system=CLASSIFICATION_SYSTEM_PROMPT,
            model=use_model,
            temperature=0.0,
            max_tokens=200,
            fast_route=False,  # Gemini not on Cerebras/Groq/SambaNova
        )
    except Exception as e:
        logger.error(f"Memory classification LLM call failed: {e}")
        return fallback

    if not response:
        logger.warning("Empty response from memory classification LLM")
        return fallback

    return _parse_classification(response)


def _parse_classification(response: str) -> dict:
    """Parse and validate the LLM classification response.

    Args:
        response: Raw LLM response (should be JSON)

    Returns:
        Validated classification dict with guaranteed keys
    """
    result = parse_json_dict(response, log_errors=True)

    if not result:
        return {
            "classification": "distinct",
            "explanation": "Failed to parse LLM response as JSON",
            "confidence": 0.0,
        }

    classification = result.get("classification", "").lower().strip()
    if classification not in VALID_CLASSIFICATIONS:
        classification = "distinct"

    confidence = result.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    explanation = result.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = ""

    return {
        "classification": classification,
        "explanation": explanation,
        "confidence": confidence,
    }
