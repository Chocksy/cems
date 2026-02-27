"""JSON parsing utilities for LLM responses.

Centralizes the repeated pattern of extracting JSON from LLM responses
that may be wrapped in markdown code blocks.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> str:
    """Extract JSON content from LLM response, handling markdown code blocks.

    LLM responses often wrap JSON in markdown code blocks like:
        ```json
        {"key": "value"}
        ```

    This function strips those wrappers to get the raw JSON string.

    Args:
        response: Raw LLM response string

    Returns:
        Cleaned JSON string ready for json.loads()

    Example:
        >>> extract_json_from_response('```json\\n{"a": 1}\\n```')
        '{"a": 1}'
        >>> extract_json_from_response('{"a": 1}')
        '{"a": 1}'
        >>> extract_json_from_response('Here is JSON:\\n```json\\n[1,2]\\n```')
        '[1,2]'
    """
    response = response.strip()

    # Try to extract from markdown code blocks anywhere in the response
    if "```" in response:
        # First try: complete code block with closing backticks
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if match:
            return match.group(1).strip()

        # Fallback: opening backticks without closing (truncated response)
        match = re.search(r"```(?:json)?\s*([\s\S]+)", response)
        if match:
            return match.group(1).strip()

    return response


def _parse_json_as(
    response: str,
    expected_type: type,
    *,
    fallback: Any,
    log_errors: bool = True,
) -> Any:
    """Parse LLM response as a specific JSON type.

    Args:
        response: Raw LLM response string
        expected_type: Expected Python type (list or dict)
        fallback: Value to return on parse failure
        log_errors: Whether to log parse failures

    Returns:
        Parsed value of expected_type, or fallback on failure
    """
    try:
        cleaned = extract_json_from_response(response)
        data = json.loads(cleaned)

        if not isinstance(data, expected_type):
            if log_errors:
                logger.warning(
                    f"Expected JSON {expected_type.__name__}, "
                    f"got {type(data).__name__}: {cleaned[:100]}"
                )
            return fallback

        return data

    except json.JSONDecodeError as e:
        if log_errors:
            logger.warning(f"JSON parse failed: {e}. Response: {response[:200]}")
        return fallback


def parse_json_list(
    response: str,
    *,
    fallback: list | None = None,
    log_errors: bool = True,
) -> list[Any]:
    """Parse LLM response as JSON list with error handling.

    Example:
        >>> parse_json_list('[1, 2, 3]')
        [1, 2, 3]
        >>> parse_json_list('invalid', fallback=[])
        []
    """
    return _parse_json_as(
        response, list, fallback=fallback if fallback is not None else [], log_errors=log_errors
    )


def parse_json_dict(
    response: str,
    *,
    fallback: dict | None = None,
    log_errors: bool = True,
) -> dict[str, Any]:
    """Parse LLM response as JSON dict with error handling.

    Example:
        >>> parse_json_dict('{"key": "value"}')
        {'key': 'value'}
        >>> parse_json_dict('invalid', fallback={})
        {}
    """
    return _parse_json_as(
        response, dict, fallback=fallback if fallback is not None else {}, log_errors=log_errors
    )
