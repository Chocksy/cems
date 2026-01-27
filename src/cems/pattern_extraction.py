"""Pattern extraction for gate rules.

This module provides utilities for extracting structured patterns from
gate rule memories for use in PreToolUse hook gating.

Gate rule memory format:
    <Tool>: <pattern> — <reason>

Optional fields in memory metadata/tags:
    - severity: block|warn|confirm (default: warn)
    - source_ref: project:<org/repo> for project-scoped rules

Example:
    Input: "Bash: coolify deploy — Never use CLI for production deployments"
    Output: {
        "tool": "bash",
        "pattern": "coolify\\s+deploy",
        "raw_pattern": "coolify deploy",
        "reason": "Never use CLI for production deployments",
        "severity": "warn"
    }
"""

from __future__ import annotations

import re
from typing import Literal

# Severity levels for gate rules
Severity = Literal["block", "warn", "confirm"]

# Pattern for parsing gate rule content
# Format: "Tool: pattern — reason"
# Uses em dash (—), en dash (–), or " - " (hyphen with spaces) as separator
# The key is requiring whitespace around hyphens to distinguish from command flags
GATE_RULE_PATTERN = re.compile(
    r"^(?P<tool>\w+):\s*(?P<pattern>.+?)\s*(?:—|–|\s-\s)\s*(?P<reason>.+)$",
    re.IGNORECASE | re.DOTALL,
)


def pattern_to_regex(pattern: str) -> re.Pattern[str]:
    r"""Convert a human-readable pattern to a regex.

    Handles common pattern conventions:
    - Spaces become ``\s+`` (flexible whitespace)
    - ``*`` becomes ``.*`` (glob-style wildcard)
    - Special regex chars are escaped

    Args:
        pattern: Human-readable pattern like "coolify deploy"

    Returns:
        Compiled regex pattern
    """
    # First, escape special regex characters (except * which we handle specially)
    escaped = re.escape(pattern)

    # Convert escaped \* back to .* for glob-style matching
    escaped = escaped.replace(r"\*", ".*")

    # Convert spaces to flexible whitespace
    escaped = re.sub(r"\\ ", r"\\s+", escaped)

    return re.compile(escaped, re.IGNORECASE)


def extract_severity_from_tags(tags: list[str] | None) -> Severity:
    """Extract severity level from memory tags.

    Args:
        tags: List of tags from memory metadata

    Returns:
        Severity level (block, warn, or confirm)
    """
    if not tags:
        return "warn"

    tags_lower = [t.lower() for t in tags]

    if "block" in tags_lower:
        return "block"
    if "confirm" in tags_lower:
        return "confirm"
    if "warn" in tags_lower:
        return "warn"

    return "warn"


def extract_gate_pattern(
    content: str,
    tags: list[str] | None = None,
    source_ref: str | None = None,
) -> dict | None:
    """Parse gate rule memory content into a structured pattern.

    Args:
        content: Memory content in format "Tool: pattern — reason"
        tags: Optional tags from memory metadata (for severity extraction)
        source_ref: Optional source reference (e.g., "project:org/repo")

    Returns:
        Structured gate pattern dict or None if parsing fails:
        {
            "tool": str,           # Tool name (lowercase)
            "pattern": str,        # Regex pattern string
            "raw_pattern": str,    # Original human-readable pattern
            "regex": re.Pattern,   # Compiled regex
            "reason": str,         # Human-readable reason
            "severity": str,       # block|warn|confirm
            "project": str|None,   # Project scope (org/repo) if any
        }
    """
    if not content or not content.strip():
        return None

    # Try to parse the content
    match = GATE_RULE_PATTERN.match(content.strip())
    if not match:
        return None

    tool = match.group("tool").strip().lower()
    raw_pattern = match.group("pattern").strip()
    reason = match.group("reason").strip()

    if not tool or not raw_pattern or not reason:
        return None

    # Convert pattern to regex
    try:
        regex = pattern_to_regex(raw_pattern)
    except re.error:
        return None

    # Extract severity from tags
    severity = extract_severity_from_tags(tags)

    # Extract project scope from source_ref
    project = None
    if source_ref and source_ref.startswith("project:"):
        project = source_ref[8:]  # Remove "project:" prefix

    return {
        "tool": tool,
        "pattern": regex.pattern,
        "raw_pattern": raw_pattern,
        "regex": regex,
        "reason": reason,
        "severity": severity,
        "project": project,
    }


def serialize_gate_pattern(pattern: dict) -> dict:
    """Serialize a gate pattern for JSON storage.

    Converts the compiled regex to a string for JSON serialization.

    Args:
        pattern: Gate pattern dict from extract_gate_pattern()

    Returns:
        JSON-serializable dict (regex compiled pattern stored as string)
    """
    return {
        "tool": pattern["tool"],
        "pattern": pattern["pattern"],
        "raw_pattern": pattern["raw_pattern"],
        "reason": pattern["reason"],
        "severity": pattern["severity"],
        "project": pattern["project"],
    }


def deserialize_gate_pattern(data: dict) -> dict:
    """Deserialize a gate pattern from JSON storage.

    Recompiles the regex from the stored pattern string.

    Args:
        data: Serialized gate pattern dict

    Returns:
        Gate pattern dict with compiled regex
    """
    return {
        "tool": data["tool"],
        "pattern": data["pattern"],
        "raw_pattern": data["raw_pattern"],
        "regex": re.compile(data["pattern"], re.IGNORECASE),
        "reason": data["reason"],
        "severity": data["severity"],
        "project": data.get("project"),
    }
