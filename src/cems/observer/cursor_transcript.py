"""Transcript extraction from Cursor IDE agent transcript files.

Cursor stores transcripts as plain text at:
    ~/.cursor/projects/{project}/agent-transcripts/{uuid}.txt

Format uses `user:` / `assistant:` delimiters with XML-like tags
for user content (<cursor_commands>, <user_query>) and bracket
markers for tool calls ([Tool call], [Tool result], [Thinking]).

Produces [USER], [ASSISTANT], [TOOL], [TOOL RESULT] lines
matching the unified transcript format.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Delimiters splitting turns
_TURN_RE = re.compile(r"^(user|assistant):\s*$", re.MULTILINE)


def extract_cursor_transcript_from_bytes(
    raw_bytes: bytes,
    max_chars: int = 100_000,
) -> str | None:
    """Parse raw Cursor transcript bytes and extract formatted text.

    Args:
        raw_bytes: Raw bytes from a Cursor transcript file.
        max_chars: Maximum output length.

    Returns:
        Formatted transcript text, or None if no useful content.
    """
    text = raw_bytes.decode("utf-8", errors="replace")
    return extract_cursor_transcript(text, max_chars=max_chars)


def extract_cursor_transcript(
    text: str,
    max_chars: int = 100_000,
) -> str | None:
    """Parse Cursor transcript text and extract formatted lines.

    Args:
        text: Full transcript text.
        max_chars: Maximum output length.

    Returns:
        Formatted transcript text, or None if no useful content.
    """
    # Split by turn delimiters
    parts = _TURN_RE.split(text)
    # parts alternates: [preamble, "user", user_content, "assistant", assistant_content, ...]

    all_lines: list[str] = []

    i = 1  # skip preamble
    while i < len(parts) - 1:
        role = parts[i].strip()
        content = parts[i + 1]
        i += 2

        if role == "user":
            all_lines.extend(_extract_user_lines(content))
        elif role == "assistant":
            all_lines.extend(_extract_assistant_lines(content))

    if not all_lines:
        return None

    # Apply max_chars limit
    result_lines = []
    total_chars = 0
    for line in all_lines:
        total_chars += len(line) + 2
        if total_chars > max_chars:
            break
        result_lines.append(line)

    return "\n\n".join(result_lines) if result_lines else None


def _extract_user_lines(content: str) -> list[str]:
    """Extract user message lines from a user turn."""
    lines = []

    # Extract <user_query> content (the actual user message)
    for match in re.finditer(r"<user_query>(.*?)</user_query>", content, re.DOTALL):
        text = match.group(1).strip()
        if text and len(text) > 5:
            # Strip slash commands
            text = re.sub(r"^/\w+\s*", "", text).strip()
            if text:
                lines.append(f"[USER]: {text[:2000]}")

    # If no <user_query> tag, look for text outside tags
    if not lines:
        # Strip all XML-like tags
        clean = re.sub(r"<[^>]+>.*?</[^>]+>", "", content, flags=re.DOTALL)
        clean = re.sub(r"<[^>]+>", "", clean)
        text = clean.strip()
        if text and len(text) > 10:
            lines.append(f"[USER]: {text[:2000]}")

    return lines


def _extract_assistant_lines(content: str) -> list[str]:
    """Extract assistant message and tool lines from an assistant turn."""
    lines = []

    # Split content by [Tool call], [Tool result], <think> markers
    segments = re.split(
        r"(\[Tool call\]|\[Tool result\]|<think>|</think>|\[Thinking\])",
        content,
    )

    in_think = False
    i = 0

    while i < len(segments):
        segment = segments[i]

        if segment == "<think>" or segment == "[Thinking]":
            in_think = True
            i += 1
            continue

        if segment == "</think>":
            in_think = False
            i += 1
            continue

        if in_think:
            i += 1
            continue

        if segment == "[Tool call]":
            # Next segment contains tool details
            if i + 1 < len(segments):
                tool_text = segments[i + 1].strip()
                tool_line = _parse_tool_call(tool_text)
                if tool_line:
                    lines.append(tool_line)
                i += 2
            else:
                i += 1
            continue

        if segment == "[Tool result]":
            # Next segment contains result
            if i + 1 < len(segments):
                result_text = segments[i + 1].strip()
                result_line = _parse_tool_result(result_text)
                if result_line:
                    lines.append(result_line)
                i += 2
            else:
                i += 1
            continue

        # Free text (assistant message)
        text = segment.strip()
        if text and len(text) > 20:
            # Skip if it's just tool result noise
            if not text.startswith(("  ", "\t")):
                lines.append(f"[ASSISTANT]: {text[:1000]}")

        i += 1

    return lines


def _parse_tool_call(text: str) -> str | None:
    """Parse a [Tool call] section into a formatted line.

    Format in transcript:
        [Tool call] ToolName
          param1: value1
          param2: value2
    """
    # First line is the tool name
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return None

    # Extract key parameters
    params = {}
    for line in text.split("\n")[1:]:
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            params[key.strip()] = value.strip()

    # Build summary based on tool name
    tool_name = first_line
    if tool_name == "Read" and "path" in params:
        return f"[TOOL] Read: {params['path']}"
    if tool_name in ("Edit", "MultiEdit") and "path" in params:
        return f"[TOOL] Edit: {params['path']}"
    if tool_name == "Write" and "path" in params:
        return f"[TOOL] Write: {params['path']}"
    if tool_name == "Bash" and "command" in params:
        cmd = params["command"][:120]
        return f"[TOOL] Bash: {cmd}"
    if tool_name == "Grep" and "pattern" in params:
        path = params.get("path", ".")
        return f"[TOOL] Grep: '{params['pattern']}' in {path}"
    if tool_name == "LS" and "target_directory" in params:
        return f"[TOOL] LS: {params['target_directory']}"

    return f"[TOOL] {tool_name}"


def _parse_tool_result(text: str) -> str | None:
    """Parse a [Tool result] section. Uses only the first line as a summary."""
    first_line = text.split("\n")[0].strip()
    if first_line:
        return f"[TOOL RESULT]: {first_line}"
    return None
