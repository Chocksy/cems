"""Transcript extraction from Claude Code JSONL session files.

Extracts readable text from JSONL entries, including:
- User text messages (raw strings and content blocks)
- Assistant text responses
- Tool action summaries (Read, Edit, Write, Bash, Grep, Glob, WebFetch, WebSearch)

After extraction, tool lines can be compacted into activity summaries
to reduce noise before sending to the observer LLM.

Shared by both the observer daemon (read_content_delta) and stop.py hook.
"""

import json
import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Tool names whose actions are worth summarizing for observations
SUMMARIZABLE_TOOLS = {
    "Read", "Edit", "Write", "Bash", "Grep", "Glob",
    "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch",
}


def _summarize_tool_use(name: str, tool_input: dict) -> str | None:
    """Create a one-line summary of a tool_use block.

    Returns a short string like:
        "Read: src/cems/server.py"
        "Edit: src/cems/config.py"
        "Bash: docker compose build"
        "Write: tests/test_new.py"
        "Grep: 'pattern' in src/"
    """
    if name not in SUMMARIZABLE_TOOLS:
        return None

    if name == "Read":
        path = tool_input.get("file_path", "")
        return f"Read: {path}" if path else None

    if name in ("Edit", "MultiEdit"):
        path = tool_input.get("file_path", "")
        return f"Edit: {path}" if path else None

    if name == "Write":
        path = tool_input.get("file_path", "")
        return f"Write: {path}" if path else None

    if name == "Bash":
        cmd = tool_input.get("command", "")
        # Truncate long commands but preserve first meaningful part
        cmd = cmd.strip().split("\n")[0][:120]
        return f"Bash: {cmd}" if cmd else None

    if name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", ".")
        return f"Grep: '{pattern}' in {path}" if pattern else None

    if name == "Glob":
        pattern = tool_input.get("pattern", "")
        return f"Glob: {pattern}" if pattern else None

    if name == "NotebookEdit":
        path = tool_input.get("notebook_path", "")
        return f"NotebookEdit: {path}" if path else None

    if name == "WebFetch":
        url = tool_input.get("url", "")
        if url:
            domain = urlparse(url).netloc
            return f"WebFetch: {domain}" if domain else f"WebFetch: {url[:80]}"
        return None

    if name == "WebSearch":
        query = tool_input.get("query", "")
        return f"WebSearch: '{query}'" if query else None

    return None


def extract_message_lines(entry: dict) -> list[str]:
    """Extract readable lines from a single JSONL entry.

    Handles both user and assistant entries, extracting:
    - Text content blocks
    - Tool action summaries (for assistant tool_use blocks)
    - Raw string user messages

    Skips: tool_result blocks, thinking blocks, progress, system entries.

    Args:
        entry: Parsed JSONL entry dict.

    Returns:
        List of formatted lines (may be empty if entry has no useful content).
    """
    msg_type = entry.get("type", "")
    if msg_type not in ("user", "assistant"):
        return []

    # Skip meta/system-injected messages
    if entry.get("isMeta"):
        return []

    message = entry.get("message", {})
    if not isinstance(message, dict):
        return []

    content = message.get("content", "")
    lines = []

    if msg_type == "user":
        # User messages come in two forms:
        # 1. Raw string (actual user typing)
        if isinstance(content, str):
            text = content.strip()[:2000]
            if text and len(text) > 5:
                # Skip system commands and compact summaries
                if not text.startswith("<") and not text.startswith("/"):
                    lines.append(f"[USER]: {text}")

        # 2. Content block array (may contain text or tool_result)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "").strip()[:2000]
                    if text and len(text) > 10:
                        # Skip skill injections and system messages
                        if not text.startswith("Base directory for this skill:"):
                            lines.append(f"[USER]: {text}")
                # Skip tool_result blocks — too verbose, low signal

    elif msg_type == "assistant":
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")

                if block_type == "text":
                    text = block.get("text", "").strip()[:1000]
                    if text and len(text) > 20:
                        lines.append(f"[ASSISTANT]: {text}")

                elif block_type == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    if isinstance(tool_input, dict):
                        summary = _summarize_tool_use(tool_name, tool_input)
                        if summary:
                            lines.append(f"[TOOL] {summary}")

                # Skip thinking blocks — internal reasoning, not observable actions

    return lines


def compact_tool_lines(lines: list[str]) -> list[str]:
    """Compact consecutive [TOOL] lines into [ACTIVITY] summaries.

    Groups consecutive tool lines and aggregates by operation type:
    - File reads/edits/writes grouped by common directory
    - Bash commands listed or summarized
    - Grep/Glob summarized as search activity
    - Web fetches/searches preserved with domains/queries

    Non-tool lines ([USER], [ASSISTANT]) are preserved as-is and act
    as group boundaries for tool compaction.

    Args:
        lines: List of formatted transcript lines.

    Returns:
        New list with [TOOL] lines replaced by [ACTIVITY] summaries.
    """
    result = []
    tool_buffer: list[str] = []

    for line in lines:
        if line.startswith("[TOOL] "):
            tool_buffer.append(line)
        else:
            if tool_buffer:
                result.extend(_flush_tool_group(tool_buffer))
                tool_buffer = []
            result.append(line)

    if tool_buffer:
        result.extend(_flush_tool_group(tool_buffer))

    return result


def _flush_tool_group(tool_lines: list[str]) -> list[str]:
    """Convert a group of consecutive [TOOL] lines into [ACTIVITY] summaries."""
    reads: list[str] = []
    edits: list[str] = []
    writes: list[str] = []
    bash_cmds: list[str] = []
    searches: list[str] = []
    web_items: list[str] = []

    for line in tool_lines:
        content = line[7:]  # Strip "[TOOL] "
        if content.startswith("Read: "):
            reads.append(content[6:])
        elif content.startswith("Edit: "):
            edits.append(content[6:])
        elif content.startswith("Write: "):
            writes.append(content[7:])
        elif content.startswith("Bash: "):
            bash_cmds.append(content[6:])
        elif content.startswith(("Grep: ", "Glob: ")):
            searches.append(content)
        elif content.startswith(("WebFetch: ", "WebSearch: ")):
            web_items.append(content)
        elif content.startswith("NotebookEdit: "):
            edits.append(content[14:])

    activities: list[str] = []

    if reads:
        activities.append(_compact_file_ops("Read", reads))
    if edits:
        activities.append(_compact_file_ops("Edited", edits))
    if writes:
        activities.append(_compact_file_ops("Created", writes))
    if bash_cmds:
        activities.append(_compact_bash(bash_cmds))
    if searches:
        activities.append(_compact_searches(searches))
    if web_items:
        activities.extend(_compact_web(web_items))

    return [f"[ACTIVITY] {a}" for a in activities]


def _compact_file_ops(verb: str, paths: list[str]) -> str:
    """Summarize file operations by common directory prefix."""
    if len(paths) == 1:
        return f"{verb} {paths[0]}"

    # Find common directory
    try:
        dirs = [os.path.dirname(p) for p in paths]
        common = os.path.commonpath(dirs) if all(dirs) else ""
    except ValueError:
        common = ""

    filenames = [os.path.basename(p) for p in paths]

    if common and common not in (".", "/"):
        if len(filenames) <= 3:
            return f"{verb} {len(paths)} files in {common}/ ({', '.join(filenames)})"
        return f"{verb} {len(paths)} files in {common}/"

    if len(filenames) <= 3:
        return f"{verb} {len(paths)} files ({', '.join(filenames)})"
    return f"{verb} {len(paths)} files across multiple directories"


def _compact_bash(commands: list[str]) -> str:
    """Summarize bash commands."""
    if len(commands) == 1:
        return f"Ran: {commands[0]}"

    # Check if all commands share a common prefix (e.g., "docker compose")
    first_words = [c.split()[0] if c.split() else "?" for c in commands]
    if len(set(first_words)) == 1:
        return f"Ran {len(commands)} {first_words[0]} commands"

    if len(commands) <= 3:
        short = [c[:60] for c in commands]
        return f"Ran commands: {'; '.join(short)}"
    return f"Ran {len(commands)} shell commands"


def _compact_searches(searches: list[str]) -> str:
    """Summarize grep/glob searches."""
    if len(searches) == 1:
        return f"Searched: {searches[0]}"
    return f"Searched codebase ({len(searches)} queries)"


def _compact_web(web_items: list[str]) -> list[str]:
    """Summarize web visits, preserving domains and queries."""
    results: list[str] = []
    for item in web_items:
        if item.startswith("WebFetch: "):
            domain = item[10:]
            results.append(f"Visited {domain}")
        elif item.startswith("WebSearch: "):
            query = item[11:]
            results.append(f"Web search: {query}")
    return results


def extract_transcript_text(
    entries: list[dict],
    max_chars: int = 100_000,
    compact: bool = False,
) -> str:
    """Convert a list of JSONL entries into readable transcript text.

    Used by both stop.py (full transcript) and observer daemon (delta).

    Args:
        entries: List of parsed JSONL entry dicts.
        max_chars: Maximum output length in characters.
        compact: If True, aggregate consecutive tool lines into activity summaries.

    Returns:
        Formatted transcript text with [USER], [ASSISTANT], and
        [TOOL] or [ACTIVITY] prefixes.
    """
    all_lines = []

    for entry in entries:
        all_lines.extend(extract_message_lines(entry))

    if compact:
        all_lines = compact_tool_lines(all_lines)

    # Apply max_chars limit
    result_lines = []
    total_chars = 0
    for line in all_lines:
        total_chars += len(line) + 2  # +2 for newline separator
        if total_chars > max_chars:
            break
        result_lines.append(line)

    return "\n\n".join(result_lines)


def extract_transcript_from_bytes(
    raw_bytes: bytes,
    max_chars: int = 100_000,
    compact: bool = False,
) -> str | None:
    """Parse raw JSONL bytes and extract transcript text.

    Used by the observer daemon's read_content_delta which reads
    raw bytes from a specific offset.

    Args:
        raw_bytes: Raw bytes from a JSONL file.
        max_chars: Maximum output length.
        compact: If True, aggregate tool lines into activity summaries.

    Returns:
        Formatted transcript text, or None if no useful content.
    """
    entries = []
    for raw_line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entries.append(json.loads(raw_line))
        except json.JSONDecodeError:
            continue

    if not entries:
        return None

    text = extract_transcript_text(entries, max_chars=max_chars, compact=compact)
    return text if text else None
