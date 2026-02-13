"""Transcript extraction from Codex CLI JSONL session files.

Handles two JSONL formats:
- New format (2026+): Uses "type" field with session_meta, response_item, event_msg, turn_context
- Old format (2025): Uses "record_type" field with direct message objects

Produces [USER], [ASSISTANT], [TOOL], [ACTIVITY] lines matching Claude transcript format.
"""

import json
import logging

logger = logging.getLogger(__name__)


def detect_format(first_line: dict) -> str:
    """Detect Codex JSONL format version from the first record.

    Returns "new" or "old".
    """
    if first_line.get("type") == "session_meta":
        return "new"
    if "record_type" in first_line:
        return "old"
    # Default to new format if ambiguous
    return "new"


def extract_text_from_record(record: dict, fmt: str) -> list[str]:
    """Extract readable lines from a single Codex JSONL record.

    Args:
        record: Parsed JSONL record dict.
        fmt: Format version ("new" or "old").

    Returns:
        List of formatted lines.
    """
    if fmt == "new":
        return _extract_new_format(record)
    return _extract_old_format(record)


def _extract_new_format(record: dict) -> list[str]:
    """Extract from new Codex format (2026+)."""
    record_type = record.get("type", "")
    payload = record.get("payload", {})

    if record_type == "session_meta":
        return []  # metadata only, no content

    if record_type == "turn_context":
        return []  # context metadata, not content

    if record_type == "event_msg":
        event_type = payload.get("type", "")

        if event_type == "user_message":
            text = payload.get("message", "").strip()
            if text and len(text) > 5:
                return [f"[USER]: {text[:2000]}"]

        if event_type == "agent_message":
            text = payload.get("message", "").strip()
            if text and len(text) > 20:
                return [f"[ASSISTANT]: {text[:1000]}"]

        # Skip agent_reasoning, token_count
        return []

    if record_type == "response_item":
        return _extract_response_item(payload)

    return []


def _extract_response_item(payload: dict) -> list[str]:
    """Extract from a response_item payload."""
    item_type = payload.get("type", "")
    role = payload.get("role", "")
    content = payload.get("content", [])
    lines = []

    if item_type == "message":
        if role == "user":
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict):
                    text = block.get("text", "").strip()
                    if text and len(text) > 5:
                        lines.append(f"[USER]: {text[:2000]}")
        elif role == "assistant":
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type in ("text", "output_text"):
                        text = block.get("text", "").strip()
                        if text and len(text) > 20:
                            lines.append(f"[ASSISTANT]: {text[:1000]}")
        # Skip developer role (system instructions)

    elif item_type == "function_call":
        name = payload.get("name", "unknown")
        args = payload.get("arguments", "")
        # Truncate long arguments
        if isinstance(args, str) and len(args) > 200:
            args = args[:200] + "..."
        lines.append(f"[TOOL] {name}: {args}")

    elif item_type == "function_call_output":
        output = payload.get("output", "")
        if isinstance(output, str) and len(output) > 200:
            output = output[:200] + "..."
        if output:
            lines.append(f"[TOOL RESULT]: {output}")

    elif item_type == "reasoning":
        pass  # Skip reasoning/thinking

    return lines


def _extract_old_format(record: dict) -> list[str]:
    """Extract from old Codex format (2025)."""
    record_type = record.get("record_type", "")
    lines = []

    if record_type == "user_message":
        text = record.get("text", "").strip()
        if text and len(text) > 5:
            lines.append(f"[USER]: {text[:2000]}")

    elif record_type == "assistant_message":
        text = record.get("text", "").strip()
        if text and len(text) > 20:
            lines.append(f"[ASSISTANT]: {text[:1000]}")

    elif record_type == "tool_call":
        name = record.get("name", "unknown")
        args = record.get("arguments", "")
        if isinstance(args, str) and len(args) > 200:
            args = args[:200] + "..."
        lines.append(f"[TOOL] {name}: {args}")

    elif record_type == "tool_result":
        output = record.get("output", "")
        if isinstance(output, str) and len(output) > 200:
            output = output[:200] + "..."
        if output:
            lines.append(f"[TOOL RESULT]: {output}")

    return lines


def extract_codex_transcript_from_bytes(
    raw_bytes: bytes,
    max_chars: int = 100_000,
) -> str | None:
    """Parse raw Codex JSONL bytes and extract transcript text.

    Args:
        raw_bytes: Raw bytes from a Codex JSONL file.
        max_chars: Maximum output length.

    Returns:
        Formatted transcript text, or None if no useful content.
    """
    records = []
    for raw_line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            records.append(json.loads(raw_line))
        except json.JSONDecodeError:
            continue

    if not records:
        return None

    fmt = detect_format(records[0])
    all_lines = []

    for record in records:
        all_lines.extend(extract_text_from_record(record, fmt))

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
