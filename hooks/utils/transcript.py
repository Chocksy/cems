#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///
"""Utility helpers for reading Claude Code JSONL transcript files."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import List

__all__ = ["read_last_messages"]


def _extract_text(entry: dict) -> str | None:
    """Return readable text from a single transcript JSON entry."""
    if not isinstance(entry, dict):
        return None
    
    # Claude Code transcript structure: entry.message.content (string or array)
    message = entry.get("message", {})
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        elif isinstance(content, list) and content:
            # Extract text from content array (user/assistant messages)
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(str(item["text"]))
                    elif "input" in item:  # tool calls
                        texts.append(f"Tool: {item.get('name', 'unknown')}")
                elif isinstance(item, str):
                    texts.append(item)
            if texts:
                return " ".join(texts).strip()
    
    # Fallback: direct content field
    content = entry.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    
    # System messages often have plain content
    if entry.get("type") == "system" and isinstance(entry.get("content"), str):
        return entry["content"].strip()

    return None


def read_last_messages(transcript_path: str | Path, max_lines: int = 40, max_chars: int = 1000) -> str:
    """Return a concatenated string of the last few messages in a transcript.

    Only the *content* field (or similar) is extracted to keep things short.

    Args:
        transcript_path: Path to the JSONL transcript file.
        max_lines: Maximum number of lines to read from file tail.
        max_chars: Truncate output to this many characters.

    Returns:
        Concatenated text or empty string if not available.
    """
    p = Path(transcript_path).expanduser()
    if not p.exists() or not p.is_file():
        return ""

    try:
        tail: deque[str] = deque(maxlen=max_lines)
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tail.append(line.rstrip())
    except Exception:
        return ""

    parts: List[str] = []
    for raw in tail:
        try:
            entry = json.loads(raw)
        except Exception:
            continue
        txt = _extract_text(entry)
        if txt:
            parts.append(txt)

    text = "\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]  # keep last section
    return text 