#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///

"""
Stop Hook - Session Logging + Relevance Feedback

This hook runs when Claude stops (fires after EVERY assistant turn) and:
1. Logs session data (stop.json, optional chat.json)
2. Parses Claude's "Memory relevance:" line and sends feedback to CEMS

NOTE: We intentionally do NOT write stop signals here. Claude Code fires
the Stop hook after every assistant turn, not just session exit. The observer
daemon's staleness detection handles session finalization.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import ensure_session_log_dir
from utils.credentials import get_cems_key, get_cems_url
from utils.hook_logger import log_hook_event

# CEMS configuration
CEMS_API_URL = get_cems_url()
CEMS_API_KEY = get_cems_key()

RELEVANCE_CACHE_DIR = Path.home() / ".cems" / "cache" / "relevance"

# Regex to find the "Memory relevance:" line anywhere in text
_RELEVANCE_LINE_RE = re.compile(
    r'^Memory relevance:\s*(.+)$',
    re.MULTILINE | re.IGNORECASE,
)

# Regex to extract #N references
_HASH_NUM_RE = re.compile(r'#(\d+)')


def parse_relevance_line(text: str) -> dict | None:
    """Parse Claude's memory relevance feedback line.

    Handles formats like:
    - "Memory relevance: #1 #3 were relevant, #2 #4 were noise"
    - "Memory relevance: none were relevant to this task"
    - "Memory relevance: #1 was relevant, rest were noise"
    - "Memory relevance: #1 #2 were relevant (helped with X), #3 was noise"

    Returns:
        {"relevant": [1, 3], "noise": [2, 4]} or
        {"all_noise": True} or
        None if no relevance line found
    """
    if not text:
        return None

    match = _RELEVANCE_LINE_RE.search(text)
    if not match:
        return None

    body = match.group(1).strip()

    # "none were relevant" → all shown memories are noise
    if re.search(r'\bnone\b.*\brelevant\b', body, re.IGNORECASE):
        return {"all_noise": True}

    relevant = []
    noise = []

    # Split on comma to get clauses
    clauses = [c.strip() for c in body.split(',')]

    for clause in clauses:
        numbers = [int(n) for n in _HASH_NUM_RE.findall(clause)]
        is_noise = bool(re.search(r'\bnoise\b', clause, re.IGNORECASE))
        is_relevant = bool(re.search(r'\brelevant\b', clause, re.IGNORECASE))

        # "rest were noise" — handled later via set difference
        if re.search(r'\brest\b.*\bnoise\b', clause, re.IGNORECASE) and not numbers:
            # Mark as "rest_noise" for post-processing
            noise.append(-1)  # sentinel
            continue

        if is_noise and not is_relevant:
            noise.extend(numbers)
        elif is_relevant:
            relevant.extend(numbers)
        elif numbers:
            # Default: if no keyword, treat as relevant
            relevant.extend(numbers)

    # Handle "rest were noise" sentinel
    if -1 in noise:
        noise.remove(-1)
        return {"relevant": relevant, "noise": noise, "rest_noise": True}

    if relevant or noise:
        return {"relevant": relevant, "noise": noise}

    return None


def send_relevance_feedback(session_id: str, input_data: dict) -> None:
    """Parse relevance line from Claude's response and send feedback to CEMS.

    Reads `last_assistant_message` from the Stop hook's input_data (provided
    by Claude Code), parses the relevance line, maps #N to memory IDs via
    the mapping file, and POSTs feedback to CEMS.
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return

    # Get last_assistant_message from input
    last_message = input_data.get("last_assistant_message", "")
    if not last_message:
        return

    # Parse the relevance line
    parsed = parse_relevance_line(last_message)
    if parsed is None:
        return

    # Read the mapping file
    short_sid = session_id[:12] if session_id else ""
    if not short_sid:
        return

    mapping_path = RELEVANCE_CACHE_DIR / f"{short_sid}.json"
    if not mapping_path.exists():
        return

    try:
        mapping = json.loads(mapping_path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    memory_ids = mapping.get("memory_ids", [])
    if not memory_ids:
        return

    # Staleness check: skip if mapping is >60s old (prevents stale mapping from wrong turn)
    mapping_ts = mapping.get("ts", 0)
    if time.time() - mapping_ts > 60:
        _cleanup_mapping(mapping_path)
        return

    # Map #N → memory_id (1-indexed)
    relevant_ids = []
    noise_ids = []

    if parsed.get("all_noise"):
        # "none were relevant" → all shown memories are noise
        noise_ids = list(memory_ids)
    else:
        for n in parsed.get("relevant", []):
            if 1 <= n <= len(memory_ids):
                relevant_ids.append(memory_ids[n - 1])
        for n in parsed.get("noise", []):
            if 1 <= n <= len(memory_ids):
                noise_ids.append(memory_ids[n - 1])

        # Handle "rest were noise" — all IDs not in relevant set
        if parsed.get("rest_noise"):
            relevant_set = set(relevant_ids)
            for mid in memory_ids:
                if mid not in relevant_set and mid not in noise_ids:
                    noise_ids.append(mid)

    if not relevant_ids and not noise_ids:
        _cleanup_mapping(mapping_path)
        return

    # Fire-and-forget POST to CEMS
    try:
        httpx.post(
            f"{CEMS_API_URL}/api/memory/log-relevance",
            json={"relevant_ids": relevant_ids, "noise_ids": noise_ids},
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=3.0,
        )
    except (httpx.RequestError, httpx.TimeoutException):
        pass

    log_hook_event("RelevanceFeedback", session_id, {
        "relevant_count": len(relevant_ids),
        "noise_count": len(noise_ids),
        "all_noise": parsed.get("all_noise", False),
    })

    _cleanup_mapping(mapping_path)


def _cleanup_mapping(mapping_path: Path) -> None:
    """Delete the mapping file after processing."""
    try:
        mapping_path.unlink(missing_ok=True)
    except OSError:
        pass


def read_transcript(transcript_path: str) -> list[dict] | None:
    """Read transcript from .jsonl file and return as list of messages."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        messages = []
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return messages if messages else None
    except Exception:
        return None


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--chat", action="store_true", help="Copy transcript to chat.json"
        )
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract required fields
        session_id = input_data.get("session_id", "")
        transcript_path = input_data.get("transcript_path", "")
        cwd = input_data.get("cwd", os.getcwd())

        log_hook_event("Stop", session_id, {"cwd": cwd}, input_data=input_data)

        # --- Relevance feedback (parse + send before session logging) ---
        send_relevance_feedback(session_id, input_data)

        # --- Session logging ---
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "stop.json"

        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        log_data.append(input_data)

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Handle --chat switch
        if args.chat and transcript_path:
            transcript = read_transcript(transcript_path)
            if transcript:
                chat_file = log_dir / "chat.json"
                with open(chat_file, "w") as f:
                    json.dump(transcript, f, indent=2)

        # NOTE: We intentionally do NOT write a "stop" signal here.
        # Claude Code fires the Stop hook after EVERY assistant turn, not just
        # session exit. Writing a stop signal would permanently mark the session
        # as done (is_done=True) after the first response. The observer daemon's
        # staleness detection handles finalization when the session truly goes idle.

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
