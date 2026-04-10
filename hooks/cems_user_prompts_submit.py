#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///

"""
CEMS UserPromptSubmit Hook - Memory Awareness + Ultrathink + Gate Cache

This hook runs on every user prompt and:
1. Searches CEMS for relevant memories and injects them as context
2. Appends ultrathink instruction if -u flag is present
3. Populates gate rule cache on session start (for PreToolUse hook)

Configuration (environment variables):
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from utils.credentials import CEMSClient
from utils.hook_logger import log_hook_event
from utils.project import get_project_id
from utils.transcript import read_last_assistant_message

CONFIRMATORY = re.compile(
    r'^(yes|yeah|yep|yup|ok|okay|sure|go|do it|proceed|'
    r'go ahead|go for it|sounds good|lgtm|ship it|'
    r'yes please|confirmed|let\'s do it|approve)[\s!.]*$',
    re.IGNORECASE,
)


def is_confirmatory(prompt: str) -> bool:
    """Check if prompt is a short confirmatory response like 'yes' or 'go ahead'."""
    clean = prompt.strip().removesuffix('-u').strip()
    if CONFIRMATORY.match(clean):
        return True
    # Very short non-question, non-slash prompts are also confirmatory
    if len(clean) < 8 and '?' not in clean and not clean.startswith('/'):
        return True
    return False


_URL_RE = re.compile(r'https?://\S+')
_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
_INLINE_CODE_RE = re.compile(r'`[^`]{10,}`')  # Only strip long inline code (10+ chars)
_FILE_PATH_RE = re.compile(r'(?:^|\s)(?:/[\w./-]+|~[\w./-]+)')
_FILLER_RE = re.compile(
    r'^(?:ok(?:ay)?|alright|so|now|also|and|but|well|right|cool|great|nice|perfect|awesome)'
    r'(?:\s+(?:all good|good|then|let\'?s|,|\.))?\s*[.,;]?\s*',
    re.IGNORECASE,
)


def _clean_search_query(text: str) -> str:
    """Clean raw user text into a better search query.

    Strips URLs, code blocks, file paths, and conversational filler
    that hurt embedding similarity. Conservative — only removes
    patterns that are clearly noise for search.
    """
    cleaned = _URL_RE.sub('', text)
    cleaned = _CODE_BLOCK_RE.sub('', cleaned)
    cleaned = _INLINE_CODE_RE.sub('', cleaned)
    cleaned = _FILE_PATH_RE.sub(' ', cleaned)
    cleaned = _FILLER_RE.sub('', cleaned)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # If cleaning removed too much, fall back to original
    if len(cleaned) < 10:
        return text.strip()
    return cleaned


def _format_agentic_response(data: dict) -> tuple[str | None, list[str], list[bool], list[dict]]:
    """Format agentic search results with entity-first layout.

    Returns (formatted_string, memory_ids, truncation_flags, score_details) tuple.
    """
    parts = []
    memory_ids = []
    truncation_flags = []
    score_details = []

    entities = data.get("entities", [])
    memories = data.get("memories", data.get("results", []))[:3]

    if entities:
        parts.append("KNOWLEDGE TOPICS matching your query:\n")
        entity_ids = []
        for i, e in enumerate(entities, 1):
            sources = f" ({e['sources']} sources)" if e.get("sources") else ""
            parts.append(f"{i}. {e['title']}{sources}")
            if e.get("summary"):
                parts.append(f"   {e['summary']}")
            short_id = e.get("id", "")[:8]
            entity_ids.append(short_id)
            if e.get("id"):
                memory_ids.append(e["id"])
        parts.append("")

        # Imperative recall instructions — entities are pre-curated documents
        # that the LLM MUST read before responding
        parts.append("REQUIRED: Fetch these knowledge pages before responding:")
        for short_id in entity_ids:
            parts.append(f"  /recall {short_id}")
        parts.append("These are curated documents relevant to your task. Read them first.")
        parts.append("")

    if memories:
        parts.append("RELEVANT MEMORIES:")
        for i, m in enumerate(memories, 1):
            category = m.get("category", "general")
            content = m.get("content", m.get("memory", ""))
            mem_id = m.get("memory_id", m.get("id", ""))
            short_id = mem_id[:8] if mem_id else ""
            score = m.get("score", 0.0)
            truncated = m.get("truncated", False)
            full_len = m.get("full_length", 0)
            suffix = f" [truncated \u2014 full doc: {full_len} chars]" if truncated else ""
            parts.append(f"{i}. [{category}] (score: {score:.2f}) {content}{suffix} (id: {short_id})")
            if truncated:
                parts.append(f"   Use /recall {short_id} to read the full document.")
            truncation_flags.append(truncated)
            if mem_id:
                memory_ids.append(mem_id)
            score_details.append({"id": short_id, "score": round(score, 3), "category": category, "content": content})

    if not entities and not memories:
        return None, [], [], []

    # Add retrieval summary
    n_entities = len(entities)
    n_memories = len(memories)
    total_candidates = data.get("total_candidates", 0)
    entity_candidates = data.get("entity_candidates", 0)
    parts.append(
        f"\n--- Agentic retrieval: {n_entities} topics, {n_memories} memories "
        f"from {total_candidates} memories + {entity_candidates} entity pages ---"
    )
    parts.append("Use /recall <id> to read full topic pages for deeper context.")

    return "\n".join(parts), memory_ids, truncation_flags, score_details


def search_cems(client: CEMSClient, query: str, project: str | None = None) -> tuple[str | None, list[str], list[bool], list[dict]]:
    """Search CEMS for relevant memories.

    Returns (formatted_string, memory_ids, truncation_flags, score_details) tuple.
    """
    if not query or len(query) < 3:
        return None, [], [], []

    try:
        payload = {"query": query, "scope": "both", "limit": 5}
        if project:
            payload["project"] = project  # Boosts same-project memories via source_ref scoring

        # Pass search mode from credentials if configured
        if client.search_mode:
            payload["mode"] = client.search_mode

        response = client.post("/api/memory/search", json=payload)

        if response.status_code != 200:
            return None, [], [], []

        data = response.json()
        if not data.get("success"):
            return None, [], [], []

        # Agentic mode returns structured {entities, memories} — use entity-first formatting
        if data.get("entities") is not None or data.get("mode") == "agentic":
            return _format_agentic_response(data)

        results = data.get("results", [])
        if not results:
            return None, [], [], []

        # Client-side score filter: drop low-relevance results
        results = [r for r in results if r.get("score", 0) >= 0.45]
        if not results:
            return None, [], [], []

        # Session dedup: if multiple results share a session tag, keep only highest-scoring
        seen_sessions: dict[str, dict] = {}
        deduped: list[dict] = []
        for r in results:
            tags = r.get("tags", [])
            session_tag = next((t for t in tags if t.startswith("session:")), None)
            if session_tag:
                # Extract base session ID (strip epoch suffix like ":e1")
                base_session = session_tag.split(":")[0] + ":" + session_tag.split(":")[1]
                existing = seen_sessions.get(base_session)
                if existing is None or r.get("score", 0) > existing.get("score", 0):
                    if existing is not None:
                        deduped.remove(existing)
                    seen_sessions[base_session] = r
                    deduped.append(r)
                # else: skip lower-scoring duplicate from same session
            else:
                deduped.append(r)
        results = deduped

        if not results:
            return None, [], [], []

        # Format results for Claude and collect memory IDs + scores
        formatted = []
        memory_ids = []
        truncation_flags = []
        score_details = []
        truncated_hints = []
        for i, r in enumerate(results, 1):
            content = r.get("content", r.get("memory", ""))
            category = r.get("category", "general")
            mem_id = r.get("memory_id", r.get("id", ""))
            short_id = mem_id[:8] if mem_id else ""
            score = r.get("score", 0.0)
            truncated = r.get("truncated", False)
            full_len = r.get("full_length", 0)
            suffix = f" [truncated — full doc: {full_len} chars]" if truncated else ""
            formatted.append(f"{i}. [{category}] (score: {score:.2f}) {content}{suffix} (id: {short_id})")
            if truncated and full_len:
                truncated_hints.append(
                    f"  #{i} ({category}, {full_len} chars) — use `/recall {short_id}` to read full document"
                )
            truncation_flags.append(truncated)
            if mem_id:
                memory_ids.append(mem_id)
            score_details.append({"id": short_id, "score": round(score, 3), "category": category, "content": content})

        # Add retrieval summary footer
        scores = [d["score"] for d in score_details]
        avg_score = sum(scores) / len(scores) if scores else 0
        top_score = max(scores) if scores else 0
        formatted.append(f"\n--- Retrieval: {len(results)} results, avg score {avg_score:.2f}, top {top_score:.2f} ---")
        if truncated_hints:
            formatted.append("Truncated memories — you're seeing a snippet, not the full document.")
            formatted.append("If any truncated memory looks partially relevant, read the full version BEFORE proceeding:")
            formatted.extend(truncated_hints)
            formatted.append("Do NOT mark truncated memories as noise without reading the full document first.")

        return "\n".join(formatted), memory_ids, truncation_flags, score_details

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return None, [], [], []


def log_shown_memories(client: CEMSClient, memory_ids: list[str]) -> None:
    """Fire-and-forget: log that memories were shown to the user."""
    if not memory_ids:
        return
    try:
        client.post("/api/memory/log-shown", json={"memory_ids": memory_ids}, timeout=2.0)
    except (httpx.RequestError, httpx.TimeoutException):
        pass


# =============================================================================
# Relevance Mapping File
# =============================================================================

RELEVANCE_CACHE_DIR = Path.home() / ".cems" / "cache" / "relevance"


def write_relevance_mapping(
    session_id: str,
    memory_ids: list[str],
    truncated: list[bool] | None = None,
) -> None:
    """Write a mapping file so the Stop hook can map #N → memory_id.

    The mapping file maps positional numbers (#1, #2, ...) to memory IDs.
    The Stop hook reads this to resolve Claude's relevance feedback line.

    Args:
        session_id: Current session ID
        memory_ids: List of memory IDs in display order
        truncated: Per-ID flag indicating if the shown content was truncated.
                   When True, noise feedback is recorded as snippet-level (lighter signal).

    Also cleans up stale mapping files (>1 hour old).
    """
    if not session_id or not memory_ids:
        return

    try:
        RELEVANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Clean stale files (>1 hour old)
        cutoff = time.time() - 3600
        for f in RELEVANCE_CACHE_DIR.glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

        # Write mapping: index 0 = #1, index 1 = #2, etc.
        mapping = {
            "memory_ids": memory_ids,
            "truncated": truncated or [False] * len(memory_ids),
            "ts": time.time(),
        }
        mapping_path = RELEVANCE_CACHE_DIR / f"{session_id[:12]}.json"
        mapping_path.write_text(json.dumps(mapping))
    except Exception:
        pass  # Non-critical — never block prompts


# =============================================================================
# Gate Rule Cache Functions
# =============================================================================

GATE_CACHE_DIR = Path.home() / ".cems" / "cache" / "gate_rules"

# Pattern for parsing gate rule content: "Tool: pattern — reason"
# Uses em dash (—), en dash (–), or " - " (hyphen with spaces) as separator
GATE_RULE_PATTERN = re.compile(
    r"^(?P<tool>\w+):\s*(?P<pattern>.+?)\s*(?:—|–|\s-\s)\s*(?P<reason>.+)$",
    re.IGNORECASE | re.DOTALL,
)


def get_cache_path(project: str | None) -> Path:
    """Get cache file path for a project.

    Args:
        project: Project ID (org/repo) or None for global rules

    Returns:
        Path to cache file
    """
    GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if project:
        # Sanitize project ID for filesystem (org/repo -> org_repo)
        safe_name = project.replace("/", "_").replace("\\", "_")
        return GATE_CACHE_DIR / f"{safe_name}.json"
    else:
        return GATE_CACHE_DIR / "global.json"


def pattern_to_regex(pattern: str) -> str:
    """Convert a human-readable pattern to a regex string.

    Args:
        pattern: Human-readable pattern like "coolify deploy"

    Returns:
        Regex pattern string
    """
    # Escape special regex characters (except * which we handle specially)
    escaped = re.escape(pattern)

    # Convert escaped \* back to .* for glob-style matching
    escaped = escaped.replace(r"\*", ".*")

    # Convert spaces to flexible whitespace
    escaped = re.sub(r"\\ ", r"\\s+", escaped)

    return escaped


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
        Structured gate pattern dict or None if parsing fails
    """
    if not content or not content.strip():
        return None

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
        regex_pattern = pattern_to_regex(raw_pattern)
    except re.error:
        return None

    # Extract severity from tags (default: warn)
    severity = "warn"
    if tags:
        tags_lower = [t.lower() for t in tags]
        if "block" in tags_lower:
            severity = "block"
        elif "confirm" in tags_lower:
            severity = "confirm"

    # Extract project scope from source_ref
    project = None
    if source_ref and source_ref.startswith("project:"):
        project = source_ref[8:]

    return {
        "tool": tool,
        "pattern": regex_pattern,
        "raw_pattern": raw_pattern,
        "reason": reason,
        "severity": severity,
        "project": project,
    }


def search_gate_rules(client: CEMSClient, project: str | None = None) -> list[dict]:
    """Fetch gate rules from CEMS using dedicated endpoint."""
    try:
        url = "/api/memory/gate-rules"
        if project:
            url += f"?project={project}"

        response = client.get(url)

        if response.status_code != 200:
            return []

        data = response.json()
        if not data.get("success"):
            return []

        return data.get("rules", [])

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return []


def populate_gate_cache(client: CEMSClient, project: str | None = None) -> int:
    """Populate gate rule cache for a project."""
    cache_path = get_cache_path(project)

    # Skip if cache exists and is fresh (< 5 minutes old)
    if cache_path.exists():
        try:
            mtime = cache_path.stat().st_mtime
            import time
            if time.time() - mtime < 300:  # 5 minute TTL
                return -1  # Cache is fresh, skip
        except OSError:
            pass

    # Fetch gate rules from CEMS
    rules = search_gate_rules(client, project)

    # If server returned empty AND we have an existing cache, keep the old cache
    # rather than overwriting with nothing (protects against temporary outages)
    if not rules and cache_path.exists():
        return 0

    # Extract patterns from rules
    # New endpoint format has content/tags/source_ref at top level
    patterns = []
    for r in rules:
        content = r.get("content", "")
        tags = r.get("tags", [])
        source_ref = r.get("source_ref", "")

        pattern = extract_gate_pattern(content, tags, source_ref)
        if pattern:
            patterns.append(pattern)

    # Write cache
    try:
        cache_path.write_text(json.dumps(patterns, indent=2))
    except OSError:
        pass

    return len(patterns)


def output_result(text: str, is_cursor: bool):
    """Output result in the correct format for the environment."""
    if is_cursor:
        # Cursor expects JSON output with systemPrompt field
        print(json.dumps({"systemPrompt": text}))
    else:
        # Claude Code CLI: use hookSpecificOutput JSON format for reliable context injection
        # Plain text output has known issues where it may not reliably reach Claude's context.
        # JSON additionalContext format matches what SessionStart uses and is consistently parsed.
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": text
            }
        }))


def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        prompt = input_data.get('prompt', '')
        session_id = input_data.get('session_id', '')
        cwd = input_data.get('cwd', '')

        # Strip system-injected XML tags from the prompt field.
        # Claude Code includes wire-format tags (task notifications, system reminders,
        # previous hook output) in the prompt — these are not user text.
        user_text = re.sub(
            r'<(?:task-notification|system-reminder|user-prompt-submit-hook|memory-recall|'
            r'recent-observations|cems-profile|cems-foundation)[^>]*>.*?'
            r'</(?:task-notification|system-reminder|user-prompt-submit-hook|memory-recall|'
            r'recent-observations|cems-profile|cems-foundation)>',
            '', prompt, flags=re.DOTALL
        ).strip()

        log_hook_event("UserPromptSubmit", session_id, {
            "prompt_len": len(user_text),
        }, input_data=input_data)

        # Detect if running in Cursor (vs Claude Code CLI)
        is_cursor = 'cursor_version' in input_data

        # Skip for subagents
        if os.environ.get('CLAUDE_AGENT_ID'):
            return

        # Resolve credentials using CWD (per-project or global)
        client = CEMSClient.from_cwd(cwd)

        # Ensure observer daemon is running (rate-limited, checks every 5 min)
        try:
            from utils.observer_manager import ensure_daemon_running
            ensure_daemon_running()
        except Exception:
            pass  # Observer is nice-to-have, never block prompts

        # Handle confirmatory prompts ("yes", "go ahead", etc.)
        # Instead of skipping, derive search intent from what Claude proposed
        if is_confirmatory(user_text):
            output_parts = []
            if user_text.rstrip().endswith('-u'):
                output_parts.append("Use the maximum amount of ultrathink. Take all the time you need.")

            # Derive search query from what Claude just proposed — send raw text
            transcript_path = input_data.get('transcript_path', '')
            if transcript_path and client:
                assistant_text = read_last_assistant_message(transcript_path, max_chars=5000)
                if assistant_text and len(assistant_text.strip()) >= 3:
                    project = get_project_id(cwd) if cwd else None
                    clean_assistant = _clean_search_query(assistant_text.strip())
                    memories, memory_ids, _, _ = search_cems(client, clean_assistant, project=project)
                    if memories:
                        output_parts.append(f"""<memory-recall>
CONTEXT for confirmed action:

{memories}

Review these memories before proceeding.
</memory-recall>""")
                        log_shown_memories(client, memory_ids)
                        write_relevance_mapping(session_id, memory_ids)

            if output_parts:
                combined = '\n\n'.join(output_parts)
                log_hook_event("UserPromptSubmitOutput", session_id, {
                    "output_len": len(combined),
                    "has_memories": "<memory-recall>" in combined,
                }, output_text=combined)
                output_result(combined, is_cursor)
            return

        # Skip remaining short prompts (greetings, gibberish) — not worth searching
        if len(user_text) < 15:
            return

        # Skip slash commands
        if user_text.startswith('/'):
            return

        # Skip if CEMS is not configured (after ultrathink processing)
        if not client:
            return

        output_parts = []

        # Extract project ID from git remote for project-scoped search
        project = get_project_id(cwd) if cwd else None

        # 0. Gate cache population (runs once per project, cached for 5 min)
        # This ensures PreToolUse hook has cached gate rules to check
        populate_gate_cache(client, project)

        # 1. Memory awareness - search CEMS (clean query for better embedding match)
        if len(user_text) >= 3:
            search_query = _clean_search_query(user_text)
            memories, memory_ids, truncation_flags, score_details = search_cems(client, search_query, project=project)
            if memories:
                memory_context = f"""<memory-recall>
RELEVANT MEMORIES found for "{search_query}":

{memories}

After responding, note which memories (by number) were relevant vs noise.
</memory-recall>"""
                output_parts.append(memory_context)

                # Log retrieval scores for offline analysis
                if score_details:
                    scores = [d["score"] for d in score_details]
                    log_hook_event("MemoryRetrieval", session_id, {
                        "query": user_text,
                        "result_count": len(score_details),
                        "avg_score": round(sum(scores) / len(scores), 3),
                        "top_score": round(max(scores), 3),
                        "details": score_details,
                    }, input_data={"query": user_text, "score_details": score_details})

                # Log that these memories were shown (fire-and-forget)
                log_shown_memories(client, memory_ids)

                # Write mapping file for Stop hook relevance feedback
                write_relevance_mapping(session_id, memory_ids, truncation_flags)

        # 2. Ultrathink flag
        if user_text.endswith('-u'):
            output_parts.append("Use the maximum amount of ultrathink. Take all the time you need. It's much better if you do too much research and thinking than not enough.")

        # Output combined context
        if output_parts:
            combined = '\n\n'.join(output_parts)
            log_hook_event("UserPromptSubmitOutput", session_id, {
                "output_len": len(combined),
                "has_memories": "<memory-recall>" in combined,
                "has_ultrathink": "ultrathink" in combined.lower(),
            }, output_text=combined)
            output_result(combined, is_cursor)

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"CEMS hook error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
