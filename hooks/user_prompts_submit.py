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
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from utils.credentials import get_cems_key, get_cems_url
from utils.hook_logger import log_hook_event
from utils.transcript import read_last_assistant_message

# CEMS configuration — env vars first, then ~/.cems/credentials fallback
CEMS_API_URL = get_cems_url()
CEMS_API_KEY = get_cems_key()

CONFIRMATORY = re.compile(
    r'^(yes|yeah|yep|yup|ok|okay|sure|go|do it|proceed|'
    r'go ahead|go for it|sounds good|lgtm|ship it|'
    r'yes please|confirmed|let\'s do it|approve)[\s!.]*$',
    re.IGNORECASE,
)


def is_confirmatory(prompt: str) -> bool:
    """Check if prompt is a short confirmatory response like 'yes' or 'go ahead'."""
    clean = prompt.strip().rstrip('-u').strip()
    if CONFIRMATORY.match(clean):
        return True
    # Very short non-question, non-slash prompts are also confirmatory
    if len(clean) < 8 and '?' not in clean and not clean.startswith('/'):
        return True
    return False


def extract_intent(prompt: str) -> str:
    """
    Extract the INTENT from user prompt - what they're actually asking about.
    Removes meta-language to get core topic.

    For long prompts (>200 chars after stripping), falls back to keyword
    extraction to avoid sending noisy multi-sentence queries to the
    embedding model.
    """
    # Strip -u flag before processing
    clean_prompt = prompt.rstrip()
    if clean_prompt.endswith('-u'):
        clean_prompt = clean_prompt[:-2].rstrip()

    # Meta-phrases to remove
    meta_patterns = [
        r'^(can you|could you|would you|please|help me|i want to|i need to|let\'s|lets)\s+',
        r'^(show me|tell me|find|search for|look for|recall|remember)\s+',
        r'^(how do i|how can i|how to|what is|what are|where is|where are)\s+',
        r'\s+(for me|please|thanks|thank you)$',
        r'\?$',
    ]

    intent = clean_prompt.strip().lower()

    for pattern in meta_patterns:
        intent = re.sub(pattern, '', intent, flags=re.IGNORECASE)

    intent = intent.strip()

    # If too short, extract keywords
    if len(intent) < 5:
        return extract_keywords(clean_prompt)

    # If still too long, extract keywords instead — long intents produce
    # noisy embeddings that return irrelevant results
    if len(intent) > 200:
        return extract_keywords(clean_prompt)

    return intent


def extract_keywords(prompt: str) -> str:
    """Extract meaningful keywords from prompt."""
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'them', 'it',
        'this', 'that', 'these', 'what', 'which', 'who', 'and', 'but', 'if',
        'or', 'because', 'help', 'want', 'need', 'please', 'just', 'now',
        'get', 'make', 'use', 'like', 'know', 'think', 'take', 'go', 'see',
    }

    words = re.sub(r'[^\w\s-]', ' ', prompt.lower()).split()
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]

    return ' '.join(list(dict.fromkeys(keywords))[:5])


def get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo').

    Parses the git remote origin URL to extract the org/repo identifier.
    Works with both SSH and HTTPS formats.

    Args:
        cwd: Current working directory (project root)

    Returns:
        Project ID like 'org/repo' or None if not a git repo
    """
    if not cwd:
        return None

    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # SSH: git@github.com:org/repo.git → org/repo
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                # HTTPS: https://github.com/org/repo.git → org/repo
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix('.git')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def search_cems(query: str, project: str | None = None) -> tuple[str | None, list[str]]:
    """
    Search CEMS for relevant memories.
    Returns (formatted_string, memory_ids) tuple.

    Args:
        query: Search query string
        project: Optional project ID (e.g., 'org/repo') to boost project-scoped memories

    Note: No limit imposed here - the API handles relevance filtering and limits.
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return None, []

    if not query or len(query) < 3:
        return None, []

    try:
        payload = {"query": query, "scope": "both"}
        if project:
            payload["project"] = project  # Boosts same-project memories via source_ref scoring

        response = httpx.post(
            f"{CEMS_API_URL}/api/memory/search",
            json=payload,
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=5.0,
        )

        if response.status_code != 200:
            return None, []

        data = response.json()
        if not data.get("success") or not data.get("results"):
            return None, []

        results = data["results"]
        if not results:
            return None, []

        # Format results for Claude and collect memory IDs
        formatted = []
        memory_ids = []
        for i, r in enumerate(results, 1):
            content = r.get("content", r.get("memory", ""))
            category = r.get("category", "general")
            mem_id = r.get("memory_id", r.get("id", ""))
            short_id = mem_id[:8] if mem_id else ""
            formatted.append(f"{i}. [{category}] {content} (id: {short_id})")
            if mem_id:
                memory_ids.append(mem_id)

        return "\n".join(formatted), memory_ids

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return None, []


def fetch_recent_observations(project: str | None = None, limit: int = 5) -> str | None:
    """Fetch recent observations for the current project.

    Searches for observations tagged with the project's source_ref.
    Returns formatted string for context injection, or None.
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return None

    try:
        query = f"recent observations"
        if project:
            query += f" for {project}"

        payload = {
            "query": query,
            "scope": "personal",
            "max_results": limit,
        }
        if project:
            payload["project"] = project

        response = httpx.post(
            f"{CEMS_API_URL}/api/memory/search",
            json=payload,
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=3.0,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        results = data.get("results", [])

        # Filter to only observation category
        observations = [
            r for r in results
            if r.get("category") == "observation"
        ]

        if not observations:
            return None

        lines = [f"- {r.get('content', r.get('memory', ''))}" for r in observations[:limit]]
        project_label = project or "current project"
        return f"""<recent-observations>
Recent observations ({project_label}):
{chr(10).join(lines)}
</recent-observations>"""

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return None


def log_shown_memories(memory_ids: list[str]) -> None:
    """Fire-and-forget: log that memories were shown to the user.

    Calls /api/memory/log-shown to increment shown_count and update last_shown_at.
    Failures are silently ignored (non-critical telemetry).
    """
    if not memory_ids or not CEMS_API_URL or not CEMS_API_KEY:
        return

    try:
        httpx.post(
            f"{CEMS_API_URL}/api/memory/log-shown",
            json={"memory_ids": memory_ids},
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=2.0,
        )
    except (httpx.RequestError, httpx.TimeoutException):
        pass


# =============================================================================
# Gate Rule Cache Functions
# =============================================================================

GATE_CACHE_DIR = Path.home() / ".claude" / "cache" / "gate_rules"

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


def search_gate_rules(project: str | None = None) -> list[dict]:
    """Fetch gate rules from CEMS using dedicated endpoint.

    Uses /api/memory/gate-rules which queries by category directly,
    bypassing semantic search for reliable results.

    Args:
        project: Optional project ID to filter rules (e.g., "org/repo")

    Returns:
        List of gate rule memories
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return []

    try:
        # Use dedicated gate-rules endpoint that queries by category
        url = f"{CEMS_API_URL}/api/memory/gate-rules"
        if project:
            url += f"?project={project}"

        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=5.0,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        if not data.get("success"):
            return []

        return data.get("rules", [])

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return []


def populate_gate_cache(project: str | None = None) -> int:
    """Populate gate rule cache for a project.

    Fetches gate rules from CEMS and caches them locally.

    Args:
        project: Project ID (org/repo) or None for global

    Returns:
        Number of rules cached
    """
    cache_path = get_cache_path(project)

    # Skip if cache exists and is fresh (< 5 minutes old)
    if cache_path.exists():
        try:
            age = cache_path.stat().st_mtime
            import time
            if time.time() - age < 300:  # 5 minute TTL
                return -1  # Cache is fresh, skip
        except OSError:
            pass

    # Fetch gate rules from CEMS
    rules = search_gate_rules(project)

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

        log_hook_event("UserPromptSubmit", session_id, {
            "prompt_len": len(prompt),
        }, input_data=input_data)

        # Detect if running in Cursor (vs Claude Code CLI)
        is_cursor = 'cursor_version' in input_data

        # Skip for subagents
        if os.environ.get('CLAUDE_AGENT_ID'):
            return

        # Ensure observer daemon is running (rate-limited, checks every 5 min)
        try:
            from utils.observer_manager import ensure_daemon_running
            ensure_daemon_running()
        except Exception:
            pass  # Observer is nice-to-have, never block prompts

        # Handle confirmatory prompts ("yes", "go ahead", etc.)
        # Instead of skipping, derive search intent from what Claude proposed
        if is_confirmatory(prompt):
            output_parts = []
            if prompt.rstrip().endswith('-u'):
                output_parts.append("Use the maximum amount of ultrathink. Take all the time you need.")

            # Derive search intent from what Claude just proposed
            transcript_path = input_data.get('transcript_path', '')
            if transcript_path:
                assistant_text = read_last_assistant_message(transcript_path, max_chars=500)
                if assistant_text:
                    # Extract keywords from last 1-2 sentences (the proposal)
                    sentences = re.split(r'[.!?]\s+', assistant_text.strip())
                    proposal = ' '.join(sentences[-2:])
                    intent = extract_keywords(proposal)
                    if intent and len(intent) >= 3:
                        project = get_project_id(cwd) if cwd else None
                        memories, memory_ids = search_cems(intent, project=project)
                        if memories:
                            output_parts.append(f"""<memory-recall>
CONTEXT for confirmed action "{intent}":

{memories}

Review these memories before proceeding.
</memory-recall>""")
                            log_shown_memories(memory_ids)

            if output_parts:
                output_result('\n\n'.join(output_parts), is_cursor)
            return

        # Skip remaining short prompts (greetings, gibberish) — not worth searching
        if len(prompt) < 15:
            return

        # Skip slash commands
        if prompt.strip().startswith('/'):
            return

        output_parts = []

        # Extract project ID from git remote for project-scoped search
        project = get_project_id(cwd) if cwd else None

        # 0. Gate cache population (runs once per project, cached for 5 min)
        # This ensures PreToolUse hook has cached gate rules to check
        populate_gate_cache(project)

        # 1. Memory awareness - search CEMS
        # Enrich the search query with assistant context (what Claude was doing)
        intent = extract_intent(prompt)

        # Read last assistant message to add context about what Claude proposed/discussed
        transcript_path = input_data.get('transcript_path', '')
        if transcript_path and intent:
            assistant_text = read_last_assistant_message(transcript_path, max_chars=500)
            if assistant_text:
                sentences = re.split(r'[.!?]\s+', assistant_text.strip())
                assistant_keywords = extract_keywords(' '.join(sentences[-2:]))
                if assistant_keywords and len(assistant_keywords) >= 3:
                    # Blend: user intent first (primary), assistant context second
                    intent = f"{intent} {assistant_keywords}"
                    # Cap total length to avoid noisy embeddings
                    if len(intent) > 200:
                        intent = intent[:200]

        if intent and len(intent) >= 3:
            memories, memory_ids = search_cems(intent, project=project)
            if memories:
                # Show the user-facing intent (without assistant keywords clutter)
                display_intent = extract_intent(prompt)
                memory_context = f"""<memory-recall>
RELEVANT MEMORIES found for "{display_intent}":

{memories}

If these memories are helpful to the current task, you may reference them.
Use /recall "{display_intent}" for more detailed results.
</memory-recall>"""
                output_parts.append(memory_context)

                # Log that these memories were shown (fire-and-forget)
                log_shown_memories(memory_ids)

        # 2. Recent observations for project context
        obs_context = fetch_recent_observations(project=project)
        if obs_context:
            output_parts.append(obs_context)

        # 3. Ultrathink flag
        if prompt.rstrip().endswith('-u'):
            output_parts.append("Use the maximum amount of ultrathink. Take all the time you need. It's much better if you do too much research and thinking than not enough.")

        # Output combined context
        if output_parts:
            output_result('\n\n'.join(output_parts), is_cursor)

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"CEMS hook error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
