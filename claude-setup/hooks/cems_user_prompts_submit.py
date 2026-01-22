#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///

"""
CEMS UserPromptSubmit Hook - Memory Awareness + Ultrathink

This hook runs on every user prompt and:
1. Searches CEMS for relevant memories and injects them as context
2. Appends ultrathink instruction if -u flag is present

Configuration (environment variables):
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
"""

import json
import os
import re
import sys

import httpx

# CEMS configuration from environment (no defaults - set CEMS_API_URL and CEMS_API_KEY)
CEMS_API_URL = os.getenv("CEMS_API_URL", "")
CEMS_API_KEY = os.getenv("CEMS_API_KEY", "")


def extract_intent(prompt: str) -> str:
    """
    Extract the INTENT from user prompt - what they're actually asking about.
    Removes meta-language to get core topic.
    """
    # Meta-phrases to remove
    meta_patterns = [
        r'^(can you|could you|would you|please|help me|i want to|i need to|let\'s|lets)\s+',
        r'^(show me|tell me|find|search for|look for|recall|remember)\s+',
        r'^(how do i|how can i|how to|what is|what are|where is|where are)\s+',
        r'\s+(for me|please|thanks|thank you)$',
        r'\?$',
    ]

    intent = prompt.strip().lower()

    for pattern in meta_patterns:
        intent = re.sub(pattern, '', intent, flags=re.IGNORECASE)

    intent = intent.strip()

    # If too short, extract keywords
    if len(intent) < 5:
        return extract_keywords(prompt)

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


def search_cems(query: str) -> str | None:
    """
    Search CEMS for relevant memories.
    Returns formatted string or None if search fails.

    Note: No limit imposed here - the API handles relevance filtering and limits.
    """
    if not CEMS_API_URL or not CEMS_API_KEY:
        return None

    if not query or len(query) < 3:
        return None

    try:
        response = httpx.post(
            f"{CEMS_API_URL}/api/memory/search",
            json={"query": query, "scope": "both"},
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=5.0,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        if not data.get("success") or not data.get("results"):
            return None

        results = data["results"]
        if not results:
            return None

        # Format results for Claude - use whatever the API returns
        formatted = []
        for i, r in enumerate(results, 1):
            content = r.get("content", r.get("memory", ""))
            category = r.get("category", "general")
            mem_id = r.get("memory_id", r.get("id", ""))[:8] if r.get("memory_id") or r.get("id") else ""
            formatted.append(f"{i}. [{category}] {content} (id: {mem_id})")

        return "\n".join(formatted)

    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError):
        return None


def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        prompt = input_data.get('prompt', '')

        # Skip for subagents
        if os.environ.get('CLAUDE_AGENT_ID'):
            return

        # Skip very short prompts
        if len(prompt) < 15:
            # Still check for -u flag
            if prompt.rstrip().endswith('-u'):
                print("Use the maximum amount of ultrathink. Take all the time you need.")
            return

        # Skip slash commands
        if prompt.strip().startswith('/'):
            return

        output_parts = []

        # 1. Memory awareness - search CEMS
        intent = extract_intent(prompt)
        if intent and len(intent) >= 3:
            memories = search_cems(intent)
            if memories:
                memory_context = f"""<memory-recall>
RELEVANT MEMORIES found for "{intent}":

{memories}

If these memories are helpful to the current task, you may reference them.
Use /recall "{intent}" for more detailed results.
</memory-recall>"""
                output_parts.append(memory_context)

        # 2. Ultrathink flag
        if prompt.rstrip().endswith('-u'):
            output_parts.append("Use the maximum amount of ultrathink. Take all the time you need. It's much better if you do too much research and thinking than not enough.")

        # Output combined context
        if output_parts:
            print('\n\n'.join(output_parts))

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"CEMS hook error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
