#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///

"""
CEMS PostToolUse Hook - Incremental Tool-Based Learning

Runs after significant tool completions and sends learnable events to CEMS.
This enables SuperMemory-style incremental learning - no need to wait for
session end to capture insights.

Triggers on:
- Edit/Write: Code changes
- Bash: Commands that affect state (git commit, npm install, etc.)
- Task: Completed sub-agent tasks

Skips:
- Read/Glob/Grep: Pure reads don't generate learnings
- Background agents: Already handled by parent
- Very short sessions: No context to learn from

Configuration:
  CEMS_API_URL - CEMS server URL (required)
  CEMS_API_KEY - Your CEMS API key (required)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from utils.hook_logger import log_hook_event

CEMS_API_URL = os.getenv("CEMS_API_URL", "")
CEMS_API_KEY = os.getenv("CEMS_API_KEY", "")

# Tools that might produce learnable events
LEARNABLE_TOOLS = {
    "Edit",
    "Write",
    "Bash",
    "Task",
    "MultiEdit",
}

# Bash commands that might produce learnable events
LEARNABLE_BASH_PATTERNS = [
    "git commit",
    "git push",
    "npm install",
    "pip install",
    "docker",
    "kubectl",
    "make",
    "cargo build",
    "cargo test",
    "pytest",
    "npm test",
    "npm run",
    "yarn",
    "pnpm",
]


def read_recent_context(transcript_path: str, max_chars: int = 1500) -> str:
    """Read recent conversation context from transcript."""
    if not transcript_path or not Path(transcript_path).exists():
        return ""

    try:
        messages = []
        with open(transcript_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entry_type = entry.get("type", "")

                    if entry_type == "user":
                        msg = entry.get("message", {})
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            text_parts = [
                                p.get("text", "")
                                for p in content
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            content = "\n".join(text_parts)
                        if content:
                            messages.append(f"User: {content[:500]}")

                    elif entry_type == "assistant":
                        msg = entry.get("message", {})
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            text_parts = [
                                p.get("text", "")
                                for p in content
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            content = "\n".join(text_parts)
                        if content:
                            messages.append(f"Assistant: {content[:500]}")

                except json.JSONDecodeError:
                    continue

        # Return last few messages as context (most recent)
        recent = messages[-5:] if len(messages) > 5 else messages
        context = "\n---\n".join(recent)
        return context[:max_chars]

    except Exception:
        return ""


def should_process_tool(tool_name: str, tool_input: dict) -> bool:
    """Determine if this tool usage might produce learnable content."""
    if tool_name not in LEARNABLE_TOOLS:
        return False

    # For Bash, check if it's a potentially learnable command
    if tool_name == "Bash":
        command = tool_input.get("command", "").lower()
        description = tool_input.get("description", "").lower()

        # Skip simple file operations
        if command.startswith(("ls ", "cd ", "pwd", "cat ", "head ", "tail ")):
            return False

        # Check for learnable patterns
        for pattern in LEARNABLE_BASH_PATTERNS:
            if pattern in command or pattern in description:
                return True

        # Skip if no recognizable pattern
        return False

    # Edit/Write always potentially learnable
    if tool_name in ("Edit", "Write", "MultiEdit"):
        return True

    # Task completions are interesting
    if tool_name == "Task":
        return True

    return False


def extract_tool_output_summary(tool_response: dict) -> str:
    """Extract a brief summary from tool response."""
    if not tool_response:
        return ""

    if isinstance(tool_response, str):
        return tool_response[:500]

    # Handle dict responses — always pass real output, not just "Success"
    if "output" in tool_response:
        return str(tool_response["output"]).strip()[:500]

    if "content" in tool_response:
        return str(tool_response["content"]).strip()[:500]

    return ""


def get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo')."""
    if not cwd:
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            import re
            url = result.stdout.strip()
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix('.git')
    except Exception:
        pass
    return None


def send_to_cems(
    tool_name: str,
    tool_input: dict,
    tool_output: str,
    session_id: str,
    context_snippet: str,
    cwd: str,
) -> bool:
    """Send tool learning to CEMS API."""
    if not CEMS_API_URL or not CEMS_API_KEY:
        return False

    try:
        payload = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "session_id": session_id,
            "context_snippet": context_snippet,
            "working_dir": cwd,
        }

        # Add project source_ref if we can determine it
        project = get_project_id(cwd)
        if project:
            payload["source_ref"] = f"project:{project}"

        response = httpx.post(
            f"{CEMS_API_URL}/api/tool/learning",
            json=payload,
            headers={"Authorization": f"Bearer {CEMS_API_KEY}"},
            timeout=10.0,  # Allow time for LLM extraction
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("stored"):
                print(
                    f"CEMS: Captured learning from {tool_name}",
                    file=sys.stderr,
                )
            return True
        return False

    except (httpx.RequestError, httpx.TimeoutException) as e:
        print(f"CEMS tool learning error: {e}", file=sys.stderr)
        return False


def main():
    try:
        input_data = json.load(sys.stdin)

        # Skip if CEMS not configured
        if not CEMS_API_URL or not CEMS_API_KEY:
            sys.exit(0)

        # Extract fields
        session_id = input_data.get("session_id", "unknown")
        tool_name = input_data.get("tool_name") or input_data.get("tool", "unknown")
        tool_input = input_data.get("tool_input", {})
        tool_response = input_data.get("tool_response", {})
        transcript_path = input_data.get("transcript_path", "")
        cwd = input_data.get("cwd", os.getcwd())
        is_background_agent = input_data.get("is_background_agent", False)

        log_hook_event("PostToolUse", session_id, {"tool": tool_name}, input_data=input_data)

        # Skip background agents (parent handles)
        if is_background_agent:
            sys.exit(0)

        # Check if this tool usage is worth processing
        if not should_process_tool(tool_name, tool_input):
            sys.exit(0)

        # Read recent context from transcript
        context_snippet = read_recent_context(transcript_path)

        # Extract tool output summary
        tool_output = extract_tool_output_summary(tool_response)

        # Send to CEMS for learning extraction
        send_to_cems(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            session_id=session_id,
            context_snippet=context_snippet,
            cwd=cwd,
        )

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as e:
        print(f"CEMS PostToolUse error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
