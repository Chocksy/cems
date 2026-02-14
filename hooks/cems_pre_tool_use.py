#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

"""
CEMS PreToolUse Hook - Gate Rule Checking

This hook runs before every tool use and:
1. Checks cached gate rules for matching patterns
2. Blocks tool execution (exit 2) for "block" severity rules
3. Warns Claude (stdout) for "warn" severity rules
4. Logs tool calls for debugging

Gate rules are cached by the UserPromptSubmit hook.
Cache location: ~/.cems/cache/gate_rules/{project}.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.hook_logger import log_hook_event


# =============================================================================
# Gate Rule Functions
# =============================================================================

GATE_CACHE_DIR = Path.home() / ".cems" / "cache" / "gate_rules"


def get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo')."""
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


def get_cache_path(project: str | None) -> Path:
    """Get cache file path for a project."""
    if project:
        safe_name = project.replace("/", "_").replace("\\", "_")
        return GATE_CACHE_DIR / f"{safe_name}.json"
    else:
        return GATE_CACHE_DIR / "global.json"


def load_gate_rules(project: str | None) -> list[dict]:
    """Load cached gate rules for a project.

    Args:
        project: Project ID (org/repo) or None for global

    Returns:
        List of gate rule patterns with compiled regex
    """
    cache_path = get_cache_path(project)

    if not cache_path.exists():
        return []

    try:
        data = json.loads(cache_path.read_text())
        # Compile regex patterns
        rules = []
        for rule in data:
            rule["regex"] = re.compile(rule["pattern"], re.IGNORECASE)
            rules.append(rule)
        return rules
    except (OSError, json.JSONDecodeError, KeyError, re.error):
        return []


def check_gate_rules(
    tool_name: str,
    command: str,
    project: str | None,
) -> dict:
    """Check if a command violates any cached gate rules.

    Args:
        tool_name: Tool being called (e.g., "Bash")
        command: Command or tool input to check
        project: Current project ID (org/repo)

    Returns:
        {
            "allowed": bool,
            "rule": dict|None,      # Matching rule if any
            "severity": str|None,   # block|warn|confirm
        }
    """
    # Load rules for current project
    rules = load_gate_rules(project)

    # Also load global rules
    if project:
        rules.extend(load_gate_rules(None))

    # Check each rule
    for rule in rules:
        # Skip if tool doesn't match
        if rule["tool"].lower() != tool_name.lower():
            continue

        # Skip project-scoped rules that don't match current project
        rule_project = rule.get("project")
        if rule_project and rule_project != project:
            continue

        # Check if pattern matches command
        if rule["regex"].search(command):
            return {
                "allowed": False,
                "rule": rule,
                "severity": rule.get("severity", "warn"),
            }

    return {"allowed": True, "rule": None, "severity": None}


# =============================================================================
# Logging Functions
# =============================================================================


def log_tool_use(input_data: dict, session_id: str) -> None:
    """Log tool use for debugging."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.constants import ensure_session_log_dir

        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / 'pre_tool_use.json'

        if log_path.exists():
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        log_data.append(input_data)

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception:
        pass  # Don't fail on logging errors


# =============================================================================
# Main
# =============================================================================


def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        session_id = input_data.get('session_id', 'default')
        tool_name = input_data.get('tool_name', '')
        tool_input = input_data.get('tool_input', {})
        cwd = input_data.get('cwd', '')

        log_hook_event("PreToolUse", session_id, {"tool": tool_name}, input_data=input_data)

        # Get project ID from git remote
        project = get_project_id(cwd) if cwd else None

        # Check gate rules for Bash commands
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            result = check_gate_rules("bash", command, project)

            if not result["allowed"]:
                rule = result["rule"]
                severity = result["severity"]

                if severity == "block":
                    # Block: stderr is fed to Claude on exit 2 (stdout is ignored)
                    msg = f"BLOCKED by gate rule: {rule['reason']} (pattern: {rule['raw_pattern']})"
                    print(msg, file=sys.stderr)
                    sys.exit(2)

                elif severity == "warn":
                    # Warn: use hookSpecificOutput.additionalContext so Claude sees it
                    # (plain stdout is only shown in verbose mode for PreToolUse)
                    warn_msg = f"WARNING - Gate rule triggered: {rule['reason']} (pattern: {rule['raw_pattern']}). Consider if this command is appropriate."
                    print(json.dumps({
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "additionalContext": warn_msg
                        }
                    }))
                    sys.exit(0)

                elif severity == "confirm":
                    # Confirm: escalate to user via "ask" permission decision
                    print(json.dumps({
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "ask",
                            "permissionDecisionReason": f"Gate rule: {rule['reason']} (pattern: {rule['raw_pattern']})"
                        }
                    }))
                    sys.exit(0)

        # Log tool use (regardless of gate check result)
        log_tool_use(input_data, session_id)

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == '__main__':
    main()