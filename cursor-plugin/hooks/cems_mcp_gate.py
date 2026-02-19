#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS Gate Rules — beforeMCPExecution Hook (Cursor)

Checks MCP tool calls against cached gate rules before execution.
Uses the same gate rule cache as shell execution hook.

Cursor hook output format:
  {"decision": "allow"}                  — let MCP call proceed
  {"decision": "deny", "reason": "..."}  — block MCP call
"""

import json
import re
import subprocess
import sys
from pathlib import Path

GATE_CACHE_DIR = Path.home() / ".cems" / "cache" / "gate_rules"


def get_project_id() -> str | None:
    """Extract project ID from git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix(".git")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def load_gate_rules(project: str | None) -> list[dict]:
    """Load cached gate rules for a project."""
    if project:
        safe_name = project.replace("/", "_").replace("\\", "_")
        cache_path = GATE_CACHE_DIR / f"{safe_name}.json"
    else:
        cache_path = GATE_CACHE_DIR / "global.json"

    if not cache_path.exists():
        return []

    try:
        data = json.loads(cache_path.read_text())
        rules = []
        for rule in data:
            rule["regex"] = re.compile(rule["pattern"], re.IGNORECASE)
            rules.append(rule)
        return rules
    except (OSError, json.JSONDecodeError, KeyError, re.error):
        return []


def check_mcp_call(tool_name: str, args_str: str, project: str | None) -> dict:
    """Check an MCP tool call against gate rules."""
    rules = load_gate_rules(project)
    if project:
        rules.extend(load_gate_rules(None))

    # Check against both tool name and serialized arguments
    check_text = f"{tool_name} {args_str}"

    for rule in rules:
        rule_tool = rule.get("tool", "").lower()
        # Match MCP-specific rules or catch-all rules
        if rule_tool not in ("mcp", "all", tool_name.lower()):
            continue

        rule_project = rule.get("project")
        if rule_project and rule_project != project:
            continue

        if rule["regex"].search(check_text):
            return {
                "matched": True,
                "severity": rule.get("severity", "warn"),
                "reason": rule.get("reason", "Gate rule matched"),
                "pattern": rule.get("raw_pattern", rule["pattern"]),
            }

    return {"matched": False}


def main():
    try:
        input_data = json.load(sys.stdin)
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        if not tool_name:
            print(json.dumps({"decision": "allow"}))
            return

        args_str = json.dumps(tool_input) if tool_input else ""
        project = get_project_id()
        result = check_mcp_call(tool_name, args_str, project)

        if not result["matched"]:
            print(json.dumps({"decision": "allow"}))
            return

        severity = result["severity"]
        reason = result["reason"]
        pattern = result["pattern"]

        if severity == "block":
            print(json.dumps({
                "decision": "deny",
                "reason": f"CEMS gate rule: {reason} (pattern: {pattern})"
            }))
        elif severity == "confirm":
            # beforeMCPExecution only supports allow/deny — block on confirm
            print(json.dumps({
                "decision": "deny",
                "reason": f"CEMS gate rule (confirm): {reason} (pattern: {pattern})"
            }))
        else:
            # warn — allow the call but log for visibility
            # (unlike Claude Code which can inject warnings, Cursor MCP hooks
            # only support allow/deny — so we allow and trust the rule is advisory)
            print(json.dumps({"decision": "allow"}))

    except json.JSONDecodeError:
        print(json.dumps({"decision": "allow"}))
    except Exception:
        print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
