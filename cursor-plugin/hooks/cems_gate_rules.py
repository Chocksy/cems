#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS Gate Rules — beforeShellExecution Hook (Cursor)

Checks shell commands against cached gate rules before execution.
Returns allow/deny/ask decisions per Cursor's hook protocol.

Gate rules are cached at ~/.cems/cache/gate_rules/{project}.json
by the CEMS system (populated via API or observer).

Cursor hook output format:
  {"decision": "allow"}              — let command run
  {"decision": "deny", "reason": "..."} — block command
  {"decision": "ask", "message": "..."}  — prompt user
"""

import json
import re
import subprocess
import sys
from pathlib import Path

GATE_CACHE_DIR = Path.home() / ".cems" / "cache" / "gate_rules"


def get_project_id() -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo')."""
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


def check_command(command: str, project: str | None) -> dict:
    """Check a shell command against gate rules."""
    rules = load_gate_rules(project)
    if project:
        rules.extend(load_gate_rules(None))

    for rule in rules:
        if rule.get("tool", "").lower() != "bash":
            continue

        rule_project = rule.get("project")
        if rule_project and rule_project != project:
            continue

        if rule["regex"].search(command):
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
        command = input_data.get("command", "")

        if not command:
            print(json.dumps({"decision": "allow"}))
            return

        project = get_project_id()
        result = check_command(command, project)

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
            print(json.dumps({
                "decision": "ask",
                "message": f"CEMS gate rule: {reason} (pattern: {pattern})"
            }))
        else:
            # warn — allow but Cursor doesn't support injecting warnings,
            # so we allow and rely on beforeShellExecution's ask for visibility
            print(json.dumps({
                "decision": "ask",
                "message": f"CEMS warning: {reason} (pattern: {pattern})"
            }))

    except json.JSONDecodeError:
        print(json.dumps({"decision": "allow"}))
    except Exception:
        print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
