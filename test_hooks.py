#!/usr/bin/env python3
"""Hook validation tests for CEMS - validates all hooks and their API dependencies.

Tests hooks via direct subprocess invocation (no Claude Code needed) and validates
the CEMS API endpoints that hooks depend on.

Requires: Docker instance running with cems-server container.
Usage: python test_hooks.py

Test groups:
  1. Direct hook testing (subprocess) - pipe JSON stdin, check stdout/stderr/exitcode
  2. CEMS API validation - endpoints hooks depend on
  3. Project scoring validation - source_ref boost in search results
  4. Gate cache validation - populate cache, then verify pre_tool_use blocks
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "http://localhost:8765"
CONTAINER_NAME = "cems-server"
API_KEY = ""  # Set dynamically in setup_api_key()

HOOKS_DIR = Path(__file__).parent / "hooks"
GATE_CACHE_DIR = Path.home() / ".cems" / "cache" / "gate_rules"

# Unique tag so we can clean up test artifacts
TEST_RUN_TAG = f"hooktest_{int(time.time())}"


# ---------------------------------------------------------------------------
# Helpers - API
# ---------------------------------------------------------------------------

def call_api(method: str, endpoint: str, data: dict | None = None) -> dict:
    """Call the REST API directly via docker exec."""
    curl_cmd = [
        "docker", "exec", CONTAINER_NAME,
        "curl", "-s", "-X", method,
        f"{API_URL}{endpoint}",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {API_KEY}",
    ]

    if data:
        curl_cmd.extend(["-d", json.dumps(data)])

    result = subprocess.run(curl_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"curl failed: {result.stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON response: {result.stdout}")


def check_docker() -> bool:
    """Check if Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        if CONTAINER_NAME not in result.stdout:
            print(f"FAIL: Docker container '{CONTAINER_NAME}' is not running")
            print("   Start it with: docker compose up -d")
            return False
        return True
    except Exception as e:
        print(f"FAIL: Error checking Docker: {e}")
        return False


def setup_api_key() -> bool:
    """Resolve API key: CEMS_TEST_API_KEY env var > auto-provision via admin API."""
    global API_KEY

    # 1. Check env var
    env_key = os.environ.get("CEMS_TEST_API_KEY")
    if env_key:
        API_KEY = env_key
        print(f"Using API key from CEMS_TEST_API_KEY env var (prefix: {API_KEY[:16]}...)")
        return True

    # 2. Auto-provision: get admin key from Docker, create a test user
    print("No CEMS_TEST_API_KEY set, auto-provisioning test user...")
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "printenv", "CEMS_ADMIN_KEY"],
            capture_output=True, text=True,
        )
        admin_key = result.stdout.strip()
        if not admin_key:
            print("FAIL: Could not get CEMS_ADMIN_KEY from container")
            return False

        username = f"hook_test_{int(time.time())}"
        create_cmd = [
            "docker", "exec", CONTAINER_NAME,
            "curl", "-s", "-X", "POST",
            f"{API_URL}/admin/users",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {admin_key}",
            "-d", json.dumps({"username": username, "email": f"{username}@test.local"}),
        ]
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        response = json.loads(result.stdout)

        if "api_key" in response:
            API_KEY = response["api_key"]
            print(f"Auto-provisioned test user '{username}' (key prefix: {API_KEY[:16]}...)")
            return True
        else:
            print(f"FAIL: Failed to create test user: {response.get('error', 'unknown')}")
            return False

    except Exception as e:
        print(f"FAIL: Error auto-provisioning API key: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers - Hook subprocess invocation
# ---------------------------------------------------------------------------

def run_hook(
    script_name: str,
    stdin_data: dict,
    env_overrides: dict | None = None,
    args: list[str] | None = None,
    timeout: int = 15,
) -> tuple[int, str, str]:
    """Run a hook script via subprocess, piping JSON to stdin.

    Args:
        script_name: Hook filename inside HOOKS_DIR (e.g., "pre_tool_use.py")
        stdin_data: Dict to serialize as JSON and pipe to stdin
        env_overrides: Extra env vars to set (merged with inherited env)
        args: Extra CLI arguments after the script path
        timeout: Seconds before killing the process

    Returns:
        (exit_code, stdout, stderr)
    """
    script_path = HOOKS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Hook script not found: {script_path}")

    cmd = ["uv", "run", str(script_path)]
    if args:
        cmd.extend(args)

    env = os.environ.copy()
    # Ensure hooks can reach the CEMS API through Docker
    env["CEMS_API_URL"] = API_URL
    env["CEMS_API_KEY"] = API_KEY
    if env_overrides:
        env.update(env_overrides)

    proc = subprocess.run(
        cmd,
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    return proc.returncode, proc.stdout, proc.stderr


def parse_hook_output(stdout: str) -> dict | None:
    """Parse JSON from hook stdout. Returns None if not JSON."""
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Helpers - Cleanup
# ---------------------------------------------------------------------------

# Accumulate memory IDs for cleanup at end
_cleanup_memory_ids: list[str] = []


def add_test_memory(content: str, **kwargs) -> str | None:
    """Add a memory and register it for cleanup. Returns memory_id or None."""
    payload = {"content": content, "infer": False, **kwargs}
    result = call_api("POST", "/api/memory/add", payload)
    if not result.get("success"):
        return None
    mem_result = result.get("result", {})
    results_list = mem_result.get("results", [])
    if not results_list:
        return None
    memory_id = results_list[0].get("id")
    if memory_id:
        _cleanup_memory_ids.append(memory_id)
    return memory_id


def cleanup_test_memories():
    """Hard-delete all test memories created during this run."""
    for mid in _cleanup_memory_ids:
        try:
            call_api("POST", "/api/memory/forget", {"memory_id": mid, "hard_delete": True})
        except Exception:
            pass
    _cleanup_memory_ids.clear()


# ==========================================================================
# GROUP 1: Direct hook testing (subprocess)
# ==========================================================================


def test_session_start_produces_context() -> tuple[bool, str]:
    """SessionStart hook should output hookSpecificOutput JSON with profile context."""
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "source": "startup",
        "is_background_agent": False,
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("cems_session_start.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}, stderr: {stderr[:200]}"

    parsed = parse_hook_output(stdout)
    if parsed is None:
        # No output is acceptable if profile is empty
        return True, "no output (empty profile or unconfigured)"

    hook_output = parsed.get("hookSpecificOutput", {})
    if hook_output.get("hookEventName") != "SessionStart":
        return False, f"unexpected hookEventName: {hook_output.get('hookEventName')}"

    context = hook_output.get("additionalContext", "")
    if "<cems-profile>" not in context:
        return False, f"missing <cems-profile> in context (got {len(context)} chars)"

    return True, f"injected {len(context)} chars of profile context"


def test_session_start_skips_background_agent() -> tuple[bool, str]:
    """SessionStart hook should exit silently for background agents."""
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "source": "startup",
        "is_background_agent": True,
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("cems_session_start.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    if stdout.strip():
        return False, f"expected no output for background agent, got: {stdout[:100]}"

    return True, "correctly skipped background agent"


def test_session_start_skips_resume() -> tuple[bool, str]:
    """SessionStart hook should exit silently for resume/compact sources."""
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "source": "resume",
        "is_background_agent": False,
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, _ = run_hook("cems_session_start.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    if stdout.strip():
        return False, f"expected no output for resume, got: {stdout[:100]}"

    return True, "correctly skipped resume source"


def test_user_prompt_submit_injects_memories() -> tuple[bool, str]:
    """UserPromptSubmit hook should search CEMS and inject memory context.

    Requires at least one memory to exist that matches the test query.
    We add one first, then run the hook.
    """
    # Add a test memory so the search has something to find
    mem_id = add_test_memory(
        f"Hook test memory {TEST_RUN_TAG}: user prefers dark mode and vim keybindings",
        category="preferences",
    )
    if not mem_id:
        return False, "could not add test memory for search"

    # Small delay for vector indexing
    time.sleep(0.5)

    stdin = {
        "prompt": "What are my preferences for dark mode and vim keybindings?",
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("user_prompts_submit.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}, stderr: {stderr[:200]}"

    parsed = parse_hook_output(stdout)
    if parsed is None:
        return False, "no JSON output (search may have returned empty)"

    hook_output = parsed.get("hookSpecificOutput", {})
    context = hook_output.get("additionalContext", "")

    if "<memory-recall>" not in context:
        return False, f"missing <memory-recall> tag (got: {context[:100]})"

    return True, f"injected memory context ({len(context)} chars)"


def test_user_prompt_submit_ultrathink_flag() -> tuple[bool, str]:
    """UserPromptSubmit hook should detect -u flag and append ultrathink instruction."""
    stdin = {
        "prompt": "Explain how the search pipeline works in detail -u",
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, _ = run_hook("user_prompts_submit.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    parsed = parse_hook_output(stdout)
    if parsed is None:
        return False, "no output"

    context = parsed.get("hookSpecificOutput", {}).get("additionalContext", "")
    if "ultrathink" not in context.lower():
        return False, f"ultrathink instruction not found in output"

    return True, "ultrathink flag detected and injected"


def test_user_prompt_submit_short_prompt_with_u() -> tuple[bool, str]:
    """Short prompts (<15 chars) with -u should still get ultrathink."""
    stdin = {
        "prompt": "fix it -u",
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, _ = run_hook("user_prompts_submit.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    parsed = parse_hook_output(stdout)
    if parsed is None:
        return False, "no output for short -u prompt"

    context = parsed.get("hookSpecificOutput", {}).get("additionalContext", "")
    if "ultrathink" not in context.lower():
        return False, "ultrathink not injected for short -u prompt"

    return True, "short prompt with -u correctly handled"


def test_user_prompt_submit_skips_slash_commands() -> tuple[bool, str]:
    """UserPromptSubmit should skip slash commands (e.g., /recall)."""
    stdin = {
        "prompt": "/recall some query here",
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, _ = run_hook("user_prompts_submit.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    if stdout.strip():
        return False, f"expected no output for slash command, got: {stdout[:100]}"

    return True, "slash command correctly skipped"


def test_user_prompt_submit_skips_subagent() -> tuple[bool, str]:
    """UserPromptSubmit should skip when CLAUDE_AGENT_ID is set (subagent)."""
    stdin = {
        "prompt": "This is a subagent prompt that should be ignored",
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, _ = run_hook(
        "user_prompts_submit.py",
        stdin,
        env_overrides={"CLAUDE_AGENT_ID": "subagent-123"},
    )

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    if stdout.strip():
        return False, f"expected no output for subagent, got: {stdout[:100]}"

    return True, "subagent correctly skipped"


def test_pre_tool_use_allows_safe_command() -> tuple[bool, str]:
    """PreToolUse should allow commands that match no gate rules (exit 0, no block)."""
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("pre_tool_use.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode} for safe command"

    # stdout may have logging JSON, but should NOT have block/warn
    parsed = parse_hook_output(stdout)
    if parsed:
        hook_output = parsed.get("hookSpecificOutput", {})
        if "permissionDecision" in hook_output:
            return False, "safe command should not trigger permission decision"

    return True, "safe command allowed (exit 0)"


def test_pre_tool_use_blocks_matching_command() -> tuple[bool, str]:
    """PreToolUse should block commands matching a cached 'block' gate rule.

    This test writes a gate rule cache file directly, then invokes pre_tool_use.
    """
    # Write a test gate rule cache
    test_cache_dir = GATE_CACHE_DIR
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = test_cache_dir / "global.json"

    # Backup existing cache if present
    backup = None
    if cache_file.exists():
        backup = cache_file.read_text()

    rules = [
        {
            "tool": "bash",
            "pattern": "rm\\s+-rf\\s+/",
            "raw_pattern": "rm -rf /",
            "reason": "Dangerous recursive delete at root",
            "severity": "block",
            "project": None,
        }
    ]
    cache_file.write_text(json.dumps(rules, indent=2))

    try:
        stdin = {
            "session_id": f"test-{TEST_RUN_TAG}",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /tmp/old-data"},
            "cwd": str(Path.cwd()),
        }

        exitcode, stdout, stderr = run_hook("pre_tool_use.py", stdin)

        if exitcode != 2:
            return False, f"expected exit code 2 (block), got {exitcode}. stderr={stderr[:100]}"

        if "BLOCKED" not in stderr:
            return False, f"expected BLOCKED in stderr, got: {stderr[:200]}"

        return True, "dangerous command blocked (exit 2)"

    finally:
        # Restore original cache
        if backup is not None:
            cache_file.write_text(backup)
        else:
            cache_file.unlink(missing_ok=True)


def test_pre_tool_use_warns_matching_command() -> tuple[bool, str]:
    """PreToolUse should warn (not block) for 'warn' severity gate rules."""
    test_cache_dir = GATE_CACHE_DIR
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = test_cache_dir / "global.json"

    backup = None
    if cache_file.exists():
        backup = cache_file.read_text()

    rules = [
        {
            "tool": "bash",
            "pattern": "docker\\s+compose\\s+down",
            "raw_pattern": "docker compose down",
            "reason": "This will stop all containers",
            "severity": "warn",
            "project": None,
        }
    ]
    cache_file.write_text(json.dumps(rules, indent=2))

    try:
        stdin = {
            "session_id": f"test-{TEST_RUN_TAG}",
            "tool_name": "Bash",
            "tool_input": {"command": "docker compose down -v"},
            "cwd": str(Path.cwd()),
        }

        exitcode, stdout, stderr = run_hook("pre_tool_use.py", stdin)

        if exitcode != 0:
            return False, f"expected exit code 0 (warn, not block), got {exitcode}"

        parsed = parse_hook_output(stdout)
        if not parsed:
            return False, "expected JSON warn output on stdout"

        context = parsed.get("hookSpecificOutput", {}).get("additionalContext", "")
        if "WARNING" not in context:
            return False, f"expected WARNING in additionalContext, got: {context[:100]}"

        return True, "matching command triggered warn (exit 0 with warning)"

    finally:
        if backup is not None:
            cache_file.write_text(backup)
        else:
            cache_file.unlink(missing_ok=True)


def test_stop_hook_logs_session() -> tuple[bool, str]:
    """Stop hook should read stdin JSON and log session data without errors.

    We skip the transcript analysis and TTS parts by not providing transcript_path.
    """
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "transcript_path": "",
        "cwd": str(Path.cwd()),
        "stop_reason": "end_turn",
    }

    # The stop hook writes to logs/ dir under cwd. Use a temp dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        exitcode, stdout, stderr = run_hook(
            "stop.py",
            stdin,
            env_overrides={
                # Unset TTS keys so it skips announce_completion quickly
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )

    if exitcode != 0:
        return False, f"exit code {exitcode}, stderr: {stderr[:200]}"

    return True, "stop hook executed cleanly (exit 0)"


def test_stop_hook_with_mock_transcript() -> tuple[bool, str]:
    """Stop hook should send transcript to /api/session/analyze when provided."""
    # Create a mock transcript file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"type": "user", "message": {"content": "Hello"}}) + "\n")
        f.write(json.dumps({"type": "assistant", "message": {"content": "Hi there!"}}) + "\n")
        f.write(json.dumps({"type": "user", "message": {"content": "What is CEMS?"}}) + "\n")
        f.write(json.dumps({"type": "assistant", "message": {"content": "CEMS is a memory system."}}) + "\n")
        transcript_path = f.name

    try:
        stdin = {
            "session_id": f"test-{TEST_RUN_TAG}",
            "transcript_path": transcript_path,
            "cwd": str(Path.cwd()),
        }

        exitcode, stdout, stderr = run_hook(
            "stop.py",
            stdin,
            env_overrides={
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )

        if exitcode != 0:
            return False, f"exit code {exitcode}, stderr: {stderr[:200]}"

        return True, "stop hook with transcript executed (exit 0)"

    finally:
        os.unlink(transcript_path)


def test_post_tool_use_skips_reads() -> tuple[bool, str]:
    """PostToolUse hook should skip Read tool (non-learnable)."""
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/test.py"},
        "tool_response": {"content": "file contents"},
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("cems_post_tool_use.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    return True, "Read tool correctly skipped (exit 0)"


def test_post_tool_use_processes_edit() -> tuple[bool, str]:
    """PostToolUse hook should process Edit tool (learnable)."""
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "tool_name": "Edit",
        "tool_input": {"file_path": "/src/test.py"},
        "tool_response": {"output": "File edited successfully"},
        "transcript_path": "",
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("cems_post_tool_use.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}, stderr: {stderr[:200]}"

    # It should attempt to send to CEMS (may or may not store depending on LLM)
    return True, "Edit tool processed (exit 0)"


def test_post_tool_use_skips_background_agent() -> tuple[bool, str]:
    """PostToolUse hook should skip background agents."""
    stdin = {
        "session_id": f"test-{TEST_RUN_TAG}",
        "tool_name": "Edit",
        "tool_input": {"file_path": "/src/test.py"},
        "tool_response": {"output": "Success"},
        "is_background_agent": True,
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("cems_post_tool_use.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    return True, "background agent correctly skipped (exit 0)"


# ==========================================================================
# GROUP 2: CEMS API validation (what hooks depend on)
# ==========================================================================


def test_api_search_returns_source_ref() -> tuple[bool, str]:
    """Search results MUST include source_ref field (used by hooks for project scoring)."""
    # Add a memory with source_ref
    mem_id = add_test_memory(
        f"Source ref test {TEST_RUN_TAG}: API returns source_ref in search",
        category="test",
        source_ref="project:hooktest/repo",
    )
    if not mem_id:
        return False, "could not add test memory"

    time.sleep(0.5)

    result = call_api("POST", "/api/memory/search", {
        "query": f"source ref test {TEST_RUN_TAG}",
        "limit": 5,
        "raw": True,
    })

    if not result.get("success"):
        return False, f"search failed: {result.get('error')}"

    results = result.get("results", [])
    if not results:
        return False, "no search results"

    # Check that at least one result has source_ref
    found_source_ref = False
    for r in results:
        metadata = r.get("metadata", {})
        if metadata.get("source_ref") == "project:hooktest/repo":
            found_source_ref = True
            break

    if not found_source_ref:
        # Check alternative response formats (enhanced search uses flat keys)
        for r in results:
            if r.get("source_ref") == "project:hooktest/repo":
                found_source_ref = True
                break

    if not found_source_ref:
        sample = json.dumps(results[0], indent=2)[:300]
        return False, f"source_ref not found in results. Sample: {sample}"

    return True, "search results include source_ref"


def test_api_add_stores_source_ref() -> tuple[bool, str]:
    """POST /api/memory/add should store source_ref and return it."""
    mem_id = add_test_memory(
        f"Store source_ref test {TEST_RUN_TAG}",
        category="test",
        source_ref="project:hooktest/addtest",
    )
    if not mem_id:
        return False, "could not add memory with source_ref"

    # Verify by searching
    time.sleep(0.5)
    result = call_api("POST", "/api/memory/search", {
        "query": f"Store source_ref test {TEST_RUN_TAG}",
        "limit": 3,
        "raw": True,
    })

    if not result.get("success"):
        return False, f"search failed: {result.get('error')}"

    results = result.get("results", [])
    for r in results:
        metadata = r.get("metadata", {})
        sr = metadata.get("source_ref") or r.get("source_ref")
        if sr == "project:hooktest/addtest":
            return True, f"source_ref correctly stored and retrieved"

    return False, "source_ref not found in search results after add"


def test_api_gate_rules_returns_rules() -> tuple[bool, str]:
    """GET /api/memory/gate-rules should return rules list."""
    result = call_api("GET", "/api/memory/gate-rules")

    if not result.get("success"):
        return False, f"failed: {result.get('error')}"

    rules = result.get("rules", [])
    count = result.get("count", 0)

    if not isinstance(rules, list):
        return False, f"rules is not a list: {type(rules)}"

    return True, f"returned {count} gate rules"


def test_api_gate_rules_with_project_filter() -> tuple[bool, str]:
    """GET /api/memory/gate-rules?project=X should filter by project."""
    # Add a project-scoped gate rule
    mem_id = add_test_memory(
        f"Bash: dangerous_command_{TEST_RUN_TAG} -- test gate rule for project filtering",
        category="gate-rules",
        source_ref="project:hooktest/gatefilter",
        tags=["block"],
    )
    if not mem_id:
        return False, "could not add gate rule memory"

    time.sleep(0.5)

    # Fetch with project filter
    result = call_api("GET", "/api/memory/gate-rules?project=hooktest/gatefilter")

    if not result.get("success"):
        return False, f"failed: {result.get('error')}"

    rules = result.get("rules", [])

    # Check our test rule is present
    found = any(TEST_RUN_TAG in r.get("content", "") for r in rules)
    if not found:
        return False, f"test gate rule not found in {len(rules)} rules"

    return True, f"project-filtered gate rules returned ({len(rules)} rules)"


def test_api_log_shown_works() -> tuple[bool, str]:
    """POST /api/memory/log-shown should increment shown_count."""
    # Add a memory to log as shown
    mem_id = add_test_memory(
        f"Log shown test {TEST_RUN_TAG}",
        category="test",
    )
    if not mem_id:
        return False, "could not add test memory"

    result = call_api("POST", "/api/memory/log-shown", {
        "memory_ids": [mem_id],
    })

    if not result.get("success"):
        return False, f"failed: {result.get('error')}"

    updated = result.get("updated", 0)
    if updated < 1:
        return False, f"expected at least 1 updated, got {updated}"

    return True, f"updated {updated} memory shown count"


def test_api_log_shown_empty_list() -> tuple[bool, str]:
    """POST /api/memory/log-shown with empty list should succeed."""
    result = call_api("POST", "/api/memory/log-shown", {
        "memory_ids": [],
    })

    if not result.get("success"):
        return False, f"failed: {result.get('error')}"

    updated = result.get("updated", -1)
    if updated != 0:
        return False, f"expected 0 updated for empty list, got {updated}"

    return True, "empty memory_ids correctly handled"


def test_api_profile_returns_context() -> tuple[bool, str]:
    """GET /api/memory/profile should return structured profile context."""
    result = call_api("GET", "/api/memory/profile?token_budget=2000")

    if not result.get("success"):
        return False, f"failed: {result.get('error')}"

    if "context" not in result:
        return False, "missing 'context' field"
    if "components" not in result:
        return False, "missing 'components' field"
    if "token_estimate" not in result:
        return False, "missing 'token_estimate' field"

    components = result["components"]
    expected_keys = {"preferences", "guidelines", "recent_memories", "gate_rules_count"}
    missing = expected_keys - set(components.keys())
    if missing:
        return False, f"missing component keys: {missing}"

    tokens = result.get("token_estimate", 0)
    return True, f"profile returned (~{tokens} tokens)"


def test_api_profile_with_project() -> tuple[bool, str]:
    """GET /api/memory/profile?project=X should return project-scoped context."""
    result = call_api("GET", "/api/memory/profile?project=hooktest/repo&token_budget=1500")

    if not result.get("success"):
        return False, f"failed: {result.get('error')}"

    if "context" not in result or "components" not in result:
        return False, "missing expected fields"

    return True, "project-scoped profile returned"


# ==========================================================================
# GROUP 3: Project scoring validation
# ==========================================================================


def test_project_scoring_order() -> tuple[bool, str]:
    """Same-project memories should rank higher than non-project memories.

    Strategy:
    1. Add memory A with source_ref=project:hooktest/scoring
    2. Add memory B with identical content but NO source_ref
    3. Search with project=hooktest/scoring
    4. Memory A should appear before or equal to Memory B
    """
    unique = f"project scoring validation {TEST_RUN_TAG}"

    # Memory A: project-scoped
    mem_a = add_test_memory(
        f"{unique} - uses pytest for testing framework",
        category="test",
        source_ref="project:hooktest/scoring",
    )
    if not mem_a:
        return False, "could not add project-scoped memory"

    # Memory B: no project scope (global)
    mem_b = add_test_memory(
        f"{unique} - uses pytest for testing framework globally",
        category="test",
    )
    if not mem_b:
        return False, "could not add global memory"

    time.sleep(0.5)

    # Search with project parameter
    result = call_api("POST", "/api/memory/search", {
        "query": f"{unique} pytest testing",
        "limit": 10,
        "project": "hooktest/scoring",
    })

    if not result.get("success"):
        return False, f"search failed: {result.get('error')}"

    results = result.get("results", [])
    if not results:
        return False, "no search results"

    # Find positions of our two memories
    pos_a = None
    pos_b = None
    for i, r in enumerate(results):
        content = r.get("content", r.get("memory", ""))
        mid = r.get("memory_id", r.get("id", ""))
        if mid == mem_a:
            pos_a = i
        elif mid == mem_b:
            pos_b = i

    if pos_a is None:
        return False, f"project-scoped memory not in results (checked {len(results)} results)"

    if pos_b is None:
        # Project memory found, global not found -- that is fine (project ranked higher)
        return True, f"project memory at pos {pos_a}, global memory filtered out"

    if pos_a <= pos_b:
        return True, f"project memory ranked {pos_a} vs global at {pos_b}"
    else:
        return False, f"project memory at {pos_a} ranked LOWER than global at {pos_b}"


# ==========================================================================
# GROUP 4: Gate cache validation (end-to-end)
# ==========================================================================


def test_gate_cache_population() -> tuple[bool, str]:
    """UserPromptSubmit should populate gate cache file on first run.

    The hook calls populate_gate_cache() which fetches from /api/memory/gate-rules
    and writes to ~/.cems/cache/gate_rules/{project}.json.
    """
    # We need a project identifier. Use the current repo.
    cwd = str(Path.cwd())

    # First, clear any existing cache for this project to force refresh
    # We detect the project the same way hooks do (git remote)
    try:
        git_result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2,
        )
        if git_result.returncode == 0:
            url = git_result.stdout.strip()
            import re
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            project_id = match.group(1).removesuffix('.git') if match else None
        else:
            project_id = None
    except Exception:
        project_id = None

    if not project_id:
        return True, "skipped (not in a git repo with remote)"

    # Clear existing cache
    safe_name = project_id.replace("/", "_").replace("\\", "_")
    cache_file = GATE_CACHE_DIR / f"{safe_name}.json"
    if cache_file.exists():
        cache_file.unlink()

    # Also clear global cache
    global_cache = GATE_CACHE_DIR / "global.json"
    global_backup = None
    if global_cache.exists():
        global_backup = global_cache.read_text()
        global_cache.unlink()

    try:
        # Run UserPromptSubmit -- this triggers populate_gate_cache()
        stdin = {
            "prompt": "A simple prompt to trigger gate cache population for testing purposes here",
            "cwd": cwd,
        }

        exitcode, stdout, stderr = run_hook("user_prompts_submit.py", stdin)

        if exitcode != 0:
            return False, f"hook failed: exit {exitcode}, stderr: {stderr[:200]}"

        # Check if either project or global cache was created
        project_cache_exists = cache_file.exists()
        global_cache_exists = global_cache.exists()

        if not project_cache_exists and not global_cache_exists:
            return False, f"no cache file created at {cache_file} or {global_cache}"

        # Read whichever cache exists
        active_cache = cache_file if project_cache_exists else global_cache
        try:
            cache_data = json.loads(active_cache.read_text())
        except json.JSONDecodeError:
            return False, f"cache file exists but contains invalid JSON"

        if not isinstance(cache_data, list):
            return False, f"cache data is not a list: {type(cache_data)}"

        return True, f"cache populated at {active_cache.name} ({len(cache_data)} rules)"

    finally:
        # Restore global cache
        if global_backup is not None:
            global_cache.write_text(global_backup)


def test_gate_cache_blocks_command() -> tuple[bool, str]:
    """End-to-end: add gate rule via API, populate cache via hook, verify block.

    1. Add a gate rule memory to CEMS
    2. Clear cache
    3. Run user_prompts_submit to populate cache
    4. Run pre_tool_use with matching command
    5. Verify exit code 2 (blocked)
    """
    # Step 1: Add gate rule to CEMS
    unique_cmd = f"dangerous_e2e_{TEST_RUN_TAG}"
    mem_id = add_test_memory(
        f"Bash: {unique_cmd} -- E2E test: must never run this command",
        category="gate-rules",
        tags=["block"],
    )
    if not mem_id:
        return False, "could not add gate rule memory"

    time.sleep(0.5)

    # Step 2: Clear global cache to force re-fetch
    global_cache = GATE_CACHE_DIR / "global.json"
    backup = None
    if global_cache.exists():
        backup = global_cache.read_text()
        global_cache.unlink()

    try:
        # Step 3: Run user_prompts_submit to populate cache
        stdin = {
            "prompt": "This prompt triggers gate cache population for our end-to-end test",
            "cwd": str(Path.cwd()),
        }
        run_hook("user_prompts_submit.py", stdin)

        # Verify cache was populated with our rule
        if not global_cache.exists():
            # Also check project-specific cache
            cwd = str(Path.cwd())
            try:
                git_result = subprocess.run(
                    ["git", "-C", cwd, "remote", "get-url", "origin"],
                    capture_output=True, text=True, timeout=2,
                )
                if git_result.returncode == 0:
                    url = git_result.stdout.strip()
                    import re
                    if url.startswith("git@"):
                        match = re.search(r":(.+?)(?:\.git)?$", url)
                    else:
                        match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
                    if match:
                        project_id = match.group(1).removesuffix('.git')
                        safe_name = project_id.replace("/", "_")
                        project_cache = GATE_CACHE_DIR / f"{safe_name}.json"
                        if not project_cache.exists():
                            return False, "no cache file created after hook run"
                    else:
                        return False, "no cache file created (global or project)"
                else:
                    return False, "no cache file created (global)"
            except Exception:
                return False, "no cache file created"

        # Step 4: Run pre_tool_use with matching command
        stdin = {
            "session_id": f"test-{TEST_RUN_TAG}",
            "tool_name": "Bash",
            "tool_input": {"command": f"{unique_cmd} --force"},
            "cwd": str(Path.cwd()),
        }

        exitcode, stdout, stderr = run_hook("pre_tool_use.py", stdin)

        # Step 5: Verify block
        if exitcode != 2:
            # Check if the rule made it to cache
            for cache_f in GATE_CACHE_DIR.glob("*.json"):
                try:
                    data = json.loads(cache_f.read_text())
                    rules_str = json.dumps(data)
                    if unique_cmd in rules_str:
                        return False, f"rule in cache ({cache_f.name}) but not blocked (exit {exitcode})"
                except Exception:
                    pass
            return False, f"expected exit 2 (block), got {exitcode}. Rule may not be in cache."

        if "BLOCKED" not in stderr:
            return False, f"exit 2 but 'BLOCKED' not in stderr: {stderr[:200]}"

        return True, "end-to-end: gate rule added, cached, and blocked command"

    finally:
        # Restore original cache
        if backup is not None:
            global_cache.write_text(backup)
        elif global_cache.exists():
            global_cache.unlink()


# ==========================================================================
# Helpers - intent extraction unit tests
# ==========================================================================


def test_intent_extraction_basic() -> tuple[bool, str]:
    """Verify the hook handles prompts and sends them for memory search.

    The hook sends the raw prompt to the server — no local processing.
    This is a smoke test that the hook runs without error.
    """
    stdin = {
        "prompt": "Can you help me with Python async patterns and asyncio best practices?",
        "cwd": str(Path.cwd()),
    }

    exitcode, stdout, stderr = run_hook("user_prompts_submit.py", stdin)

    if exitcode != 0:
        return False, f"exit code {exitcode}"

    # We cannot directly check the intent extraction without importing the module,
    # but we verify the hook runs without error
    return True, "intent extraction ran without error"


# ==========================================================================
# Test runner
# ==========================================================================


def run_all_tests():
    """Run all hook validation tests with pass/fail summary."""
    print("\n" + "=" * 60)
    print("CEMS HOOK VALIDATION TESTS")
    print("=" * 60 + "\n")

    # Pre-checks
    if not check_docker():
        sys.exit(1)

    if not setup_api_key():
        sys.exit(1)

    # Verify hooks directory exists
    if not HOOKS_DIR.exists():
        print(f"FAIL: Hooks directory not found: {HOOKS_DIR}")
        sys.exit(1)

    print(f"Hooks directory: {HOOKS_DIR}")
    print(f"Gate cache dir: {GATE_CACHE_DIR}")
    print()

    tests = [
        # Group 1: Direct hook testing
        ("GROUP 1: DIRECT HOOK TESTING", None),
        ("SessionStart: profile injection", test_session_start_produces_context),
        ("SessionStart: skip background agent", test_session_start_skips_background_agent),
        ("SessionStart: skip resume", test_session_start_skips_resume),
        ("UserPromptSubmit: memory injection", test_user_prompt_submit_injects_memories),
        ("UserPromptSubmit: ultrathink -u flag", test_user_prompt_submit_ultrathink_flag),
        ("UserPromptSubmit: short prompt -u", test_user_prompt_submit_short_prompt_with_u),
        ("UserPromptSubmit: skip slash commands", test_user_prompt_submit_skips_slash_commands),
        ("UserPromptSubmit: skip subagent", test_user_prompt_submit_skips_subagent),
        ("PreToolUse: allow safe command", test_pre_tool_use_allows_safe_command),
        ("PreToolUse: block matching command", test_pre_tool_use_blocks_matching_command),
        ("PreToolUse: warn matching command", test_pre_tool_use_warns_matching_command),
        ("Stop: log session data", test_stop_hook_logs_session),
        ("Stop: mock transcript", test_stop_hook_with_mock_transcript),
        ("PostToolUse: skip Read", test_post_tool_use_skips_reads),
        ("PostToolUse: process Edit", test_post_tool_use_processes_edit),
        ("PostToolUse: skip background agent", test_post_tool_use_skips_background_agent),
        ("Intent extraction: smoke test", test_intent_extraction_basic),

        # Group 2: CEMS API validation
        ("GROUP 2: CEMS API VALIDATION", None),
        ("API: search returns source_ref", test_api_search_returns_source_ref),
        ("API: add stores source_ref", test_api_add_stores_source_ref),
        ("API: gate-rules returns rules", test_api_gate_rules_returns_rules),
        ("API: gate-rules project filter", test_api_gate_rules_with_project_filter),
        ("API: log-shown works", test_api_log_shown_works),
        ("API: log-shown empty list", test_api_log_shown_empty_list),
        ("API: profile returns context", test_api_profile_returns_context),
        ("API: profile with project", test_api_profile_with_project),

        # Group 3: Project scoring
        ("GROUP 3: PROJECT SCORING", None),
        ("Project scoring: same-project ranks higher", test_project_scoring_order),

        # Group 4: Gate cache end-to-end
        ("GROUP 4: GATE CACHE E2E", None),
        ("Gate cache: population via hook", test_gate_cache_population),
        ("Gate cache: add rule -> populate -> block", test_gate_cache_blocks_command),
    ]

    results = []
    for test_name, test_func in tests:
        if test_func is None:
            # Group header
            print(f"\n--- {test_name} ---")
            continue

        print(f"  {test_name}...", end=" ", flush=True)
        try:
            passed, message = test_func()
            if passed:
                print(f"OK: {message}")
                results.append((test_name, True, message))
            else:
                print(f"FAIL: {message}")
                results.append((test_name, False, message))
        except subprocess.TimeoutExpired:
            print("FAIL: timeout")
            results.append((test_name, False, "timeout"))
        except Exception as e:
            print(f"FAIL: {e}")
            results.append((test_name, False, str(e)))

    # Cleanup
    print("\nCleaning up test memories...", end=" ", flush=True)
    cleanup_test_memories()
    print("done")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)

    print(f"\n  Passed: {passed_count}/{total_count}")
    print(f"  Failed: {total_count - passed_count}/{total_count}\n")

    if passed_count < total_count:
        print("Failed tests:")
        for test_name, passed, message in results:
            if not passed:
                print(f"  - {test_name}: {message}")
        print()
        sys.exit(1)
    else:
        print("All hook tests passed!\n")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
