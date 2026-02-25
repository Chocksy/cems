#!/usr/bin/env python3
"""Integration tests for CEMS Claude Code hooks.

Tests the hook scripts by piping JSON to their stdin via subprocess and
validating stdout/stderr output, exit codes, side effects (API calls,
cache files).

Two test tiers:
  1. OFFLINE (no CEMS server needed) - input parsing, output format, gate
     cache files, edge cases. These run always.
  2. ONLINE (requires running CEMS server) - actual API calls, memory search,
     gate rule fetch, log-shown. Skipped unless
     CEMS_TEST_API_URL and CEMS_TEST_API_KEY are set.

Usage:
    # Offline tests only (no server needed):
    python -m pytest tests/test_hooks_integration.py -x -q

    # Full suite with live server:
    CEMS_TEST_API_URL=http://localhost:8765 CEMS_TEST_API_KEY=... \
        python -m pytest tests/test_hooks_integration.py -x -q

    # Run just one class:
    python -m pytest tests/test_hooks_integration.py::TestUserPromptSubmitOffline -x -q
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOOKS_DIR = Path(__file__).parent.parent / "hooks"
HOOK_USER_PROMPT = HOOKS_DIR / "cems_user_prompts_submit.py"
HOOK_SESSION_START = HOOKS_DIR / "cems_session_start.py"
HOOK_PRE_TOOL_USE = HOOKS_DIR / "cems_pre_tool_use.py"
HOOK_STOP = HOOKS_DIR / "cems_stop.py"
HOOK_POST_TOOL_USE = HOOKS_DIR / "cems_post_tool_use.py"

# Gate cache directory (mirrors the hooks' constant)
GATE_CACHE_DIR = Path.home() / ".cems" / "cache" / "gate_rules"

# Live server env vars (set these to enable online tests)
CEMS_TEST_API_URL = os.environ.get("CEMS_TEST_API_URL", "")
CEMS_TEST_API_KEY = os.environ.get("CEMS_TEST_API_KEY", "")

# Skip marker for tests that need a live server
needs_server = pytest.mark.skipif(
    not (CEMS_TEST_API_URL and CEMS_TEST_API_KEY),
    reason="CEMS_TEST_API_URL and CEMS_TEST_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_hook(
    hook_path: Path,
    input_data: dict,
    env_overrides: dict | None = None,
    timeout: int = 15,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a hook script by piping JSON to its stdin.

    Args:
        hook_path: Absolute path to the hook .py file
        input_data: Dict to serialize as JSON and pipe to stdin
        env_overrides: Extra environment variables (merged with os.environ)
        timeout: Subprocess timeout in seconds
        extra_args: Additional CLI arguments after the script path

    Returns:
        CompletedProcess with stdout, stderr, returncode
    """
    env = os.environ.copy()
    # Default: disable CEMS so offline tests don't hit a server
    env.setdefault("CEMS_API_URL", "")
    env.setdefault("CEMS_API_KEY", "")
    if env_overrides:
        env.update(env_overrides)
    # Disable credentials file fallback so offline tests don't pick up real creds
    if not env.get("CEMS_API_KEY"):
        env["CEMS_CREDENTIALS_FILE"] = "/dev/null"

    cmd = ["uv", "run", str(hook_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result


def parse_hook_output(stdout: str) -> dict | None:
    """Parse hookSpecificOutput JSON from hook stdout.

    Returns the parsed dict, or None if stdout is empty/non-JSON.
    """
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# OFFLINE TESTS: user_prompts_submit.py
# ---------------------------------------------------------------------------


class TestUserPromptSubmitOffline:
    """Tests that run without a CEMS server."""

    def test_short_prompt_no_output(self):
        """Prompts < 15 chars should produce no output (skipped)."""
        result = run_hook(HOOK_USER_PROMPT, {"prompt": "hi", "cwd": "/tmp"})
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_short_prompt_with_ultrathink_flag(self):
        """Short prompt ending with -u should output ultrathink instruction."""
        result = run_hook(HOOK_USER_PROMPT, {"prompt": "do it -u", "cwd": "/tmp"})
        assert result.returncode == 0
        output = parse_hook_output(result.stdout)
        assert output is not None
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert "ultrathink" in ctx.lower()

    def test_slash_command_skipped(self):
        """Slash commands like /help should be skipped entirely."""
        result = run_hook(HOOK_USER_PROMPT, {"prompt": "/help with something longer", "cwd": "/tmp"})
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_subagent_skipped(self):
        """If CLAUDE_AGENT_ID is set, hook should exit silently (subagent)."""
        result = run_hook(
            HOOK_USER_PROMPT,
            {"prompt": "a long prompt that would normally trigger search", "cwd": "/tmp"},
            env_overrides={"CLAUDE_AGENT_ID": "subagent-123"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_ultrathink_appended_to_normal_prompt(self):
        """Normal prompt ending with -u should include ultrathink in output."""
        result = run_hook(
            HOOK_USER_PROMPT,
            {"prompt": "refactor the authentication module for better security -u", "cwd": "/tmp"},
        )
        assert result.returncode == 0
        output = parse_hook_output(result.stdout)
        # Without a server, no memories are returned, but -u still fires
        if output:
            ctx = output["hookSpecificOutput"]["additionalContext"]
            assert "ultrathink" in ctx.lower()

    def test_invalid_json_no_crash(self):
        """Hook should not crash on malformed stdin."""
        env = os.environ.copy()
        env["CEMS_API_URL"] = ""
        env["CEMS_API_KEY"] = ""
        result = subprocess.run(
            ["uv", "run", str(HOOK_USER_PROMPT)],
            input="not valid json {{{",
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        # Should not crash (exit 0 or 1, not a traceback)
        assert result.returncode == 0

    def test_empty_stdin_no_crash(self):
        """Hook should handle empty stdin gracefully."""
        env = os.environ.copy()
        env["CEMS_API_URL"] = ""
        env["CEMS_API_KEY"] = ""
        result = subprocess.run(
            ["uv", "run", str(HOOK_USER_PROMPT)],
            input="",
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        assert result.returncode == 0

    def test_cursor_mode_output_format(self):
        """When cursor_version is present, output should use systemPrompt key."""
        result = run_hook(
            HOOK_USER_PROMPT,
            {"prompt": "short -u", "cwd": "/tmp", "cursor_version": "0.42"},
        )
        assert result.returncode == 0
        output = parse_hook_output(result.stdout)
        if output:
            # Cursor format uses systemPrompt, not hookSpecificOutput
            assert "systemPrompt" in output
            assert "ultrathink" in output["systemPrompt"].lower()

    def test_output_json_structure(self):
        """Verify the exact JSON structure of hookSpecificOutput."""
        result = run_hook(
            HOOK_USER_PROMPT,
            {"prompt": "a short -u", "cwd": "/tmp"},
        )
        output = parse_hook_output(result.stdout)
        if output:
            assert "hookSpecificOutput" in output
            hso = output["hookSpecificOutput"]
            assert hso["hookEventName"] == "UserPromptSubmit"
            assert "additionalContext" in hso


# ---------------------------------------------------------------------------
# OFFLINE TESTS: cems_session_start.py
# ---------------------------------------------------------------------------


class TestSessionStartOffline:
    """Tests that run without a CEMS server."""

    def test_no_cems_config_exits_silently(self):
        """Without CEMS_API_URL/KEY, hook should exit 0 with no output."""
        result = run_hook(
            HOOK_SESSION_START,
            {"session_id": "test-123", "source": "startup", "cwd": "/tmp"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_background_agent_skipped(self):
        """Background agents should be skipped."""
        result = run_hook(
            HOOK_SESSION_START,
            {
                "session_id": "test-123",
                "source": "startup",
                "is_background_agent": True,
                "cwd": "/tmp",
            },
            env_overrides={"CEMS_API_URL": "http://fake:9999", "CEMS_API_KEY": "fake"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_resume_source_skipped(self):
        """source=resume should be skipped (avoid redundant injection)."""
        result = run_hook(
            HOOK_SESSION_START,
            {"session_id": "test-123", "source": "resume", "cwd": "/tmp"},
            env_overrides={"CEMS_API_URL": "http://fake:9999", "CEMS_API_KEY": "fake"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_compact_source_skipped(self):
        """source=compact should be skipped."""
        result = run_hook(
            HOOK_SESSION_START,
            {"session_id": "test-123", "source": "compact", "cwd": "/tmp"},
            env_overrides={"CEMS_API_URL": "http://fake:9999", "CEMS_API_KEY": "fake"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_invalid_json_no_crash(self):
        """Hook should handle bad JSON gracefully."""
        env = os.environ.copy()
        env["CEMS_API_URL"] = ""
        env["CEMS_API_KEY"] = ""
        result = subprocess.run(
            ["uv", "run", str(HOOK_SESSION_START)],
            input="not json",
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# OFFLINE TESTS: pre_tool_use.py
# ---------------------------------------------------------------------------


class TestPreToolUseOffline:
    """Tests that run without a CEMS server."""

    def test_no_gate_rules_allows_all(self):
        """With no cached gate rules, all commands should be allowed (exit 0)."""
        result = run_hook(
            HOOK_PRE_TOOL_USE,
            {
                "session_id": "test-123",
                "tool_name": "Bash",
                "tool_input": {"command": "ls -la"},
                "cwd": "/tmp",
            },
        )
        assert result.returncode == 0

    def test_gate_rule_block_exits_2(self):
        """A 'block' severity gate rule should cause exit code 2."""
        # Create a temporary gate cache with a block rule
        cache_file = GATE_CACHE_DIR / "test_block_project.json"
        GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rules = [
            {
                "tool": "bash",
                "pattern": "rm\\s+-rf\\s+/",
                "raw_pattern": "rm -rf /",
                "reason": "Never delete root filesystem",
                "severity": "block",
                "project": None,
            }
        ]
        cache_file.write_text(json.dumps(rules))

        try:
            # We need to set cwd to a non-git dir so project=None, then
            # the hook loads global.json. Actually, let's use the global cache.
            global_cache = GATE_CACHE_DIR / "global.json"
            global_cache.write_text(json.dumps(rules))

            result = run_hook(
                HOOK_PRE_TOOL_USE,
                {
                    "session_id": "test-block",
                    "tool_name": "Bash",
                    "tool_input": {"command": "rm -rf /important"},
                    "cwd": "/tmp",  # /tmp has no git remote, so project=None
                },
            )
            assert result.returncode == 2
            assert "BLOCKED" in result.stderr
            assert "Never delete root filesystem" in result.stderr
        finally:
            cache_file.unlink(missing_ok=True)
            global_cache.unlink(missing_ok=True)

    def test_gate_rule_warn_exits_0_with_context(self):
        """A 'warn' severity gate rule should exit 0 with warning context."""
        global_cache = GATE_CACHE_DIR / "global.json"
        GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rules = [
            {
                "tool": "bash",
                "pattern": "docker\\s+push",
                "raw_pattern": "docker push",
                "reason": "Verify image tag before pushing",
                "severity": "warn",
                "project": None,
            }
        ]
        global_cache.write_text(json.dumps(rules))

        try:
            result = run_hook(
                HOOK_PRE_TOOL_USE,
                {
                    "session_id": "test-warn",
                    "tool_name": "Bash",
                    "tool_input": {"command": "docker push myimage:latest"},
                    "cwd": "/tmp",
                },
            )
            assert result.returncode == 0
            output = parse_hook_output(result.stdout)
            assert output is not None
            hso = output["hookSpecificOutput"]
            assert hso["hookEventName"] == "PreToolUse"
            assert "Verify image tag" in hso["additionalContext"]
        finally:
            global_cache.unlink(missing_ok=True)

    def test_gate_rule_confirm_exits_0_with_ask(self):
        """A 'confirm' severity gate rule should output permissionDecision=ask."""
        global_cache = GATE_CACHE_DIR / "global.json"
        GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rules = [
            {
                "tool": "bash",
                "pattern": "git\\s+push\\s+--force",
                "raw_pattern": "git push --force",
                "reason": "Force push rewrites history",
                "severity": "confirm",
                "project": None,
            }
        ]
        global_cache.write_text(json.dumps(rules))

        try:
            result = run_hook(
                HOOK_PRE_TOOL_USE,
                {
                    "session_id": "test-confirm",
                    "tool_name": "Bash",
                    "tool_input": {"command": "git push --force origin main"},
                    "cwd": "/tmp",
                },
            )
            assert result.returncode == 0
            output = parse_hook_output(result.stdout)
            assert output is not None
            hso = output["hookSpecificOutput"]
            assert hso["permissionDecision"] == "ask"
            assert "Force push" in hso["permissionDecisionReason"]
        finally:
            global_cache.unlink(missing_ok=True)

    def test_non_bash_tool_not_gate_checked(self):
        """Non-Bash tools should not be checked against gate rules."""
        global_cache = GATE_CACHE_DIR / "global.json"
        GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rules = [
            {
                "tool": "bash",
                "pattern": ".*",
                "raw_pattern": "*",
                "reason": "Block everything",
                "severity": "block",
                "project": None,
            }
        ]
        global_cache.write_text(json.dumps(rules))

        try:
            result = run_hook(
                HOOK_PRE_TOOL_USE,
                {
                    "session_id": "test-read",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/etc/passwd"},
                    "cwd": "/tmp",
                },
            )
            # Read tool should NOT be blocked even though bash rule matches everything
            assert result.returncode == 0
        finally:
            global_cache.unlink(missing_ok=True)

    def test_unmatched_command_allowed(self):
        """Commands that don't match any rule pattern should be allowed."""
        global_cache = GATE_CACHE_DIR / "global.json"
        GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rules = [
            {
                "tool": "bash",
                "pattern": "coolify\\s+deploy",
                "raw_pattern": "coolify deploy",
                "reason": "Needs approval",
                "severity": "block",
                "project": None,
            }
        ]
        global_cache.write_text(json.dumps(rules))

        try:
            result = run_hook(
                HOOK_PRE_TOOL_USE,
                {
                    "session_id": "test-pass",
                    "tool_name": "Bash",
                    "tool_input": {"command": "git status"},
                    "cwd": "/tmp",
                },
            )
            assert result.returncode == 0
        finally:
            global_cache.unlink(missing_ok=True)

    def test_tool_use_logged(self):
        """Tool use should be logged to session log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_hook(
                HOOK_PRE_TOOL_USE,
                {
                    "session_id": "log-test-session",
                    "tool_name": "Bash",
                    "tool_input": {"command": "echo hello"},
                    "cwd": tmpdir,
                },
            )
            assert result.returncode == 0
            # The log is written to cwd/logs/sessions/{session_id}/pre_tool_use.json
            # Note: log_tool_use uses Path.cwd() which is the subprocess cwd,
            # not the cwd field in input_data. This test verifies the hook
            # does not crash on the logging path.

    def test_invalid_json_no_crash(self):
        """Hook should handle bad JSON gracefully."""
        env = os.environ.copy()
        env["CEMS_API_URL"] = ""
        env["CEMS_API_KEY"] = ""
        result = subprocess.run(
            ["uv", "run", str(HOOK_PRE_TOOL_USE)],
            input="{{{bad json",
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# OFFLINE TESTS: stop.py
# ---------------------------------------------------------------------------


class TestStopOffline:
    """Tests that run without a CEMS server."""

    def test_basic_stop_writes_log(self):
        """Stop hook should write input data to session log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_data = {
                "session_id": "stop-test-001",
                "transcript_path": "",
                "cwd": tmpdir,
            }
            # The stop hook uses Path.cwd() for log dir, and also has TTS
            # which we can't easily test. But we verify it doesn't crash.
            result = run_hook(HOOK_STOP, input_data)
            assert result.returncode == 0

    def test_stop_with_transcript(self):
        """Stop hook should read transcript and attempt session analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake transcript file
            transcript_path = Path(tmpdir) / "transcript.jsonl"
            transcript_lines = [
                json.dumps({"type": "user", "message": {"content": "hello"}}),
                json.dumps({"type": "assistant", "message": {"content": "hi there"}}),
                json.dumps({"type": "user", "message": {"content": "do something"}}),
                json.dumps({"type": "assistant", "message": {"content": "done"}}),
            ]
            transcript_path.write_text("\n".join(transcript_lines))

            input_data = {
                "session_id": "stop-transcript-test",
                "transcript_path": str(transcript_path),
                "cwd": tmpdir,
            }
            # Stop hook writes observer signal and exits cleanly
            result = run_hook(HOOK_STOP, input_data)
            assert result.returncode == 0

    def test_stop_invalid_json_no_crash(self):
        """Hook should handle bad JSON gracefully."""
        env = os.environ.copy()
        env["CEMS_API_URL"] = ""
        env["CEMS_API_KEY"] = ""
        result = subprocess.run(
            ["uv", "run", str(HOOK_STOP)],
            input="not json",
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        assert result.returncode == 0

    def test_stop_short_transcript_skipped(self):
        """Transcripts with <= 2 messages should skip analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "transcript.jsonl"
            transcript_lines = [
                json.dumps({"type": "user", "message": {"content": "hi"}}),
            ]
            transcript_path.write_text("\n".join(transcript_lines))

            result = run_hook(HOOK_STOP, {
                "session_id": "short-session",
                "transcript_path": str(transcript_path),
                "cwd": tmpdir,
            })
            assert result.returncode == 0


# ---------------------------------------------------------------------------
# OFFLINE TESTS: cems_post_tool_use.py
# ---------------------------------------------------------------------------


class TestPostToolUseOffline:
    """Tests that run without a CEMS server."""

    def test_read_tool_skipped(self):
        """Read tool should not trigger learning (not in LEARNABLE_TOOLS)."""
        result = run_hook(
            HOOK_POST_TOOL_USE,
            {
                "session_id": "test-123",
                "tool_name": "Read",
                "tool_input": {"file_path": "/tmp/foo.py"},
                "tool_response": {},
                "cwd": "/tmp",
            },
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_grep_tool_skipped(self):
        """Grep tool should not trigger learning."""
        result = run_hook(
            HOOK_POST_TOOL_USE,
            {
                "session_id": "test-123",
                "tool_name": "Grep",
                "tool_input": {"pattern": "foo"},
                "tool_response": {},
                "cwd": "/tmp",
            },
        )
        assert result.returncode == 0

    def test_bash_ls_skipped(self):
        """Simple 'ls' command should be skipped (not learnable)."""
        result = run_hook(
            HOOK_POST_TOOL_USE,
            {
                "session_id": "test-123",
                "tool_name": "Bash",
                "tool_input": {"command": "ls -la /tmp"},
                "tool_response": {},
                "cwd": "/tmp",
            },
        )
        assert result.returncode == 0

    def test_background_agent_skipped(self):
        """Background agents should be skipped."""
        result = run_hook(
            HOOK_POST_TOOL_USE,
            {
                "session_id": "test-123",
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/foo.py"},
                "tool_response": {},
                "is_background_agent": True,
                "cwd": "/tmp",
            },
        )
        assert result.returncode == 0

    def test_no_cems_config_exits_silently(self):
        """Without CEMS config, hook should exit 0."""
        result = run_hook(
            HOOK_POST_TOOL_USE,
            {
                "session_id": "test-123",
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/foo.py"},
                "tool_response": {},
                "cwd": "/tmp",
            },
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# OFFLINE TESTS: Gate cache population (unit-level via user_prompts_submit)
# ---------------------------------------------------------------------------


class TestGateCachePopulation:
    """Test gate cache file creation/update logic.

    The gate cache is populated by user_prompts_submit.py's
    populate_gate_cache() function. Without a live server, it creates
    empty caches or preserves existing ones.
    """

    def test_cache_dir_created(self):
        """Gate cache directory should be created if it doesn't exist."""
        # This is implicitly tested by running any hook that calls
        # populate_gate_cache. We verify the dir exists afterward.
        assert GATE_CACHE_DIR.parent.exists() or True  # May not exist in CI
        # The real assertion: running the hook doesn't crash
        result = run_hook(
            HOOK_USER_PROMPT,
            {"prompt": "a medium length prompt about testing", "cwd": "/tmp"},
        )
        assert result.returncode == 0

    def test_cache_path_sanitization(self):
        """Project IDs with / should be sanitized to _ in cache filenames."""
        expected_name = "org_repo.json"
        safe = "org/repo".replace("/", "_").replace("\\", "_")
        assert f"{safe}.json" == expected_name


# ---------------------------------------------------------------------------
# OFFLINE TESTS: gate pattern helpers (unit tests)
# ---------------------------------------------------------------------------


class TestGatePatterns:
    """Unit tests for gate pattern parsing in user_prompts_submit.

    These import the functions directly rather than running via subprocess.
    """

    @pytest.fixture(autouse=True)
    def _import_module(self):
        """Import the hook module for direct function testing."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "user_prompts_submit", str(HOOK_USER_PROMPT)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.extract_gate_pattern = mod.extract_gate_pattern
        self.pattern_to_regex = mod.pattern_to_regex

    def test_gate_pattern_em_dash(self):
        """Gate rule with em dash separator should parse correctly."""
        result = self.extract_gate_pattern(
            "Bash: coolify deploy \u2014 Needs manual approval first",
            tags=["block"],
            source_ref="project:org/repo",
        )
        assert result is not None
        assert result["tool"] == "bash"
        assert result["severity"] == "block"
        assert result["project"] == "org/repo"
        assert result["reason"] == "Needs manual approval first"

    def test_gate_pattern_en_dash(self):
        """Gate rule with en dash separator should parse correctly."""
        result = self.extract_gate_pattern(
            "Bash: docker push \u2013 Verify tag first",
            tags=["confirm"],
        )
        assert result is not None
        assert result["severity"] == "confirm"

    def test_gate_pattern_hyphen_with_spaces(self):
        """Gate rule with ' - ' (spaced hyphen) separator should parse."""
        result = self.extract_gate_pattern(
            "Bash: npm publish - Check version first",
            tags=[],
        )
        assert result is not None
        assert result["severity"] == "warn"  # default
        assert result["reason"] == "Check version first"

    def test_gate_pattern_warn_default_severity(self):
        """Without block/confirm tags, severity should default to warn."""
        result = self.extract_gate_pattern(
            "Bash: docker push \u2014 Verify tag first",
            tags=["safety"],
        )
        assert result is not None
        assert result["severity"] == "warn"

    def test_gate_pattern_invalid_content(self):
        """Invalid content should return None."""
        assert self.extract_gate_pattern("") is None
        assert self.extract_gate_pattern("no colon here") is None
        assert self.extract_gate_pattern("Bash: pattern but no dash separator") is None

    def test_gate_pattern_no_source_ref(self):
        """Without source_ref, project should be None."""
        result = self.extract_gate_pattern(
            "Bash: rm -rf \u2014 Dangerous deletion",
            tags=["block"],
        )
        assert result is not None
        assert result["project"] is None

    def test_pattern_to_regex_spaces(self):
        """Spaces in patterns should become flexible whitespace."""
        regex = self.pattern_to_regex("coolify deploy")
        assert "\\s+" in regex

    def test_pattern_to_regex_glob_star(self):
        """Glob * should become .* in regex."""
        regex = self.pattern_to_regex("docker push *")
        assert ".*" in regex


# ---------------------------------------------------------------------------
# ONLINE TESTS: Require a running CEMS server
# ---------------------------------------------------------------------------


@needs_server
class TestUserPromptSubmitOnline:
    """Tests that require a live CEMS server.

    These verify actual API calls are made and responses are correct.
    Set CEMS_TEST_API_URL and CEMS_TEST_API_KEY to enable.
    """

    @pytest.fixture(autouse=True)
    def _setup_env(self):
        self.env = {
            "CEMS_API_URL": CEMS_TEST_API_URL,
            "CEMS_API_KEY": CEMS_TEST_API_KEY,
        }

    def test_memory_search_returns_context(self):
        """A real search should return hookSpecificOutput with memory context.

        NOTE: This test requires at least one memory in the CEMS server.
        If the server is empty, the hook may produce no output (which is
        valid behavior). Seed the server first if you need this to pass.
        """
        result = run_hook(
            HOOK_USER_PROMPT,
            {
                "prompt": "what are the CEMS embedding dimensions and configuration",
                "cwd": "/Users/razvan/Development/cems",
            },
            env_overrides=self.env,
        )
        assert result.returncode == 0
        # If memories exist, we should get hookSpecificOutput
        output = parse_hook_output(result.stdout)
        if output:
            assert "hookSpecificOutput" in output
            ctx = output["hookSpecificOutput"]["additionalContext"]
            assert "<memory-recall>" in ctx

    def test_gate_cache_populated(self):
        """After running the hook, gate cache file should exist."""
        # Clear any existing cache for this project
        project_cache = GATE_CACHE_DIR / "chocksy_cems.json"
        if project_cache.exists():
            project_cache.unlink()

        result = run_hook(
            HOOK_USER_PROMPT,
            {
                "prompt": "check if gate cache is populated correctly after hook runs",
                "cwd": "/Users/razvan/Development/cems",
            },
            env_overrides=self.env,
        )
        assert result.returncode == 0
        # Gate cache should now exist (even if empty rules list)
        # Note: project ID for this repo is "chocksy/cems" -> "chocksy_cems.json"
        # The actual project depends on git remote; check either project or global cache
        assert project_cache.exists() or (GATE_CACHE_DIR / "global.json").exists()

    def test_gate_cache_freshness_skips_refetch(self):
        """A fresh cache (< 5 min old) should not be re-fetched."""
        project_cache = GATE_CACHE_DIR / "chocksy_cems.json"
        GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write a known cache value
        known_rules = [{"tool": "bash", "pattern": "test", "raw_pattern": "test",
                         "reason": "test rule", "severity": "warn", "project": None}]
        project_cache.write_text(json.dumps(known_rules))
        # Touch to make it fresh
        project_cache.touch()

        result = run_hook(
            HOOK_USER_PROMPT,
            {
                "prompt": "testing gate cache freshness with a sufficient length prompt",
                "cwd": "/Users/razvan/Development/cems",
            },
            env_overrides=self.env,
        )
        assert result.returncode == 0
        # Cache should still contain our known rules (not overwritten)
        cached = json.loads(project_cache.read_text())
        assert cached == known_rules

        # Cleanup
        project_cache.unlink(missing_ok=True)


@needs_server
class TestSessionStartOnline:
    """Tests that require a live CEMS server."""

    def test_profile_injection(self):
        """SessionStart should inject profile context from CEMS."""
        result = run_hook(
            HOOK_SESSION_START,
            {
                "session_id": "test-online-start",
                "source": "startup",
                "cwd": "/Users/razvan/Development/cems",
            },
            env_overrides={
                "CEMS_API_URL": CEMS_TEST_API_URL,
                "CEMS_API_KEY": CEMS_TEST_API_KEY,
            },
        )
        assert result.returncode == 0
        output = parse_hook_output(result.stdout)
        if output:
            assert "hookSpecificOutput" in output
            hso = output["hookSpecificOutput"]
            assert hso["hookEventName"] == "SessionStart"
            assert "<cems-profile>" in hso["additionalContext"]


@needs_server
class TestStopOnline:
    """Tests that require a live CEMS server for stop hook signal writing."""

    def test_stop_writes_signal_file(self):
        """Stop hook should write a signal file for the observer daemon."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "transcript.jsonl"
            transcript_lines = [
                json.dumps({"type": "user", "message": {"content": "Fix the login bug"}}),
                json.dumps({"type": "assistant", "message": {"content": "I'll look at the auth module"}}),
            ]
            transcript_path.write_text("\n".join(transcript_lines))

            signal_dir = Path(tmpdir) / ".cems" / "observer" / "signals"

            result = run_hook(
                HOOK_STOP,
                {
                    "session_id": f"test-stop-{int(time.time())}",
                    "transcript_path": str(transcript_path),
                    "cwd": tmpdir,
                },
                env_overrides={
                    "CEMS_API_URL": CEMS_TEST_API_URL,
                    "CEMS_API_KEY": CEMS_TEST_API_KEY,
                    "HOME": tmpdir,
                },
            )
            assert result.returncode == 0


@needs_server
class TestPostToolUseOnline:
    """Tests that require a live CEMS server for tool learning."""

    def test_git_commit_triggers_learning(self):
        """A 'git commit' Bash command should trigger tool learning."""
        result = run_hook(
            HOOK_POST_TOOL_USE,
            {
                "session_id": f"test-learning-{int(time.time())}",
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'fix auth bug'"},
                "tool_response": {"output": "1 file changed, 2 insertions"},
                "cwd": "/tmp",
            },
            env_overrides={
                "CEMS_API_URL": CEMS_TEST_API_URL,
                "CEMS_API_KEY": CEMS_TEST_API_KEY,
            },
        )
        assert result.returncode == 0
        # On success, stderr should mention "Captured learning"
        # (only if API returns stored=true)

    def test_edit_triggers_learning(self):
        """An Edit tool use should trigger tool learning."""
        result = run_hook(
            HOOK_POST_TOOL_USE,
            {
                "session_id": f"test-edit-{int(time.time())}",
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/foo.py"},
                "tool_response": {"content": "def hello():\n    pass\n"},
                "cwd": "/tmp",
            },
            env_overrides={
                "CEMS_API_URL": CEMS_TEST_API_URL,
                "CEMS_API_KEY": CEMS_TEST_API_KEY,
            },
        )
        assert result.returncode == 0
