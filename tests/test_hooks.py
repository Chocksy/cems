"""
Tests for Claude Code CEMS hooks.

These tests invoke hooks as subprocesses (exactly as Claude Code does),
using a recording HTTP server to capture and verify all API interactions.
No Claude Code session needed -- pure subprocess + HTTP assertions.

Run: .venv/bin/python3 -m pytest tests/test_hooks.py -x -v
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tests.hook_harness import (
    CannedResponse,
    HookResult,
    RecordingServer,
    run_hook,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cems_server():
    """Spin up a recording server for the duration of a test."""
    with RecordingServer() as server:
        yield server


@pytest.fixture
def cwd_path(tmp_path):
    """A temporary directory to use as cwd for hooks."""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session_start_input(
    session_id: str = "test-session-001",
    cwd: str = "/tmp/test-project",
    source: str = "startup",
) -> dict:
    return {
        "session_id": session_id,
        "cwd": cwd,
        "source": source,
        "is_background_agent": False,
    }


def make_user_prompt_input(
    prompt: str = "How do I configure Docker for this project?",
    cwd: str = "/tmp/test-project",
    session_id: str = "test-session-001",
) -> dict:
    return {
        "session_id": session_id,
        "prompt": prompt,
        "cwd": cwd,
    }


def make_pre_tool_use_input(
    tool_name: str = "Bash",
    command: str = "ls -la",
    cwd: str = "/tmp/test-project",
    session_id: str = "test-session-001",
) -> dict:
    return {
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_input": {"command": command},
        "cwd": cwd,
    }


def make_stop_input(
    session_id: str = "test-session-001",
    cwd: str = "/tmp/test-project",
    transcript_path: str = "",
) -> dict:
    return {
        "session_id": session_id,
        "cwd": cwd,
        "transcript_path": transcript_path,
    }


def make_post_tool_use_input(
    tool_name: str = "Bash",
    tool_input: dict | None = None,
    tool_response: dict | None = None,
    session_id: str = "test-session-001",
    cwd: str = "/tmp/test-project",
    transcript_path: str = "",
) -> dict:
    return {
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_input": tool_input or {"command": "git commit -m 'test'"},
        "tool_response": tool_response or {"output": "committed successfully"},
        "cwd": cwd,
        "transcript_path": transcript_path,
        "is_background_agent": False,
    }


# ============================================================================
# SessionStart hook tests
# ============================================================================


class TestSessionStart:
    """Tests for cems_session_start.py"""

    def test_injects_profile_context(self, cems_server: RecordingServer):
        """When CEMS returns a profile, the hook should output it as additionalContext."""
        cems_server.set_response("/api/memory/profile", CannedResponse(
            body={
                "success": True,
                "context": "User prefers Python. Uses Docker for deployment.",
            }
        ))

        result = run_hook("cems_session_start.py", make_session_start_input(), server=cems_server)

        assert result.exit_code == 0
        assert result.additional_context is not None
        assert "cems-profile" in result.additional_context
        assert "User prefers Python" in result.additional_context

        # Verify it called the profile endpoint
        profile_reqs = cems_server.get_requests("/api/memory/profile")
        assert len(profile_reqs) == 1
        assert "Bearer test-key-12345" in profile_reqs[0].headers.get("authorization", "")

    def test_silent_when_no_profile(self, cems_server: RecordingServer):
        """When CEMS returns no profile, the hook should exit silently."""
        cems_server.set_response("/api/memory/profile", CannedResponse(
            body={"success": True, "context": ""}
        ))

        result = run_hook("cems_session_start.py", make_session_start_input(), server=cems_server)

        assert result.exit_code == 0
        assert result.stdout == "" or result.additional_context is None

    def test_skips_background_agents(self, cems_server: RecordingServer):
        """Background agents should not trigger profile injection."""
        input_data = make_session_start_input()
        input_data["is_background_agent"] = True

        result = run_hook("cems_session_start.py", input_data, server=cems_server)

        assert result.exit_code == 0
        assert len(cems_server.requests) == 0  # No API calls made

    def test_skips_resume_source(self, cems_server: RecordingServer):
        """Resume sessions should not re-inject profile."""
        input_data = make_session_start_input(source="resume")

        result = run_hook("cems_session_start.py", input_data, server=cems_server)

        assert result.exit_code == 0
        assert len(cems_server.requests) == 0

    def test_graceful_without_api_config(self):
        """With no CEMS_API_URL, hook exits cleanly without errors."""
        result = run_hook(
            "cems_session_start.py",
            make_session_start_input(),
            server=None,  # No server = empty CEMS_API_URL
        )

        assert result.exit_code == 0
        assert result.stdout == ""

    def test_graceful_on_api_error(self, cems_server: RecordingServer):
        """When CEMS returns 500, hook exits cleanly."""
        cems_server.set_response("/api/memory/profile", CannedResponse(
            status_code=500,
            body={"error": "internal error"}
        ))

        result = run_hook("cems_session_start.py", make_session_start_input(), server=cems_server)

        assert result.exit_code == 0


# ============================================================================
# UserPromptSubmit hook tests
# ============================================================================


class TestUserPromptSubmit:
    """Tests for user_prompts_submit.py

    All tests pass HOME=tmp_path to isolate the gate cache from the real
    user HOME directory. The hook writes gate cache to ~/.cems/cache/gate_rules/
    and has a 5-minute TTL -- using real HOME would cause tests to read stale
    caches or skip API calls unexpectedly.
    """

    def test_searches_memories_and_injects_context(self, cems_server: RecordingServer, tmp_path):
        """When memories are found, they should be injected as context."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [
                    {
                        "memory_id": "abc12345-6789-0000-0000-000000000000",
                        "content": "Docker config uses port 8765",
                        "category": "technical",
                    },
                    {
                        "memory_id": "def12345-6789-0000-0000-000000000000",
                        "content": "Use docker compose build before deploy",
                        "category": "workflow",
                    },
                ],
            }
        ))
        cems_server.set_response("/api/memory/log-shown", CannedResponse(body={"success": True}))
        cems_server.set_response("/api/memory/gate-rules", CannedResponse(
            body={"success": True, "rules": []}
        ))

        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="How do I configure Docker for this project?"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0
        ctx = result.additional_context
        assert ctx is not None
        assert "memory-recall" in ctx
        assert "Docker config uses port 8765" in ctx
        assert "docker compose build" in ctx

        # Verify search was called with intent extraction
        # (2 requests: memory search + observation fetch)
        search_reqs = cems_server.get_requests("/api/memory/search")
        assert len(search_reqs) == 2
        assert "query" in search_reqs[0].body

        # Verify log-shown was called with the memory IDs
        log_reqs = cems_server.get_requests("/api/memory/log-shown")
        assert len(log_reqs) == 1
        logged_ids = log_reqs[0].body.get("memory_ids", [])
        assert "abc12345-6789-0000-0000-000000000000" in logged_ids
        assert "def12345-6789-0000-0000-000000000000" in logged_ids

    def test_no_memories_found(self, cems_server: RecordingServer, tmp_path):
        """When no memories match, no context should be injected."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={"success": True, "results": []}
        ))
        cems_server.set_response("/api/memory/gate-rules", CannedResponse(
            body={"success": True, "rules": []}
        ))

        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="Tell me about quantum physics in detail"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0
        # No memory-recall block
        if result.additional_context:
            assert "memory-recall" not in result.additional_context

    def test_ultrathink_flag(self, cems_server: RecordingServer, tmp_path):
        """Prompt ending with -u should inject ultrathink instruction."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={"success": True, "results": []}
        ))
        cems_server.set_response("/api/memory/gate-rules", CannedResponse(
            body={"success": True, "rules": []}
        ))

        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="Explain the architecture of this project -u"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0
        ctx = result.additional_context
        assert ctx is not None
        assert "ultrathink" in ctx.lower()

    def test_short_prompts_skip_search(self, cems_server: RecordingServer, tmp_path):
        """Very short prompts (<15 chars) should skip memory search."""
        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="hi there"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0
        # No search request should have been made
        search_reqs = cems_server.get_requests("/api/memory/search")
        assert len(search_reqs) == 0

    def test_slash_commands_skip(self, cems_server: RecordingServer, tmp_path):
        """Slash commands should be skipped entirely."""
        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="/recall something from last week about docker"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0
        assert len(cems_server.requests) == 0

    def test_populates_gate_cache(self, cems_server: RecordingServer, tmp_path):
        """Hook should populate gate rule cache for PreToolUse to consume."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={"success": True, "results": []}
        ))
        cems_server.set_response("/api/memory/log-shown", CannedResponse(body={"success": True}))
        cems_server.set_response("/api/memory/gate-rules", CannedResponse(
            body={
                "success": True,
                "rules": [
                    {
                        "content": "Bash: rm -rf * - Never delete with glob patterns",
                        "tags": ["block"],
                        "source_ref": "",
                    }
                ],
            }
        ))

        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="Help me clean up old build artifacts from the project"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0
        # Verify gate-rules endpoint was called
        gate_reqs = cems_server.get_requests("/api/memory/gate-rules")
        assert len(gate_reqs) == 1

        # Verify cache file was actually written
        cache_dir = tmp_path / ".cems" / "cache" / "gate_rules"
        cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        assert len(cache_files) > 0, "Gate cache file should have been created"

    def test_graceful_without_api(self):
        """Without CEMS configured, hook exits cleanly."""
        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(),
            server=None,
        )

        assert result.exit_code == 0

    def test_subagent_skips(self, cems_server: RecordingServer, tmp_path):
        """Subagents (CLAUDE_AGENT_ID set) should be skipped."""
        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(),
            server=cems_server,
            extra_env={
                "CLAUDE_AGENT_ID": "subagent-123",
                "HOME": str(tmp_path),
            },
        )

        assert result.exit_code == 0
        assert len(cems_server.requests) == 0


# ============================================================================
# PreToolUse hook tests
# ============================================================================


class TestPreToolUse:
    """Tests for pre_tool_use.py"""

    def test_allows_normal_commands(self, cems_server: RecordingServer, tmp_path):
        """Normal commands with no gate rules should pass through."""
        result = run_hook(
            "cems_pre_tool_use.py",
            make_pre_tool_use_input(command="git status"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0

    def test_blocks_matching_gate_rule(self, cems_server: RecordingServer, tmp_path):
        """Commands matching a 'block' gate rule should exit 2 with stderr."""
        # First, write a gate cache file that the hook will read
        cache_dir = tmp_path / ".cems" / "cache" / "gate_rules"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "global.json"
        cache_file.write_text(json.dumps([
            {
                "tool": "bash",
                "pattern": "rm\\s+-rf",
                "raw_pattern": "rm -rf",
                "reason": "Dangerous recursive delete",
                "severity": "block",
            }
        ]))

        result = run_hook(
            "cems_pre_tool_use.py",
            make_pre_tool_use_input(command="rm -rf /important-data"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 2
        assert "BLOCKED" in result.stderr
        assert "Dangerous recursive delete" in result.stderr

    def test_warns_on_matching_warn_rule(self, cems_server: RecordingServer, tmp_path):
        """Commands matching a 'warn' gate rule should warn via additionalContext."""
        cache_dir = tmp_path / ".cems" / "cache" / "gate_rules"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "global.json"
        cache_file.write_text(json.dumps([
            {
                "tool": "bash",
                "pattern": "docker\\s+push",
                "raw_pattern": "docker push",
                "reason": "Confirm before pushing Docker images",
                "severity": "warn",
            }
        ]))

        result = run_hook(
            "cems_pre_tool_use.py",
            make_pre_tool_use_input(command="docker push myapp:latest"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0
        ctx = result.additional_context
        assert ctx is not None
        assert "WARNING" in ctx
        assert "Confirm before pushing Docker images" in ctx

    def test_non_bash_tools_skip_gate_check(self, cems_server: RecordingServer, tmp_path):
        """Non-Bash tools should not be checked against gate rules."""
        cache_dir = tmp_path / ".cems" / "cache" / "gate_rules"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "global.json"
        cache_file.write_text(json.dumps([
            {
                "tool": "bash",
                "pattern": ".*",
                "raw_pattern": "*",
                "reason": "Block everything",
                "severity": "block",
            }
        ]))

        result = run_hook(
            "cems_pre_tool_use.py",
            {
                "session_id": "test-session-001",
                "tool_name": "Read",
                "tool_input": {"file_path": "/etc/passwd"},
                "cwd": "/tmp/test-project",
            },
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        # Read tool should not be blocked by Bash gate rules
        assert result.exit_code == 0

    def test_no_cache_allows_all(self, cems_server: RecordingServer, tmp_path):
        """With no gate cache file, all commands should be allowed."""
        result = run_hook(
            "cems_pre_tool_use.py",
            make_pre_tool_use_input(command="rm -rf /"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        # No cache = no rules = allow
        assert result.exit_code == 0


# ============================================================================
# Stop hook tests
# ============================================================================


class TestStop:
    """Tests for stop.py"""

    def test_sends_transcript_to_cems(self, cems_server: RecordingServer, tmp_path):
        """Stop hook should send transcript to /api/session/analyze."""
        cems_server.set_response("/api/session/analyze", CannedResponse(
            body={"success": True, "memories_created": 2}
        ))

        # Create a mock transcript
        transcript_path = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"content": "Hello"}}),
            json.dumps({"type": "assistant", "message": {"content": "Hi there"}}),
            json.dumps({"type": "user", "message": {"content": "Help me with Docker"}}),
            json.dumps({"type": "assistant", "message": {"content": "Sure, here is how..."}}),
        ]
        transcript_path.write_text("\n".join(lines))

        result = run_hook(
            "cems_stop.py",
            make_stop_input(
                transcript_path=str(transcript_path),
                cwd=str(tmp_path),
            ),
            server=cems_server,
            extra_env={
                # Disable TTS/LLM so the test doesn't hang
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )

        assert result.exit_code == 0

        # Verify transcript was sent to CEMS
        analyze_reqs = cems_server.get_requests("/api/session/analyze")
        assert len(analyze_reqs) == 1
        body = analyze_reqs[0].body
        assert "transcript" in body
        assert len(body["transcript"]) == 4
        assert body["session_id"] == "test-session-001"

    def test_skips_short_transcripts(self, cems_server: RecordingServer, tmp_path):
        """Transcripts with 2 or fewer messages should not be sent."""
        transcript_path = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"content": "Hi"}}),
            json.dumps({"type": "assistant", "message": {"content": "Hello"}}),
        ]
        transcript_path.write_text("\n".join(lines))

        result = run_hook(
            "cems_stop.py",
            make_stop_input(
                transcript_path=str(transcript_path),
                cwd=str(tmp_path),
            ),
            server=cems_server,
            extra_env={
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )

        assert result.exit_code == 0
        # Should NOT have called analyze (only 2 messages)
        analyze_reqs = cems_server.get_requests("/api/session/analyze")
        assert len(analyze_reqs) == 0

    def test_graceful_without_transcript(self, cems_server: RecordingServer, tmp_path):
        """No transcript path should not crash."""
        result = run_hook(
            "cems_stop.py",
            make_stop_input(cwd=str(tmp_path)),
            server=cems_server,
            extra_env={
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )

        assert result.exit_code == 0


# ============================================================================
# PostToolUse hook tests
# ============================================================================


class TestPostToolUse:
    """Tests for cems_post_tool_use.py"""

    def test_sends_learnable_bash_command(self, cems_server: RecordingServer, tmp_path):
        """Git commit should be sent to CEMS for learning."""
        cems_server.set_response("/api/tool/learning", CannedResponse(
            body={"success": True, "stored": True}
        ))

        result = run_hook(
            "cems_post_tool_use.py",
            make_post_tool_use_input(
                tool_name="Bash",
                tool_input={"command": "git commit -m 'feat: add user auth'"},
                cwd=str(tmp_path),
            ),
            server=cems_server,
        )

        assert result.exit_code == 0
        learn_reqs = cems_server.get_requests("/api/tool/learning")
        assert len(learn_reqs) == 1
        body = learn_reqs[0].body
        assert body["tool_name"] == "Bash"
        assert "git commit" in body["tool_input"]["command"]

    def test_skips_read_only_tools(self, cems_server: RecordingServer):
        """Read/Glob/Grep should not trigger learning."""
        for tool in ["Read", "Glob", "Grep"]:
            cems_server.clear()
            result = run_hook(
                "cems_post_tool_use.py",
                {
                    "session_id": "test-001",
                    "tool_name": tool,
                    "tool_input": {"file_path": "/some/file.py"},
                    "tool_response": {},
                    "cwd": "/tmp",
                    "transcript_path": "",
                    "is_background_agent": False,
                },
                server=cems_server,
            )

            assert result.exit_code == 0
            assert len(cems_server.requests) == 0, f"{tool} should not trigger API call"

    def test_skips_non_learnable_bash(self, cems_server: RecordingServer):
        """Simple bash commands (ls, cd, cat) should not trigger learning."""
        result = run_hook(
            "cems_post_tool_use.py",
            make_post_tool_use_input(
                tool_name="Bash",
                tool_input={"command": "ls -la /tmp"},
            ),
            server=cems_server,
        )

        assert result.exit_code == 0
        assert len(cems_server.requests) == 0

    def test_skips_background_agents(self, cems_server: RecordingServer):
        """Background agents should not trigger learning."""
        input_data = make_post_tool_use_input()
        input_data["is_background_agent"] = True

        result = run_hook("cems_post_tool_use.py", input_data, server=cems_server)

        assert result.exit_code == 0
        assert len(cems_server.requests) == 0

    def test_write_tool_triggers_learning(self, cems_server: RecordingServer, tmp_path):
        """Write tool should trigger learning."""
        cems_server.set_response("/api/tool/learning", CannedResponse(
            body={"success": True, "stored": False}
        ))

        result = run_hook(
            "cems_post_tool_use.py",
            make_post_tool_use_input(
                tool_name="Write",
                tool_input={"file_path": "/src/app.py", "content": "print('hello')"},
                cwd=str(tmp_path),
            ),
            server=cems_server,
        )

        assert result.exit_code == 0
        learn_reqs = cems_server.get_requests("/api/tool/learning")
        assert len(learn_reqs) == 1


# ============================================================================
# Cross-hook integration tests
# ============================================================================


class TestHookIntegration:
    """Tests that verify hooks work together correctly."""

    def test_gate_cache_populated_then_consumed(self, cems_server: RecordingServer, tmp_path):
        """UserPromptSubmit populates gate cache, PreToolUse reads it.

        This simulates a real Claude Code flow:
        1. User sends a prompt -> UserPromptSubmit fires, populates gate cache
        2. Claude tries to run a command -> PreToolUse fires, reads gate cache
        """
        # Step 1: Configure canned responses
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={"success": True, "results": []}
        ))
        cems_server.set_response("/api/memory/gate-rules", CannedResponse(
            body={
                "success": True,
                "rules": [
                    {
                        "content": "Bash: coolify deploy - Must confirm before deploying to Coolify",
                        "tags": ["block"],
                        "source_ref": "",
                    }
                ],
            }
        ))

        # Step 2: Run UserPromptSubmit (populates gate cache)
        run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(
                prompt="Help me deploy this application to the production server",
            ),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        # Step 3: Verify cache file was created
        cache_dir = tmp_path / ".cems" / "cache" / "gate_rules"
        cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        assert len(cache_files) > 0, "Gate cache should have been created"

        # Step 4: Run PreToolUse with a command that should be blocked
        cems_server.clear()
        result = run_hook(
            "cems_pre_tool_use.py",
            make_pre_tool_use_input(command="coolify deploy myapp"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 2
        assert "BLOCKED" in result.stderr
        assert "coolify deploy" in result.stderr.lower() or "Must confirm" in result.stderr

    def test_full_session_lifecycle(self, cems_server: RecordingServer, tmp_path):
        """Simulate a complete session: start -> prompt -> tool -> stop.

        Verifies that each hook fires in the right order and makes the
        expected API calls.
        """
        # Configure all endpoints
        cems_server.set_response("/api/memory/profile", CannedResponse(
            body={"success": True, "context": "User prefers verbose explanations."}
        ))
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [{
                    "memory_id": "mem-001",
                    "content": "Project uses FastAPI",
                    "category": "technical",
                }],
            }
        ))
        cems_server.set_response("/api/memory/log-shown", CannedResponse(body={"success": True}))
        cems_server.set_response("/api/memory/gate-rules", CannedResponse(
            body={"success": True, "rules": []}
        ))
        cems_server.set_response("/api/tool/learning", CannedResponse(
            body={"success": True, "stored": True}
        ))
        cems_server.set_response("/api/session/analyze", CannedResponse(
            body={"success": True, "memories_created": 1}
        ))

        # 1. SessionStart
        r1 = run_hook(
            "cems_session_start.py",
            make_session_start_input(cwd=str(tmp_path)),
            server=cems_server,
        )
        assert r1.exit_code == 0
        assert r1.additional_context is not None
        assert "User prefers verbose" in r1.additional_context

        # 2. UserPromptSubmit
        r2 = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(
                prompt="What framework does this project use for the API server?",
                cwd=str(tmp_path),
            ),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )
        assert r2.exit_code == 0
        assert "FastAPI" in (r2.additional_context or "")

        # 3. PreToolUse (no gate rules = allow)
        r3 = run_hook(
            "cems_pre_tool_use.py",
            make_pre_tool_use_input(command="grep -r 'FastAPI' src/", cwd=str(tmp_path)),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )
        assert r3.exit_code == 0

        # 4. PostToolUse (git commit = learnable)
        r4 = run_hook(
            "cems_post_tool_use.py",
            make_post_tool_use_input(
                tool_name="Bash",
                tool_input={"command": "git commit -m 'docs: update API readme'"},
                cwd=str(tmp_path),
            ),
            server=cems_server,
        )
        assert r4.exit_code == 0

        # 5. Stop (with transcript)
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "What framework?"}}),
            json.dumps({"type": "assistant", "message": {"content": "FastAPI"}}),
            json.dumps({"type": "user", "message": {"content": "Show me the routes"}}),
            json.dumps({"type": "assistant", "message": {"content": "Here are the routes..."}}),
        ]))
        r5 = run_hook(
            "cems_stop.py",
            make_stop_input(
                transcript_path=str(transcript_path),
                cwd=str(tmp_path),
            ),
            server=cems_server,
            extra_env={
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )
        assert r5.exit_code == 0

        # Verify the full set of API calls across the session
        all_paths = [r.path for r in cems_server.requests]
        assert "/api/memory/profile" in all_paths
        assert "/api/memory/search" in all_paths
        assert "/api/memory/log-shown" in all_paths
        assert "/api/memory/gate-rules" in all_paths
        assert "/api/tool/learning" in all_paths
        assert "/api/session/analyze" in all_paths


# ============================================================================
# Edge case / robustness tests
# ============================================================================


class TestEdgeCases:
    """Tests for malformed input, missing fields, and other edge cases."""

    def test_empty_stdin_all_hooks(self, cems_server: RecordingServer, tmp_path):
        """All hooks should handle empty/invalid JSON gracefully."""
        for hook in [
            "cems_session_start.py",
            "cems_user_prompts_submit.py",
            "cems_pre_tool_use.py",
            "cems_stop.py",
            "cems_post_tool_use.py",
        ]:
            result = run_hook(
                hook,
                {},  # Empty but valid JSON
                server=cems_server,
                extra_env={
                    "HOME": str(tmp_path),
                    "ELEVENLABS_API_KEY": "",
                    "OPENAI_API_KEY": "",
                    "ANTHROPIC_API_KEY": "",
                },
            )
            assert result.exit_code == 0, f"{hook} crashed on empty input: {result.stderr}"

    def test_hooks_respond_within_timeout(self, cems_server: RecordingServer, tmp_path):
        """All hooks should respond within a reasonable time (5s with server)."""
        cems_server.set_response("/api/memory/profile", CannedResponse(
            body={"success": True, "context": "fast"}
        ))
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={"success": True, "results": []}
        ))
        cems_server.set_response("/api/memory/gate-rules", CannedResponse(
            body={"success": True, "rules": []}
        ))

        hooks_and_inputs = [
            ("cems_session_start.py", make_session_start_input()),
            ("cems_user_prompts_submit.py", make_user_prompt_input()),
            ("cems_pre_tool_use.py", make_pre_tool_use_input()),
        ]

        for hook, input_data in hooks_and_inputs:
            result = run_hook(
                hook,
                input_data,
                server=cems_server,
                timeout=10.0,
                extra_env={"HOME": str(tmp_path)},
            )
            assert result.exit_code != -1, f"{hook} timed out ({result.duration_ms:.0f}ms)"
            # uv run has startup overhead, so be generous but flag anything over 8s
            assert result.duration_ms < 8000, f"{hook} too slow: {result.duration_ms:.0f}ms"
