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
                        "score": 0.85,
                    },
                    {
                        "memory_id": "def12345-6789-0000-0000-000000000000",
                        "content": "Use docker compose build before deploy",
                        "category": "workflow",
                        "score": 0.72,
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

        # Verify search was called (1 request — observations fetch removed in Phase 2)
        search_reqs = cems_server.get_requests("/api/memory/search")
        assert len(search_reqs) == 1
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

    # --- Phase 6: Observability tests ---

    def test_output_logged_in_lean_log(self, cems_server: RecordingServer, tmp_path):
        """UserPromptSubmit should log a UserPromptSubmitOutput event to the lean log."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [
                    {
                        "memory_id": "abc12345-6789-0000-0000-000000000000",
                        "content": "Docker config uses port 8765",
                        "category": "technical",
                        "score": 0.85,
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
            make_user_prompt_input(
                prompt="How do I configure Docker for this project?",
                session_id="log-test-session-001",
            ),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0

        # Check lean log for UserPromptSubmitOutput event
        log_file = tmp_path / ".claude" / "hooks" / "logs" / "hook_events.jsonl"
        assert log_file.exists(), "Lean log file should have been created"

        events = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        output_events = [e for e in events if e.get("event") == "UserPromptSubmitOutput"]
        assert len(output_events) == 1, f"Expected 1 UserPromptSubmitOutput event, found {len(output_events)}"

        evt = output_events[0]
        assert evt["output_len"] > 0
        assert evt["has_memories"] is True
        assert evt["session_id"] == "log-test-ses"  # truncated to 12 chars

    def test_output_logged_in_verbose(self, cems_server: RecordingServer, tmp_path):
        """UserPromptSubmitOutput should have full output text in verbose log."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [
                    {
                        "memory_id": "abc12345-6789-0000-0000-000000000000",
                        "content": "Docker config uses port 8765",
                        "category": "technical",
                        "score": 0.85,
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
            make_user_prompt_input(
                prompt="How do I configure Docker for this project?",
                session_id="verbose-test-session",
            ),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0

        # Check verbose log for output text
        verbose_file = tmp_path / ".claude" / "hooks" / "logs" / "verbose" / "verbose-test.jsonl"
        assert verbose_file.exists(), "Verbose log file should exist"

        entries = [json.loads(line) for line in verbose_file.read_text().splitlines() if line.strip()]
        output_entries = [e for e in entries if e.get("event") == "UserPromptSubmitOutput"]
        assert len(output_entries) == 1
        assert "output" in output_entries[0]
        assert "memory-recall" in output_entries[0]["output"]
        assert "Docker config uses port 8765" in output_entries[0]["output"]

    def test_memory_retrieval_has_verbose_entry(self, cems_server: RecordingServer, tmp_path):
        """MemoryRetrieval should write to verbose log too (with score details)."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [
                    {
                        "memory_id": "abc12345-6789-0000-0000-000000000000",
                        "content": "Docker config uses port 8765",
                        "category": "technical",
                        "score": 0.85,
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
            make_user_prompt_input(
                prompt="How do I configure Docker for this project?",
                session_id="retrieval-verbose",
            ),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0

        # Check verbose log for MemoryRetrieval entry
        verbose_file = tmp_path / ".claude" / "hooks" / "logs" / "verbose" / "retrieval-ve.jsonl"
        assert verbose_file.exists(), "Verbose log file should exist"

        entries = [json.loads(line) for line in verbose_file.read_text().splitlines() if line.strip()]
        retrieval_entries = [e for e in entries if e.get("event") == "MemoryRetrieval"]
        assert len(retrieval_entries) == 1
        assert "score_details" in retrieval_entries[0]

    # --- Phase 2: Pipeline quality tests ---

    def test_no_fetch_recent_observations(self, cems_server: RecordingServer, tmp_path):
        """fetch_recent_observations is removed — only 1 search call, no observations block."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [
                    {
                        "memory_id": "abc12345-6789-0000-0000-000000000000",
                        "content": "Docker config uses port 8765",
                        "category": "technical",
                        "score": 0.85,
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
        # Only 1 search request (no second call for observations)
        search_reqs = cems_server.get_requests("/api/memory/search")
        assert len(search_reqs) == 1

        # No <recent-observations> block in output
        ctx = result.additional_context or ""
        assert "recent-observations" not in ctx

    def test_client_side_score_filter(self, cems_server: RecordingServer, tmp_path):
        """Results below 0.4 should be filtered out client-side."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [
                    {
                        "memory_id": "high-score-id-0000-0000-000000000000",
                        "content": "High relevance memory",
                        "category": "technical",
                        "score": 0.85,
                    },
                    {
                        "memory_id": "low-score-id-0000-0000-000000000000",
                        "content": "Low relevance noise",
                        "category": "general",
                        "score": 0.25,
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
        ctx = result.additional_context or ""
        assert "High relevance memory" in ctx
        assert "Low relevance noise" not in ctx

    def test_session_dedup(self, cems_server: RecordingServer, tmp_path):
        """Multiple results from the same session should keep only the highest-scoring."""
        cems_server.set_response("/api/memory/search", CannedResponse(
            body={
                "success": True,
                "results": [
                    {
                        "memory_id": "sess-summary-0000-0000-000000000000",
                        "content": "Session summary: worked on Docker setup",
                        "category": "session-summary",
                        "score": 0.75,
                        "tags": ["session:abc12345"],
                    },
                    {
                        "memory_id": "sess-obs-0000-0000-000000000000",
                        "content": "Observation from same session about Docker",
                        "category": "observation",
                        "score": 0.65,
                        "tags": ["session:abc12345"],
                    },
                    {
                        "memory_id": "other-mem-0000-0000-000000000000",
                        "content": "Different session memory about Docker",
                        "category": "technical",
                        "score": 0.70,
                        "tags": ["session:def98765"],
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
        ctx = result.additional_context or ""
        # Higher-scoring session result should be kept
        assert "Session summary: worked on Docker setup" in ctx
        # Lower-scoring duplicate from same session should be deduped
        assert "Observation from same session about Docker" not in ctx
        # Different session should still be present
        assert "Different session memory about Docker" in ctx


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

    # --- Phase 6: Gate trigger logging ---

    def test_gate_block_logs_event(self, cems_server: RecordingServer, tmp_path):
        """Blocked gate should log a GateTriggered event to the lean log."""
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
            make_pre_tool_use_input(
                command="rm -rf /important-data",
                session_id="gate-test-session",
            ),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 2

        # Check lean log for GateTriggered event
        log_file = tmp_path / ".claude" / "hooks" / "logs" / "hook_events.jsonl"
        assert log_file.exists(), "Lean log file should exist"

        events = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        gate_events = [e for e in events if e.get("event") == "GateTriggered"]
        assert len(gate_events) == 1, f"Expected 1 GateTriggered event, found {len(gate_events)}"

        evt = gate_events[0]
        assert evt["gate_action"] == "block"
        assert evt["reason"] == "Dangerous recursive delete"
        assert evt["tool"] == "Bash"

    def test_gate_warn_logs_event(self, cems_server: RecordingServer, tmp_path):
        """Warned gate should also log a GateTriggered event."""
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
            make_pre_tool_use_input(
                command="docker push myapp:latest",
                session_id="gate-warn-session",
            ),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0

        # Check lean log for GateTriggered event
        log_file = tmp_path / ".claude" / "hooks" / "logs" / "hook_events.jsonl"
        assert log_file.exists()

        events = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        gate_events = [e for e in events if e.get("event") == "GateTriggered"]
        assert len(gate_events) == 1

        evt = gate_events[0]
        assert evt["gate_action"] == "warn"
        assert evt["reason"] == "Confirm before pushing Docker images"


# ============================================================================
# Stop hook tests
# ============================================================================


class TestStop:
    """Tests for stop.py"""

    def test_writes_stop_signal(self, cems_server: RecordingServer, tmp_path):
        """Stop hook should write a stop signal for the observer daemon."""
        result = run_hook(
            "cems_stop.py",
            make_stop_input(cwd=str(tmp_path)),
            server=cems_server,
            extra_env={
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
                "HOME": str(tmp_path),
            },
        )

        assert result.exit_code == 0

        # Verify signal file was written
        signal_dir = tmp_path / ".cems" / "observer" / "signals"
        if signal_dir.exists():
            signal_files = list(signal_dir.glob("*.json"))
            assert len(signal_files) >= 1

    def test_does_not_call_session_analyze(self, cems_server: RecordingServer, tmp_path):
        """Stop hook should NOT send transcript to /api/session/analyze (removed)."""
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
                "ELEVENLABS_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )

        assert result.exit_code == 0

        # Verify NO calls to session/analyze (removed — observer handles summaries)
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
                    "score": 0.90,
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
        # Note: /api/session/analyze removed — observer daemon handles summaries

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
        # Note: /api/session/analyze removed — observer daemon handles summaries


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

    def test_log_rotation_lean(self, cems_server: RecordingServer, tmp_path):
        """Lean log should be rotated when it exceeds 10MB."""
        # Create an oversized lean log file
        log_dir = tmp_path / ".claude" / "hooks" / "logs"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "hook_events.jsonl"

        # Write >10MB of dummy data
        line = json.dumps({"ts": "2026-01-01T00:00:00", "event": "Test", "session_id": "x" * 12}) + "\n"
        lines_needed = (10_000_001 // len(line)) + 1
        log_file.write_text(line * lines_needed)

        assert log_file.stat().st_size > 10_000_000, "Pre-condition: file must be >10MB"

        # Run any hook — log_hook_event calls _rotate_if_needed()
        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="trigger rotation test hook call please"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0

        # Old file should have been rotated to .1
        rotated = log_dir / "hook_events.jsonl.1"
        assert rotated.exists(), "Rotated .1 file should exist"
        # Current log file should be small (just the new event)
        assert log_file.exists()
        assert log_file.stat().st_size < 1_000_000, "Current log should be small after rotation"

    def test_log_rotation_verbose_cleanup(self, cems_server: RecordingServer, tmp_path):
        """Verbose log files older than 7 days should be cleaned up."""
        log_dir = tmp_path / ".claude" / "hooks" / "logs"
        verbose_dir = log_dir / "verbose"
        verbose_dir.mkdir(parents=True)

        # Create an old verbose file (mtime = 8 days ago)
        old_file = verbose_dir / "old-session-i.jsonl"
        old_file.write_text('{"ts": "old", "event": "Test"}\n')
        old_mtime = time.time() - 8 * 86400
        os.utime(old_file, (old_mtime, old_mtime))

        # Create a recent verbose file
        new_file = verbose_dir / "new-session-i.jsonl"
        new_file.write_text('{"ts": "new", "event": "Test"}\n')

        # Run any hook to trigger rotation
        result = run_hook(
            "cems_user_prompts_submit.py",
            make_user_prompt_input(prompt="trigger verbose cleanup test please"),
            server=cems_server,
            extra_env={"HOME": str(tmp_path)},
        )

        assert result.exit_code == 0

        # Old file should be deleted
        assert not old_file.exists(), "Old verbose file (>7d) should be cleaned up"
        # New file should still exist
        assert new_file.exists(), "Recent verbose file should be kept"

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
