"""Tests for observer transcript extraction."""

import json

from cems.observer.transcript import (
    _summarize_tool_use,
    compact_tool_lines,
    extract_message_lines,
    extract_transcript_from_bytes,
    extract_transcript_text,
)


# --- _summarize_tool_use tests ---

def test_summarize_read():
    assert _summarize_tool_use("Read", {"file_path": "/src/main.py"}) == "Read: /src/main.py"


def test_summarize_edit():
    assert _summarize_tool_use("Edit", {"file_path": "/src/config.py"}) == "Edit: /src/config.py"


def test_summarize_write():
    assert _summarize_tool_use("Write", {"file_path": "/tests/new.py"}) == "Write: /tests/new.py"


def test_summarize_bash():
    result = _summarize_tool_use("Bash", {"command": "docker compose build\necho done"})
    assert result == "Bash: docker compose build"


def test_summarize_bash_long_command_truncated():
    long_cmd = "x" * 200
    result = _summarize_tool_use("Bash", {"command": long_cmd})
    assert len(result) <= 126  # "Bash: " + 120 chars


def test_summarize_grep():
    result = _summarize_tool_use("Grep", {"pattern": "def main", "path": "src/"})
    assert result == "Grep: 'def main' in src/"


def test_summarize_glob():
    assert _summarize_tool_use("Glob", {"pattern": "**/*.py"}) == "Glob: **/*.py"


def test_summarize_unknown_tool_returns_none():
    assert _summarize_tool_use("Task", {"prompt": "do something"}) is None


def test_summarize_empty_input_returns_none():
    assert _summarize_tool_use("Read", {}) is None
    assert _summarize_tool_use("Bash", {"command": ""}) is None


# --- extract_message_lines tests ---

def test_extract_user_raw_string():
    entry = {
        "type": "user",
        "message": {"role": "user", "content": "please fix the bug"},
    }
    lines = extract_message_lines(entry)
    assert lines == ["[USER]: please fix the bug"]


def test_extract_user_text_block():
    entry = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "deploy to production please"}],
        },
    }
    lines = extract_message_lines(entry)
    assert lines == ["[USER]: deploy to production please"]


def test_extract_user_tool_result_skipped():
    """tool_result blocks should be skipped — too verbose."""
    entry = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "abc", "content": "lots of output..."}
            ],
        },
    }
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_user_meta_skipped():
    """isMeta messages should be skipped."""
    entry = {
        "type": "user",
        "isMeta": True,
        "message": {"role": "user", "content": "system injected message"},
    }
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_user_command_skipped():
    """Messages starting with / should be skipped."""
    entry = {
        "type": "user",
        "message": {"role": "user", "content": "/compact"},
    }
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_user_xml_skipped():
    """Messages starting with < should be skipped."""
    entry = {
        "type": "user",
        "message": {"role": "user", "content": "<local-command-caveat>some system text</local-command-caveat>"},
    }
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_user_skill_injection_skipped():
    """Skill injections should be skipped."""
    entry = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "Base directory for this skill: /Users/x/.claude/skills/foo\n\n# Some Skill"},
            ],
        },
    }
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_assistant_text():
    entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me check the configuration file for that setting."}],
        },
    }
    lines = extract_message_lines(entry)
    assert lines == ["[ASSISTANT]: Let me check the configuration file for that setting."]


def test_extract_assistant_tool_use():
    entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/src/config.py"}},
            ],
        },
    }
    lines = extract_message_lines(entry)
    assert lines == ["[TOOL] Read: /src/config.py"]


def test_extract_assistant_mixed_text_and_tools():
    entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll update the configuration now."},
                {"type": "tool_use", "id": "t1", "name": "Edit", "input": {"file_path": "/src/config.py", "old_string": "a", "new_string": "b"}},
            ],
        },
    }
    lines = extract_message_lines(entry)
    assert len(lines) == 2
    assert "[ASSISTANT]:" in lines[0]
    assert "[TOOL] Edit: /src/config.py" == lines[1]


def test_extract_assistant_thinking_skipped():
    entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Let me think about this..."}],
        },
    }
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_progress_skipped():
    entry = {"type": "progress", "data": {"type": "bash_progress"}}
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_system_skipped():
    entry = {"type": "system", "subtype": "turn_duration", "durationMs": 5000}
    lines = extract_message_lines(entry)
    assert lines == []


def test_extract_short_text_skipped():
    """Very short assistant text should be skipped."""
    entry = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "OK"}]},
    }
    lines = extract_message_lines(entry)
    assert lines == []


# --- extract_transcript_text tests ---

def test_extract_transcript_full_session():
    entries = [
        {"type": "user", "message": {"content": "deploy the app to production"}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "I'll deploy the application now."},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "docker compose up -d"}},
        ]}},
        {"type": "progress", "data": {"type": "bash_progress"}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Container started"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "The deployment is complete. All containers are running."},
        ]}},
    ]
    text = extract_transcript_text(entries)
    assert "[USER]: deploy the app to production" in text
    assert "[ASSISTANT]: I'll deploy the application now." in text
    assert "[TOOL] Bash: docker compose up -d" in text
    assert "[ASSISTANT]: The deployment is complete." in text
    # tool_result and progress should NOT be in output
    assert "Container started" not in text
    assert "bash_progress" not in text


def test_extract_transcript_max_chars():
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "x" * 500}]}}
        for _ in range(100)
    ]
    text = extract_transcript_text(entries, max_chars=1000)
    assert len(text) <= 1100  # Some slack for line separators


# --- extract_transcript_from_bytes tests ---

def test_extract_from_bytes():
    entries = [
        {"type": "user", "message": {"content": "hello world, this is a test message"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "I'll help you with that task."}]}},
    ]
    raw = "\n".join(json.dumps(e) for e in entries).encode()
    result = extract_transcript_from_bytes(raw)
    assert result is not None
    assert "[USER]: hello world" in result
    assert "[ASSISTANT]: I'll help you" in result


def test_extract_from_bytes_empty():
    assert extract_transcript_from_bytes(b"") is None


def test_extract_from_bytes_only_progress():
    entries = [
        {"type": "progress", "data": {"type": "bash_progress"}},
        {"type": "progress", "data": {"type": "hook_progress"}},
    ]
    raw = "\n".join(json.dumps(e) for e in entries).encode()
    assert extract_transcript_from_bytes(raw) is None


# --- _summarize_tool_use: WebFetch/WebSearch tests ---

def test_summarize_webfetch():
    result = _summarize_tool_use("WebFetch", {"url": "https://mastra.ai/docs/memory"})
    assert result == "WebFetch: mastra.ai"


def test_summarize_webfetch_no_scheme():
    result = _summarize_tool_use("WebFetch", {"url": "mastra.ai/docs"})
    assert result == "WebFetch: mastra.ai/docs"  # Falls back to truncated URL


def test_summarize_websearch():
    result = _summarize_tool_use("WebSearch", {"query": "Claude Code JSONL format"})
    assert result == "WebSearch: 'Claude Code JSONL format'"


def test_summarize_webfetch_empty():
    assert _summarize_tool_use("WebFetch", {"url": ""}) is None
    assert _summarize_tool_use("WebFetch", {}) is None


def test_summarize_websearch_empty():
    assert _summarize_tool_use("WebSearch", {"query": ""}) is None


# --- compact_tool_lines tests ---

def test_compact_single_read_unchanged():
    lines = ["[TOOL] Read: /src/main.py"]
    result = compact_tool_lines(lines)
    assert result == ["[ACTIVITY] Read /src/main.py"]


def test_compact_multiple_reads_same_dir():
    lines = [
        "[TOOL] Read: /src/cems/server.py",
        "[TOOL] Read: /src/cems/config.py",
        "[TOOL] Read: /src/cems/app.py",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 1
    assert "[ACTIVITY]" in result[0]
    assert "Read 3 files" in result[0]
    assert "/src/cems" in result[0]


def test_compact_reads_different_dirs():
    lines = [
        "[TOOL] Read: /src/server.py",
        "[TOOL] Read: /tests/test_server.py",
        "[TOOL] Read: /hooks/stop.py",
        "[TOOL] Read: /config/settings.py",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 1
    assert "Read 4 files" in result[0]


def test_compact_preserves_user_assistant_lines():
    lines = [
        "[USER]: deploy the app",
        "[TOOL] Read: /src/server.py",
        "[TOOL] Read: /src/config.py",
        "[ASSISTANT]: I'll deploy now.",
        "[TOOL] Bash: docker compose up -d",
    ]
    result = compact_tool_lines(lines)
    assert result[0] == "[USER]: deploy the app"
    assert "[ACTIVITY]" in result[1]
    assert "Read 2 files" in result[1]
    assert result[2] == "[ASSISTANT]: I'll deploy now."
    assert "[ACTIVITY]" in result[3]
    assert "docker compose up" in result[3]


def test_compact_groups_by_boundaries():
    """Tool lines separated by non-tool lines form separate groups."""
    lines = [
        "[TOOL] Read: /src/a.py",
        "[TOOL] Read: /src/b.py",
        "[ASSISTANT]: Checking the code now.",
        "[TOOL] Edit: /src/a.py",
        "[TOOL] Edit: /src/b.py",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 3
    assert "Read 2" in result[0]
    assert "[ASSISTANT]" in result[1]
    assert "Edited 2" in result[2]


def test_compact_bash_single():
    lines = ["[TOOL] Bash: docker compose build"]
    result = compact_tool_lines(lines)
    assert result == ["[ACTIVITY] Ran: docker compose build"]


def test_compact_bash_multiple_same_prefix():
    lines = [
        "[TOOL] Bash: docker compose build",
        "[TOOL] Bash: docker compose up -d",
        "[TOOL] Bash: docker compose logs",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 1
    assert "3 docker commands" in result[0]


def test_compact_bash_multiple_different():
    lines = [
        "[TOOL] Bash: git status",
        "[TOOL] Bash: npm test",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 1
    assert "Ran commands:" in result[0]


def test_compact_searches():
    lines = [
        "[TOOL] Grep: 'def main' in src/",
        "[TOOL] Grep: 'class Server' in src/",
        "[TOOL] Glob: **/*.py",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 1
    assert "3 queries" in result[0]


def test_compact_web_preserved():
    lines = [
        "[TOOL] WebFetch: mastra.ai",
        "[TOOL] WebSearch: 'Claude Code sessions'",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 2
    assert "[ACTIVITY] Visited mastra.ai" == result[0]
    assert "[ACTIVITY] Web search: 'Claude Code sessions'" == result[1]


def test_compact_mixed_tool_types():
    """A single group with reads, edits, bash, and web should produce multiple activities."""
    lines = [
        "[TOOL] Read: /src/server.py",
        "[TOOL] Read: /src/config.py",
        "[TOOL] Edit: /src/server.py",
        "[TOOL] Bash: docker compose build",
        "[TOOL] WebFetch: docs.python.org",
    ]
    result = compact_tool_lines(lines)
    assert any("Read 2" in r for r in result)
    assert any("Edited" in r for r in result)
    assert any("docker compose" in r for r in result)
    assert any("Visited docs.python.org" in r for r in result)


def test_compact_writes():
    lines = [
        "[TOOL] Write: /tests/test_new.py",
        "[TOOL] Write: /tests/test_other.py",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 1
    assert "Created 2 files" in result[0]


def test_compact_notebook_edit_grouped_with_edits():
    lines = [
        "[TOOL] NotebookEdit: /notebooks/analysis.ipynb",
        "[TOOL] Edit: /src/utils.py",
    ]
    result = compact_tool_lines(lines)
    assert len(result) == 1
    assert "Edited 2" in result[0]


def test_compact_empty_input():
    assert compact_tool_lines([]) == []


def test_compact_no_tool_lines():
    lines = ["[USER]: hello there friend", "[ASSISTANT]: How can I help you today?"]
    assert compact_tool_lines(lines) == lines


# --- extract_transcript_text with compact=True ---

def test_extract_transcript_compact_mode():
    entries = [
        {"type": "user", "message": {"content": "check the server configuration"}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "I'll read the server files now."},
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/src/server.py"}},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/src/config.py"}},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "The server is configured correctly for production."},
        ]}},
    ]
    text = extract_transcript_text(entries, compact=True)
    assert "[USER]:" in text
    assert "[ACTIVITY]" in text
    assert "[TOOL]" not in text  # No raw tool lines in compact mode
    assert "Read 2 files" in text


def test_extract_transcript_non_compact_preserves_tools():
    entries = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/src/server.py"}},
        ]}},
    ]
    text = extract_transcript_text(entries, compact=False)
    assert "[TOOL] Read: /src/server.py" in text
    assert "[ACTIVITY]" not in text


# --- extract_transcript_from_bytes with compact=True ---

def test_extract_from_bytes_compact():
    entries = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/src/a.py"}},
            {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/src/b.py"}},
        ]}},
    ]
    raw = "\n".join(json.dumps(e) for e in entries).encode()
    result = extract_transcript_from_bytes(raw, compact=True)
    assert result is not None
    assert "[ACTIVITY]" in result
    assert "Read 2" in result
