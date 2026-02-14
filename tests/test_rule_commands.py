"""Tests for rule CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cems.commands.rule import rule


def test_rule_add_interactive_constitution() -> None:
    runner = CliRunner()
    mock_client = MagicMock()
    mock_client.add.return_value = {
        "success": True,
        "result": {"results": [{"event": "ADD", "id": "mem-123"}]},
    }

    wizard_input = "\n".join([
        "constitution",  # kind
        "13",  # principle number
        "Transparent Reasoning",  # title
        "Record assumptions and evidence for major decisions.",  # statement
        "guidelines",  # category
        "personal",  # scope
        "foundation:constitution:v2",  # source_ref
        "audit,trace",  # extra tags
        "y",  # pinned?
        "foundational constitution memory",  # pin reason
        "y",  # confirm add
    ]) + "\n"

    with patch("cems.commands.rule.get_client", return_value=mock_client):
        result = runner.invoke(rule, ["add"], input=wizard_input)

    assert result.exit_code == 0, result.output
    mock_client.add.assert_called_once()

    kwargs = mock_client.add.call_args.kwargs
    assert kwargs["category"] == "guidelines"
    assert kwargs["scope"] == "personal"
    assert kwargs["source_ref"] == "foundation:constitution:v2"
    assert kwargs["pinned"] is True
    assert "foundation" in kwargs["tags"]
    assert "constitution" in kwargs["tags"]
    assert "principle:13" in kwargs["tags"]
    assert "rule-id:p13" in kwargs["tags"]
    assert kwargs["content"].startswith("Foundation 13 Transparent Reasoning:")


def test_rule_load_dry_run_does_not_call_api(tmp_path: Path) -> None:
    runner = CliRunner()
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps({
        "name": "test_pack",
        "scope": "personal",
        "memories": [
            {
                "content": "Test rule",
                "category": "guidelines",
                "tags": ["playbook", "u1"],
                "source_ref": "playbook:test:v1",
            }
        ],
    }))

    with patch("cems.commands.rule.get_client") as mock_get_client:
        result = runner.invoke(rule, ["load", "--file", str(seed_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Mode: dry-run" in result.output
    assert "Test rule" in result.output
    mock_get_client.assert_not_called()


def test_rule_load_calls_api_and_counts_events(tmp_path: Path) -> None:
    runner = CliRunner()
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps({
        "name": "test_apply",
        "scope": "personal",
        "memories": [
            {
                "content": "Rule A",
                "category": "guidelines",
                "tags": ["foundation"],
                "source_ref": "foundation:constitution:v2",
            },
            {
                "content": "Rule B",
                "category": "guidelines",
                "tags": ["foundation"],
                "source_ref": "foundation:constitution:v2",
            },
        ],
    }))

    mock_client = MagicMock()
    mock_client.add.side_effect = [
        {"success": True, "result": {"results": [{"event": "ADD", "id": "a"}]}},
        {"success": True, "result": {"results": [{"event": "DUPLICATE", "id": "b"}]}},
    ]

    with patch("cems.commands.rule.get_client", return_value=mock_client):
        result = runner.invoke(rule, ["load", "--file", str(seed_path)])

    assert result.exit_code == 0, result.output
    assert "Added: 1" in result.output
    assert "Duplicates: 1" in result.output
    assert mock_client.add.call_count == 2
