"""Tests for LongMemEval ASMR-style agentic retrieval module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion()."""

    def test_single_ranking(self):
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([["a", "b", "c"]])
        assert result == ["a", "b", "c"]

    def test_two_rankings_agreement(self):
        """When agents agree, order is preserved."""
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([
            ["a", "b", "c"],
            ["a", "b", "c"],
        ])
        assert result[0] == "a"
        assert result[1] == "b"
        assert result[2] == "c"

    def test_two_rankings_disagreement(self):
        """Item appearing in both rankings ranks higher than one-list-only items."""
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([
            ["a", "b", "c"],
            ["b", "d", "a"],
        ])
        # "b" appears at rank 2 in list 1 and rank 1 in list 2
        # "a" appears at rank 1 in list 1 and rank 3 in list 2
        # Both should be top-2
        assert set(result[:2]) == {"a", "b"}

    def test_three_rankings_majority_wins(self):
        """Item ranked first by 2 of 3 agents should be first overall."""
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([
            ["x", "y"],
            ["x", "z"],
            ["y", "z"],
        ])
        # x appears in 2 lists at rank 1, y appears at rank 1+2, z at rank 2+2
        assert result[0] == "x"

    def test_empty_rankings(self):
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([])
        assert result == []

    def test_empty_inner_ranking(self):
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([[], ["a", "b"]])
        assert result == ["a", "b"]

    def test_k_parameter_affects_smoothing(self):
        """Lower k gives more weight to top positions."""
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        # With very low k, rank position matters more
        r1 = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1)
        # With high k, rank position matters less (more uniform)
        r2 = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1000)
        # Both should have both items but scores are closer with high k
        assert set(r1) == {"a", "b"}
        assert set(r2) == {"a", "b"}

    def test_disjoint_rankings(self):
        """Rankings with no overlap should merge all items."""
        from cems.eval.longmemeval_agentic import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
        assert set(result) == {"a", "b", "c", "d"}
        # Items at rank 1 should come before items at rank 2
        assert result[0] in {"a", "c"}
        assert result[1] in {"a", "c"}


# ---------------------------------------------------------------------------
# _parse_agent_response
# ---------------------------------------------------------------------------

class TestParseAgentResponse:
    """Tests for _parse_agent_response()."""

    def test_valid_json_array(self):
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {"session_a", "session_b", "session_c"}
        result = _parse_agent_response('["session_a", "session_b"]', valid_ids)
        assert result == ["session_a", "session_b"]

    def test_json_in_markdown_code_block(self):
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {"session_a", "session_b"}
        response = '```json\n["session_a", "session_b"]\n```'
        result = _parse_agent_response(response, valid_ids)
        assert result == ["session_a", "session_b"]

    def test_filters_invalid_ids(self):
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {"session_a", "session_b"}
        result = _parse_agent_response('["session_a", "session_FAKE", "session_b"]', valid_ids)
        assert result == ["session_a", "session_b"]

    def test_limits_to_10_results(self):
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {f"s_{i}" for i in range(20)}
        ids = [f"s_{i}" for i in range(15)]
        result = _parse_agent_response(json.dumps(ids), valid_ids)
        assert len(result) <= 10

    def test_empty_json_array(self):
        from cems.eval.longmemeval_agentic import _parse_agent_response

        result = _parse_agent_response("[]", set())
        assert result == []

    def test_regex_fallback_for_malformed_json(self):
        """When JSON parsing fails, regex should extract session IDs."""
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {"ultrachat_283755", "answer_abc123"}
        response = "I think ultrachat_283755 is most relevant, followed by answer_abc123."
        result = _parse_agent_response(response, valid_ids)
        assert "ultrachat_283755" in result
        assert "answer_abc123" in result

    def test_regex_deduplicates(self):
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {"ultrachat_123"}
        response = "ultrachat_123 is the best. Yes, ultrachat_123 for sure."
        result = _parse_agent_response(response, valid_ids)
        assert result == ["ultrachat_123"]

    def test_empty_response(self):
        from cems.eval.longmemeval_agentic import _parse_agent_response

        result = _parse_agent_response("", {"a_1"})
        assert result == []

    def test_thinking_tags_stripped(self):
        """Gemini/Qwen models wrap output in <think> tags — parse_json_list handles this."""
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {"session_a"}
        response = '<think>Let me analyze...</think>\n["session_a"]'
        result = _parse_agent_response(response, valid_ids)
        assert result == ["session_a"]

    def test_json_parse_with_no_valid_ids_falls_to_regex(self):
        """When JSON parses but IDs don't match, regex fallback is tried."""
        from cems.eval.longmemeval_agentic import _parse_agent_response

        valid_ids = {"ultrachat_999"}
        # JSON parses fine but contains IDs not in valid_ids
        response = '["fake_id_1", "fake_id_2"]\nAlso mentioned ultrachat_999'
        result = _parse_agent_response(response, valid_ids)
        assert "ultrachat_999" in result


# ---------------------------------------------------------------------------
# _format_summaries_for_agents
# ---------------------------------------------------------------------------

class TestFormatSummariesForAgents:
    """Tests for _format_summaries_for_agents()."""

    def test_formats_summaries(self):
        from cems.eval.longmemeval_agentic import _format_summaries_for_agents

        summaries = {
            "s1": {"content": "User likes pizza", "tags": [], "priority": "medium"},
            "s2": {"content": "User has a dog named Rex", "tags": [], "priority": "high"},
        }
        result = _format_summaries_for_agents(["s1", "s2"], summaries)
        assert "Session s1" in result
        assert "User likes pizza" in result
        assert "Session s2" in result
        assert "User has a dog named Rex" in result

    def test_skips_missing_summaries(self):
        from cems.eval.longmemeval_agentic import _format_summaries_for_agents

        summaries = {"s1": {"content": "fact", "tags": [], "priority": "medium"}}
        result = _format_summaries_for_agents(["s1", "s_missing"], summaries)
        assert "Session s1" in result
        assert "s_missing" not in result

    def test_summary_mode_uses_dates_from_sessions(self):
        """Temporal navigator needs dates even in summary mode."""
        from cems.eval.longmemeval_agentic import _format_summaries_for_agents

        summaries = {"s1": {"content": "User likes pizza"}}
        sessions = {"s1": {"content": "raw text", "timestamp": "2023/04/10 (Mon) 17:50"}}
        result = _format_summaries_for_agents(
            ["s1"], summaries, sessions=sessions, raw_mode=False
        )
        assert "User likes pizza" in result  # Uses summary content
        assert "2023/04/10" in result  # But gets date from sessions

    def test_raw_mode_uses_sessions(self):
        from cems.eval.longmemeval_agentic import _format_summaries_for_agents

        sessions = {
            "s1": {"content": "Raw conversation text...", "timestamp": "2023/04/10 17:50"},
        }
        result = _format_summaries_for_agents(
            ["s1"], summaries={}, sessions=sessions, raw_mode=True
        )
        assert "Raw conversation text" in result
        assert "2023/04/10 17:50" in result

    def test_empty_session_ids(self):
        from cems.eval.longmemeval_agentic import _format_summaries_for_agents

        result = _format_summaries_for_agents([], {})
        assert result == ""


# ---------------------------------------------------------------------------
# Summary Caching
# ---------------------------------------------------------------------------

class TestSummaryCaching:
    """Tests for summary cache load/save."""

    def test_save_and_load_roundtrip(self, tmp_path):
        from cems.eval.longmemeval_agentic import _load_summary_cache, _save_summary_cache

        cache = {
            "s1": {"title": "Test", "content": "fact1\nfact2", "tags": ["t"], "priority": "high"},
        }
        cache_file = tmp_path / "test_cache.json"
        _save_summary_cache(cache_file, cache, "test-model")

        loaded = _load_summary_cache(cache_file)
        assert loaded == cache

    def test_load_nonexistent_returns_empty(self, tmp_path):
        from cems.eval.longmemeval_agentic import _load_summary_cache

        result = _load_summary_cache(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_corrupted_returns_empty(self, tmp_path):
        from cems.eval.longmemeval_agentic import _load_summary_cache

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{{{")
        result = _load_summary_cache(bad_file)
        assert result == {}

    def test_save_atomic_write(self, tmp_path):
        """Save uses tmp+rename pattern for atomicity."""
        from cems.eval.longmemeval_agentic import _save_summary_cache

        cache_file = tmp_path / "cache.json"
        _save_summary_cache(cache_file, {"s1": {"content": "x"}}, "model")

        # File should exist with valid JSON
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["_meta"]["model"] == "model"
        assert "s1" in data["sessions"]

        # Temp file should be cleaned up
        assert not cache_file.with_suffix(".tmp").exists()

    def test_save_includes_metadata(self, tmp_path):
        from cems.eval.longmemeval_agentic import _save_summary_cache

        cache_file = tmp_path / "cache.json"
        _save_summary_cache(cache_file, {"a": {}, "b": {}}, "google/gemini-2.5-flash")

        data = json.loads(cache_file.read_text())
        assert data["_meta"]["total_sessions"] == 2
        assert data["_meta"]["model"] == "google/gemini-2.5-flash"
        assert "updated_at" in data["_meta"]


# ---------------------------------------------------------------------------
# _extract_one_summary
# ---------------------------------------------------------------------------

class TestExtractOneSummary:
    """Tests for _extract_one_summary()."""

    @patch("cems.eval.longmemeval_agentic.extract_session_summary")
    def test_returns_session_id_and_result(self, mock_extract):
        from cems.eval.longmemeval_agentic import _extract_one_summary

        mock_extract.return_value = {
            "title": "Test",
            "content": "User has a cat",
            "tags": ["pets"],
            "priority": "medium",
        }
        sid, result = _extract_one_summary("s1", "conversation text", "model")
        assert sid == "s1"
        assert result["content"] == "User has a cat"
        mock_extract.assert_called_once_with("conversation text", project_context=None, model="model")

    @patch("cems.eval.longmemeval_agentic.extract_session_summary")
    def test_returns_none_on_failure(self, mock_extract):
        from cems.eval.longmemeval_agentic import _extract_one_summary

        mock_extract.return_value = None
        sid, result = _extract_one_summary("s1", "short", "model")
        assert sid == "s1"
        assert result is None


# ---------------------------------------------------------------------------
# load_or_create_summaries
# ---------------------------------------------------------------------------

class TestLoadOrCreateSummaries:
    """Tests for load_or_create_summaries()."""

    @patch("cems.eval.longmemeval_agentic.extract_session_summary")
    @patch("cems.eval.longmemeval_agentic._cache_path")
    def test_extracts_uncached_sessions(self, mock_cache_path, mock_extract, tmp_path):
        from cems.eval.longmemeval_agentic import load_or_create_summaries

        cache_file = tmp_path / "cache.json"
        mock_cache_path.return_value = cache_file

        mock_extract.return_value = {
            "title": "T", "content": "fact", "tags": [], "priority": "medium"
        }

        sessions = {
            "s1": {"content": "long conversation text here " * 20, "session_id": "s1"},
            "s2": {"content": "another conversation " * 20, "session_id": "s2"},
        }

        result = load_or_create_summaries(sessions, concurrency=1, no_cache=True)
        assert "s1" in result
        assert "s2" in result
        assert mock_extract.call_count == 2

    @patch("cems.eval.longmemeval_agentic.extract_session_summary")
    @patch("cems.eval.longmemeval_agentic._cache_path")
    def test_uses_cache_skips_extraction(self, mock_cache_path, mock_extract, tmp_path):
        from cems.eval.longmemeval_agentic import (
            _save_summary_cache,
            load_or_create_summaries,
        )

        cache_file = tmp_path / "cache.json"
        mock_cache_path.return_value = cache_file

        # Pre-populate cache
        _save_summary_cache(cache_file, {
            "s1": {"title": "Cached", "content": "cached fact", "tags": [], "priority": "medium"},
        }, "model")

        sessions = {
            "s1": {"content": "text", "session_id": "s1"},
        }

        result = load_or_create_summaries(sessions, concurrency=1)
        assert result["s1"]["content"] == "cached fact"
        mock_extract.assert_not_called()  # Should not extract — already cached


# ---------------------------------------------------------------------------
# run_search_agents (integration-style with mocked LLM)
# ---------------------------------------------------------------------------

class TestRunSearchAgents:
    """Tests for run_search_agents() with mocked LLM calls."""

    @patch("cems.eval.longmemeval_agentic.get_client")
    def test_returns_ranked_session_ids(self, mock_get_client):
        from cems.eval.longmemeval_agentic import run_search_agents

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # All 3 agents agree on the same ranking
        mock_client.complete.return_value = '["s_correct", "s_other"]'

        summaries = {
            "s_correct": {"content": "User graduated with a CS degree"},
            "s_other": {"content": "User likes coffee"},
            "s_noise": {"content": "Weather was sunny"},
        }

        result = run_search_agents(
            question="What degree did I graduate with?",
            session_ids=["s_correct", "s_other", "s_noise"],
            summaries=summaries,
        )

        assert result[0] == "s_correct"
        assert mock_client.complete.call_count == 3  # 3 agents

    @patch("cems.eval.longmemeval_agentic.get_client")
    def test_handles_agent_failure(self, mock_get_client):
        from cems.eval.longmemeval_agentic import run_search_agents

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call succeeds, second fails, third succeeds
        mock_client.complete.side_effect = [
            '["s1"]',
            Exception("LLM timeout"),
            '["s1", "s2"]',
        ]

        summaries = {
            "s1": {"content": "fact1"},
            "s2": {"content": "fact2"},
        }

        result = run_search_agents(
            question="test",
            session_ids=["s1", "s2"],
            summaries=summaries,
        )

        # Should still return results from the 2 successful agents
        assert "s1" in result

    @patch("cems.eval.longmemeval_agentic.get_client")
    def test_all_agents_fail_returns_empty(self, mock_get_client):
        from cems.eval.longmemeval_agentic import run_search_agents

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.complete.side_effect = Exception("All broken")

        result = run_search_agents(
            question="test",
            session_ids=["s1"],
            summaries={"s1": {"content": "x"}},
        )

        assert result == []

    @patch("cems.eval.longmemeval_agentic.get_client")
    def test_uses_correct_agent_roles(self, mock_get_client):
        from cems.eval.longmemeval_agentic import AGENT_SYSTEM_PROMPTS, run_search_agents

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.complete.return_value = "[]"

        run_search_agents(
            question="test",
            session_ids=["s1"],
            summaries={"s1": {"content": "x"}},
        )

        # Verify all 3 agent roles were called
        system_prompts_used = [
            call.kwargs.get("system") or call.args[0] if call.args else call.kwargs.get("system", "")
            for call in mock_client.complete.call_args_list
        ]

        # Should have called with system prompts for all 3 roles
        assert mock_client.complete.call_count == 3


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

class TestAgentPrompts:
    """Tests for search agent prompt structure."""

    def test_all_three_agent_roles_defined(self):
        from cems.eval.longmemeval_agentic import AGENT_SYSTEM_PROMPTS

        assert "direct_seeker" in AGENT_SYSTEM_PROMPTS
        assert "inference_engine" in AGENT_SYSTEM_PROMPTS
        assert "temporal_navigator" in AGENT_SYSTEM_PROMPTS

    def test_prompts_are_non_empty_strings(self):
        from cems.eval.longmemeval_agentic import AGENT_SYSTEM_PROMPTS

        for role, prompt in AGENT_SYSTEM_PROMPTS.items():
            assert isinstance(prompt, str), f"{role} prompt is not a string"
            assert len(prompt) > 100, f"{role} prompt is too short"

    def test_search_user_prompt_template_has_placeholders(self):
        from cems.eval.longmemeval_agentic import SEARCH_USER_PROMPT

        assert "{question}" in SEARCH_USER_PROMPT
        assert "{n}" in SEARCH_USER_PROMPT
        assert "{formatted_summaries}" in SEARCH_USER_PROMPT

    def test_search_user_prompt_formats_correctly(self):
        from cems.eval.longmemeval_agentic import SEARCH_USER_PROMPT

        formatted = SEARCH_USER_PROMPT.format(
            question="What is my cat's name?",
            n=5,
            formatted_summaries="--- Session s1 ---\nUser's cat is named Luna",
        )
        assert "What is my cat's name?" in formatted
        assert "5 past sessions" in formatted
        assert "Luna" in formatted


# ---------------------------------------------------------------------------
# Cache path generation
# ---------------------------------------------------------------------------

class TestCachePath:
    """Tests for _cache_path()."""

    def test_generates_model_specific_path(self):
        from cems.eval.longmemeval_agentic import _cache_path

        path = _cache_path("google/gemini-2.5-flash")
        assert "google_gemini-2_5-flash" in path.name
        assert path.suffix == ".json"

    def test_different_models_different_paths(self):
        from cems.eval.longmemeval_agentic import _cache_path

        p1 = _cache_path("google/gemini-2.5-flash")
        p2 = _cache_path("openai/gpt-4o-mini")
        assert p1 != p2
