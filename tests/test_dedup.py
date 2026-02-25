"""Tests for LLM-powered memory pair classification (Phase 4).

Tests the classify_memory_pair() function that classifies two memories
as duplicate, related, conflicting, or distinct.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from cems.llm.dedup import classify_memory_pair, CLASSIFICATION_SYSTEM_PROMPT


class TestClassifyMemoryPair:
    """Tests for classify_memory_pair()."""

    @patch("cems.llm.dedup.get_client")
    def test_returns_duplicate_classification(self, mock_get_client):
        """Classifies near-identical memories as duplicate."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "duplicate",
            "explanation": "Both describe the same Python backend preference",
            "confidence": 0.95,
        })
        mock_get_client.return_value = mock_client

        result = classify_memory_pair(
            "User prefers Python for backend development",
            "User likes Python for backend work",
        )

        assert result["classification"] == "duplicate"
        assert result["confidence"] == 0.95
        assert "explanation" in result

    @patch("cems.llm.dedup.get_client")
    def test_returns_conflicting_classification(self, mock_get_client):
        """Classifies contradictory memories as conflicting."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "conflicting",
            "explanation": "Memory A says Hetzner, Memory B says Railway",
            "confidence": 0.88,
        })
        mock_get_client.return_value = mock_client

        result = classify_memory_pair(
            "User deployed CEMS to Coolify on Hetzner",
            "User deployed CEMS to Coolify on Railway",
        )

        assert result["classification"] == "conflicting"
        assert result["confidence"] == 0.88

    @patch("cems.llm.dedup.get_client")
    def test_returns_related_classification(self, mock_get_client):
        """Classifies same-topic-different-info as related."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "related",
            "explanation": "Both about Python but different aspects",
            "confidence": 0.82,
        })
        mock_get_client.return_value = mock_client

        result = classify_memory_pair(
            "User prefers Python for backend development",
            "User uses pytest for testing Python code",
        )

        assert result["classification"] == "related"

    @patch("cems.llm.dedup.get_client")
    def test_returns_distinct_classification(self, mock_get_client):
        """Classifies unrelated memories as distinct."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "distinct",
            "explanation": "Completely unrelated topics",
            "confidence": 0.97,
        })
        mock_get_client.return_value = mock_client

        result = classify_memory_pair(
            "User prefers dark mode in IDEs",
            "Project deadline is next Friday",
        )

        assert result["classification"] == "distinct"

    @patch("cems.llm.dedup.get_client")
    def test_handles_markdown_wrapped_json(self, mock_get_client):
        """Parses JSON wrapped in markdown code blocks."""
        mock_client = MagicMock()
        mock_client.complete.return_value = '```json\n{"classification": "duplicate", "explanation": "Same fact", "confidence": 0.9}\n```'
        mock_get_client.return_value = mock_client

        result = classify_memory_pair("Memory A", "Memory B")

        assert result["classification"] == "duplicate"
        assert result["confidence"] == 0.9

    @patch("cems.llm.dedup.get_client")
    def test_fallback_on_invalid_json(self, mock_get_client):
        """Returns 'distinct' with low confidence on unparseable response."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "I think these are duplicates"
        mock_get_client.return_value = mock_client

        result = classify_memory_pair("Memory A", "Memory B")

        assert result["classification"] == "distinct"
        assert result["confidence"] == 0.0
        assert "parse" in result["explanation"].lower() or "failed" in result["explanation"].lower()

    @patch("cems.llm.dedup.get_client")
    def test_fallback_on_missing_fields(self, mock_get_client):
        """Fills in defaults when response JSON is missing fields."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "duplicate",
        })
        mock_get_client.return_value = mock_client

        result = classify_memory_pair("Memory A", "Memory B")

        assert result["classification"] == "duplicate"
        assert result["confidence"] == 0.5  # default when missing
        assert result["explanation"] == ""  # default when missing

    @patch("cems.llm.dedup.get_client")
    def test_fallback_on_invalid_classification(self, mock_get_client):
        """Treats unknown classification as 'distinct'."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "maybe_duplicate",
            "explanation": "Not sure",
            "confidence": 0.5,
        })
        mock_get_client.return_value = mock_client

        result = classify_memory_pair("Memory A", "Memory B")

        assert result["classification"] == "distinct"

    @patch("cems.llm.dedup.get_client")
    def test_fallback_on_llm_exception(self, mock_get_client):
        """Returns safe fallback when LLM call raises."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API timeout")
        mock_get_client.return_value = mock_client

        result = classify_memory_pair("Memory A", "Memory B")

        assert result["classification"] == "distinct"
        assert result["confidence"] == 0.0

    @patch("cems.llm.dedup.get_client")
    def test_uses_gemini_flash_model(self, mock_get_client):
        """Uses Gemini 2.5 Flash with fast_route=False."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "distinct",
            "explanation": "Different",
            "confidence": 0.9,
        })
        mock_get_client.return_value = mock_client

        classify_memory_pair("A", "B")

        call_kwargs = mock_client.complete.call_args[1]
        assert call_kwargs["model"] == "google/gemini-2.5-flash"
        assert call_kwargs["fast_route"] is False
        assert call_kwargs["temperature"] == 0.0

    @patch("cems.llm.dedup.get_client")
    def test_model_override(self, mock_get_client):
        """Allows model override via parameter."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "distinct",
            "explanation": "Different",
            "confidence": 0.9,
        })
        mock_get_client.return_value = mock_client

        classify_memory_pair("A", "B", model="openai/gpt-4o-mini")

        call_kwargs = mock_client.complete.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o-mini"

    def test_system_prompt_has_classifications(self):
        """System prompt mentions all four classification types."""
        for cls in ["duplicate", "related", "conflicting", "distinct"]:
            assert cls in CLASSIFICATION_SYSTEM_PROMPT

    @patch("cems.llm.dedup.get_client")
    def test_confidence_clamped_to_0_1(self, mock_get_client):
        """Confidence values outside 0-1 are clamped."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "classification": "duplicate",
            "explanation": "Same",
            "confidence": 1.5,
        })
        mock_get_client.return_value = mock_client

        result = classify_memory_pair("A", "B")

        assert result["confidence"] == 1.0
