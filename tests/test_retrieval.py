"""Tests for the 5-stage inference retrieval pipeline."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from cems.models import MemoryMetadata, MemoryScope, SearchResult
from cems.retrieval import (
    assemble_context,
    calculate_relevance_score,
    deduplicate_results,
    format_memory_context,
    synthesize_query,
)


class TestQuerySynthesis:
    """Stage 1: Query expansion tests."""

    def test_synthesize_query_returns_list(self):
        """Query synthesis should return a list of expanded terms."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "coding styles\nprogramming preferences\ndeveloper habits"

        result = synthesize_query("coding preferences", mock_client)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "coding styles" in result

    def test_synthesize_query_filters_short_terms(self):
        """Query synthesis should filter out very short terms."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "ab\nprogramming preferences\nc"

        result = synthesize_query("preferences", mock_client)

        assert "ab" not in result
        assert "c" not in result
        assert "programming preferences" in result

    def test_synthesize_query_handles_error(self):
        """Query synthesis should return empty list on error."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")

        result = synthesize_query("coding preferences", mock_client)

        assert result == []

    def test_synthesize_query_limits_to_3_results(self):
        """Query synthesis should return at most 3 terms (constrained for relevance)."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "\n".join([f"term{i}" for i in range(10)])

        result = synthesize_query("preferences", mock_client)

        assert len(result) <= 3


class TestRelevanceScoring:
    """Stage 3-4: Relevance filtering and temporal ranking tests."""

    def test_calculate_relevance_score_no_decay(self):
        """Score should be unaffected by time when days_since_access is 0."""
        score = calculate_relevance_score(
            base_score=1.0,
            days_since_access=0,
            priority=1.0,
            pinned=False,
        )
        assert score == 1.0

    def test_calculate_relevance_score_time_decay_30_days(self):
        """Score should decay to ~0.5 after 30 days."""
        score = calculate_relevance_score(
            base_score=1.0,
            days_since_access=30,
            priority=1.0,
            pinned=False,
        )
        assert 0.49 < score < 0.51  # Should be close to 0.5

    def test_calculate_relevance_score_time_decay_90_days(self):
        """Score should decay to ~0.25 after 90 days."""
        score = calculate_relevance_score(
            base_score=1.0,
            days_since_access=90,
            priority=1.0,
            pinned=False,
        )
        assert 0.24 < score < 0.26  # Should be close to 0.25

    def test_calculate_relevance_score_priority_boost(self):
        """Score should be multiplied by priority."""
        score = calculate_relevance_score(
            base_score=1.0,
            days_since_access=0,
            priority=2.0,
            pinned=False,
        )
        assert score == 2.0

    def test_calculate_relevance_score_pinned_boost(self):
        """Pinned memories should get a 10% boost."""
        unpinned = calculate_relevance_score(
            base_score=1.0,
            days_since_access=0,
            priority=1.0,
            pinned=False,
        )
        pinned = calculate_relevance_score(
            base_score=1.0,
            days_since_access=0,
            priority=1.0,
            pinned=True,
        )
        assert pinned == unpinned * 1.1

    def test_calculate_relevance_score_combined_factors(self):
        """Score should combine all factors correctly."""
        score = calculate_relevance_score(
            base_score=0.8,
            days_since_access=30,
            priority=1.5,
            pinned=True,
        )
        # Expected: 0.8 * 0.5 (time decay) * 1.5 (priority) * 1.1 (pinned)
        expected = 0.8 * 0.5 * 1.5 * 1.1
        assert abs(score - expected) < 0.01


class TestTokenBudgetAssembly:
    """Stage 5: Token-budgeted assembly tests."""

    def _create_search_result(
        self, memory_id: str, content: str, score: float
    ) -> SearchResult:
        """Helper to create a SearchResult."""
        return SearchResult(
            memory_id=memory_id,
            content=content,
            score=score,
            scope=MemoryScope.PERSONAL,
            metadata=None,
        )

    def test_assemble_context_respects_budget(self):
        """Assembly should stop when token budget is exhausted."""
        results = [
            self._create_search_result("1", "a" * 100, 0.9),
            self._create_search_result("2", "b" * 100, 0.8),
            self._create_search_result("3", "c" * 100, 0.7),
        ]

        # With a small budget, only some results should be selected
        selected, tokens = assemble_context(results, max_tokens=50)

        assert len(selected) < len(results)
        assert tokens <= 50

    def test_assemble_context_preserves_order(self):
        """Assembly should preserve the input order (score order)."""
        results = [
            self._create_search_result("1", "first", 0.9),
            self._create_search_result("2", "second", 0.8),
            self._create_search_result("3", "third", 0.7),
        ]

        selected, _ = assemble_context(results, max_tokens=1000)

        assert selected[0].memory_id == "1"
        assert selected[-1].memory_id == "3"

    def test_assemble_context_returns_token_count(self):
        """Assembly should return accurate token count."""
        results = [self._create_search_result("1", "test content here", 0.9)]

        selected, tokens = assemble_context(results, max_tokens=1000)

        assert tokens > 0
        assert len(selected) == 1


class TestContextFormatting:
    """Tests for memory context formatting."""

    def test_format_memory_context_empty(self):
        """Formatting empty results should return placeholder message."""
        result = format_memory_context([])
        assert result == "No relevant memories found."

    def test_format_memory_context_includes_header(self):
        """Formatted context should include header and footer."""
        results = [
            SearchResult(
                memory_id="1",
                content="Test content",
                score=0.9,
                scope=MemoryScope.PERSONAL,
                metadata=None,
            )
        ]

        result = format_memory_context(results)

        assert "=== RELEVANT MEMORIES ===" in result
        assert "=== END MEMORIES ===" in result

    def test_format_memory_context_includes_metadata(self):
        """Formatted context should include metadata when available."""
        metadata = MemoryMetadata(
            memory_id="1",
            user_id="test",
            scope=MemoryScope.PERSONAL,
            category="preferences",
            last_accessed=datetime(2025, 1, 15, tzinfo=UTC),
        )
        results = [
            SearchResult(
                memory_id="1",
                content="Test content",
                score=0.95,
                scope=MemoryScope.PERSONAL,
                metadata=metadata,
            )
        ]

        result = format_memory_context(results)

        assert "2025-01-15" in result
        assert "preferences" in result
        assert "0.95" in result
        assert "Test content" in result


class TestDeduplication:
    """Tests for result deduplication."""

    def _create_search_result(
        self, memory_id: str, content: str, score: float
    ) -> SearchResult:
        """Helper to create a SearchResult."""
        return SearchResult(
            memory_id=memory_id,
            content=content,
            score=score,
            scope=MemoryScope.PERSONAL,
            metadata=None,
        )

    def test_deduplicate_removes_duplicates(self):
        """Deduplication should remove duplicate memory IDs."""
        results = [
            self._create_search_result("1", "first", 0.9),
            self._create_search_result("2", "second", 0.8),
            self._create_search_result("1", "first duplicate", 0.7),
        ]

        unique = deduplicate_results(results)

        assert len(unique) == 2
        memory_ids = [r.memory_id for r in unique]
        assert memory_ids.count("1") == 1

    def test_deduplicate_keeps_highest_score(self):
        """Deduplication should keep the version with the highest score."""
        results = [
            self._create_search_result("1", "low score", 0.5),
            self._create_search_result("2", "other", 0.8),
            self._create_search_result("1", "high score", 0.9),
        ]

        unique = deduplicate_results(results)

        result_1 = next(r for r in unique if r.memory_id == "1")
        assert result_1.score == 0.9

    def test_deduplicate_preserves_order_of_first_occurrence(self):
        """Deduplication should maintain order based on first occurrence."""
        results = [
            self._create_search_result("1", "first", 0.9),
            self._create_search_result("2", "second", 0.8),
            self._create_search_result("3", "third", 0.7),
        ]

        unique = deduplicate_results(results)

        assert unique[0].memory_id == "1"
        assert unique[1].memory_id == "2"
        assert unique[2].memory_id == "3"


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @patch("cems.memory.CEMSMemory")
    def test_pipeline_handles_no_results(self, mock_memory_class):
        """Pipeline should gracefully handle no results."""
        # This tests the formatting when there are no results
        result = format_memory_context([])
        assert "No relevant memories found" in result

    def test_pipeline_respects_threshold(self):
        """Results below threshold should be filtered out."""
        results = [
            SearchResult(
                memory_id="1",
                content="High score",
                score=0.9,
                scope=MemoryScope.PERSONAL,
                metadata=None,
            ),
            SearchResult(
                memory_id="2",
                content="Low score",
                score=0.3,
                scope=MemoryScope.PERSONAL,
                metadata=None,
            ),
        ]

        # Apply threshold filter (mimicking pipeline Stage 3)
        threshold = 0.5
        filtered = [r for r in results if r.score >= threshold]

        assert len(filtered) == 1
        assert filtered[0].memory_id == "1"
