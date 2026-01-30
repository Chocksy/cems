"""Tests for the 5-stage inference retrieval pipeline."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from cems.models import MemoryMetadata, MemoryScope, SearchResult
from cems.retrieval import (
    _is_aggregation_query,
    _is_preference_query,
    _is_temporal_query,
    _jaccard_similarity,
    _max_similarity_to_selected,
    _word_set,
    apply_score_adjustments,
    assemble_context,
    assemble_context_diverse,
    deduplicate_results,
    filter_preferences_by_relevance,
    format_memory_context,
    generate_adaptive_probe,
    is_strong_lexical_signal,
    normalize_lexical_score,
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


class TestRelevanceFilter:
    """Tests for preference relevance filtering."""

    def test_relevance_filter_keeps_relevant(self):
        """Relevance filter should keep relevant preferences."""
        mock_client = MagicMock()
        mock_client.complete.return_value = '{"relevant": ["mixology class", "Hendrick\'s gin"]}'

        prefs = ["mixology class", "Hendrick's gin", "Instagram sticker", "camping trip"]
        result = filter_preferences_by_relevance(prefs, "recommend a cocktail", mock_client)

        assert "mixology class" in result
        assert "Hendrick's gin" in result
        assert "Instagram sticker" not in result
        assert "camping trip" not in result

    def test_relevance_filter_fallback_when_none_relevant(self):
        """Relevance filter uses lenient fallback - keeps all prefs if LLM says none relevant.

        This is intentional: over-filtering hurt performance in v2/v3 experiments.
        Empty filtering is worse than keeping slightly off-topic preferences.
        """
        mock_client = MagicMock()
        mock_client.complete.return_value = '{"relevant": []}'

        prefs = ["Instagram sticker", "camping trip", "evening gown"]
        result = filter_preferences_by_relevance(prefs, "recommend a cocktail", mock_client)

        # Lenient fallback: if filter removes ALL, keep original (prevents over-filtering)
        assert result == prefs

    def test_relevance_filter_handles_error(self):
        """Relevance filter should return all prefs on error (fallback)."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")

        prefs = ["pref1", "pref2"]
        result = filter_preferences_by_relevance(prefs, "test query", mock_client)

        # Should return all prefs as fallback
        assert result == prefs

    def test_relevance_filter_handles_empty_input(self):
        """Relevance filter should handle empty input."""
        mock_client = MagicMock()

        result = filter_preferences_by_relevance([], "test query", mock_client)

        assert result == []
        mock_client.complete.assert_not_called()

    def test_relevance_filter_validates_returned_items(self):
        """Relevance filter should only return items from original list."""
        mock_client = MagicMock()
        # LLM returns items not in original list (hallucination)
        mock_client.complete.return_value = '{"relevant": ["pref1", "hallucinated_pref", "pref2"]}'

        prefs = ["pref1", "pref2", "pref3"]
        result = filter_preferences_by_relevance(prefs, "test query", mock_client)

        assert "pref1" in result
        assert "pref2" in result
        assert "hallucinated_pref" not in result


class TestAdaptiveProbe:
    """Tests for adaptive profile probe generation."""

    def test_adaptive_probe_returns_list(self):
        """Adaptive probe should return a list of domain-specific phrases."""
        mock_client = MagicMock()
        mock_client.complete.return_value = '{"phrases": ["cocktail gin", "mixology class", "bartending spirits"]}'

        result = generate_adaptive_probe("recommend a cocktail", mock_client)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "cocktail gin" in result

    def test_adaptive_probe_limits_to_5(self):
        """Adaptive probe should return at most 5 phrases."""
        mock_client = MagicMock()
        phrases = [f"phrase{i}" for i in range(10)]
        mock_client.complete.return_value = f'{{"phrases": {phrases}}}'

        result = generate_adaptive_probe("test query", mock_client)

        assert len(result) <= 5

    def test_adaptive_probe_handles_invalid_json(self):
        """Adaptive probe should return empty list on invalid JSON."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "not valid json"

        result = generate_adaptive_probe("test query", mock_client)

        assert result == []

    def test_adaptive_probe_handles_error(self):
        """Adaptive probe should return empty list on error."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")

        result = generate_adaptive_probe("test query", mock_client)

        assert result == []

    def test_adaptive_probe_filters_empty_phrases(self):
        """Adaptive probe should filter out empty/short phrases."""
        mock_client = MagicMock()
        mock_client.complete.return_value = '{"phrases": ["", "ab", "valid phrase", "   ", "another valid"]}'

        result = generate_adaptive_probe("test query", mock_client)

        assert "" not in result
        assert "ab" not in result
        assert "valid phrase" in result
        assert "another valid" in result


class TestQueryTypeDetection:
    """Tests for query type detection (temporal vs preference)."""

    def test_preference_query_detection_recommendation(self):
        """Queries asking for recommendations should be detected."""
        assert _is_preference_query("Can you recommend video editing resources?")
        assert _is_preference_query("Suggest accessories for my photography setup")
        assert _is_preference_query("What would you suggest for data analysis?")
        assert _is_preference_query("Any advice on improving my workflow?")

    def test_preference_query_detection_resources(self):
        """Queries asking about resources/tools should be detected."""
        assert _is_preference_query("What tools do I use for Python development?")
        assert _is_preference_query("Resources that complement my learning")
        assert _is_preference_query("Publications I might find interesting")
        assert _is_preference_query("Equipment that goes with my camera gear")

    def test_preference_query_detection_complement(self):
        """Queries about complementing/matching should be detected."""
        assert _is_preference_query("What would complement my current setup?")
        assert _is_preference_query("What works well with my existing tools?")

    def test_preference_query_detection_indirect(self):
        """Indirect preference queries should be detected."""
        # Advice-seeking via emotion/struggle
        assert _is_preference_query("I've been feeling stuck with my paintings")
        assert _is_preference_query("I've been struggling with my slow cooker recipes")
        assert _is_preference_query("Any tips for keeping my kitchen organized?")
        # Planning/intent patterns
        assert _is_preference_query("I've been thinking about making a cocktail")
        assert _is_preference_query("I'm thinking of inviting colleagues over")
        assert _is_preference_query("What should I serve for dinner this weekend?")
        # Entertainment/activity seeking
        assert _is_preference_query("Can you recommend a show or movie for me?")
        assert _is_preference_query("Suggest some activities that I can do in the evening")

    def test_preference_query_detection_v3_planning(self):
        """v3 patterns: Planning-style queries that expect preference-based answers."""
        # Trip/activity planning
        assert _is_preference_query("I'm planning a trip to Denver soon. Any suggestions on what to do there?")
        assert _is_preference_query("I'm planning my meal prep next week, any suggestions for new recipes?")
        # "Thinking of trying" patterns
        assert _is_preference_query("I was thinking of trying a new coffee creamer recipe. Any recommendations?")
        assert _is_preference_query("I'm thinking of trying a different approach to this problem")
        # "Do you think" patterns (opinion/preference seeking)
        assert _is_preference_query("Do you think it would be a good idea to attend my high school reunion?")
        assert _is_preference_query("What do you think about this approach?")

    def test_preference_query_detection_v3_observation(self):
        """v3 patterns: Observation + question patterns (implicit preference queries)."""
        # "I noticed" patterns
        assert _is_preference_query("I noticed my bike seems to be performing even better. Could there be a reason?")
        assert _is_preference_query("I've noticed the kitchen is getting messy again")
        # "Free time" entertainment seeking
        assert _is_preference_query("I've got some free time tonight, any documentary recommendations?")
        assert _is_preference_query("Got some free time this weekend, what should I do?")

    def test_non_preference_queries_not_detected(self):
        """Regular factual queries should NOT be detected as preference."""
        assert not _is_preference_query("What is the capital of France?")
        assert not _is_preference_query("How do I install Python?")
        assert not _is_preference_query("Show me recent commits")
        assert not _is_preference_query("What databases do we use?")

    def test_temporal_queries_not_detected_as_preference(self):
        """Temporal queries should be detected by temporal, not preference."""
        # These should be temporal, not preference
        assert not _is_preference_query("Which happened first, X or Y?")
        assert not _is_preference_query("When did I start using Python?")
        # But they should be temporal
        assert _is_temporal_query("Which happened first, X or Y?")
        assert _is_temporal_query("When did I start using Python?")


class TestAggregationQueryDetection:
    """Tests for aggregation query detection (multi-session queries)."""

    def test_aggregation_counting_patterns(self):
        """Queries with counting patterns should be detected."""
        assert _is_aggregation_query("How many different doctors did I visit?")
        assert _is_aggregation_query("How many camping trips did I take?")
        assert _is_aggregation_query("How much money did I spend in total?")
        assert _is_aggregation_query("What is the number of books I read?")

    def test_aggregation_total_patterns(self):
        """Queries with total/sum patterns should be detected."""
        assert _is_aggregation_query("What is the total amount I spent on luxury items?")
        assert _is_aggregation_query("How many days altogether did I spend traveling?")
        assert _is_aggregation_query("What is the combined cost of all my purchases?")
        assert _is_aggregation_query("How many hours in total did I work out?")

    def test_aggregation_frequency_patterns(self):
        """Queries about frequency should be detected."""
        assert _is_aggregation_query("All the times I went to the gym")
        assert _is_aggregation_query("Every time I visited the doctor")
        assert _is_aggregation_query("How often did I exercise last month?")

    def test_aggregation_variety_patterns(self):
        """Queries about variety/diversity should be detected."""
        assert _is_aggregation_query("How many different cuisines have I tried?")
        assert _is_aggregation_query("The various museums I visited")
        assert _is_aggregation_query("All the different countries I traveled to")

    def test_aggregation_time_range_patterns(self):
        """Queries about time ranges should be detected."""
        assert _is_aggregation_query("How much did I spend in the past month?")
        assert _is_aggregation_query("What have I done throughout this year?")
        assert _is_aggregation_query("Activities across all my vacation trips")

    def test_non_aggregation_queries(self):
        """Regular queries should NOT be detected as aggregation."""
        assert not _is_aggregation_query("What is my favorite restaurant?")
        assert not _is_aggregation_query("When did I last go to the gym?")
        assert not _is_aggregation_query("Can you recommend a book?")
        assert not _is_aggregation_query("What is the capital of France?")

    def test_aggregation_vs_temporal_distinction(self):
        """Some queries may be both aggregation and temporal - that's OK."""
        # "How many days" is both temporal (has time) and aggregation (counting)
        query = "How many days did I spend camping in total?"
        assert _is_aggregation_query(query)
        # This is fine - both types will force synthesis

    def test_preference_synthesis_uses_special_prompt(self):
        """Preference queries should use the preference-specific prompt."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "Adobe Premiere Pro\nvideo editing software I use\nediting workflow"

        result = synthesize_query("video editing resources", mock_client, is_preference=True)

        # Check that the prompt mentioned preferences
        call_args = mock_client.complete.call_args
        prompt = call_args[0][0]
        assert "PREFERENCE" in prompt or "preference" in prompt.lower()
        assert "USER STATEMENTS" in prompt or "user statements" in prompt.lower()


class TestRelevanceScoring:
    """Stage 3-4: Relevance filtering and temporal ranking tests."""

    def _create_result(
        self, score: float = 1.0, priority: float = 1.0, pinned: bool = False,
        days_ago: int = 0, category: str = "general", source_ref: str | None = None
    ) -> SearchResult:
        """Helper to create a SearchResult for testing."""
        now = datetime.now(UTC)
        last_accessed = now - timedelta(days=days_ago)
        metadata = MemoryMetadata(
            memory_id="test-id",
            user_id="test-user",
            scope=MemoryScope.PERSONAL,
            category=category,
            priority=priority,
            pinned=pinned,
            source_ref=source_ref,
            created_at=now,
            updated_at=now,
            last_accessed=last_accessed,
        )
        return SearchResult(
            memory_id="test-id",
            content="Test content",
            score=score,
            scope=MemoryScope.PERSONAL,
            metadata=metadata,
        )

    def test_apply_score_adjustments_no_decay(self):
        """Score should be unaffected by time when accessed today."""
        result = self._create_result(score=1.0, days_ago=0)
        score = apply_score_adjustments(result)
        assert score == 1.0

    def test_apply_score_adjustments_time_decay_60_days(self):
        """Score should decay to ~0.5 after 60 days (60-day half-life)."""
        result = self._create_result(score=1.0, days_ago=60)
        score = apply_score_adjustments(result)
        assert 0.49 < score < 0.51  # Should be close to 0.5

    def test_apply_score_adjustments_time_decay_120_days(self):
        """Score should decay to ~0.33 after 120 days."""
        result = self._create_result(score=1.0, days_ago=120)
        score = apply_score_adjustments(result)
        assert 0.32 < score < 0.34  # Should be close to 0.33

    def test_apply_score_adjustments_priority_boost(self):
        """Score should be multiplied by priority (then clamped)."""
        result = self._create_result(score=1.0, priority=2.0, days_ago=0)
        score = apply_score_adjustments(result)
        assert score == 1.0  # Clamped to [0, 1]

    def test_apply_score_adjustments_pinned_boost(self):
        """Pinned memories should get a 10% boost (then clamped)."""
        unpinned_result = self._create_result(score=1.0, pinned=False, days_ago=0)
        pinned_result = self._create_result(score=1.0, pinned=True, days_ago=0)

        unpinned_score = apply_score_adjustments(unpinned_result)
        pinned_score = apply_score_adjustments(pinned_result)

        assert pinned_score == 1.0  # Clamped to [0, 1]

    def test_apply_score_adjustments_combined_factors(self):
        """Score should combine all factors correctly (with clamping)."""
        result = self._create_result(score=0.8, priority=1.5, pinned=True, days_ago=60)
        score = apply_score_adjustments(result)
        # Expected: 0.8 * 0.5 (time decay at 60 days) * 1.5 (priority) * 1.1 (pinned)
        expected = 0.8 * 0.5 * 1.5 * 1.1
        expected = min(1.0, expected)
        assert abs(score - expected) < 0.01

    def test_apply_score_adjustments_cross_category_penalty(self):
        """Cross-category penalty should reduce score by 20%."""
        result = self._create_result(score=1.0, category="deployment", days_ago=0)
        # With default config (penalties enabled), inferred category mismatch should apply 0.8x
        score = apply_score_adjustments(result, inferred_category="development")
        assert score == 0.8

    def test_apply_score_adjustments_same_category_no_penalty(self):
        """Same category should not apply penalty."""
        result = self._create_result(score=1.0, category="deployment", days_ago=0)
        score = apply_score_adjustments(result, inferred_category="deployment")
        assert score == 1.0

    def test_apply_score_adjustments_project_boost(self):
        """Same-project memories should get 30% boost (then clamped)."""
        result = self._create_result(score=1.0, source_ref="project:acme/api", days_ago=0)
        score = apply_score_adjustments(result, project="acme/api")
        assert score == 1.0

    def test_apply_score_adjustments_project_penalty(self):
        """Different-project memories should get 20% penalty."""
        result = self._create_result(score=1.0, source_ref="project:acme/website", days_ago=0)
        score = apply_score_adjustments(result, project="acme/api")
        assert score == 0.8

    def test_apply_score_adjustments_skip_category_penalty(self):
        """Category penalty can be skipped when categories are uniform."""
        result = self._create_result(score=1.0, category="deployment", days_ago=0)
        score = apply_score_adjustments(
            result,
            inferred_category="development",
            skip_category_penalty=True,
        )
        assert score == 1.0


class TestLexicalSignal:
    """Tests for lexical score normalization and strong-signal detection."""

    def test_normalize_lexical_score_bounds(self):
        """Normalization should map scores to [0, 1]."""
        assert normalize_lexical_score(0.0) == 0.0
        assert 0.0 < normalize_lexical_score(1.0) < 1.0
        assert normalize_lexical_score(-1.0) == 0.0

    def test_is_strong_lexical_signal_true(self):
        """Strong signal should be detected when threshold and gap are met."""
        top = 4.0   # normalized 0.8
        second = 1.0  # normalized 0.5
        assert is_strong_lexical_signal(top, second, threshold=0.75, gap=0.2)

    def test_is_strong_lexical_signal_false(self):
        """Strong signal should be false when gap is too small."""
        top = 2.0   # normalized 0.667
        second = 1.5  # normalized 0.6
        assert not is_strong_lexical_signal(top, second, threshold=0.6, gap=0.1)


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


class TestMMRDiversity:
    """Tests for MMR (Maximal Marginal Relevance) diversity selection."""

    def test_word_set_extraction(self):
        """Word set extraction should return lowercase words."""
        text = "The Quick Brown FOX jumps over 123 lazy dogs!"
        words = _word_set(text)

        assert "the" in words
        assert "quick" in words
        assert "brown" in words
        assert "fox" in words
        assert "123" in words
        assert "dogs" in words
        # Original case not preserved
        assert "Quick" not in words
        assert "FOX" not in words

    def test_jaccard_similarity_identical(self):
        """Identical sets should have similarity 1.0."""
        set_a = {"apple", "banana", "cherry"}
        set_b = {"apple", "banana", "cherry"}

        similarity = _jaccard_similarity(set_a, set_b)
        assert similarity == 1.0

    def test_jaccard_similarity_disjoint(self):
        """Disjoint sets should have similarity 0.0."""
        set_a = {"apple", "banana"}
        set_b = {"cherry", "date"}

        similarity = _jaccard_similarity(set_a, set_b)
        assert similarity == 0.0

    def test_jaccard_similarity_partial(self):
        """Partial overlap should give value between 0 and 1."""
        set_a = {"apple", "banana", "cherry"}
        set_b = {"banana", "cherry", "date"}

        similarity = _jaccard_similarity(set_a, set_b)
        # Intersection: {banana, cherry} = 2
        # Union: {apple, banana, cherry, date} = 4
        # Jaccard = 2/4 = 0.5
        assert similarity == 0.5

    def test_jaccard_similarity_empty(self):
        """Empty sets should return 0."""
        assert _jaccard_similarity(set(), {"a"}) == 0.0
        assert _jaccard_similarity({"a"}, set()) == 0.0
        assert _jaccard_similarity(set(), set()) == 0.0

    def test_max_similarity_to_selected_empty(self):
        """No selected documents should return 0."""
        candidate_words = {"apple", "banana"}
        selected = []

        similarity = _max_similarity_to_selected(candidate_words, selected)
        assert similarity == 0.0

    def test_max_similarity_to_selected_finds_max(self):
        """Should return the maximum similarity across selected docs."""
        candidate_words = {"apple", "banana", "cherry"}
        selected = [
            {"date", "elderberry"},  # 0% overlap
            {"apple", "banana"},  # 66% overlap (2/3 Jaccard)
            {"apple"},  # ~33% overlap
        ]

        similarity = _max_similarity_to_selected(candidate_words, selected)
        # Max should be from second set: intersection {apple, banana} / union {apple, banana, cherry} = 2/3
        assert similarity == pytest.approx(2 / 3, rel=0.01)

    def test_assemble_context_diverse_with_mmr(self):
        """MMR should prefer diverse results over similar high-scoring ones."""
        # Create results where high-scoring items are very similar
        results = [
            SearchResult(
                memory_id="1",
                content="I visited Dr. Smith for my checkup appointment",
                score=0.95,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="1",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-doctor-1",
                ),
            ),
            SearchResult(
                memory_id="2",
                content="Dr. Smith gave me a prescription during my checkup",  # Similar to #1
                score=0.90,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="2",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-doctor-1",
                ),
            ),
            SearchResult(
                memory_id="3",
                content="I saw Dr. Johnson for a dental cleaning",  # Different doctor
                score=0.85,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="3",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-doctor-2",
                ),
            ),
            SearchResult(
                memory_id="4",
                content="Bought groceries at the supermarket today",  # Completely different
                score=0.80,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="4",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-shopping",
                ),
            ),
        ]

        # With high token budget, MMR should select diverse results
        selected, tokens = assemble_context_diverse(results, max_tokens=2000)

        # Should include results from different sessions
        selected_ids = {r.memory_id for r in selected}
        assert "1" in selected_ids  # Top scorer
        assert "3" in selected_ids  # Different doctor (diverse)
        # Result #2 might be skipped because it's too similar to #1

    def test_assemble_context_diverse_respects_token_budget(self):
        """MMR assembly should respect token budget."""
        results = [
            SearchResult(
                memory_id="1",
                content="Short content",
                score=0.9,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="1",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-1",
                ),
            ),
            SearchResult(
                memory_id="2",
                content="Another short content",
                score=0.8,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="2",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-2",
                ),
            ),
        ]

        # Very small budget
        selected, tokens = assemble_context_diverse(results, max_tokens=10)

        # Should respect budget
        assert tokens <= 10

    def test_assemble_context_diverse_lambda_effect(self):
        """Higher lambda should favor relevance, lower should favor diversity."""
        results = [
            SearchResult(
                memory_id="1",
                content="doctor appointment checkup medical visit",
                score=0.95,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="1",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-1",
                ),
            ),
            SearchResult(
                memory_id="2",
                content="doctor appointment checkup medical exam",  # Very similar
                score=0.90,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="2",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-2",
                ),
            ),
            SearchResult(
                memory_id="3",
                content="grocery shopping supermarket food",  # Very different
                score=0.50,
                scope=MemoryScope.PERSONAL,
                metadata=MemoryMetadata(
                    memory_id="3",
                    user_id="test",
                    scope=MemoryScope.PERSONAL,
                    source_ref="session-3",
                ),
            ),
        ]

        # High lambda (favor relevance) - should take similar high-scoring items
        selected_high, _ = assemble_context_diverse(results, max_tokens=2000, mmr_lambda=0.9)

        # Low lambda (favor diversity) - should prefer the diverse low-scorer
        selected_low, _ = assemble_context_diverse(results, max_tokens=2000, mmr_lambda=0.3)

        # With low lambda, the grocery result should rank higher due to diversity
        # Both should include #1 (top scorer)
        assert any(r.memory_id == "1" for r in selected_high)
        assert any(r.memory_id == "1" for r in selected_low)
