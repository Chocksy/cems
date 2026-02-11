"""Tests for LongMemEval end-to-end evaluation module."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestParseJudgeResponse:
    """Tests for parse_judge_response()."""

    def test_yes_at_start(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, explanation = parse_judge_response("YES. The answer matches.")
        assert verdict is True
        assert "YES" in explanation

    def test_no_at_start(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, explanation = parse_judge_response("NO. The answer is wrong.")
        assert verdict is False
        assert "NO" in explanation

    def test_yes_case_insensitive(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, _ = parse_judge_response("yes, the response is correct")
        assert verdict is True

    def test_no_case_insensitive(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, _ = parse_judge_response("no, the information is incorrect")
        assert verdict is False

    def test_yes_in_first_line(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, _ = parse_judge_response(
            "The answer is YES.\nThe model correctly identified the preference."
        )
        assert verdict is True

    def test_no_in_first_line(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, _ = parse_judge_response(
            "The answer is NO.\nThe model fabricated information."
        )
        assert verdict is False

    def test_empty_response_defaults_to_no(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, explanation = parse_judge_response("")
        assert verdict is False
        assert "Empty" in explanation

    def test_ambiguous_defaults_to_no(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, explanation = parse_judge_response(
            "The model provided a partial answer that is somewhat correct."
        )
        assert verdict is False
        assert "AMBIGUOUS" in explanation

    def test_whitespace_stripped(self):
        from cems.eval.longmemeval_e2e import parse_judge_response

        verdict, _ = parse_judge_response("  YES  \n  The answer is correct.  ")
        assert verdict is True


class TestFormatContext:
    """Tests for format_context()."""

    def test_formats_memories_with_session_labels(self):
        from cems.eval.longmemeval_e2e import format_context

        results = [
            {
                "content": "user: I like pizza\nassistant: Good choice!",
                "source_ref": "project:longmemeval:session_123",
            },
            {
                "content": "user: My dog is named Rex",
                "source_ref": "project:longmemeval:session_456",
            },
        ]

        context = format_context(results)
        assert "Memory 1 (Session session_123)" in context
        assert "Memory 2 (Session session_456)" in context
        assert "I like pizza" in context
        assert "dog is named Rex" in context

    def test_empty_results(self):
        from cems.eval.longmemeval_e2e import format_context

        context = format_context([])
        assert "No relevant" in context

    def test_truncates_at_max_chars(self):
        from cems.eval.longmemeval_e2e import format_context

        results = [
            {"content": "A" * 5000, "source_ref": "project:longmemeval:s1"},
            {"content": "B" * 5000, "source_ref": "project:longmemeval:s2"},
            {"content": "C" * 5000, "source_ref": "project:longmemeval:s3"},
            {"content": "D" * 5000, "source_ref": "project:longmemeval:s4"},
        ]

        context = format_context(results, max_chars=12000)
        # Should include first 2 entries but not all 4
        assert "Memory 1" in context
        assert "Memory 2" in context
        # 3rd might be partial but 4th definitely cut
        assert "DDDD" not in context

    def test_handles_missing_source_ref(self):
        from cems.eval.longmemeval_e2e import format_context

        results = [
            {"content": "Some memory content", "source_ref": None},
        ]

        context = format_context(results)
        assert "Memory 1" in context
        assert "Some memory content" in context


class TestJudgePromptSelection:
    """Tests for type-specific judge prompt selection."""

    def test_standard_types_use_standard_prompt(self):
        from cems.eval.longmemeval_e2e import QUESTION_TYPE_TO_JUDGE

        assert QUESTION_TYPE_TO_JUDGE["single-session-user"] == "standard"
        assert QUESTION_TYPE_TO_JUDGE["single-session-assistant"] == "standard"
        assert QUESTION_TYPE_TO_JUDGE["multi-session"] == "standard"

    def test_temporal_uses_temporal_prompt(self):
        from cems.eval.longmemeval_e2e import QUESTION_TYPE_TO_JUDGE

        assert QUESTION_TYPE_TO_JUDGE["temporal-reasoning"] == "temporal-reasoning"

    def test_knowledge_update_uses_update_prompt(self):
        from cems.eval.longmemeval_e2e import QUESTION_TYPE_TO_JUDGE

        assert QUESTION_TYPE_TO_JUDGE["knowledge-update"] == "knowledge-update"

    def test_preference_uses_preference_prompt(self):
        from cems.eval.longmemeval_e2e import QUESTION_TYPE_TO_JUDGE

        assert QUESTION_TYPE_TO_JUDGE["single-session-preference"] == "single-session-preference"

    def test_all_judge_prompts_have_placeholders(self):
        from cems.eval.longmemeval_e2e import JUDGE_PROMPTS

        for key, template in JUDGE_PROMPTS.items():
            if key != "abstention":
                assert "{question}" in template, f"{key} missing {{question}}"
                assert "{answer}" in template, f"{key} missing {{answer}}"
                assert "{response}" in template, f"{key} missing {{response}}"


class TestE2ESummary:
    """Tests for E2ESummary scoring."""

    def test_accuracy(self):
        from cems.eval.longmemeval_e2e import E2ESummary

        s = E2ESummary(total_questions=10, correct_count=8)
        assert s.accuracy == 0.8

    def test_accuracy_empty(self):
        from cems.eval.longmemeval_e2e import E2ESummary

        s = E2ESummary()
        assert s.accuracy == 0

    def test_macro_accuracy_single_type(self):
        from cems.eval.longmemeval_e2e import E2ESummary

        s = E2ESummary(
            total_questions=10,
            correct_count=8,
            by_type={"temporal-reasoning": {"total": 10, "correct": 8, "recall_any": 9}},
        )
        assert s.macro_accuracy == 0.8

    def test_macro_accuracy_multiple_types(self):
        from cems.eval.longmemeval_e2e import E2ESummary

        s = E2ESummary(
            total_questions=20,
            correct_count=14,
            by_type={
                "temporal-reasoning": {"total": 10, "correct": 8, "recall_any": 9},
                "multi-session": {"total": 10, "correct": 6, "recall_any": 8},
            },
        )
        # Macro = average of (8/10=0.8, 6/10=0.6) = 0.7
        assert abs(s.macro_accuracy - 0.7) < 0.001

    def test_macro_accuracy_differs_from_micro(self):
        """Macro accuracy weights all types equally, unlike micro (overall)."""
        from cems.eval.longmemeval_e2e import E2ESummary

        s = E2ESummary(
            total_questions=110,
            correct_count=95,
            by_type={
                "type_a": {"total": 100, "correct": 90, "recall_any": 100},  # 90%
                "type_b": {"total": 10, "correct": 5, "recall_any": 10},   # 50%
            },
        )
        # Micro = 95/110 = 86.4%
        assert abs(s.accuracy - 95 / 110) < 0.001
        # Macro = average of (90/100, 5/10) = average of (0.9, 0.5) = 0.7
        assert abs(s.macro_accuracy - 0.7) < 0.001

    def test_macro_accuracy_empty(self):
        from cems.eval.longmemeval_e2e import E2ESummary

        s = E2ESummary()
        assert s.macro_accuracy == 0


class TestJudgeAnswer:
    """Tests for judge_answer() with mocked LLM."""

    def test_judge_yes(self):
        from cems.eval.longmemeval_e2e import judge_answer

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "YES. The response correctly states GPS issues."

        verdict, explanation = judge_answer(
            mock_llm,
            question="What was wrong with my car?",
            correct_answer="GPS system not functioning correctly",
            generated_answer="Your car had GPS issues after the first service.",
            question_type="single-session-user",
        )

        assert verdict is True
        assert "GPS" in explanation

    def test_judge_no(self):
        from cems.eval.longmemeval_e2e import judge_answer

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "NO. The model mentioned brakes, not GPS."

        verdict, explanation = judge_answer(
            mock_llm,
            question="What was wrong with my car?",
            correct_answer="GPS system not functioning correctly",
            generated_answer="Your brakes needed replacing.",
            question_type="single-session-user",
        )

        assert verdict is False

    def test_judge_uses_correct_prompt_for_temporal(self):
        from cems.eval.longmemeval_e2e import judge_answer

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "YES"

        judge_answer(
            mock_llm,
            question="When did X happen?",
            correct_answer="March 15",
            generated_answer="Around March 15th",
            question_type="temporal-reasoning",
        )

        # Check the prompt sent to LLM contains temporal-specific text
        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt") or call_args[0][0]
        assert "off by one" in prompt.lower() or "date" in prompt.lower()

    def test_judge_uses_correct_prompt_for_knowledge_update(self):
        from cems.eval.longmemeval_e2e import judge_answer

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "YES"

        judge_answer(
            mock_llm,
            question="What's my phone?",
            correct_answer="iPhone 15",
            generated_answer="You recently upgraded to iPhone 15",
            question_type="knowledge-update",
        )

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt") or call_args[0][0]
        assert "UPDATED" in prompt


class TestGenerateAnswer:
    """Tests for generate_answer() with mocked LLM."""

    def test_generates_with_context(self):
        from cems.eval.longmemeval_e2e import generate_answer

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Your favorite color is blue."

        result = generate_answer(
            mock_llm,
            question="What is my favorite color?",
            context="--- Memory 1 ---\nuser: I love the color blue\nassistant: Nice choice!\n",
        )

        assert result == "Your favorite color is blue."
        # Verify context was included in the prompt
        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt") or call_args[0][0]
        assert "blue" in prompt

    def test_uses_temperature_zero(self):
        from cems.eval.longmemeval_e2e import generate_answer

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Answer."

        generate_answer(mock_llm, "Q?", "Context")

        call_args = mock_llm.complete.call_args
        assert call_args.kwargs.get("temperature") == 0


class TestDownloadDataset:
    """Tests for download_dataset()."""

    def test_invalid_variant_raises(self):
        from cems.eval.longmemeval_e2e import download_dataset

        with pytest.raises(ValueError, match="Unknown variant"):
            download_dataset(variant="invalid")

    def test_dataset_urls_valid(self):
        from cems.eval.longmemeval_e2e import DATASET_URLS

        assert "oracle" in DATASET_URLS
        assert "s" in DATASET_URLS
        assert "longmemeval_oracle" in DATASET_URLS["oracle"]
        assert "longmemeval_s_cleaned" in DATASET_URLS["s"]


class TestLoadQuestions:
    """Tests for load_questions()."""

    def test_loads_with_limit(self, tmp_path):
        from cems.eval.longmemeval_e2e import load_questions

        data = [
            {"question_id": f"q{i}", "question_type": "multi-session", "question": f"Q{i}", "answer": f"A{i}"}
            for i in range(10)
        ]
        data_file = tmp_path / "test.json"
        data_file.write_text(json.dumps(data))

        result = load_questions(data_file, limit=3)
        assert len(result) == 3

    def test_loads_all_types_including_abstention(self, tmp_path):
        """E2E eval includes abstention questions (unlike retrieval-only)."""
        from cems.eval.longmemeval_e2e import load_questions

        data = [
            {"question_id": "q1", "question_type": "multi-session", "question": "Q1", "answer": "A1"},
            {"question_id": "q2_abs", "question_type": "abstention", "question": "Q2", "answer": ""},
        ]
        data_file = tmp_path / "test.json"
        data_file.write_text(json.dumps(data))

        result = load_questions(data_file)
        assert len(result) == 2  # Both included


class TestE2EResult:
    """Tests for E2EResult dataclass."""

    def test_creates_result(self):
        from cems.eval.longmemeval_e2e import E2EResult

        r = E2EResult(
            question_id="q1",
            question_type="temporal-reasoning",
            question="What happened first?",
            ground_truth="GPS broke",
            generated_answer="The GPS system malfunctioned",
            judge_verdict=True,
            judge_explanation="YES. Correct.",
            retrieved_session_ids=["s1", "s2"],
            correct_session_ids=["s1"],
            recall_any=True,
            search_time_ms=100,
            answer_time_ms=1200,
            judge_time_ms=800,
            mode_used="vector",
            num_results=5,
        )
        assert r.judge_verdict is True
        assert r.recall_any is True
        assert r.answer_time_ms == 1200
