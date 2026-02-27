#!/usr/bin/env python3
"""
LongMemEval End-to-End Benchmark Runner for CEMS

Full end-to-end eval with LLM answer generation and LLM-as-judge scoring.
Produces scores directly comparable to the Mastra leaderboard.

Pipeline:
    S dataset (40 sessions/question) → Bulk ingest → Search → LLM answer → LLM judge → Accuracy

Usage:
    python -m cems.eval.longmemeval_e2e --questions 5   # Quick test
    python -m cems.eval.longmemeval_e2e --questions 500  # Full eval (~$35, ~45min)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# Reuse shared infrastructure from retrieval-only eval
from cems.eval.longmemeval import (
    CEMSEvalClient,
    collect_all_sessions,
    format_session_content,
)

# Configuration
DEFAULT_API_URL = os.getenv("CEMS_API_URL", "http://localhost:8765")
DEFAULT_API_KEY = os.getenv("CEMS_API_KEY", "")
DATA_DIR = Path(__file__).parent / "data"

# Dataset URLs
DATASET_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
}

# Default models
DEFAULT_READER_MODEL = "openai/gpt-4o"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o"

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# --- Type-specific judge prompts ---

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating whether a model's response "
    "correctly answers a question based on conversation history. "
    "You will be given the question, the correct answer, and the model's response. "
    "Evaluate carefully and respond with YES or NO followed by a brief explanation."
)

JUDGE_PROMPTS = {
    "standard": (
        "Question: {question}\n"
        "Correct Answer: {answer}\n"
        "Model Response: {response}\n\n"
        "Does the model's response contain the correct answer? "
        "The response doesn't need to be word-for-word identical, but must convey "
        "the same factual information.\n"
        "Answer YES or NO, then explain briefly."
    ),
    "temporal-reasoning": (
        "Question: {question}\n"
        "Correct Answer: {answer}\n"
        "Model Response: {response}\n\n"
        "Does the model's response contain the correct answer? "
        "For temporal/date questions, allow minor discrepancies (off by one day or "
        "slight time differences). The core temporal fact must be correct.\n"
        "Answer YES or NO, then explain briefly."
    ),
    "knowledge-update": (
        "Question: {question}\n"
        "Correct Answer (most recent): {answer}\n"
        "Model Response: {response}\n\n"
        "Does the model's response contain the UPDATED (most recent) answer? "
        "It is acceptable if both old and new information appear, as long as "
        "the updated answer is clearly present. The model should ideally indicate "
        "that the information was updated.\n"
        "Answer YES or NO, then explain briefly."
    ),
    "single-session-preference": (
        "Question: {question}\n"
        "Correct Answer: {answer}\n"
        "Model Response: {response}\n\n"
        "Does the model's response correctly reflect the user's preference or "
        "personal information? Be lenient in evaluation — partial matches and "
        "reasonable paraphrasing are acceptable as long as the core preference "
        "is captured.\n"
        "Answer YES or NO, then explain briefly."
    ),
    "abstention": (
        "Question: {question}\n"
        "Note: This information was NEVER discussed in the conversation history.\n"
        "Model Response: {response}\n\n"
        "Did the model correctly identify that this information is not available "
        "in the conversation history? The model should NOT fabricate an answer. "
        "A correct response would indicate uncertainty or state that the information "
        "was not discussed.\n"
        "Answer YES or NO, then explain briefly."
    ),
}

# Map question types to judge prompt keys
QUESTION_TYPE_TO_JUDGE = {
    "single-session-user": "standard",
    "single-session-assistant": "standard",
    "single-session-preference": "single-session-preference",
    "multi-session": "standard",
    "knowledge-update": "knowledge-update",
    "temporal-reasoning": "temporal-reasoning",
    "abstention": "abstention",
}

READER_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to extensive conversation history. "
    "Use the provided context from past conversations to answer the user's question. "
    "If the information is not available in the context, say that you don't have "
    "that information rather than guessing. Be concise and direct in your answer."
)


# --- Data structures ---


@dataclass
class E2EResult:
    """Result for a single end-to-end question."""
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    generated_answer: str
    judge_verdict: bool
    judge_explanation: str
    retrieved_session_ids: list[str]
    correct_session_ids: list[str]
    recall_any: bool
    search_time_ms: int
    answer_time_ms: int
    judge_time_ms: int
    mode_used: str
    num_results: int


@dataclass
class E2ESummary:
    """Summary statistics for the end-to-end eval run."""
    total_questions: int = 0
    correct_count: int = 0
    total_search_time_ms: int = 0
    total_answer_time_ms: int = 0
    total_judge_time_ms: int = 0
    recall_any_count: int = 0
    by_type: dict[str, dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.total_questions if self.total_questions else 0

    @property
    def macro_accuracy(self) -> float:
        """Average of per-type accuracies. This is the leaderboard metric."""
        if not self.by_type:
            return 0
        type_accuracies = []
        for stats in self.by_type.values():
            total = stats["total"]
            if total > 0:
                type_accuracies.append(stats["correct"] / total)
        return sum(type_accuracies) / len(type_accuracies) if type_accuracies else 0

    @property
    def avg_search_time_ms(self) -> float:
        return self.total_search_time_ms / self.total_questions if self.total_questions else 0

    @property
    def avg_answer_time_ms(self) -> float:
        return self.total_answer_time_ms / self.total_questions if self.total_questions else 0

    @property
    def avg_judge_time_ms(self) -> float:
        return self.total_judge_time_ms / self.total_questions if self.total_questions else 0


# --- LLM functions ---


class LLMClient:
    """Thin OpenRouter client for eval (doesn't depend on CEMS internals)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
            )
        self._client = httpx.Client(timeout=120)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str = DEFAULT_READER_MODEL,
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> str:
        """Call OpenRouter chat completion API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/cems",
                "X-Title": "CEMS LongMemEval",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


def format_context(search_results: list[dict], max_chars: int = 16000) -> str:
    """Format search results into context for the reader LLM.

    Args:
        search_results: List of memory dicts from CEMS search API
        max_chars: Maximum characters for context (prevent token overflow)

    Returns:
        Formatted context string
    """
    if not search_results:
        return "[No relevant conversation history found]"

    parts = []
    total_chars = 0

    for i, mem in enumerate(search_results):
        content = mem.get("content", "")
        source_ref = mem.get("source_ref", "") or ""

        # Extract a label from source_ref
        label = ""
        if source_ref.startswith("project:longmemeval:"):
            label = f"Session {source_ref.replace('project:longmemeval:', '')}"
        elif source_ref:
            label = source_ref

        entry = f"--- Memory {i+1}"
        if label:
            entry += f" ({label})"
        entry += f" ---\n{content}\n"

        if total_chars + len(entry) > max_chars:
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


def generate_answer(
    llm: LLMClient,
    question: str,
    context: str,
    model: str = DEFAULT_READER_MODEL,
) -> str:
    """Generate an answer using retrieved context."""
    prompt = (
        f"Here is relevant context from past conversations:\n\n"
        f"{context}\n\n"
        f"Based on the above context, please answer this question:\n"
        f"{question}"
    )

    return llm.complete(
        prompt=prompt,
        system=READER_SYSTEM_PROMPT,
        model=model,
        temperature=0,
        max_tokens=512,
    )


def judge_answer(
    llm: LLMClient,
    question: str,
    correct_answer: str,
    generated_answer: str,
    question_type: str,
    model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[bool, str]:
    """Judge whether the generated answer is correct.

    Returns:
        Tuple of (verdict: bool, explanation: str)
    """
    # Select the appropriate judge prompt template
    judge_key = QUESTION_TYPE_TO_JUDGE.get(question_type, "standard")
    prompt_template = JUDGE_PROMPTS[judge_key]

    prompt = prompt_template.format(
        question=question,
        answer=correct_answer,
        response=generated_answer,
    )

    response = llm.complete(
        prompt=prompt,
        system=JUDGE_SYSTEM_PROMPT,
        model=model,
        temperature=0,
        max_tokens=256,
    )

    # Parse verdict
    verdict, explanation = parse_judge_response(response)
    return verdict, explanation


def parse_judge_response(response: str) -> tuple[bool, str]:
    """Parse YES/NO verdict from judge response.

    Args:
        response: Raw judge response text

    Returns:
        Tuple of (verdict: bool, explanation: str)
    """
    if not response:
        return False, "Empty judge response"

    response_stripped = response.strip()
    upper = response_stripped.upper()

    # Check for clear YES/NO at the start
    if upper.startswith("YES"):
        return True, response_stripped
    if upper.startswith("NO"):
        return False, response_stripped

    # Check for YES/NO anywhere in first line
    first_line = upper.split("\n")[0]
    if "YES" in first_line and "NO" not in first_line:
        return True, response_stripped
    if "NO" in first_line and "YES" not in first_line:
        return False, response_stripped

    # Ambiguous — default to NO (conservative)
    return False, f"[AMBIGUOUS] {response_stripped}"


# --- Dataset functions ---


def download_dataset(variant: str = "s", force: bool = False) -> Path:
    """Download LongMemEval dataset variant.

    Args:
        variant: Dataset variant ("oracle" or "s")
        force: Force re-download even if file exists

    Returns:
        Path to downloaded file
    """
    if variant not in DATASET_URLS:
        raise ValueError(f"Unknown variant: {variant}. Use: {list(DATASET_URLS.keys())}")

    url = DATASET_URLS[variant]
    filename = url.split("/")[-1]
    data_file = DATA_DIR / filename

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if data_file.exists() and not force:
        size_mb = data_file.stat().st_size / 1024 / 1024
        print(f"Using cached data: {data_file} ({size_mb:.1f} MB)")
        return data_file

    print(f"Downloading LongMemEval {variant} variant...")
    print(f"  URL: {url}")

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        # Stream download for large files
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(data_file, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

    size_mb = data_file.stat().st_size / 1024 / 1024
    print(f"\n  Saved to: {data_file} ({size_mb:.1f} MB)")
    return data_file


def load_questions(data_file: Path, limit: int | None = None) -> list[dict]:
    """Load LongMemEval questions from JSON file.

    Includes abstention questions (unlike retrieval-only eval) since
    the end-to-end eval tests whether the model correctly abstains.

    Args:
        data_file: Path to dataset JSON
        limit: Max questions to load (None = all)

    Returns:
        List of question dicts
    """
    print(f"Loading data from {data_file}...")

    with open(data_file) as f:
        data = json.load(f)

    print(f"  Total questions: {len(data)}")

    if limit:
        data = data[:limit]
        print(f"  Using first {limit} questions")

    # Count by type
    by_type: dict[str, int] = {}
    for q in data:
        qtype = q.get("question_type", "unknown")
        by_type[qtype] = by_type.get(qtype, 0) + 1

    print(f"  Question types:")
    for qtype, count in sorted(by_type.items()):
        print(f"    - {qtype}: {count}")

    return data


# --- Main eval ---


def run_e2e_eval(
    cems_client: CEMSEvalClient,
    llm: LLMClient,
    questions: list[dict],
    reader_model: str = DEFAULT_READER_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    verbose: bool = False,
    search_limit: int = 10,
) -> tuple[list[E2EResult], E2ESummary]:
    """Run the end-to-end evaluation.

    Three-phase approach:
    1. Collect all unique sessions → bulk ingest
    2. For each question: search → generate answer → judge answer
    3. Compute per-type and macro accuracy

    Args:
        cems_client: CEMS API client for ingestion and search
        llm: LLM client for answer generation and judging
        questions: List of question dicts
        reader_model: Model for answer generation
        judge_model: Model for judging
        verbose: Print detailed output
        search_limit: Number of search results to retrieve

    Returns:
        Tuple of (results list, summary)
    """
    results: list[E2EResult] = []
    summary = E2ESummary()

    # Phase 1: Bulk ingest all sessions
    print("\nPhase 1: Collecting all unique sessions...")
    all_sessions = collect_all_sessions(questions)
    print(f"  Found {len(all_sessions)} unique sessions across {len(questions)} questions")

    if all_sessions:
        print("\nPhase 2: Bulk ingesting all sessions...")
        ingest_start = time.time()

        memories_to_add = list(all_sessions.values())
        cems_client.add_memories_batch(memories_to_add)

        ingest_elapsed = time.time() - ingest_start
        per_session_ms = ingest_elapsed / len(memories_to_add) * 1000 if memories_to_add else 0
        print(
            f"  Ingested {len(memories_to_add)} sessions in {ingest_elapsed:.1f}s "
            f"({per_session_ms:.0f}ms/session)"
        )

    # Phase 3: Run end-to-end eval
    print(f"\nPhase 3: Running end-to-end eval ({reader_model} + {judge_model})...")

    for i, q in enumerate(questions):
        qid = q["question_id"]
        qtype = q.get("question_type", "unknown")
        question = q["question"]
        answer = q.get("answer", "")
        session_ids = q.get("haystack_session_ids", [])
        correct_ids = q.get("answer_session_ids", [])

        # Step A: Search CEMS
        search_result = cems_client.search(question, limit=search_limit)
        search_ms = search_result.get("_elapsed_ms", 0)
        mode_used = search_result.get("mode", "unknown")

        # Extract retrieved session IDs and format context
        retrieved_ids: list[str] = []
        search_memories: list[dict] = []
        if search_result.get("success"):
            for mem in search_result.get("results", []):
                search_memories.append(mem)
                source_ref = mem.get("source_ref", "") or ""
                if source_ref.startswith("project:longmemeval:"):
                    sid = source_ref.replace("project:longmemeval:", "")
                    retrieved_ids.append(sid)
                else:
                    tags = mem.get("tags", []) or []
                    for tag in tags:
                        if tag in session_ids:
                            retrieved_ids.append(tag)
                            break

        # Calculate retrieval recall
        correct_set = set(correct_ids)
        retrieved_set = set(retrieved_ids[:5])
        recall_any = bool(correct_set & retrieved_set)

        # Step B: Generate answer
        context = format_context(search_memories)
        answer_start = time.time()
        try:
            generated = generate_answer(llm, question, context, model=reader_model)
        except Exception as e:
            generated = f"[Error generating answer: {e}]"
            if verbose:
                print(f"  Answer generation error: {e}", file=sys.stderr)
        answer_ms = int((time.time() - answer_start) * 1000)

        # Step C: Judge answer
        judge_start = time.time()
        try:
            verdict, explanation = judge_answer(
                llm, question, answer, generated, qtype, model=judge_model
            )
        except Exception as e:
            verdict, explanation = False, f"[Error judging: {e}]"
            if verbose:
                print(f"  Judge error: {e}", file=sys.stderr)
        judge_ms = int((time.time() - judge_start) * 1000)

        # Record result
        result = E2EResult(
            question_id=qid,
            question_type=qtype,
            question=question,
            ground_truth=str(answer),
            generated_answer=generated,
            judge_verdict=verdict,
            judge_explanation=explanation,
            retrieved_session_ids=retrieved_ids[:5],
            correct_session_ids=correct_ids,
            recall_any=recall_any,
            search_time_ms=search_ms,
            answer_time_ms=answer_ms,
            judge_time_ms=judge_ms,
            mode_used=mode_used,
            num_results=len(search_memories),
        )
        results.append(result)

        # Update summary
        summary.total_questions += 1
        summary.total_search_time_ms += search_ms
        summary.total_answer_time_ms += answer_ms
        summary.total_judge_time_ms += judge_ms

        if verdict:
            summary.correct_count += 1
        if recall_any:
            summary.recall_any_count += 1

        if qtype not in summary.by_type:
            summary.by_type[qtype] = {"total": 0, "correct": 0, "recall_any": 0}
        summary.by_type[qtype]["total"] += 1
        if verdict:
            summary.by_type[qtype]["correct"] += 1
        if recall_any:
            summary.by_type[qtype]["recall_any"] += 1

        # Progress output
        status = "+" if verdict else "-"
        print(
            f"  [{i+1}/{len(questions)}] {qtype}: {status} "
            f"(search {search_ms}ms, answer {answer_ms}ms, judge {judge_ms}ms)"
        )

        if verbose:
            print(f"    Q: {question[:80]}...")
            print(f"    A: {generated[:80]}...")
            print(f"    J: {explanation[:80]}...")
            if not recall_any:
                print(f"    Retrieval miss: expected {correct_ids}, got {retrieved_ids[:5]}")

    return results, summary


def print_e2e_summary(summary: E2ESummary, reader_model: str, judge_model: str) -> None:
    """Print end-to-end evaluation summary."""
    print("\n" + "=" * 60)
    print("END-TO-END EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nModels: reader={reader_model}, judge={judge_model}")
    print(f"\nOverall ({summary.total_questions} questions):")
    print(f"  Accuracy:        {summary.accuracy:.1%} ({summary.correct_count}/{summary.total_questions})")
    print(f"  Macro accuracy:  {summary.macro_accuracy:.1%} (leaderboard metric)")
    print(f"  Retrieval hit:   {summary.recall_any_count}/{summary.total_questions} ({summary.recall_any_count/summary.total_questions:.1%})")
    print(f"\n  Avg search:  {summary.avg_search_time_ms:.0f}ms")
    print(f"  Avg answer:  {summary.avg_answer_time_ms:.0f}ms")
    print(f"  Avg judge:   {summary.avg_judge_time_ms:.0f}ms")

    print(f"\nBy Question Type:")
    for qtype, stats in sorted(summary.by_type.items()):
        total = stats["total"]
        correct = stats["correct"]
        recall = stats["recall_any"]
        acc = correct / total if total else 0
        rec = recall / total if total else 0
        print(f"  {qtype}:")
        print(f"    Accuracy:  {acc:.1%} ({correct}/{total})")
        print(f"    Retrieval: {rec:.1%} ({recall}/{total})")

    # Leaderboard comparison
    print(f"\nLeaderboard Comparison (macro accuracy):")
    leaderboard = [
        ("Mastra OM (gpt-4o)", 84.23),
        ("CEMS (this run)", summary.macro_accuracy * 100),
        ("Supermemory (gpt-4o)", 81.6),
        ("Mastra RAG (topK=20)", 80.0),
        ("Zep (gpt-4o)", 71.2),
        ("GPT-4o full context", 60.6),
    ]
    # Sort descending
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    for name, score in leaderboard:
        marker = " <-- us" if "this run" in name else ""
        print(f"  {name:30s} {score:5.1f}%{marker}")


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Run LongMemEval end-to-end benchmark on CEMS"
    )
    parser.add_argument(
        "--questions", "-n", type=int, default=500,
        help="Number of questions to evaluate (default: 500)"
    )
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL,
        help=f"CEMS API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY,
        help="CEMS API key (default: from CEMS_API_KEY env)"
    )
    parser.add_argument(
        "--reader-model", default=DEFAULT_READER_MODEL,
        help=f"Model for answer generation (default: {DEFAULT_READER_MODEL})"
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=f"Model for judging (default: {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--dataset", choices=["oracle", "s"], default="s",
        help="Dataset variant (default: s)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file for detailed results (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Don't delete eval memories after run"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Force re-download of dataset"
    )
    parser.add_argument(
        "--no-clean-stale", action="store_true",
        help="Don't clean up stale data from previous eval runs"
    )
    parser.add_argument(
        "--search-limit", type=int, default=10,
        help="Number of search results to retrieve per question (default: 10)"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key. Set CEMS_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    # Check for OpenRouter key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY env var required for LLM calls", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("LONGMEMEVAL END-TO-END BENCHMARK FOR CEMS")
    print("=" * 60)
    print(f"API URL:      {args.api_url}")
    print(f"Dataset:      {args.dataset} variant")
    print(f"Questions:    {args.questions}")
    print(f"Reader model: {args.reader_model}")
    print(f"Judge model:  {args.judge_model}")

    # Initialize clients
    cems_client = CEMSEvalClient(args.api_url, args.api_key)
    llm = LLMClient()

    # Health check
    print("\nChecking CEMS connection...")
    if not cems_client.health_check():
        print("Error: Cannot connect to CEMS. Is it running?", file=sys.stderr)
        sys.exit(1)
    print("  Connected!")

    # Clean up stale data
    if not args.no_clean_stale:
        print("\nCleaning up stale eval data from previous runs...")
        cems_client.cleanup_stale_eval_data()

    # Download/load data
    try:
        data_file = download_dataset(variant=args.dataset, force=args.download)
        questions = load_questions(data_file, limit=args.questions)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not questions:
        print("No questions to evaluate!", file=sys.stderr)
        sys.exit(1)

    # Run eval
    print(f"\nStarting end-to-end evaluation...")
    start_time = time.time()

    try:
        results, summary = run_e2e_eval(
            cems_client=cems_client,
            llm=llm,
            questions=questions,
            reader_model=args.reader_model,
            judge_model=args.judge_model,
            verbose=args.verbose,
            search_limit=args.search_limit,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        summary = E2ESummary()
        results = []

    elapsed = time.time() - start_time

    # Print summary
    if summary.total_questions > 0:
        print_e2e_summary(summary, args.reader_model, args.judge_model)
        print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Estimate cost
        answer_calls = summary.total_questions
        judge_calls = summary.total_questions
        est_cost = answer_calls * 0.04 + judge_calls * 0.02  # rough GPT-4o pricing
        print(f"Estimated cost: ~${est_cost:.2f}")

    # Cleanup
    if not args.no_cleanup:
        print(f"\nCleaning up eval memories...")
        deleted = cems_client.cleanup_eval_memories()
        print(f"  Deleted {deleted} memories")

    # Save results
    if args.output and results:
        output_data = {
            "summary": {
                "total_questions": summary.total_questions,
                "accuracy": summary.accuracy,
                "macro_accuracy": summary.macro_accuracy,
                "correct_count": summary.correct_count,
                "recall_any_count": summary.recall_any_count,
                "avg_search_time_ms": summary.avg_search_time_ms,
                "avg_answer_time_ms": summary.avg_answer_time_ms,
                "avg_judge_time_ms": summary.avg_judge_time_ms,
                "by_type": summary.by_type,
                "elapsed_seconds": elapsed,
                "reader_model": args.reader_model,
                "judge_model": args.judge_model,
                "dataset": args.dataset,
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "generated_answer": r.generated_answer,
                    "judge_verdict": r.judge_verdict,
                    "judge_explanation": r.judge_explanation,
                    "retrieved_session_ids": r.retrieved_session_ids,
                    "correct_session_ids": r.correct_session_ids,
                    "recall_any": r.recall_any,
                    "search_time_ms": r.search_time_ms,
                    "answer_time_ms": r.answer_time_ms,
                    "judge_time_ms": r.judge_time_ms,
                    "mode_used": r.mode_used,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
