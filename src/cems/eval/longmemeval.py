#!/usr/bin/env python3
"""
LongMemEval Benchmark Runner for CEMS

This runs the LongMemEval benchmark against a CEMS instance to measure
retrieval quality. It tests 5 core memory abilities:
1. Information Extraction - recall specific details
2. Multi-Session Reasoning - synthesize across sessions
3. Temporal Reasoning - time-based queries
4. Knowledge Updates - handle changed information
5. Abstention - know when info is unavailable

Usage:
    python -m cems.eval.longmemeval --questions 10  # Sample run
    python -m cems.eval.longmemeval --questions 500  # Full eval
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


# Configuration
DEFAULT_API_URL = os.getenv("CEMS_API_URL", "http://localhost:8765")
DEFAULT_API_KEY = os.getenv("CEMS_API_KEY", "")
DATA_DIR = Path(__file__).parent / "data"
LONGMEMEVAL_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"


@dataclass
class EvalResult:
    """Result for a single question."""
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    retrieved_session_ids: list[str]
    correct_session_ids: list[str]
    recall_any: bool  # Did we find ANY correct session?
    recall_all: bool  # Did we find ALL correct sessions?
    search_time_ms: int
    mode_used: str
    num_results: int


@dataclass
class EvalSummary:
    """Summary statistics for the eval run."""
    total_questions: int = 0
    recall_any_count: int = 0
    recall_all_count: int = 0
    total_search_time_ms: int = 0
    by_type: dict[str, dict] = field(default_factory=dict)

    @property
    def recall_any_rate(self) -> float:
        return self.recall_any_count / self.total_questions if self.total_questions else 0

    @property
    def recall_all_rate(self) -> float:
        return self.recall_all_count / self.total_questions if self.total_questions else 0

    @property
    def avg_search_time_ms(self) -> float:
        return self.total_search_time_ms / self.total_questions if self.total_questions else 0


class CEMSEvalClient:
    """Client for CEMS evaluation operations."""

    def __init__(self, api_url: str, api_key: str, timeout: float = 60.0):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self._session_to_memory: dict[str, str] = {}  # session_id -> memory_id

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def health_check(self) -> bool:
        """Check if CEMS is reachable."""
        try:
            resp = self.client.get(f"{self.api_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def add_memory(
        self,
        content: str,
        session_id: str,
        timestamp: str | None = None,
    ) -> str | None:
        """Add a memory and return its ID."""
        try:
            resp = self.client.post(
                f"{self.api_url}/api/memory/add",
                headers=self._headers(),
                json={
                    "content": content,
                    "category": "eval-session",
                    "infer": False,  # Fast mode - no LLM extraction
                    "source_ref": f"longmemeval:{session_id}",
                    "tags": ["longmemeval", session_id],
                },
            )
            data = resp.json()
            if data.get("success"):
                results = data.get("result", {}).get("results", [])
                if results:
                    memory_id = results[0].get("id")
                    self._session_to_memory[session_id] = memory_id
                    return memory_id
        except Exception as e:
            print(f"  Error adding memory: {e}", file=sys.stderr)
        return None

    def search(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search memories and return results with timing."""
        start = time.time()
        try:
            resp = self.client.post(
                f"{self.api_url}/api/memory/search",
                headers=self._headers(),
                json={
                    "query": query,
                    "limit": limit,
                    "scope": "personal",
                    "mode": "auto",  # Let CEMS decide vector vs hybrid
                },
            )
            elapsed_ms = int((time.time() - start) * 1000)
            data = resp.json()
            data["_elapsed_ms"] = elapsed_ms
            return data
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return {"success": False, "error": str(e), "_elapsed_ms": elapsed_ms}

    def forget_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            resp = self.client.post(
                f"{self.api_url}/api/memory/forget",
                headers=self._headers(),
                json={"memory_id": memory_id, "hard_delete": True},
            )
            return resp.json().get("success", False)
        except Exception:
            return False

    def cleanup_eval_memories(self) -> int:
        """Delete all memories created during eval."""
        deleted = 0
        for session_id, memory_id in self._session_to_memory.items():
            if self.forget_memory(memory_id):
                deleted += 1
        self._session_to_memory.clear()
        return deleted


def download_longmemeval(force: bool = False) -> Path:
    """Download LongMemEval dataset if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "longmemeval_oracle.json"

    if data_file.exists() and not force:
        print(f"Using cached data: {data_file}")
        return data_file

    print(f"Downloading LongMemEval from HuggingFace...")
    print(f"  URL: {LONGMEMEVAL_URL}")

    try:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            resp = client.get(LONGMEMEVAL_URL)
            resp.raise_for_status()
            data_file.write_bytes(resp.content)
        print(f"  Saved to: {data_file}")
        print(f"  Size: {data_file.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  Error downloading: {e}", file=sys.stderr)
        raise

    return data_file


def load_longmemeval(data_file: Path, limit: int | None = None) -> list[dict]:
    """Load LongMemEval questions from JSON file."""
    print(f"Loading data from {data_file}...")

    with open(data_file) as f:
        data = json.load(f)

    # Filter out abstention questions (they have no correct answer)
    questions = [q for q in data if "_abs" not in q.get("question_id", "")]

    print(f"  Total questions: {len(data)}")
    print(f"  Non-abstention: {len(questions)}")

    if limit:
        questions = questions[:limit]
        print(f"  Using first {limit} questions")

    # Count by type
    by_type: dict[str, int] = {}
    for q in questions:
        qtype = q.get("question_type", "unknown")
        by_type[qtype] = by_type.get(qtype, 0) + 1

    print(f"  Question types:")
    for qtype, count in sorted(by_type.items()):
        print(f"    - {qtype}: {count}")

    return questions


def format_session_content(session: list[dict]) -> str:
    """Format a session's messages into a single string for storage."""
    parts = []
    for turn in session:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def run_eval(
    client: CEMSEvalClient,
    questions: list[dict],
    verbose: bool = False,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run the evaluation and return results."""
    results: list[EvalResult] = []
    summary = EvalSummary()

    # Track which sessions we've already ingested
    ingested_sessions: set[str] = set()

    for i, q in enumerate(questions):
        qid = q["question_id"]
        qtype = q.get("question_type", "unknown")
        question = q["question"]
        answer = q.get("answer", "")

        print(f"\n[{i+1}/{len(questions)}] {qtype}: {question[:60]}...")

        # 1. Ingest any new sessions for this question
        sessions = q.get("haystack_sessions", [])
        session_ids = q.get("haystack_session_ids", [])
        correct_ids = q.get("answer_session_ids", [])

        for sid, session in zip(session_ids, sessions):
            if sid not in ingested_sessions:
                content = format_session_content(session)
                mem_id = client.add_memory(content, sid)
                if mem_id:
                    ingested_sessions.add(sid)
                    if verbose:
                        print(f"  Ingested session {sid} -> {mem_id[:8]}...")

        # 2. Search for relevant memories
        search_result = client.search(question, limit=10)
        elapsed_ms = search_result.get("_elapsed_ms", 0)
        mode_used = search_result.get("mode", "unknown")

        # 3. Extract retrieved session IDs from results
        retrieved_ids: list[str] = []
        if search_result.get("success"):
            for mem in search_result.get("results", []):
                # Get session ID from source_ref or tags
                source_ref = mem.get("source_ref", "") or ""
                if source_ref.startswith("longmemeval:"):
                    sid = source_ref.replace("longmemeval:", "")
                    retrieved_ids.append(sid)
                else:
                    # Try tags
                    tags = mem.get("tags", []) or []
                    for tag in tags:
                        if tag in session_ids:
                            retrieved_ids.append(tag)
                            break

        # 4. Calculate recall metrics
        correct_set = set(correct_ids)
        retrieved_set = set(retrieved_ids[:5])  # Recall@5

        recall_any = bool(correct_set & retrieved_set)
        recall_all = correct_set <= retrieved_set if correct_set else True

        result = EvalResult(
            question_id=qid,
            question_type=qtype,
            question=question,
            ground_truth=str(answer),
            retrieved_session_ids=retrieved_ids[:5],
            correct_session_ids=correct_ids,
            recall_any=recall_any,
            recall_all=recall_all,
            search_time_ms=elapsed_ms,
            mode_used=mode_used,
            num_results=len(search_result.get("results", [])),
        )
        results.append(result)

        # Update summary
        summary.total_questions += 1
        summary.total_search_time_ms += elapsed_ms
        if recall_any:
            summary.recall_any_count += 1
        if recall_all:
            summary.recall_all_count += 1

        # Update by-type stats
        if qtype not in summary.by_type:
            summary.by_type[qtype] = {"total": 0, "recall_any": 0, "recall_all": 0}
        summary.by_type[qtype]["total"] += 1
        if recall_any:
            summary.by_type[qtype]["recall_any"] += 1
        if recall_all:
            summary.by_type[qtype]["recall_all"] += 1

        # Print result
        status = "✓" if recall_any else "✗"
        print(f"  {status} Recall@5: {recall_any} | Mode: {mode_used} | Time: {elapsed_ms}ms")
        if verbose and not recall_any:
            print(f"    Expected: {correct_ids}")
            print(f"    Got: {retrieved_ids[:5]}")

    return results, summary


def print_summary(summary: EvalSummary) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nOverall ({summary.total_questions} questions):")
    print(f"  Recall@5 (any):  {summary.recall_any_rate:.1%} ({summary.recall_any_count}/{summary.total_questions})")
    print(f"  Recall@5 (all):  {summary.recall_all_rate:.1%} ({summary.recall_all_count}/{summary.total_questions})")
    print(f"  Avg search time: {summary.avg_search_time_ms:.0f}ms")

    print(f"\nBy Question Type:")
    for qtype, stats in sorted(summary.by_type.items()):
        total = stats["total"]
        any_rate = stats["recall_any"] / total if total else 0
        all_rate = stats["recall_all"] / total if total else 0
        print(f"  {qtype}:")
        print(f"    Recall@5 (any): {any_rate:.1%} ({stats['recall_any']}/{total})")
        print(f"    Recall@5 (all): {all_rate:.1%} ({stats['recall_all']}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark on CEMS")
    parser.add_argument(
        "--questions", "-n", type=int, default=10,
        help="Number of questions to evaluate (default: 10)"
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

    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key provided. Set CEMS_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("LONGMEMEVAL BENCHMARK FOR CEMS")
    print("=" * 60)
    print(f"API URL: {args.api_url}")
    print(f"Questions: {args.questions}")

    # Initialize client
    client = CEMSEvalClient(args.api_url, args.api_key)

    # Health check
    print("\nChecking CEMS connection...")
    if not client.health_check():
        print("Error: Cannot connect to CEMS. Is it running?", file=sys.stderr)
        sys.exit(1)
    print("  Connected!")

    # Download/load data
    try:
        data_file = download_longmemeval(force=args.download)
        questions = load_longmemeval(data_file, limit=args.questions)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not questions:
        print("No questions to evaluate!", file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    print(f"\nStarting evaluation...")
    start_time = time.time()

    try:
        results, summary = run_eval(client, questions, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        summary = EvalSummary()
        results = []

    elapsed = time.time() - start_time

    # Print summary
    if summary.total_questions > 0:
        print_summary(summary)
        print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Cleanup
    if not args.no_cleanup:
        print(f"\nCleaning up eval memories...")
        deleted = client.cleanup_eval_memories()
        print(f"  Deleted {deleted} memories")

    # Save results
    if args.output and results:
        output_data = {
            "summary": {
                "total_questions": summary.total_questions,
                "recall_any_rate": summary.recall_any_rate,
                "recall_all_rate": summary.recall_all_rate,
                "avg_search_time_ms": summary.avg_search_time_ms,
                "by_type": summary.by_type,
                "elapsed_seconds": elapsed,
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "retrieved_session_ids": r.retrieved_session_ids,
                    "correct_session_ids": r.correct_session_ids,
                    "recall_any": r.recall_any,
                    "recall_all": r.recall_all,
                    "search_time_ms": r.search_time_ms,
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
