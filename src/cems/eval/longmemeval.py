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
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import math

import httpx


# Configuration
DEFAULT_API_URL = os.getenv("CEMS_API_URL", "http://localhost:8765")
DEFAULT_API_KEY = os.getenv("CEMS_API_KEY", "")
DATA_DIR = Path(__file__).parent / "data"

# Dataset URLs
LONGMEMEVAL_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
}
# Legacy alias for backwards compatibility
LONGMEMEVAL_URL = LONGMEMEVAL_URLS["oracle"]


@dataclass
class EvalResult:
    """Result for a single question."""
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    retrieved_session_ids: list[str]
    correct_session_ids: list[str]
    recall_at_5: bool   # Did we find ANY correct session in top 5?
    recall_at_10: bool  # Did we find ANY correct session in top 10?
    recall_all: bool    # Did we find ALL correct sessions in top 5?
    ndcg_at_5: float    # Position-sensitive score (0-1)
    search_time_ms: int
    mode_used: str
    num_results: int


@dataclass
class EvalSummary:
    """Summary statistics for the eval run."""
    total_questions: int = 0
    recall_at_5_count: int = 0
    recall_at_10_count: int = 0
    recall_all_count: int = 0
    total_ndcg_at_5: float = 0.0
    total_search_time_ms: int = 0
    by_type: dict[str, dict] = field(default_factory=dict)

    @property
    def recall_at_5_rate(self) -> float:
        return self.recall_at_5_count / self.total_questions if self.total_questions else 0

    @property
    def recall_at_10_rate(self) -> float:
        return self.recall_at_10_count / self.total_questions if self.total_questions else 0

    @property
    def recall_all_rate(self) -> float:
        return self.recall_all_count / self.total_questions if self.total_questions else 0

    @property
    def avg_ndcg_at_5(self) -> float:
        return self.total_ndcg_at_5 / self.total_questions if self.total_questions else 0

    @property
    def avg_search_time_ms(self) -> float:
        return self.total_search_time_ms / self.total_questions if self.total_questions else 0


class CEMSEvalClient:
    """Client for CEMS evaluation operations."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: float = 60.0,
        enable_synthesis: bool = False,
        enable_hyde: bool = False,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.enable_synthesis = enable_synthesis
        self.enable_hyde = enable_hyde
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
        """Add a memory and return its ID.

        Args:
            content: Memory content
            session_id: Session identifier for tracking
            timestamp: Optional timestamp in format "2023/04/10 (Mon) 17:50"
        """
        try:
            payload = {
                "content": content,
                "category": "eval-session",
                "infer": False,  # Fast mode - no LLM extraction
                # Use project: prefix so project-scoped scoring works correctly
                # All eval memories are in the same "longmemeval" project
                "source_ref": f"project:longmemeval:{session_id}",
                "tags": ["longmemeval", session_id],
            }

            # Convert LongMemEval timestamp to ISO format if provided
            if timestamp:
                iso_timestamp = self._parse_timestamp(timestamp)
                if iso_timestamp:
                    payload["timestamp"] = iso_timestamp

            resp = self.client.post(
                f"{self.api_url}/api/memory/add",
                headers=self._headers(),
                json=payload,
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

    def add_memories_batch(
        self,
        memories: list[dict],
    ) -> dict[str, str]:
        """Add multiple memories in a single request.

        Args:
            memories: List of memory dicts with keys:
                - content: Memory content
                - session_id: Session identifier for tracking
                - timestamp: Optional timestamp in LongMemEval format

        Returns:
            Dict mapping session_id -> document_id for successfully added memories
        """
        if not memories:
            return {}

        # Prepare batch payload
        batch_memories = []
        session_ids = []

        for mem in memories:
            session_id = mem["session_id"]
            session_ids.append(session_id)

            payload = {
                "content": mem["content"],
                "category": "eval-session",
                "source_ref": f"project:longmemeval:{session_id}",
                "tags": ["longmemeval", session_id],
            }

            # Convert timestamp if provided
            if mem.get("timestamp"):
                iso_timestamp = self._parse_timestamp(mem["timestamp"])
                if iso_timestamp:
                    payload["timestamp"] = iso_timestamp

            batch_memories.append(payload)

        try:
            # Use extended timeout for large batches
            # Each memory may have multiple chunks, and embedding is slow (~4s/chunk on llama.cpp)
            # Conservative estimate: 10 seconds per memory for chunking + embedding + DB insert
            timeout = max(self.timeout, len(memories) * 10)
            resp = self.client.post(
                f"{self.api_url}/api/memory/add_batch",
                headers=self._headers(),
                json={"memories": batch_memories},
                timeout=timeout,
            )
            data = resp.json()

            if data.get("success"):
                document_ids = data.get("document_ids", [])
                # Map session_ids to document_ids
                result_map = {}
                for session_id, doc_id in zip(session_ids, document_ids):
                    result_map[session_id] = doc_id
                    self._session_to_memory[session_id] = doc_id
                return result_map
            else:
                print(f"  Batch add failed: {data.get('error')}", file=sys.stderr)
                return {}

        except Exception as e:
            print(f"  Error in batch add: {e}", file=sys.stderr)
            return {}

    def _parse_timestamp(self, ts: str) -> str | None:
        """Parse LongMemEval timestamp format to ISO format.

        Input: "2023/04/10 (Mon) 17:50"
        Output: "2023-04-10T17:50:00Z"
        """
        try:
            # Remove day of week: "2023/04/10 (Mon) 17:50" -> "2023/04/10 17:50"
            clean = re.sub(r"\s*\([^)]+\)\s*", " ", ts).strip()
            # Parse and convert
            dt = datetime.strptime(clean, "%Y/%m/%d %H:%M")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    def search(self, query: str, limit: int = 10, max_tokens: int = 4000) -> dict[str, Any]:
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
                    "max_tokens": max_tokens,
                    # Pass project so same-project boost (1.3x) applies to all eval memories
                    "project": "longmemeval",
                    "enable_query_synthesis": self.enable_synthesis,
                    "enable_hyde": self.enable_hyde,
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

    def _admin_eval_cleanup(self, admin_key: str | None = None) -> dict | None:
        """Call admin eval cleanup API. Returns response dict or None on failure."""
        effective_key = admin_key or os.getenv("CEMS_ADMIN_KEY") or self.api_key
        try:
            resp = self.client.delete(
                f"{self.api_url}/admin/eval/cleanup",
                headers={
                    "Authorization": f"Bearer {effective_key}",
                    "Content-Type": "application/json",
                },
                params={"source_prefix": "project:longmemeval"},
            )
            data = resp.json()
            if data.get("success"):
                return data
        except Exception:
            pass
        return None

    def cleanup_eval_memories(self, admin_key: str | None = None) -> int:
        """Delete all memories created during eval.

        Uses admin bulk delete API for efficiency instead of individual forget calls.
        Falls back to individual deletes if admin API fails.
        """
        data = self._admin_eval_cleanup(admin_key)
        if data:
            self._session_to_memory.clear()
            return data.get("documents_deleted", 0)

        # Fallback: individual deletes (slow but reliable)
        deleted = 0
        for session_id, memory_id in self._session_to_memory.items():
            if self.forget_memory(memory_id):
                deleted += 1
        self._session_to_memory.clear()
        return deleted

    def cleanup_stale_eval_data(self, admin_key: str | None = None) -> int:
        """Delete ALL old eval data from previous runs.

        Uses the admin API endpoint for reliable bulk deletion.
        Falls back to search-based cleanup if admin API fails.

        Should be called before starting a fresh eval run to prevent
        data contamination from previous runs.
        """
        data = self._admin_eval_cleanup(admin_key)
        if data:
            docs = data.get("documents_deleted", 0)
            chunks = data.get("chunks_deleted", 0)
            if docs > 0:
                print(f"  Deleted {docs} documents ({chunks} chunks)")
            else:
                print("  No stale eval data found")
            return docs

        print("  Admin API unavailable, trying search-based cleanup...")

        # Fallback: search-based cleanup (less reliable)
        return self._search_based_cleanup()

    def _search_based_cleanup(self) -> int:
        """Fallback cleanup using search API (less reliable)."""
        deleted = 0
        found_ids: set[str] = set()

        # Try multiple search terms to find eval memories
        search_queries = [
            "eval session longmemeval",
            "temporal reasoning first",
            "seeds tomatoes marigolds",
            "device Samsung Galaxy MacBook",
        ]

        try:
            for query in search_queries:
                resp = self.client.post(
                    f"{self.api_url}/api/memory/search",
                    headers=self._headers(),
                    json={
                        "query": query,
                        "limit": 200,
                        "scope": "personal",
                        "raw": True,
                    },
                )
                data = resp.json()

                if not data.get("success"):
                    continue

                for mem in data.get("results", []):
                    source_ref = mem.get("source_ref") or ""
                    tags = mem.get("tags") or []

                    is_eval_memory = (
                        source_ref.startswith("project:longmemeval:")
                        or "longmemeval" in tags
                        or "eval-session" in str(mem.get("category", ""))
                    )

                    if is_eval_memory:
                        memory_id = mem.get("memory_id") or mem.get("id")
                        if memory_id:
                            found_ids.add(memory_id)

            if not found_ids:
                print("  No stale eval data found (search-based)")
                return 0

            print(f"  Found {len(found_ids)} stale memories (search-based)...")

            for memory_id in found_ids:
                if self.forget_memory(memory_id):
                    deleted += 1

            print(f"  Deleted {deleted} stale memories")

        except Exception as e:
            print(f"  Warning: Search-based cleanup failed: {e}")

        return deleted

    def summarize_session(
        self,
        content: str,
        session_id: str,
        source_ref: str,
        mode: str = "finalize",
        model: str | None = None,
    ) -> dict:
        """Ingest a session through /api/session/summarize (observer pipeline).

        This runs the full session summary extraction pipeline:
        extract_session_summary() → chunk → embed → store.
        """
        body = {
            "content": content,
            "session_id": session_id,
            "source_ref": source_ref,
            "mode": mode,
            # CRITICAL: Pass full session_id as tag to avoid truncation.
            # Default handler truncates to session_id[:12], but LongMemEval
            # IDs like "ultrachat_283755" share a 10-char prefix, causing
            # massive collisions (32% data loss on full dataset).
            "session_tag": f"lme:{session_id}",
        }
        if model:
            body["model"] = model
        resp = self.client.post(
            f"{self.api_url}/api/session/summarize",
            headers=self._headers(),
            json=body,
            timeout=120,
        )
        return resp.json()


def download_dataset(variant: str = "s", force: bool = False) -> Path:
    """Download LongMemEval dataset variant.

    Args:
        variant: Dataset variant ("oracle" or "s")
        force: Force re-download even if file exists

    Returns:
        Path to downloaded file
    """
    if variant not in LONGMEMEVAL_URLS:
        raise ValueError(f"Unknown variant: {variant}. Use: {list(LONGMEMEVAL_URLS.keys())}")

    url = LONGMEMEVAL_URLS[variant]
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


def load_longmemeval(
    data_file: Path,
    limit: int | None = None,
    variant: str = "oracle",
) -> list[dict]:
    """Load LongMemEval questions from JSON file.

    Args:
        data_file: Path to dataset JSON
        limit: Max questions to load
        variant: Dataset variant. For "oracle", abstention questions are filtered out
            (they have no answer sessions). For "s", all questions are included since
            the S variant contains answer sessions for abstention questions too.
    """
    print(f"Loading data from {data_file}...")

    with open(data_file) as f:
        data = json.load(f)

    print(f"  Total questions: {len(data)}")

    if variant == "oracle":
        # Filter out abstention questions (they have no correct answer in oracle)
        questions = [q for q in data if "_abs" not in q.get("question_id", "")]
        print(f"  Non-abstention: {len(questions)}")
    else:
        # S variant includes abstention questions (they have answer sessions)
        questions = data
        abs_count = sum(1 for q in data if "_abs" in q.get("question_id", ""))
        print(f"  Including {abs_count} abstention questions")

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


def collect_all_sessions(questions: list[dict]) -> dict[str, dict]:
    """Collect all unique sessions across all questions.

    Args:
        questions: List of LongMemEval questions

    Returns:
        Dict mapping session_id -> {content, timestamp} for all unique sessions
    """
    all_sessions: dict[str, dict] = {}

    for q in questions:
        sessions = q.get("haystack_sessions", [])
        session_ids = q.get("haystack_session_ids", [])
        session_dates = q.get("haystack_dates", [])

        # Pad dates if needed
        dates_padded = session_dates + [None] * (len(sessions) - len(session_dates))

        for sid, session, session_date in zip(session_ids, sessions, dates_padded):
            if sid not in all_sessions:
                all_sessions[sid] = {
                    "content": format_session_content(session),
                    "session_id": sid,
                    "timestamp": session_date,
                }

    return all_sessions


def compute_ndcg_at_k(retrieved_ids: list[str], correct_ids: list[str], k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Position-sensitive metric: finding correct session at rank 1 scores higher
    than finding it at rank 5.

    Args:
        retrieved_ids: Ordered list of retrieved session IDs
        correct_ids: List of correct session IDs (ground truth)
        k: Number of top results to consider

    Returns:
        NDCG score between 0 and 1
    """
    if not correct_ids:
        return 0.0

    correct_set = set(correct_ids)
    top_k = retrieved_ids[:k]

    # DCG: sum of 1/log2(rank+1) for each correct result
    dcg = 0.0
    for i, sid in enumerate(top_k):
        if sid in correct_set:
            dcg += 1.0 / math.log2(i + 2)  # rank is 1-indexed, so i+2

    # Ideal DCG: assume all correct results are at the top positions
    ideal_hits = min(len(correct_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def ingest_via_summarize(
    client: CEMSEvalClient,
    sessions: dict[str, dict],
    concurrency: int = 1,
    summary_model: str | None = None,
) -> int:
    """Ingest sessions through /api/session/summarize (observer pipeline).

    Each session goes through: extract_session_summary()
    → chunk → embed → store as session-summary category.

    Args:
        client: CEMS eval client
        sessions: Dict mapping session_id → {content, session_id, timestamp}
        concurrency: Number of parallel requests (default 1, sequential)
        summary_model: Optional LLM model override for summary extraction

    Returns:
        Number of successfully ingested sessions
    """
    total = len(sessions)
    success_count = 0
    errors: list[str] = []
    start_time = time.time()
    completed = 0

    session_items = list(sessions.items())

    def _ingest_one(item: tuple[str, dict]) -> tuple[str, dict | None, str | None]:
        session_id, session_data = item
        try:
            result = client.summarize_session(
                content=session_data["content"],
                session_id=session_id,
                source_ref=f"project:longmemeval:{session_id}",
                mode="finalize",
                model=summary_model,
            )
            if result.get("success"):
                return (session_id, result, None)
            elif result.get("error"):
                return (session_id, None, result["error"])
            return (session_id, result, None)
        except Exception as e:
            return (session_id, None, str(e))

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_ingest_one, item): item[0] for item in session_items}
            for future in as_completed(futures):
                sid, result, error = future.result()
                if error:
                    errors.append(f"{sid}: {error}")
                elif result:
                    success_count += 1
                completed += 1

                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    avg = elapsed / completed
                    remaining = avg * (total - completed)
                    print(
                        f"  [{completed}/{total}] Ingesting sessions... "
                        f"({avg:.1f}s/session, ~{remaining/60:.1f}min remaining)"
                    )
    else:
        for i, item in enumerate(session_items):
            sid, result, error = _ingest_one(item)
            if error:
                errors.append(f"{sid}: {error}")
            elif result:
                success_count += 1
            completed += 1

            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                avg = elapsed / completed
                remaining = avg * (total - completed)
                print(
                    f"  [{completed}/{total}] Ingesting sessions... "
                    f"({avg:.1f}s/session, ~{remaining/60:.1f}min remaining)"
                )

    elapsed_total = time.time() - start_time
    print(f"  Ingested {success_count}/{total} sessions in {elapsed_total:.1f}s")

    if errors:
        print(f"  {len(errors)} errors during ingestion:")
        for err in errors[:5]:  # Show first 5
            print(f"    - {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    return success_count


def run_eval(
    client: CEMSEvalClient,
    questions: list[dict],
    verbose: bool = False,
    ingestion_mode: str = "raw",
    concurrency: int = 1,
    summary_model: str | None = None,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run the evaluation and return results.

    Two-phase approach:
    1. Collect all unique sessions upfront
    2. Ingest all sessions (raw batch or observer-style summarize)
    3. Run searches against stable data

    Args:
        client: CEMS evaluation client
        questions: List of questions to evaluate
        verbose: Print detailed output
        ingestion_mode: "raw" (batch add) or "summarize" (observer pipeline)
        concurrency: Parallel requests for summarize mode
        summary_model: Optional LLM model override for summary extraction
    """
    results: list[EvalResult] = []
    summary = EvalSummary()

    # Phase 1: Collect and ingest all sessions
    print("\nPhase 1: Collecting all unique sessions...")
    all_sessions = collect_all_sessions(questions)
    print(f"  Found {len(all_sessions)} unique sessions across {len(questions)} questions")

    if all_sessions:
        if ingestion_mode == "summarize":
            model_label = f" with {summary_model}" if summary_model else ""
            print(f"\nPhase 2: Ingesting via /api/session/summarize{model_label}...")
            ingest_via_summarize(client, all_sessions, concurrency=concurrency, summary_model=summary_model)
        else:
            print("\nPhase 2: Bulk ingesting (raw mode)...")
            ingest_start = time.time()
            memories_to_add = list(all_sessions.values())
            result_map = client.add_memories_batch(memories_to_add)
            ingest_elapsed = time.time() - ingest_start
            per_session_ms = ingest_elapsed / len(memories_to_add) * 1000 if memories_to_add else 0
            print(
                f"  Ingested {len(result_map)} sessions in {ingest_elapsed:.1f}s "
                f"({per_session_ms:.0f}ms/session)"
            )

    # Phase 3: Run searches
    print(f"\nPhase 3: Running searches...")

    for i, q in enumerate(questions):
        qid = q["question_id"]
        qtype = q.get("question_type", "unknown")
        question = q["question"]
        answer = q.get("answer", "")

        print(f"\n[{i+1}/{len(questions)}] {qtype}: {question[:60]}...")

        # Get correct session IDs for this question
        session_ids = q.get("haystack_session_ids", [])
        correct_ids = q.get("answer_session_ids", [])

        # Search for relevant memories
        search_result = client.search(question, limit=10)
        elapsed_ms = search_result.get("_elapsed_ms", 0)
        mode_used = search_result.get("mode", "unknown")

        # Extract retrieved session IDs from results
        retrieved_ids: list[str] = []
        if search_result.get("success"):
            for mem in search_result.get("results", []):
                # Get session ID from source_ref or tags
                source_ref = mem.get("source_ref", "") or ""
                if source_ref.startswith("project:longmemeval:"):
                    sid = source_ref.replace("project:longmemeval:", "")
                    retrieved_ids.append(sid)
                else:
                    # Try tags
                    tags = mem.get("tags", []) or []
                    for tag in tags:
                        if tag in session_ids:
                            retrieved_ids.append(tag)
                            break

        # Calculate recall metrics
        correct_set = set(correct_ids)
        retrieved_at_5 = set(retrieved_ids[:5])
        retrieved_at_10 = set(retrieved_ids[:10])

        recall_at_5 = bool(correct_set & retrieved_at_5)
        recall_at_10 = bool(correct_set & retrieved_at_10)
        recall_all = bool(correct_set) and correct_set <= retrieved_at_5
        ndcg_at_5 = compute_ndcg_at_k(retrieved_ids, correct_ids, k=5)

        result = EvalResult(
            question_id=qid,
            question_type=qtype,
            question=question,
            ground_truth=str(answer),
            retrieved_session_ids=retrieved_ids[:10],
            correct_session_ids=correct_ids,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            recall_all=recall_all,
            ndcg_at_5=ndcg_at_5,
            search_time_ms=elapsed_ms,
            mode_used=mode_used,
            num_results=len(search_result.get("results", [])),
        )
        results.append(result)

        # Update summary
        summary.total_questions += 1
        summary.total_search_time_ms += elapsed_ms
        summary.total_ndcg_at_5 += ndcg_at_5
        if recall_at_5:
            summary.recall_at_5_count += 1
        if recall_at_10:
            summary.recall_at_10_count += 1
        if recall_all:
            summary.recall_all_count += 1

        # Update by-type stats
        if qtype not in summary.by_type:
            summary.by_type[qtype] = {
                "total": 0, "recall_at_5": 0, "recall_at_10": 0,
                "recall_all": 0, "ndcg_at_5_sum": 0.0,
            }
        summary.by_type[qtype]["total"] += 1
        if recall_at_5:
            summary.by_type[qtype]["recall_at_5"] += 1
        if recall_at_10:
            summary.by_type[qtype]["recall_at_10"] += 1
        if recall_all:
            summary.by_type[qtype]["recall_all"] += 1
        summary.by_type[qtype]["ndcg_at_5_sum"] += ndcg_at_5

        # Print result
        status = "+" if recall_at_5 else "-"
        print(f"  {status} R@5:{recall_at_5} R@10:{recall_at_10} NDCG@5:{ndcg_at_5:.3f} | {mode_used} | {elapsed_ms}ms")
        if verbose and not recall_at_5:
            print(f"    Expected: {correct_ids}")
            print(f"    Got: {retrieved_ids[:10]}")

    return results, summary


def print_summary(summary: EvalSummary, ingestion_mode: str = "raw", dataset: str = "oracle") -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nDataset: {dataset} variant | Ingestion: {ingestion_mode}")
    print(f"\nOverall ({summary.total_questions} questions):")
    print(f"  Recall@5:        {summary.recall_at_5_rate:.1%} ({summary.recall_at_5_count}/{summary.total_questions})")
    print(f"  Recall@10:       {summary.recall_at_10_rate:.1%} ({summary.recall_at_10_count}/{summary.total_questions})")
    print(f"  Recall@5 (all):  {summary.recall_all_rate:.1%} ({summary.recall_all_count}/{summary.total_questions})")
    print(f"  NDCG@5:          {summary.avg_ndcg_at_5:.3f}")
    print(f"  Avg search time: {summary.avg_search_time_ms:.0f}ms")

    print(f"\nBy Question Type:")
    for qtype, stats in sorted(summary.by_type.items()):
        total = stats["total"]
        r5 = stats["recall_at_5"] / total if total else 0
        r10 = stats["recall_at_10"] / total if total else 0
        r_all = stats["recall_all"] / total if total else 0
        ndcg = stats["ndcg_at_5_sum"] / total if total else 0
        print(f"  {qtype}:")
        print(f"    Recall@5: {r5:.1%} ({stats['recall_at_5']}/{total})  Recall@10: {r10:.1%}  NDCG@5: {ndcg:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval retrieval benchmark on CEMS")
    parser.add_argument(
        "--questions", "-n", type=int, default=10,
        help="Number of questions to evaluate (default: 10)"
    )
    parser.add_argument(
        "--dataset", choices=["oracle", "s"], default="s",
        help="Dataset variant (default: s)"
    )
    parser.add_argument(
        "--ingestion-mode", choices=["raw", "summarize"], default="raw",
        help="Ingestion mode: raw (batch add) or summarize (observer pipeline, default: raw)"
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
        "--enable-synthesis", action="store_true",
        help="Enable LLM query synthesis for better retrieval"
    )
    parser.add_argument(
        "--enable-hyde", action="store_true",
        help="Enable HyDE (Hypothetical Document Embeddings) for better vector matching"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Parallel requests for summarize ingestion (default: 1)"
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
        "--summary-model", type=str,
        help="Override LLM model for summary extraction (e.g. qwen/qwen3-32b for Cerebras speed)"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key provided. Set CEMS_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("LONGMEMEVAL RETRIEVAL BENCHMARK FOR CEMS")
    print("=" * 60)
    print(f"API URL:    {args.api_url}")
    print(f"Dataset:    {args.dataset} variant")
    print(f"Questions:  {args.questions}")
    print(f"Ingestion:  {args.ingestion_mode}")
    print(f"Synthesis:  {'ON' if args.enable_synthesis else 'OFF'}")
    print(f"HyDE:       {'ON' if args.enable_hyde else 'OFF'}")
    if args.summary_model:
        print(f"Summary:    {args.summary_model}")

    # Initialize client
    client = CEMSEvalClient(
        args.api_url,
        args.api_key,
        enable_synthesis=args.enable_synthesis,
        enable_hyde=args.enable_hyde,
    )

    # Health check
    print("\nChecking CEMS connection...")
    if not client.health_check():
        print("Error: Cannot connect to CEMS. Is it running?", file=sys.stderr)
        sys.exit(1)
    print("  Connected!")

    # Clean up stale data from previous runs (prevents data contamination)
    if not args.no_clean_stale:
        print("\nCleaning up stale eval data from previous runs...")
        client.cleanup_stale_eval_data()

    # Download/load data
    try:
        data_file = download_dataset(variant=args.dataset, force=args.download)
        questions = load_longmemeval(data_file, limit=args.questions, variant=args.dataset)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not questions:
        print("No questions to evaluate!", file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    print(f"\nStarting evaluation ({args.ingestion_mode} ingestion)...")
    start_time = time.time()

    try:
        results, summary = run_eval(
            client, questions,
            verbose=args.verbose,
            ingestion_mode=args.ingestion_mode,
            concurrency=args.concurrency,
            summary_model=args.summary_model,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        summary = EvalSummary()
        results = []

    elapsed = time.time() - start_time

    # Print summary
    if summary.total_questions > 0:
        print_summary(summary, ingestion_mode=args.ingestion_mode, dataset=args.dataset)
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
                "dataset": args.dataset,
                "ingestion_mode": args.ingestion_mode,
                "enable_synthesis": args.enable_synthesis,
                "enable_hyde": args.enable_hyde,
                "recall_at_5_rate": summary.recall_at_5_rate,
                "recall_at_10_rate": summary.recall_at_10_rate,
                "recall_all_rate": summary.recall_all_rate,
                "avg_ndcg_at_5": summary.avg_ndcg_at_5,
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
                    "recall_at_5": r.recall_at_5,
                    "recall_at_10": r.recall_at_10,
                    "recall_all": r.recall_all,
                    "ndcg_at_5": r.ndcg_at_5,
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
