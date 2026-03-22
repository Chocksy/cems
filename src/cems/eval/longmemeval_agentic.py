#!/usr/bin/env python3
"""
LongMemEval ASMR-Style Agentic Retrieval Benchmark

Implements Supermemory's ASMR approach: LLM agents replace embeddings for retrieval.
Three parallel search agents (Direct Seeker, Inference Engine, Temporal Navigator)
reason over session summaries to find relevant sessions.

No CEMS server needed — standalone with direct OpenRouter LLM calls.

Usage:
    python -m cems.eval.longmemeval_agentic --questions 10 --verbose
    python -m cems.eval.longmemeval_agentic --questions 500 --output results.json
    python -m cems.eval.longmemeval_agentic --questions 10 --raw-sessions --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from cems.eval.longmemeval import (
    EvalResult,
    EvalSummary,
    collect_all_sessions,
    compute_ndcg_at_k,
    download_dataset,
    format_session_content,
    load_longmemeval,
    print_summary,
)
from cems.lib.json_parsing import parse_json_list
from cems.llm.client import get_client
from cems.llm.session_summary_extraction import extract_session_summary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SEARCH_MODEL = "google/gemini-2.5-flash"
DEFAULT_SUMMARY_MODEL = "google/gemini-2.5-flash"
RRF_K = 60  # Standard RRF smoothing constant


# ---------------------------------------------------------------------------
# Search Agent Prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPTS = {
    "direct_seeker": (
        "You are a Direct Seeker — a memory retrieval specialist focused on EXACT MATCHES.\n\n"
        "Your strategy:\n"
        "- Look for sessions that EXPLICITLY mention the specific entities, names, values, "
        "or facts asked about in the question.\n"
        "- Match concrete details: names, numbers, dates, places, products, tools, versions.\n"
        "- If the question asks \"What is X?\", find sessions where X is directly stated.\n"
        "- If the question asks about a person, find sessions that mention that person by name.\n"
        "- Prefer sessions with LITERAL matches over sessions with vague topical overlap.\n"
        "- Ignore sessions that are merely in the same topic area but lack the specific answer.\n\n"
        "You are precision-focused. Better to return fewer highly-relevant sessions than many "
        "loosely-related ones."
    ),
    "inference_engine": (
        "You are an Inference Engine — a memory retrieval specialist focused on IMPLICIT CONNECTIONS.\n\n"
        "Your strategy:\n"
        "- Look for sessions where the answer can be INFERRED even if not explicitly stated.\n"
        "- Connect dots across related sessions: if session A mentions \"my sister Sarah\" and "
        "session B mentions \"Sarah's birthday\", both are relevant to \"When is my sister's birthday?\"\n"
        "- Consider social and contextual cues: relationships, roles, preferences expressed indirectly.\n"
        "- Look for sessions that provide CONTEXT needed to answer the question, even if they "
        "don't contain the literal answer.\n"
        "- For preference questions, find sessions where the user demonstrated a preference "
        "through actions or choices, not just explicit statements.\n"
        "- For multi-session questions, find ALL sessions that contribute partial information.\n\n"
        "You are recall-focused. Cast a wider net to catch sessions that a literal matcher would miss."
    ),
    "temporal_navigator": (
        "You are a Temporal Navigator — a memory retrieval specialist focused on TIME and SEQUENCE.\n\n"
        "Your strategy:\n"
        "- Pay close attention to session dates and chronological ordering.\n"
        "- For \"most recent\" or \"latest\" questions, find the LAST session that mentions the topic.\n"
        "- For \"when did\" questions, find sessions with temporal markers (dates, \"last week\", "
        "\"yesterday\").\n"
        "- For knowledge-update questions (where information changed over time), find BOTH the old "
        "and new sessions, but rank the most recent one first.\n"
        "- Track state changes: if the user switched from X to Y, the session mentioning Y is more "
        "relevant than the one mentioning X (but both matter).\n"
        "- For \"how long\" or \"how many days\" questions, find sessions that establish the start "
        "and end points.\n"
        "- Session dates are your primary signal. Use them to establish chronological order.\n\n"
        "You are temporally-focused. Your unique value is understanding WHEN things happened and "
        "what the CURRENT state is."
    ),
}

SEARCH_USER_PROMPT = """I need to find which past sessions are relevant to answering this question:

QUESTION: {question}

Below are summaries of {n} past sessions. Each summary is labeled with its session ID.

{formatted_summaries}

Return a JSON array of session IDs ranked by relevance to the question, most relevant first.
Return ONLY session IDs that are relevant. If none seem relevant, return an empty array.
Return at most 10 session IDs.

Example output: ["session_abc123", "session_def456"]"""


# ---------------------------------------------------------------------------
# Summary Caching
# ---------------------------------------------------------------------------

def _cache_path(model: str) -> Path:
    """Get cache file path for a given model."""
    slug = model.replace("/", "_").replace(".", "_")
    return DATA_DIR / f"agentic_summaries_{slug}.json"


def _load_summary_cache(path: Path) -> dict[str, dict]:
    """Load cached summaries from disk."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("sessions", {})
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: Could not load cache: {e}", file=sys.stderr)
        return {}


def _save_summary_cache(path: Path, cache: dict[str, dict], model: str) -> None:
    """Save summaries cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "_meta": {
            "model": model,
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "total_sessions": len(cache),
        },
        "sessions": cache,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.rename(path)


def _extract_one_summary(
    session_id: str,
    content: str,
    model: str,
) -> tuple[str, dict | None]:
    """Extract summary for a single session. Thread-safe."""
    result = extract_session_summary(content, project_context=None, model=model)
    return session_id, result


def load_or_create_summaries(
    sessions: dict[str, dict],
    model: str = DEFAULT_SUMMARY_MODEL,
    concurrency: int = 10,
    no_cache: bool = False,
    rebuild_cache: bool = False,
) -> dict[str, dict]:
    """Load cached summaries or extract new ones.

    Args:
        sessions: Dict mapping session_id -> {content, session_id, timestamp}
        model: Model for summary extraction
        concurrency: Parallel extraction workers
        no_cache: Don't load or save cache
        rebuild_cache: Force re-extract all sessions

    Returns:
        Dict mapping session_id -> summary dict {title, content, tags, priority}
    """
    cache_file = _cache_path(model)

    # Load existing cache
    if no_cache or rebuild_cache:
        cache: dict[str, dict] = {}
    else:
        cache = _load_summary_cache(cache_file)
        if cache:
            print(f"  Loaded {len(cache)} cached summaries from {cache_file.name}")

    # Find sessions needing extraction
    needed = {sid: s for sid, s in sessions.items() if sid not in cache}
    if not needed:
        print(f"  All {len(sessions)} sessions cached, skipping extraction")
        return {sid: cache[sid] for sid in sessions if sid in cache}

    print(f"  Need to extract {len(needed)} summaries ({len(cache)} cached)")

    # Extract in parallel
    success = 0
    failures = 0
    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_extract_one_summary, sid, s["content"], model): sid
            for sid, s in needed.items()
        }
        for future in as_completed(futures):
            sid, result = future.result()
            completed += 1
            if result:
                cache[sid] = result
                success += 1
            else:
                failures += 1

            if completed % 50 == 0 or completed == len(needed):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(needed) - completed) / rate if rate > 0 else 0
                print(
                    f"  [{completed}/{len(needed)}] Extracting summaries... "
                    f"({rate:.1f}/s, ~{remaining/60:.1f}min remaining)"
                )

    elapsed_total = time.time() - start_time
    print(f"  Extracted {success} summaries in {elapsed_total:.1f}s ({failures} failures)")

    # Save cache
    if not no_cache:
        _save_summary_cache(cache_file, cache, model)
        print(f"  Saved {len(cache)} summaries to {cache_file.name}")

    return {sid: cache[sid] for sid in sessions if sid in cache}


# ---------------------------------------------------------------------------
# Search Agents
# ---------------------------------------------------------------------------

def _format_summaries_for_agents(
    session_ids: list[str],
    summaries: dict[str, dict],
    sessions: dict[str, dict] | None = None,
    raw_mode: bool = False,
) -> str:
    """Format session summaries (or raw content) for search agent context.

    Args:
        session_ids: List of session IDs for this question's haystack
        summaries: Dict mapping session_id -> summary dict
        sessions: Dict mapping session_id -> {content, timestamp} (always passed
            so temporal navigator can use dates even in summary mode)
        raw_mode: Use raw session content instead of summaries

    Returns:
        Formatted text block with all sessions
    """
    parts = []
    for sid in session_ids:
        # Always try to get the date from sessions (critical for temporal navigator)
        date = "unknown"
        if sessions and sid in sessions:
            date = sessions[sid].get("timestamp") or "unknown"

        if raw_mode and sessions and sid in sessions:
            content = sessions[sid]["content"]
        elif sid in summaries:
            content = summaries[sid]["content"]
        else:
            continue

        parts.append(f"--- Session {sid} (date: {date}) ---\n{content}")

    return "\n\n".join(parts)


def _parse_agent_response(response: str, valid_ids: set[str]) -> list[str]:
    """Parse search agent response into a list of session IDs.

    Tries JSON parsing first, falls back to regex extraction.

    Args:
        response: Raw LLM response text
        valid_ids: Set of valid session IDs to filter against

    Returns:
        Ordered list of session IDs
    """
    # Try JSON parsing — parse_json_list returns [] on failure, not None,
    # so we check if parsed has valid IDs before falling through to regex.
    parsed = parse_json_list(response)
    if parsed:
        result = [str(s).strip() for s in parsed if str(s).strip() in valid_ids]
        if result:
            return result[:10]

    # Fallback: regex extraction of session IDs
    # LongMemEval IDs look like: ultrachat_283755, answer_abc123, etc.
    found = re.findall(r'["\']?([a-zA-Z_]+_[a-zA-Z0-9_]+)["\']?', response)
    result = [s for s in found if s in valid_ids]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for s in result:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:10]


def _run_single_agent(
    role: str,
    question: str,
    summaries_text: str,
    n_sessions: int,
    model: str,
) -> tuple[str, str | list[str]]:
    """Run a single search agent. Thread-safe.

    Args:
        role: Agent role key (e.g., "direct_seeker")
        question: The question to answer
        summaries_text: Pre-formatted summaries text
        n_sessions: Number of sessions in context
        model: LLM model to use

    Returns:
        Tuple of (role, raw_response_str_or_empty_list_on_failure).
        Caller is responsible for parsing the response via _parse_agent_response().
    """
    system = AGENT_SYSTEM_PROMPTS[role]
    user_prompt = SEARCH_USER_PROMPT.format(
        question=question,
        n=n_sessions,
        formatted_summaries=summaries_text,
    )

    client = get_client()

    try:
        response = client.complete(
            prompt=user_prompt,
            system=system,
            model=model,
            temperature=0.1,
            max_tokens=1000,
            fast_route=False,  # Gemini not on Cerebras/Groq
        )
    except Exception as e:
        print(f"    Warning: {role} agent failed: {e}", file=sys.stderr)
        return role, []

    if not response:
        return role, []

    # We need to filter against valid IDs — extract them from summaries_text
    # This is handled by the caller passing valid_ids
    return role, response  # Return raw response, caller will parse


def run_search_agents(
    question: str,
    session_ids: list[str],
    summaries: dict[str, dict],
    sessions: dict[str, dict] | None = None,
    raw_mode: bool = False,
    model: str = DEFAULT_SEARCH_MODEL,
) -> list[str]:
    """Run 3 parallel search agents and merge results via RRF.

    Args:
        question: The question to answer
        session_ids: List of session IDs in this question's haystack
        summaries: Dict mapping session_id -> summary dict
        sessions: Dict mapping session_id -> {content, timestamp} (for raw mode)
        raw_mode: Use raw session content instead of summaries
        model: LLM model for search agents

    Returns:
        Ranked list of session IDs (best first)
    """
    # Format summaries once for all agents
    summaries_text = _format_summaries_for_agents(
        session_ids, summaries, sessions, raw_mode
    )
    n_sessions = len(session_ids)
    valid_ids = set(session_ids)

    # Run 3 agents in parallel
    rankings: list[list[str]] = []
    roles = list(AGENT_SYSTEM_PROMPTS.keys())

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                _run_single_agent, role, question, summaries_text, n_sessions, model
            ): role
            for role in roles
        }

        for future in as_completed(futures):
            role = futures[future]
            try:
                _, raw_response = future.result()
                if isinstance(raw_response, str):
                    parsed = _parse_agent_response(raw_response, valid_ids)
                else:
                    parsed = raw_response  # Already a list
                if parsed:
                    rankings.append(parsed)
            except Exception as e:
                print(f"    Warning: {role} failed: {e}", file=sys.stderr)

    if not rankings:
        return []

    return reciprocal_rank_fusion(rankings, k=RRF_K)


# ---------------------------------------------------------------------------
# Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    For each item, score = sum(1 / (k + rank)) across all lists that contain it.

    Args:
        rankings: List of ranked session ID lists (one per agent).
        k: Smoothing constant (standard=60).

    Returns:
        Merged list of session IDs sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_idx, session_id in enumerate(ranking):
            if session_id not in scores:
                scores[session_id] = 0.0
            scores[session_id] += 1.0 / (k + rank_idx + 1)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ---------------------------------------------------------------------------
# Main Eval Loop
# ---------------------------------------------------------------------------

def run_agentic_eval(
    questions: list[dict],
    summaries: dict[str, dict],
    all_sessions: dict[str, dict],
    search_model: str = DEFAULT_SEARCH_MODEL,
    raw_mode: bool = False,
    verbose: bool = False,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run the agentic retrieval evaluation.

    Args:
        questions: List of LongMemEval questions
        summaries: Dict mapping session_id -> summary dict
        all_sessions: Dict mapping session_id -> {content, timestamp}
        search_model: Model for search agents
        raw_mode: Use raw session content instead of summaries
        verbose: Print detailed output

    Returns:
        Tuple of (results, summary)
    """
    results: list[EvalResult] = []
    summary = EvalSummary()

    for i, q in enumerate(questions):
        qid = q["question_id"]
        qtype = q.get("question_type", "unknown")
        question = q["question"]
        answer = q.get("answer", "")

        print(f"\n[{i+1}/{len(questions)}] {qtype}: {question[:60]}...")

        # Get this question's haystack session IDs
        session_ids = q.get("haystack_session_ids", [])
        correct_ids = q.get("answer_session_ids", [])

        # Run search agents
        start = time.time()
        retrieved_ids = run_search_agents(
            question=question,
            session_ids=session_ids,
            summaries=summaries,
            sessions=all_sessions,  # Always pass — temporal navigator needs dates
            raw_mode=raw_mode,
            model=search_model,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        # Score
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
            mode_used="agentic" + ("-raw" if raw_mode else ""),
            num_results=len(retrieved_ids),
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
        print(f"  {status} R@5:{recall_at_5} R@10:{recall_at_10} NDCG@5:{ndcg_at_5:.3f} | {elapsed_ms}ms")
        if verbose and not recall_at_5:
            print(f"    Expected: {correct_ids}")
            print(f"    Got: {retrieved_ids[:10]}")

    return results, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval ASMR-style agentic retrieval benchmark"
    )
    parser.add_argument(
        "--questions", "-n", type=int, default=10,
        help="Number of questions to evaluate (default: 10)"
    )
    parser.add_argument(
        "--dataset", choices=["oracle", "s"], default="s",
        help="Dataset variant (default: s)"
    )
    parser.add_argument(
        "--search-model", default=DEFAULT_SEARCH_MODEL,
        help=f"Model for search agents (default: {DEFAULT_SEARCH_MODEL})"
    )
    parser.add_argument(
        "--summary-model", default=DEFAULT_SUMMARY_MODEL,
        help=f"Model for summary extraction (default: {DEFAULT_SUMMARY_MODEL})"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Parallel summary extraction workers (default: 10)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file for detailed results (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed output for failures"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Don't use or save summary cache"
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Force re-extract all summaries (ignores existing cache)"
    )
    parser.add_argument(
        "--raw-sessions", action="store_true",
        help="Skip summarization, feed raw session text to agents"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Force re-download of dataset"
    )

    args = parser.parse_args()

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("LONGMEMEVAL AGENTIC RETRIEVAL BENCHMARK (ASMR)")
    print("=" * 60)
    print(f"Questions:    {args.questions}")
    print(f"Dataset:      {args.dataset} variant")
    print(f"Search model: {args.search_model}")
    print(f"Summary model:{args.summary_model}")
    print(f"Raw sessions: {'ON' if args.raw_sessions else 'OFF'}")
    print(f"Concurrency:  {args.concurrency}")

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

    # Phase 1: Collect all sessions
    print("\nPhase 1: Collecting sessions...")
    all_sessions = collect_all_sessions(questions)
    print(f"  Found {len(all_sessions)} unique sessions across {len(questions)} questions")

    # Phase 2: Extract summaries (or skip if raw mode)
    summaries: dict[str, dict] = {}
    if not args.raw_sessions:
        print(f"\nPhase 2: Extracting session summaries...")
        summaries = load_or_create_summaries(
            sessions=all_sessions,
            model=args.summary_model,
            concurrency=args.concurrency,
            no_cache=args.no_cache,
            rebuild_cache=args.rebuild_cache,
        )
        print(f"  {len(summaries)} summaries ready")
    else:
        print("\nPhase 2: Skipped (raw sessions mode)")

    # Phase 3: Run agentic eval
    mode_label = "raw" if args.raw_sessions else "agentic"
    print(f"\nPhase 3: Running agentic search ({mode_label})...")
    start_time = time.time()

    try:
        results, summary = run_agentic_eval(
            questions=questions,
            summaries=summaries,
            all_sessions=all_sessions,
            search_model=args.search_model,
            raw_mode=args.raw_sessions,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        summary = EvalSummary()
        results = []

    elapsed = time.time() - start_time

    # Print summary
    if summary.total_questions > 0:
        print_summary(summary, ingestion_mode=mode_label, dataset=args.dataset)
        print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    if args.output and results:
        output_data = {
            "summary": {
                "total_questions": summary.total_questions,
                "dataset": args.dataset,
                "mode": mode_label,
                "search_model": args.search_model,
                "summary_model": args.summary_model,
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
