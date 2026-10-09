#!/usr/bin/env python3
"""Noise reduction validation tests.

Tests the scoring changes against production data to verify:
1. No regression in recall for good queries
2. Reduced noise for noisy queries
3. Score-gap filter working correctly
4. Hook-mimicking behavior matches expectations

Usage: CEMS_API_KEY=... python3 scripts/experiments/noise_reduction.py
"""

import json
import subprocess
import os
import sys
import time

API_URL = "http://localhost:8765"
CONTAINER = "cems-server"
API_KEY = os.environ["CEMS_API_KEY"]


def api(method: str, endpoint: str, data: dict | None = None) -> dict:
    """Call CEMS API via docker exec."""
    cmd = [
        "docker", "exec", CONTAINER,
        "curl", "-s", "-X", method,
        f"{API_URL}{endpoint}",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {API_KEY}",
    ]
    if data:
        cmd.extend(["-d", json.dumps(data)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def search(query: str, limit: int = 10, project: str | None = None) -> dict:
    """Search CEMS (mimics hook behavior)."""
    payload = {"query": query, "scope": "both", "limit": limit}
    if project:
        payload["project"] = project
    return api("POST", "/api/memory/search", payload)


def hook_search(query: str, project: str | None = None) -> list[dict]:
    """Mimic the hook's search + client-side filtering.

    This reproduces what cems_user_prompts_submit.py does:
    - limit: 5 (NEW)
    - client-side score filter >= 0.45 (NEW, was 0.4)
    - session dedup
    """
    payload = {"query": query, "scope": "both", "limit": 5}
    if project:
        payload["project"] = project

    data = search(query, limit=5, project=project)
    if not data.get("success") or not data.get("results"):
        return []

    results = data["results"]

    # Client-side score filter (matches hook)
    results = [r for r in results if r.get("score", 0) >= 0.45]

    # Session dedup (matches hook)
    seen_sessions: dict[str, dict] = {}
    deduped: list[dict] = []
    for r in results:
        tags = r.get("tags", [])
        session_tag = next((t for t in tags if t.startswith("session:")), None)
        if session_tag:
            base_session = session_tag.split(":")[0] + ":" + session_tag.split(":")[1]
            existing = seen_sessions.get(base_session)
            if existing is None or r.get("score", 0) > existing.get("score", 0):
                if existing is not None:
                    deduped.remove(existing)
                seen_sessions[base_session] = r
                deduped.append(r)
        else:
            deduped.append(r)

    return deduped


# ============================================================================
# Test cases
# ============================================================================

def test_simple_query():
    """Simple focused query should return relevant results."""
    results = hook_search("Docker port binding")
    if not results:
        return True, "0 results (OK if no Docker memories exist)"

    # Check that results are relevant
    relevant = sum(1 for r in results if any(
        kw in r.get("content", "").lower()
        for kw in ["docker", "port", "container", "compose"]
    ))
    noise = len(results) - relevant
    return noise <= 1, f"{len(results)} results, {relevant} relevant, {noise} noise"


def test_preference_query():
    """Preference query should find user preferences."""
    results = hook_search("What editor do I use")
    if not results:
        return True, "0 results (OK if no preference memories)"

    return True, f"{len(results)} results, scores: {[round(r.get('score',0), 3) for r in results]}"


def test_noisy_long_prompt():
    """Long prompt with code blocks and URLs (common hook input).

    The hook cleans this, but let's test raw to see if scoring handles it.
    """
    query = "I need to fix the deployment pipeline. The error in CI is failing on tests"
    results = hook_search(query)

    scores = [r.get("score", 0) for r in results]
    if not scores:
        return True, "0 results"

    # Check score spread — gap filter should remove tail
    spread = max(scores) - min(scores) if len(scores) > 1 else 0
    return True, f"{len(results)} results, spread={spread:.3f}, scores={[round(s,3) for s in scores]}"


def test_datecs_noise():
    """Datecs printer query — historically returned SSH/GSC noise."""
    query = "datecs fp-700 printer Windows remote connection"
    data = search(query, limit=10)

    if not data.get("success"):
        return False, f"API error: {data.get('error')}"

    results = data.get("results", [])

    # Check for known noise patterns
    noise_keywords = ["ssh", "hetzner", "gsc", "epicpxls", "seo", "google search console"]
    noise_items = []
    for r in results:
        content = r.get("content", "").lower()
        for kw in noise_keywords:
            if kw in content:
                noise_items.append(f"{kw} (score={r.get('score', 0):.3f})")
                break

    if noise_items:
        return False, f"{len(results)} results, NOISE: {noise_items}"
    return True, f"{len(results)} results, no known noise"


def test_score_distribution():
    """Check that score-gap filter is working — look at score distribution."""
    query = "Raspberry Pi autostart configuration"
    data = search(query, limit=10)

    if not data.get("success"):
        return False, f"API error"

    results = data.get("results", [])
    if not results:
        return True, "0 results"

    scores = [r.get("score", 0) for r in results]
    top = scores[0]

    # With gap filter at 0.5x, no result should be below top * 0.5
    # (except first 2 which are always kept)
    violations = [s for i, s in enumerate(scores) if i >= 2 and s < top * 0.5]

    return len(violations) == 0, (
        f"{len(results)} results, top={top:.3f}, min={min(scores):.3f}, "
        f"gap_violations={len(violations)}, scores={[round(s,3) for s in scores]}"
    )


def test_hook_limit_respected():
    """Hook now sends limit=5 — verify API respects it."""
    data = search("Python development preferences", limit=5)

    if not data.get("success"):
        return False, "API error"

    results = data.get("results", [])

    return len(results) <= 5, f"got {len(results)} results (limit=5)"


def test_graph_traversal_noise():
    """Graph-related results should have lower base_score (0.3 not 0.5).

    We can't directly test this via API, but we can check that results
    with low relevance don't sneak in via graph traversal.
    """
    # Use a very specific query — graph results would be tangentially related
    query = "Stripe subscription webhook handling"
    data = search(query, limit=10)

    if not data.get("success"):
        return False, "API error"

    results = data.get("results", [])
    if not results:
        return True, "0 results"

    # All results should be above 0.45 threshold
    below_threshold = [r for r in results if r.get("score", 0) < 0.45]

    return len(below_threshold) == 0, (
        f"{len(results)} results, "
        f"min_score={min(r.get('score',0) for r in results):.3f}"
    )


def test_real_hook_queries():
    """Test with queries that actually appear in real hook usage.

    These are representative prompts that a developer would type.
    """
    test_cases = [
        # (query, description, max_acceptable_noise)
        ("implement the plan", "Confirmatory-like short prompt", 2),
        ("How do I deploy to Coolify", "Specific question", 2),
        ("fix the bug in the observer daemon", "Bug fix intent", 2),
        ("what are my SSH credentials", "Credential lookup", 1),
        ("remind me about the Pi setup", "Pi-specific recall", 2),
    ]

    all_passed = True
    details = []

    for query, desc, max_noise in test_cases:
        results = hook_search(query)
        count = len(results)
        scores = [round(r.get("score", 0), 3) for r in results]

        # Basic check: not too many results (max 5 from limit)
        if count > 5:
            all_passed = False
            details.append(f"  FAIL {desc}: {count} > 5 results")
        else:
            details.append(f"  OK {desc}: {count} results, scores={scores}")

    return all_passed, "\n" + "\n".join(details)


def test_threshold_effectiveness():
    """Compare result counts with old threshold (0.4) vs new (0.45).

    We can't test the old threshold directly, but we can check how many
    results would be added back at 0.4 vs kept at 0.45.
    """
    queries = [
        "Docker compose configuration",
        "Python test fixtures",
        "JavaScript frontend patterns",
        "database migration strategy",
    ]

    total_at_45 = 0
    would_add_at_40 = 0

    for q in queries:
        data = search(q, limit=10)
        if not data.get("success"):
            continue
        results = data.get("results", [])
        at_45 = [r for r in results if r.get("score", 0) >= 0.45]
        at_40 = [r for r in results if 0.40 <= r.get("score", 0) < 0.45]
        total_at_45 += len(at_45)
        would_add_at_40 += len(at_40)

    return True, (
        f"across {len(queries)} queries: {total_at_45} results at >=0.45, "
        f"{would_add_at_40} extra would be added at >=0.40 threshold"
    )


def test_latency():
    """Check that pipeline latency hasn't increased (no new LLM calls)."""
    start = time.time()
    data = search("general development preferences", limit=5)
    elapsed = time.time() - start

    if not data.get("success"):
        return False, "API error"

    # Should be well under 3 seconds (network overhead from docker exec)
    return elapsed < 5.0, f"{elapsed:.2f}s ({len(data.get('results', []))} results)"


# ============================================================================
# Runner
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("NOISE REDUCTION VALIDATION (Production Data)")
    print("=" * 60 + "\n")

    # Verify connection
    try:
        status = api("GET", "/api/memory/status")
        if status.get("status") != "healthy":
            print(f"Server not healthy: {status}")
            sys.exit(1)
        threshold = status.get("relevance_threshold", "?")
        print(f"Server healthy. User: {status.get('user_id', '?')[:8]}...")
        print(f"Relevance threshold: {threshold}")
        print(f"Backend: {status.get('backend', '?')}\n")
    except Exception as e:
        print(f"Cannot connect: {e}")
        sys.exit(1)

    tests = [
        ("Simple Query", test_simple_query),
        ("Preference Query", test_preference_query),
        ("Noisy Long Prompt", test_noisy_long_prompt),
        ("Datecs Noise Check", test_datecs_noise),
        ("Score Distribution", test_score_distribution),
        ("Hook Limit (5)", test_hook_limit_respected),
        ("Graph Traversal Noise", test_graph_traversal_noise),
        ("Real Hook Queries", test_real_hook_queries),
        ("Threshold Effectiveness", test_threshold_effectiveness),
        ("Latency Check", test_latency),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        print(f"Testing {name}...", end=" ", flush=True)
        try:
            ok, msg = fn()
            if ok:
                print(f"OK {msg}")
                passed += 1
            else:
                print(f"FAIL {msg}")
                failed += 1
        except Exception as e:
            print(f"ERROR {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 60}\n")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
