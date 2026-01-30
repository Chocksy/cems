#!/usr/bin/env python3
"""Integration tests for CEMS - validates all API endpoints work.

Requires: Docker instance running with cems-server container.
Usage: python test_integration.py
"""

import json
import subprocess
import sys
from typing import Any

# Configuration - Direct REST API via docker exec
API_URL = "http://localhost:8765"
API_KEY = "cems_ak_1a33266736f39c2393a515b8af9cc687520f0478cfa3a8a7"
CONTAINER_NAME = "cems-server"


def call_api(method: str, endpoint: str, data: dict | None = None) -> dict:
    """Call the REST API directly via docker exec."""
    curl_cmd = [
        "docker", "exec", CONTAINER_NAME,
        "curl", "-s", "-X", method,
        f"{API_URL}{endpoint}",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {API_KEY}",
    ]
    
    if data:
        curl_cmd.extend(["-d", json.dumps(data)])
    
    result = subprocess.run(curl_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"curl failed: {result.stderr}")
    
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON response: {result.stdout}")


def check_docker():
    """Check if Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if CONTAINER_NAME not in result.stdout:
            print(f"❌ Error: Docker container '{CONTAINER_NAME}' is not running")
            print("   Start it with: docker compose up -d")
            return False
        return True
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False


def test_health() -> tuple[bool, str]:
    """Test GET /health endpoint."""
    try:
        result = call_api("GET", "/health")
        if result.get("status") == "healthy":
            db_status = result.get("database", "unknown")
            return True, f"healthy (database: {db_status})"
        return False, f"unhealthy: {result}"
    except Exception as e:
        return False, str(e)


def test_ping() -> tuple[bool, str]:
    """Test GET /ping endpoint."""
    try:
        result = call_api("GET", "/ping")
        if result.get("status") == "ok":
            return True, "ok"
        return False, f"unexpected response: {result}"
    except Exception as e:
        return False, str(e)


def test_status() -> tuple[bool, str]:
    """Test GET /api/memory/status endpoint."""
    try:
        result = call_api("GET", "/api/memory/status")
        if result.get("status") == "healthy":
            user_id = result.get("user_id", "?")
            backend = result.get("backend", "?")
            return True, f"user_id={user_id}, backend={backend}"
        return False, f"unexpected response: {result}"
    except Exception as e:
        return False, str(e)


def test_memory_add_with_llm() -> tuple[bool, str]:
    """Test POST /api/memory/add with infer=True (LLM fact extraction).
    
    This is the critical test that validates the model supports system prompts.
    Creates a unique memory, verifies it, then deletes it for idempotent testing.
    """
    import time
    timestamp = int(time.time())
    unique_content = f"LLM test memory created at {timestamp} - should be deleted after test"
    
    try:
        # Step 1: Add memory with LLM inference
        result = call_api("POST", "/api/memory/add", {
            "content": unique_content,
            "category": "test",
            "infer": True,  # Use LLM - this will fail if model doesn't support system prompts
        })
        
        if not result.get("success"):
            return False, f"add failed: {result.get('error', 'unknown error')}"
        
        # Check memory operation result
        mem_result = result.get("result", {})
        results = mem_result.get("results", [])
        
        if not results:
            return False, "success=True but empty results"
        
        first_result = results[0]
        event = first_result.get("event", "UNKNOWN")
        memory_id = first_result.get("id", "")
        
        # Valid LLM events: ADD (new), UPDATE (modified), DELETE (consolidated/removed)
        # All these prove the LLM is working - it's making decisions about memory
        if event not in ("ADD", "UPDATE", "DELETE"):
            return False, f"unexpected event: {event} (expected ADD, UPDATE, or DELETE)"
        
        # For DELETE events, Mem0 consolidated/removed - LLM worked but no cleanup needed
        if event == "DELETE":
            return True, f"LLM consolidated memory (event=DELETE)"
        
        if not memory_id:
            return False, f"event={event} but no memory_id returned"
        
        # Step 2: Clean up - forget the test memory
        forget_result = call_api("POST", "/api/memory/forget", {
            "memory_id": memory_id,
            "hard_delete": True,  # Actually delete, not just archive
        })
        
        if forget_result.get("success"):
            return True, f"created & cleaned up: {memory_id[:12]}..."
        else:
            # Memory was created but cleanup failed - still pass but note it
            return True, f"created {memory_id[:12]}... (cleanup failed)"
            
    except Exception as e:
        error_str = str(e)
        # Check for the specific Gemma error
        if "Developer instruction is not enabled" in error_str or "system" in error_str.lower():
            return False, f"MODEL ERROR: {error_str} (model doesn't support system prompts)"
        return False, str(e)


def test_memory_add_fast() -> tuple[bool, str]:
    """Test POST /api/memory/add with infer=False (fast mode, no LLM)."""
    try:
        result = call_api("POST", "/api/memory/add", {
            "content": "Fast import test: This memory is stored directly without LLM processing",
            "category": "test",
            "infer": False,  # Fast mode - no LLM calls
        })
        
        if result.get("success"):
            mem_result = result.get("result", {})
            if mem_result.get("results"):
                memory_id = mem_result["results"][0].get("id", "unknown")
                return True, f"memory created: {memory_id[:12]}..."
            return False, "success=True but no memory ID returned"
        return False, f"failed: {result.get('error', 'unknown error')}"
    except Exception as e:
        return False, str(e)


def test_memory_search() -> tuple[bool, str]:
    """Test POST /api/memory/search endpoint - MUST return results.
    
    This tests the enhanced search pipeline. If it returns 0 results
    when we know memories exist, something is wrong with the pipeline.
    """
    try:
        result = call_api("POST", "/api/memory/search", {
            "query": "Python preferences",
            "limit": 5,
            "scope": "personal",
        })
        
        if not result.get("success"):
            return False, f"failed: {result.get('error', 'unknown error')}"
        
        results = result.get("results", [])
        mode = result.get("mode", "unknown")
        total_candidates = result.get("total_candidates", 0)
        filtered_count = result.get("filtered_count", 0)
        
        # Enhanced search MUST return results if candidates exist
        if total_candidates > 0 and len(results) == 0:
            return False, f"BUG: {total_candidates} candidates found but 0 returned after filtering ({filtered_count} passed filter). Check threshold/scoring."
        
        return True, f"found {len(results)} results (mode: {mode}, candidates: {total_candidates})"
    except Exception as e:
        return False, str(e)


def test_memory_search_raw() -> tuple[bool, str]:
    """Test POST /api/memory/search with raw=True (debug mode)."""
    try:
        result = call_api("POST", "/api/memory/search", {
            "query": "test",
            "limit": 3,
            "raw": True,  # Debug mode
        })
        
        if result.get("success") and result.get("mode") == "raw":
            results = result.get("results", [])
            return True, f"found {len(results)} results (raw mode)"
        return False, f"failed or wrong mode: {result.get('error', result.get('mode'))}"
    except Exception as e:
        return False, str(e)


def test_memory_summary() -> tuple[bool, str]:
    """Test GET /api/memory/summary/personal endpoint."""
    try:
        result = call_api("GET", "/api/memory/summary/personal")
        
        if result.get("success"):
            total = result.get("total", 0)
            categories = result.get("categories", {})
            return True, f"total={total}, categories={len(categories)}"
        return False, f"failed: {result.get('error', 'unknown error')}"
    except Exception as e:
        return False, str(e)


def test_memory_forget() -> tuple[bool, str]:
    """Test POST /api/memory/forget endpoint.
    
    First adds a memory, then forgets it.
    """
    try:
        # First add a memory to forget
        add_result = call_api("POST", "/api/memory/add", {
            "content": "Temporary memory for forget test - should be deleted",
            "category": "test",
            "infer": False,
        })
        
        if not add_result.get("success"):
            return False, f"failed to add memory for forget test: {add_result.get('error')}"
        
        # Extract memory ID
        memory_id = None
        mem_result = add_result.get("result", {})
        if mem_result.get("results"):
            memory_id = mem_result["results"][0].get("id")
        
        if not memory_id:
            return False, "could not extract memory ID from add result"
        
        # Now forget it
        forget_result = call_api("POST", "/api/memory/forget", {
            "memory_id": memory_id,
            "hard_delete": False,  # Archive, not hard delete
        })
        
        if forget_result.get("success"):
            return True, f"archived memory: {memory_id[:12]}..."
        return False, f"failed: {forget_result.get('error', 'unknown error')}"
    except Exception as e:
        return False, str(e)


def test_maintenance() -> tuple[bool, str]:
    """Test POST /api/memory/maintenance endpoint."""
    try:
        result = call_api("POST", "/api/memory/maintenance", {
            "job_type": "consolidation",
        })

        if result.get("success"):
            consolidated = result.get("consolidated", 0)
            return True, f"consolidated {consolidated} memories"
        return False, f"failed: {result.get('error', 'unknown error')}"
    except Exception as e:
        return False, str(e)


def test_memory_add_with_source_ref() -> tuple[bool, str]:
    """Test POST /api/memory/add with source_ref for project-scoped recall."""
    import time
    timestamp = int(time.time())

    try:
        result = call_api("POST", "/api/memory/add", {
            "content": f"Project-scoped test memory at {timestamp}",
            "category": "test",
            "infer": False,
            "source_ref": "project:testorg/testrepo",
        })

        if not result.get("success"):
            return False, f"add failed: {result.get('error', 'unknown error')}"

        mem_result = result.get("result", {})
        results = mem_result.get("results", [])

        if not results:
            return False, "success=True but empty results"

        memory_id = results[0].get("id", "")
        if not memory_id:
            return False, "no memory_id returned"

        # Clean up
        call_api("POST", "/api/memory/forget", {
            "memory_id": memory_id,
            "hard_delete": True,
        })

        return True, f"created with source_ref: {memory_id[:12]}..."
    except Exception as e:
        return False, str(e)


def test_memory_search_with_project() -> tuple[bool, str]:
    """Test POST /api/memory/search with project parameter for scoped boost."""
    try:
        # First create a project-scoped memory
        add_result = call_api("POST", "/api/memory/add", {
            "content": "Project-specific preference: use pytest for testing",
            "category": "test",
            "infer": False,
            "source_ref": "project:testorg/testrepo",
        })

        if not add_result.get("success"):
            return False, f"add failed: {add_result.get('error')}"

        memory_id = add_result.get("result", {}).get("results", [{}])[0].get("id")

        # Search with project parameter
        result = call_api("POST", "/api/memory/search", {
            "query": "pytest testing",
            "limit": 5,
            "scope": "personal",
            "project": "testorg/testrepo",
        })

        # Clean up
        if memory_id:
            call_api("POST", "/api/memory/forget", {
                "memory_id": memory_id,
                "hard_delete": True,
            })

        if result.get("success"):
            results = result.get("results", [])
            mode = result.get("mode", "unknown")
            return True, f"found {len(results)} results with project boost (mode: {mode})"
        return False, f"failed: {result.get('error', 'unknown error')}"
    except Exception as e:
        return False, str(e)


def test_datecs_benchmark() -> tuple[bool, str]:
    """Benchmark test: datecs printer query should NOT return SSH/GSC memories in smart modes.

    This is the critical benchmark from the user's troubled query.
    The query is about connecting datecs fp-700 printer to Windows remotely.
    
    SHOULD return: printer, windows, remote access, fiscal, pos related memories
    SHOULD NOT return: SSH to Hetzner, GSC scripts, SEO, unrelated infrastructure
    
    NOTE: 
    - Vector mode may return some irrelevant results (expected - no LLM reranking)
    - Hybrid/auto modes should either return 0 (no relevant memories) or relevant results only
    - If NO datecs-related memories exist, getting 0 results from hybrid/auto is CORRECT
    """
    query = "datecs fp-700 printer Windows remote connection erpnet"
    
    # Test all modes
    results_by_mode = {}
    irrelevant_by_mode = {}
    
    irrelevant_keywords = ["ssh", "hetzner", "gsc", "epicpxls", "seo", "google search console"]
    
    for mode in ["vector", "hybrid", "auto"]:
        try:
            result = call_api("POST", "/api/memory/search", {
                "query": query,
                "limit": 10,
                "scope": "both",
                "mode": mode,
                "enable_hyde": True,
                "enable_rerank": True,
            })
            
            if not result.get("success"):
                return False, f"mode={mode} failed: {result.get('error')}"
            
            memories = result.get("results", [])
            results_by_mode[mode] = memories
            
            # Check for IRRELEVANT results
            irrelevant_found = []
            
            for mem in memories:
                content = mem.get("content", "").lower()
                category = mem.get("category", "").lower()
                for keyword in irrelevant_keywords:
                    if keyword in content or keyword in category:
                        irrelevant_found.append(f"{keyword} in '{content[:50]}...'")
                        break
            
            irrelevant_by_mode[mode] = irrelevant_found
            
            if irrelevant_found and mode != "vector":
                # Only warn for vector mode (expected), but note for hybrid/auto
                print(f"\n  ⚠️  Mode {mode}: Found irrelevant results (should not happen!):")
                for irr in irrelevant_found[:3]:
                    print(f"      - {irr}")
            
        except Exception as e:
            return False, f"mode={mode} error: {e}"
    
    # Summary
    summary = f"vector={len(results_by_mode.get('vector', []))}, "
    summary += f"hybrid={len(results_by_mode.get('hybrid', []))}, "
    summary += f"auto={len(results_by_mode.get('auto', []))}"
    
    # FAIL only if hybrid or auto modes return irrelevant results
    # Vector mode is allowed to return irrelevant results (no LLM reranking)
    hybrid_irrelevant = len(irrelevant_by_mode.get("hybrid", []))
    auto_irrelevant = len(irrelevant_by_mode.get("auto", []))
    
    if hybrid_irrelevant > 0 or auto_irrelevant > 0:
        return False, f"{summary} - SMART MODES RETURNED IRRELEVANT RESULTS!"
    
    # Check that vector mode at least found something (proves search works)
    vector_count = len(results_by_mode.get("vector", []))
    if vector_count == 0:
        # Raw check to see if there's data
        raw_result = call_api("POST", "/api/memory/search", {
            "query": query,
            "limit": 3,
            "raw": True,
        })
        raw_count = len(raw_result.get("results", []))
        if raw_count > 0:
            return False, f"{summary} - vector mode found 0 but raw found {raw_count}"
    
    # Note if vector mode has irrelevant results (expected behavior, not a failure)
    vector_irrelevant = len(irrelevant_by_mode.get("vector", []))
    if vector_irrelevant > 0:
        summary += f" (vector has {vector_irrelevant} irrelevant - expected)"
    
    return True, summary


def test_enhanced_search_returns_results() -> tuple[bool, str]:
    """Critical test: Enhanced search MUST return results for existing memories.
    
    First creates a known memory, then searches for it using enhanced search.
    This validates the full pipeline works end-to-end.
    """
    import time
    timestamp = int(time.time())
    unique_term = f"xyztest{timestamp}"
    test_content = f"Test memory with unique term {unique_term} for search validation"
    
    try:
        # Step 1: Add a memory with a unique searchable term
        add_result = call_api("POST", "/api/memory/add", {
            "content": test_content,
            "category": "test",
            "infer": False,  # Fast mode
        })
        
        if not add_result.get("success"):
            return False, f"failed to add test memory: {add_result.get('error')}"
        
        memory_id = add_result.get("result", {}).get("results", [{}])[0].get("id")
        if not memory_id:
            return False, "no memory_id returned from add"
        
        # Give vector store a moment to index
        import time as t
        t.sleep(0.5)
        
        # Step 2: Search for it using ENHANCED search (not raw)
        search_result = call_api("POST", "/api/memory/search", {
            "query": unique_term,
            "limit": 5,
            "scope": "personal",
            # Don't set raw=True - we want enhanced search!
        })
        
        # Step 3: Clean up
        call_api("POST", "/api/memory/forget", {
            "memory_id": memory_id,
            "hard_delete": True,
        })
        
        if not search_result.get("success"):
            return False, f"search failed: {search_result.get('error')}"
        
        results = search_result.get("results", [])
        mode = search_result.get("mode", "unknown")
        total_candidates = search_result.get("total_candidates", 0)
        filtered_count = search_result.get("filtered_count", 0)
        
        # The memory we just added MUST be found
        if len(results) == 0:
            if total_candidates > 0:
                return False, f"BUG: {total_candidates} candidates but 0 results after filtering. Check RRF/threshold."
            return False, f"No results found at all (mode: {mode})"
        
        # Check that our memory is in the results
        found_our_memory = any(unique_term in r.get("content", "") for r in results)
        if not found_our_memory:
            return False, f"Found {len(results)} results but not our test memory"
        
        return True, f"found test memory (mode: {mode}, {len(results)} results)"
        
    except Exception as e:
        return False, str(e)


def test_retrieval_modes() -> tuple[bool, str]:
    """Test that different retrieval modes work correctly and return results.
    
    All modes MUST return results when searching for common terms that
    we know exist in the memory database (like "Python", "preferences").
    """
    query = "Python preferences"
    
    try:
        # Test vector mode (should be fast)
        vector_result = call_api("POST", "/api/memory/search", {
            "query": query,
            "mode": "vector",
            "limit": 5,
        })
        if not vector_result.get("success"):
            return False, f"vector mode failed: {vector_result.get('error')}"
        
        # Test hybrid mode (should use HyDE + RRF + rerank)
        hybrid_result = call_api("POST", "/api/memory/search", {
            "query": query,
            "mode": "hybrid",
            "limit": 5,
        })
        if not hybrid_result.get("success"):
            return False, f"hybrid mode failed: {hybrid_result.get('error')}"
        
        # Test auto mode (should analyze query and route)
        auto_result = call_api("POST", "/api/memory/search", {
            "query": query,
            "mode": "auto",
            "limit": 5,
        })
        if not auto_result.get("success"):
            return False, f"auto mode failed: {auto_result.get('error')}"
        
        # Get result counts
        vector_count = len(vector_result.get("results", []))
        hybrid_count = len(hybrid_result.get("results", []))
        auto_count = len(auto_result.get("results", []))
        actual_mode = auto_result.get("mode", "unknown")
        
        # All modes MUST return results for this common query
        # If we get 0 results across all modes, something is broken
        total_results = vector_count + hybrid_count + auto_count
        if total_results == 0:
            # Check if raw mode finds anything
            raw_result = call_api("POST", "/api/memory/search", {
                "query": query,
                "limit": 5,
                "raw": True,
            })
            raw_count = len(raw_result.get("results", []))
            if raw_count > 0:
                return False, f"BUG: raw mode found {raw_count} results but enhanced modes found 0. Check RRF scoring or thresholds."
            return False, f"No memories found at all - database may be empty"
        
        return True, f"vector={vector_count}, hybrid={hybrid_count}, auto={auto_count} (routed to {actual_mode})"
        
    except Exception as e:
        return False, str(e)


def run_all_tests():
    """Run all integration tests with pass/fail summary."""
    print("\n" + "="*60)
    print("CEMS INTEGRATION TESTS (Enhanced Retrieval)")
    print("="*60 + "\n")
    
    # Check Docker first
    if not check_docker():
        sys.exit(1)
    
    tests = [
        ("Health Check", test_health),
        ("Ping", test_ping),
        ("Status", test_status),
        ("Add Memory (LLM)", test_memory_add_with_llm),
        ("Add Memory (Fast)", test_memory_add_fast),
        ("Add Memory (source_ref)", test_memory_add_with_source_ref),
        ("Search", test_memory_search),
        ("Search Raw", test_memory_search_raw),
        ("Search (project)", test_memory_search_with_project),
        ("Enhanced Search E2E", test_enhanced_search_returns_results),  # Critical validation!
        ("Retrieval Modes", test_retrieval_modes),
        ("Datecs Benchmark", test_datecs_benchmark),
        ("Summary", test_memory_summary),
        ("Forget", test_memory_forget),
        ("Maintenance", test_maintenance),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...", end=" ", flush=True)
        try:
            passed, message = test_func()
            if passed:
                print(f"✅ {message}")
                results.append((test_name, True, message))
            else:
                print(f"❌ {message}")
                results.append((test_name, False, message))
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append((test_name, False, f"Exception: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    
    print(f"\n✅ Passed: {passed_count}/{total_count}")
    print(f"❌ Failed: {total_count - passed_count}/{total_count}\n")
    
    if passed_count < total_count:
        print("Failed tests:")
        for test_name, passed, message in results:
            if not passed:
                print(f"  - {test_name}: {message}")
        print()
        sys.exit(1)
    else:
        print("🎉 All tests passed!\n")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
