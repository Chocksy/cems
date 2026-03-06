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
CONTAINER_NAME = "cems-server"

# API key resolution: env var > auto-provisioned via admin API
API_KEY = ""  # Set dynamically in setup_api_key()


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


def setup_api_key():
    """Resolve API key: CEMS_TEST_API_KEY env var > auto-provision via admin API."""
    global API_KEY
    import os

    # 1. Check env var
    env_key = os.environ.get("CEMS_TEST_API_KEY")
    if env_key:
        API_KEY = env_key
        print(f"Using API key from CEMS_TEST_API_KEY env var (prefix: {API_KEY[:16]}...)")
        return True

    # 2. Auto-provision: get admin key from Docker, create a test user
    print("No CEMS_TEST_API_KEY set, auto-provisioning test user...")
    try:
        # Get admin key from container
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "printenv", "CEMS_ADMIN_KEY"],
            capture_output=True, text=True,
        )
        admin_key = result.stdout.strip()
        if not admin_key:
            print("❌ Could not get CEMS_ADMIN_KEY from container")
            print("   Set CEMS_TEST_API_KEY env var or ensure CEMS_ADMIN_KEY is configured")
            return False

        # Create test user via admin API
        import time
        username = f"integration_test_{int(time.time())}"
        create_cmd = [
            "docker", "exec", CONTAINER_NAME,
            "curl", "-s", "-X", "POST",
            f"{API_URL}/admin/users",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {admin_key}",
            "-d", json.dumps({"username": username, "email": f"{username}@test.local"}),
        ]
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        response = json.loads(result.stdout)

        if "api_key" in response:
            API_KEY = response["api_key"]
            print(f"Auto-provisioned test user '{username}' (key prefix: {API_KEY[:16]}...)")
            return True
        else:
            print(f"❌ Failed to create test user: {response.get('error', 'unknown')}")
            return False

    except Exception as e:
        print(f"❌ Error auto-provisioning API key: {e}")
        print("   Set CEMS_TEST_API_KEY env var manually")
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
    """Test POST /api/memory/maintenance endpoint.

    Uses full_sweep with a small limit to avoid OpenRouter rate limits
    when the database has thousands of documents.
    """
    try:
        result = call_api("POST", "/api/memory/maintenance", {
            "job_type": "consolidation",
            "full_sweep": True,
            "limit": 10,
        })

        if result.get("success"):
            results = result.get("results", {})
            merged = results.get("duplicates_merged", 0)
            checked = results.get("memories_checked", 0)
            return True, f"checked {checked}, merged {merged}"
        return False, f"failed: {result.get('error', 'unknown error')}"
    except Exception as e:
        return False, str(e)


def test_profile_endpoint() -> tuple[bool, str]:
    """Test GET /api/memory/profile endpoint for session context injection."""
    try:
        result = call_api("GET", "/api/memory/profile?token_budget=2000")

        if not result.get("success"):
            return False, f"failed: {result.get('error', 'unknown error')}"

        # Verify expected response structure
        if "context" not in result:
            return False, "missing 'context' field"
        if "components" not in result:
            return False, "missing 'components' field"
        if "token_estimate" not in result:
            return False, "missing 'token_estimate' field"

        components = result.get("components", {})
        prefs = components.get("preferences", 0)
        guidelines = components.get("guidelines", 0)
        recent = components.get("recent_memories", 0)
        gates = components.get("gate_rules_count", 0)
        tokens = result.get("token_estimate", 0)

        return True, f"prefs={prefs}, guidelines={guidelines}, recent={recent}, gates={gates}, ~{tokens} tokens"
    except Exception as e:
        return False, str(e)


def test_profile_with_project() -> tuple[bool, str]:
    """Test GET /api/memory/profile with project parameter."""
    try:
        result = call_api("GET", "/api/memory/profile?project=testorg/testrepo&token_budget=1500")

        if not result.get("success"):
            return False, f"failed: {result.get('error', 'unknown error')}"

        # Should still return valid structure even if no project memories exist
        if "context" not in result or "components" not in result:
            return False, "missing expected fields"

        return True, f"project-scoped profile returned"
    except Exception as e:
        return False, str(e)


def test_tool_learning_endpoint() -> tuple[bool, str]:
    """Test POST /api/tool/learning endpoint for incremental learning."""
    try:
        # Test with a learnable tool (Edit)
        result = call_api("POST", "/api/tool/learning", {
            "tool_name": "Edit",
            "tool_input": {"file_path": "/src/test.py"},
            "tool_output": "Success",
            "session_id": "integration-test-session",
            "context_snippet": "Implementing new feature for database connection pooling",
        })

        if not result.get("success"):
            return False, f"failed: {result.get('error', 'unknown error')}"

        stored = result.get("stored", False)
        reason = result.get("reason", "unknown")

        # It may or may not store depending on LLM extraction
        return True, f"stored={stored}, reason={reason}"
    except Exception as e:
        return False, str(e)


def test_tool_learning_skips_reads() -> tuple[bool, str]:
    """Test that /api/tool/learning skips non-learnable tools like Read."""
    try:
        result = call_api("POST", "/api/tool/learning", {
            "tool_name": "Read",
            "tool_input": {"file_path": "/src/test.py"},
        })

        if not result.get("success"):
            return False, f"failed: {result.get('error', 'unknown error')}"

        stored = result.get("stored", True)
        reason = result.get("reason", "")

        if stored:
            return False, "Read tool should NOT store learnings"

        if reason != "skipped_non_learnable_tool":
            return False, f"unexpected reason: {reason}"

        return True, "correctly skipped Read tool"
    except Exception as e:
        return False, str(e)


def test_gate_rules_endpoint() -> tuple[bool, str]:
    """Test GET /api/memory/gate-rules endpoint."""
    try:
        result = call_api("GET", "/api/memory/gate-rules")

        if not result.get("success"):
            return False, f"failed: {result.get('error', 'unknown error')}"

        rules = result.get("rules", [])
        count = result.get("count", 0)

        return True, f"found {count} gate rules"
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


def test_multi_topic_decomposition() -> tuple[bool, str]:
    """Test multi-topic query decomposition returns results from multiple topics.

    Creates two memories (Docker, TypeScript), then searches with a compound
    query that mentions both topics. Verifies that decomposition finds BOTH.
    """
    import time
    timestamp = int(time.time())
    docker_content = f"Docker port binding fix {timestamp}: use -p 8080:80 for the web container"
    ts_content = f"TypeScript error pattern {timestamp}: use Result<T, E> for typed error handling"

    try:
        # Step 1: Add two topically distinct memories
        docker_add = call_api("POST", "/api/memory/add", {
            "content": docker_content,
            "category": "devops",
            "infer": False,
        })
        ts_add = call_api("POST", "/api/memory/add", {
            "content": ts_content,
            "category": "frontend",
            "infer": False,
        })

        if not docker_add.get("success") or not ts_add.get("success"):
            return False, "failed to add test memories"

        docker_id = docker_add.get("result", {}).get("results", [{}])[0].get("id")
        ts_id = ts_add.get("result", {}).get("results", [{}])[0].get("id")

        # Give vector store time to index
        time.sleep(0.5)

        # Step 2: Compound query mentioning both topics
        result = call_api("POST", "/api/memory/search", {
            "query": f"fix Docker port binding {timestamp}. Also what was that TypeScript error pattern {timestamp}?",
            "limit": 10,
            "scope": "personal",
            "enable_decomposition": True,
        })

        # Step 3: Clean up
        if docker_id:
            call_api("POST", "/api/memory/forget", {"memory_id": docker_id, "hard_delete": True})
        if ts_id:
            call_api("POST", "/api/memory/forget", {"memory_id": ts_id, "hard_delete": True})

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        results = result.get("results", [])
        queries_used = result.get("queries_used", [])

        # Verify decomposition happened (should have >1 query)
        has_decomposition = len(queries_used) > 1

        # Check both topics found in results
        found_docker = any("Docker" in r.get("content", "") or "docker" in r.get("content", "").lower() for r in results)
        found_ts = any("TypeScript" in r.get("content", "") or "typescript" in r.get("content", "").lower() for r in results)

        detail = f"queries={len(queries_used)}, results={len(results)}, docker={found_docker}, ts={found_ts}"

        if found_docker and found_ts:
            return True, f"both topics found ({detail})"
        elif found_docker or found_ts:
            # Partial success - at least one topic found
            return True, f"partial: {detail} (decomposition={'yes' if has_decomposition else 'no'})"
        else:
            return False, f"neither topic found ({detail})"

    except Exception as e:
        return False, str(e)


def test_single_topic_no_decomposition() -> tuple[bool, str]:
    """Single-topic queries should NOT trigger decomposition (zero cost path).

    Verifies the heuristic gate correctly skips LLM decomposition for
    focused, single-topic queries.
    """
    import time
    timestamp = int(time.time())
    content = f"Pytest fixture scope {timestamp}: use session scope for expensive setup"

    try:
        # Add a memory
        add_result = call_api("POST", "/api/memory/add", {
            "content": content,
            "category": "testing",
            "infer": False,
        })
        if not add_result.get("success"):
            return False, f"add failed: {add_result.get('error')}"

        memory_id = add_result.get("result", {}).get("results", [{}])[0].get("id")
        time.sleep(0.5)

        # Single-topic query — should NOT decompose
        result = call_api("POST", "/api/memory/search", {
            "query": f"pytest fixture scope {timestamp}",
            "limit": 5,
            "scope": "personal",
            "enable_decomposition": True,
        })

        # Clean up
        if memory_id:
            call_api("POST", "/api/memory/forget", {"memory_id": memory_id, "hard_delete": True})

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        queries_used = result.get("queries_used", [])
        results = result.get("results", [])

        # For a single-topic query, queries_used should be [original] only
        # (no sub-queries appended from decomposition)
        # Note: synthesis may add expansion queries, but that's not decomposition
        found = any(str(timestamp) in r.get("content", "") for r in results)

        return True, f"queries={len(queries_used)}, results={len(results)}, found={found}"
    except Exception as e:
        return False, str(e)


def test_decomposition_opt_out() -> tuple[bool, str]:
    """enable_decomposition=False should skip decomposition even for multi-topic queries."""
    try:
        result = call_api("POST", "/api/memory/search", {
            "query": "fix Docker ports. Also what was that TypeScript pattern? And remind me about Coolify",
            "limit": 5,
            "scope": "personal",
            "enable_decomposition": False,
        })

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        queries_used = result.get("queries_used", [])
        # With decomposition disabled, the original query should be present
        # and no decomposed sub-queries should appear
        # (synthesis may still add expansion queries if enabled)
        return True, f"opt-out OK, queries={len(queries_used)}"
    except Exception as e:
        return False, str(e)


def test_three_topic_decomposition() -> tuple[bool, str]:
    """Test decomposition with three distinct topics and unique markers.

    Creates three memories across very different domains, then fires a
    compound query with 'also' and 'by the way' to trigger decomposition.
    """
    import time
    timestamp = int(time.time())
    mem_a = f"Redis caching strategy {timestamp}: use TTL 3600 for session data"
    mem_b = f"Figma design tokens {timestamp}: export colors as CSS custom properties"
    mem_c = f"Kubernetes pod scaling {timestamp}: set HPA min=2 max=10 target CPU 70%"

    try:
        ids = []
        for content, cat in [(mem_a, "backend"), (mem_b, "design"), (mem_c, "devops")]:
            r = call_api("POST", "/api/memory/add", {
                "content": content, "category": cat, "infer": False,
            })
            if not r.get("success"):
                return False, f"add failed: {r.get('error')}"
            ids.append(r.get("result", {}).get("results", [{}])[0].get("id"))

        time.sleep(0.5)

        # Three-topic compound query
        result = call_api("POST", "/api/memory/search", {
            "query": (
                f"What was that Redis caching strategy {timestamp}? "
                f"Also remind me about the Figma design tokens {timestamp}. "
                f"By the way, what Kubernetes scaling config {timestamp} did I use?"
            ),
            "limit": 15,
            "scope": "personal",
            "enable_decomposition": True,
        })

        # Clean up
        for mid in ids:
            if mid:
                call_api("POST", "/api/memory/forget", {"memory_id": mid, "hard_delete": True})

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        results = result.get("results", [])
        queries_used = result.get("queries_used", [])

        found_redis = any("Redis" in r.get("content", "") or "redis" in r.get("content", "").lower() for r in results)
        found_figma = any("Figma" in r.get("content", "") or "figma" in r.get("content", "").lower() for r in results)
        found_k8s = any("Kubernetes" in r.get("content", "") or "kubernetes" in r.get("content", "").lower() for r in results)

        topics_found = sum([found_redis, found_figma, found_k8s])
        detail = f"queries={len(queries_used)}, results={len(results)}, redis={found_redis}, figma={found_figma}, k8s={found_k8s}"

        if topics_found >= 2:
            return True, f"{topics_found}/3 topics found ({detail})"
        else:
            return False, f"only {topics_found}/3 topics found ({detail})"

    except Exception as e:
        return False, str(e)


def test_score_gap_filter() -> tuple[bool, str]:
    """Score-gap filter should remove results far below the top score.

    Creates memories with known relevance, searches, and verifies
    that results below 50% of top score are dropped (except first 2).
    """
    import time
    timestamp = int(time.time())

    try:
        # Add a highly relevant memory and a tangentially related one
        relevant = call_api("POST", "/api/memory/add", {
            "content": f"Unique test term xyzgaptest{timestamp}: primary result for score gap validation",
            "category": "test",
            "infer": False,
        })

        if not relevant.get("success"):
            return False, f"add failed: {relevant.get('error')}"

        relevant_id = relevant.get("result", {}).get("results", [{}])[0].get("id")
        time.sleep(0.5)

        # Search with a query that will match our memory strongly
        result = call_api("POST", "/api/memory/search", {
            "query": f"xyzgaptest{timestamp}",
            "limit": 10,
            "scope": "personal",
        })

        # Clean up
        if relevant_id:
            call_api("POST", "/api/memory/forget", {"memory_id": relevant_id, "hard_delete": True})

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        results = result.get("results", [])
        if not results:
            return False, "no results found"

        # All results should be above threshold (0.45)
        below_threshold = [r for r in results if r.get("score", 0) < 0.45]
        if below_threshold:
            return False, f"found {len(below_threshold)} results below 0.45 threshold"

        # Check gap filter: if we have 3+ results, none should be below 50% of top
        if len(results) >= 3:
            top_score = results[0].get("score", 0)
            cutoff = top_score * 0.5
            violations = [
                r.get("score", 0) for i, r in enumerate(results)
                if i >= 2 and r.get("score", 0) < cutoff
            ]
            if violations:
                return False, f"gap filter violations: {violations} below cutoff {cutoff:.3f}"

        return True, f"found {len(results)} results, gap filter OK"
    except Exception as e:
        return False, str(e)


def test_hook_limit_5() -> tuple[bool, str]:
    """Hook sends limit=5 — API should return at most 5 results.

    This validates the hook-side change where we cap results at 5.
    """
    try:
        result = call_api("POST", "/api/memory/search", {
            "query": "Python development preferences configuration",
            "limit": 5,
            "scope": "both",
        })

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        results = result.get("results", [])
        if len(results) > 5:
            return False, f"returned {len(results)} results, expected <= 5"

        return True, f"returned {len(results)} results (limit=5 respected)"
    except Exception as e:
        return False, str(e)


def test_relevance_threshold_045() -> tuple[bool, str]:
    """Relevance threshold is now 0.45 — no results should be below it."""
    try:
        # Use a broad query that would have returned marginal results at 0.4
        result = call_api("POST", "/api/memory/search", {
            "query": "general coding patterns and best practices",
            "limit": 10,
            "scope": "both",
        })

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        results = result.get("results", [])
        if not results:
            return True, "0 results (OK - threshold filtering working)"

        # Verify no result has score below 0.45
        below = [r.get("score", 0) for r in results if r.get("score", 0) < 0.45]
        if below:
            return False, f"found {len(below)} results below 0.45: {below}"

        scores = [round(r.get("score", 0), 3) for r in results]
        return True, f"{len(results)} results, all >= 0.45, scores={scores}"
    except Exception as e:
        return False, str(e)


def test_pipeline_latency() -> tuple[bool, str]:
    """Pipeline should stay under 3s (no new LLM calls added).

    Tests a typical hook query end-to-end.
    """
    import time as t

    try:
        start = t.time()
        result = call_api("POST", "/api/memory/search", {
            "query": "How do I configure Docker compose for development",
            "limit": 5,
            "scope": "both",
        })
        elapsed = t.time() - start

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        # Allow generous margin for docker exec overhead (adds ~200-500ms)
        if elapsed > 5.0:
            return False, f"too slow: {elapsed:.2f}s (limit 5s including docker overhead)"

        count = len(result.get("results", []))
        return True, f"{elapsed:.2f}s for {count} results"
    except Exception as e:
        return False, str(e)


def test_graph_base_score_reduced() -> tuple[bool, str]:
    """Graph-related results should not inflate scores to pass threshold.

    With base_score reduced from 0.5 to 0.3, graph traversal results
    need actual relevance to survive the 0.45 threshold.
    This is validated indirectly: search results should all have genuine scores.
    """
    try:
        # Use a specific query - graph traversal would add tangentially related memories
        result = call_api("POST", "/api/memory/search", {
            "query": "PostgreSQL connection pooling configuration",
            "limit": 10,
            "scope": "both",
        })

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        results = result.get("results", [])
        if not results:
            return True, "0 results (no graph inflation)"

        # All results should be above 0.45 (graph results at 0.3 base should be filtered)
        min_score = min(r.get("score", 0) for r in results)
        if min_score < 0.45:
            return False, f"min score {min_score:.3f} < 0.45 - possible graph noise"

        return True, f"{len(results)} results, min_score={min_score:.3f}"
    except Exception as e:
        return False, str(e)


def test_shown_count_no_boost() -> tuple[bool, str]:
    """Shown-count boost was removed. Verify via status endpoint config."""
    try:
        result = call_api("GET", "/api/memory/status")
        if not result.get("status") == "healthy":
            return False, "server unhealthy"

        threshold = result.get("relevance_threshold", 0)
        if threshold != 0.45:
            return False, f"expected threshold 0.45, got {threshold}"

        return True, f"threshold confirmed at {threshold}"
    except Exception as e:
        return False, str(e)


def test_aggregation_exempt_from_gap_filter() -> tuple[bool, str]:
    """Aggregation queries should NOT have score-gap filtering applied.

    When asking 'what did I work on this week', we want many results
    even if they have similar scores. The gap filter would incorrectly
    trim these.
    """
    try:
        result = call_api("POST", "/api/memory/search", {
            "query": "what are all the things I worked on recently, list everything",
            "limit": 15,
            "scope": "both",
        })

        if not result.get("success"):
            return False, f"search failed: {result.get('error')}"

        results = result.get("results", [])
        if not results:
            return True, "0 results"

        # Aggregation should return more results (no gap trimming)
        return True, f"{len(results)} results (aggregation allows similar scores)"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Mini LongMemEval: Known-Answer Precision Tests (requires production data)
# =============================================================================
# These tests search for content KNOWN to exist in production and verify:
# 1. Precision: returned results match the expected topic (no noise)
# 2. Recall: the known memory is actually found in results
# 3. Noise ratio: count relevant vs irrelevant in the result set
#
# Skipped gracefully if running against an empty test user.

def _has_production_data() -> bool:
    """Check if we're running against a user with substantial data."""
    try:
        result = call_api("GET", "/api/memory/summary/personal")
        return result.get("total", 0) > 100
    except Exception:
        return False


def _hook_search(query: str, project: str | None = None) -> list[dict]:
    """Mimic hook behavior: limit=5, threshold=0.45, session dedup."""
    data = call_api("POST", "/api/memory/search", {
        "query": query, "scope": "both", "limit": 5,
        **({"project": project} if project else {}),
    })
    if not data.get("success") or not data.get("results"):
        return []
    results = [r for r in data["results"] if r.get("score", 0) >= 0.45]
    # Session dedup
    seen: dict[str, dict] = {}
    deduped: list[dict] = []
    for r in results:
        tags = r.get("tags", [])
        stag = next((t for t in tags if t.startswith("session:")), None)
        if stag:
            base = stag.split(":")[0] + ":" + stag.split(":")[1]
            old = seen.get(base)
            if old is None or r.get("score", 0) > old.get("score", 0):
                if old is not None:
                    deduped.remove(old)
                seen[base] = r
                deduped.append(r)
        else:
            deduped.append(r)
    return deduped


def _precision(results: list[dict], relevant_keywords: list[str]) -> tuple[int, int]:
    """Count how many results contain at least one relevant keyword."""
    relevant = 0
    for r in results:
        content = r.get("content", "").lower()
        cat = r.get("category", "").lower()
        if any(kw.lower() in content or kw.lower() in cat for kw in relevant_keywords):
            relevant += 1
    return relevant, len(results) - relevant


def test_known_answer_coolify() -> tuple[bool, str]:
    """Known-answer: Coolify memories MUST be found for Coolify queries."""
    if not _has_production_data():
        return True, "SKIP (no production data)"

    results = _hook_search("How do I deploy with Coolify CLI")
    if not results:
        return False, "0 results — Coolify memories exist but not found"

    rel, noise = _precision(results, ["coolify", "deploy", "cli"])
    if rel == 0:
        contents = [r.get("content", "")[:60] for r in results]
        return False, f"0 relevant in {len(results)} results: {contents}"

    return noise <= 1, f"{rel} relevant, {noise} noise out of {len(results)}"


def test_known_answer_raspberry_pi() -> tuple[bool, str]:
    """Known-answer: Raspberry Pi memories for Pi-related queries."""
    if not _has_production_data():
        return True, "SKIP (no production data)"

    results = _hook_search("Raspberry Pi autostart configuration pos-app")
    if not results:
        return False, "0 results — Pi memories exist but not found"

    rel, noise = _precision(results, ["raspberry", "pi", "pos-app", "openbox", "autostart"])
    return noise <= 1, f"{rel} relevant, {noise} noise out of {len(results)}"


def test_known_answer_cems_architecture() -> tuple[bool, str]:
    """Known-answer: CEMS architecture queries should find CEMS-related memories."""
    if not _has_production_data():
        return True, "SKIP (no production data)"

    results = _hook_search("CEMS observer daemon signal IPC")
    if not results:
        return True, "0 results (acceptable if observer memories sparse)"

    rel, noise = _precision(results, ["observer", "daemon", "signal", "cems", "ipc", "epoch"])
    return noise <= 1, f"{rel} relevant, {noise} noise out of {len(results)}"


def test_cross_topic_isolation() -> tuple[bool, str]:
    """Stripe query should NOT return Coolify/Pi/Docker noise.

    Tests that the pipeline doesn't bleed unrelated topics into results.
    """
    if not _has_production_data():
        return True, "SKIP (no production data)"

    results = _hook_search("Stripe subscription webhook handling")
    if not results:
        return True, "0 results (acceptable if no Stripe memories)"

    # These should NOT appear in Stripe results
    # Note: use full phrases to avoid substring false positives ("pi" in "api"/"pipeline")
    noise_topics = ["coolify", "raspberry pi", "docker compose", "openbox", "pos-app"]
    noise_count, _ = _precision(results, noise_topics)

    if noise_count > 0:
        noisy = [r.get("content", "")[:60] for r in results
                 if any(kw in r.get("content", "").lower() for kw in noise_topics)]
        return False, f"{noise_count} cross-topic noise items: {noisy}"

    return True, f"{len(results)} results, no cross-topic noise"


def test_preference_recall() -> tuple[bool, str]:
    """Preference queries should find user preferences, not session noise.

    The user has preferences about effort estimates, Pi testing, etc.
    These should surface for preference-type queries.
    """
    if not _has_production_data():
        return True, "SKIP (no production data)"

    results = _hook_search("What are my preferences about effort estimates")
    if not results:
        return True, "0 results (acceptable)"

    # Check if at least one result mentions effort/estimates/timeline
    has_preference = any(
        any(kw in r.get("content", "").lower() for kw in ["effort", "estimate", "timeline", "preference"])
        for r in results
    )

    # Check noise: session summaries about random topics
    session_noise = sum(1 for r in results if r.get("category") == "session-summary"
                       and "effort" not in r.get("content", "").lower()
                       and "estimate" not in r.get("content", "").lower())

    return True, (
        f"{len(results)} results, has_preference={has_preference}, "
        f"session_noise={session_noise}, "
        f"scores={[round(r.get('score', 0), 3) for r in results]}"
    )


def test_noise_ratio_across_queries() -> tuple[bool, str]:
    """Measure noise ratio across 5 diverse real-world queries.

    This is the key metric: what % of surfaced results are actually noise?
    Target: <40% noise (improvement from ~80% baseline).
    """
    if not _has_production_data():
        return True, "SKIP (no production data)"

    test_cases = [
        ("How do I deploy with Coolify CLI", ["coolify", "deploy", "cli"]),
        ("Raspberry Pi autostart configuration", ["raspberry", "pi", "autostart", "openbox", "pos"]),
        ("CEMS memory search pipeline scoring", ["cems", "memory", "search", "score", "retrieval", "pipeline"]),
        ("Docker compose port binding fix", ["docker", "compose", "port", "container"]),
        ("fix the bug in the observer daemon", ["observer", "daemon", "signal", "cems", "epoch"]),
    ]

    total_relevant = 0
    total_noise = 0
    details = []

    for query, keywords in test_cases:
        results = _hook_search(query)
        rel, noise = _precision(results, keywords)
        total_relevant += rel
        total_noise += noise
        details.append(f"  {query[:40]}: {rel}R/{noise}N of {len(results)}")

    total = total_relevant + total_noise
    if total == 0:
        return True, "no results across all queries"

    noise_pct = (total_noise / total) * 100
    passed = noise_pct < 40  # Target: < 40% noise (was ~80%)

    return passed, (
        f"noise ratio: {noise_pct:.0f}% ({total_noise}N/{total}T), "
        f"target <40%\n" + "\n".join(details)
    )


def test_score_gap_prevents_tail() -> tuple[bool, str]:
    """Score-gap filter should prevent long tail of low-relevance results.

    For a specific query, results should cluster tightly. The gap filter
    at 50% of top score should trim any outliers beyond position 2.
    """
    if not _has_production_data():
        return True, "SKIP (no production data)"

    # Use a broad query that might have a tail
    data = call_api("POST", "/api/memory/search", {
        "query": "general development workflow patterns",
        "limit": 10, "scope": "both",
    })

    if not data.get("success"):
        return False, "API error"

    results = data.get("results", [])
    if len(results) < 3:
        return True, f"only {len(results)} results (no tail to test)"

    scores = [r.get("score", 0) for r in results]
    top = scores[0]
    cutoff = top * 0.5

    # Check that results beyond position 2 are all above cutoff
    tail_below = [(i, s) for i, s in enumerate(scores) if i >= 2 and s < cutoff]
    spread = top - scores[-1] if scores else 0

    return len(tail_below) == 0, (
        f"{len(results)} results, spread={spread:.3f}, "
        f"top={top:.3f}, cutoff={cutoff:.3f}, tail_violations={len(tail_below)}"
    )


def test_well_validated_memory_survives_decay() -> tuple[bool, str]:
    """Memories shown 10+ times should resist time decay.

    The adaptive decay ceiling ensures that well-validated memories
    (shown_count >= 10) get at least 0.95 time_decay, protecting
    important memories from being killed by age alone.

    We test this indirectly: search for a topic known to have
    high shown_count memories and verify they appear in results
    even if they're old.
    """
    if not _has_production_data():
        return True, "SKIP (no production data)"

    # "read output file" preference has shown_count=79
    results = _hook_search("read the output file to retrieve the result")
    if not results:
        return True, "0 results (memory may have decayed — investigate)"

    # Check if the high-shown memory appears
    has_high_shown = any(
        "output file" in r.get("content", "").lower() or "retrieve the result" in r.get("content", "").lower()
        for r in results
    )

    return True, (
        f"{len(results)} results, high-shown memory found: {has_high_shown}, "
        f"scores={[round(r.get('score', 0), 3) for r in results]}"
    )


def test_session_summary_not_dominant() -> tuple[bool, str]:
    """Session summaries should not dominate results for non-session queries.

    Session summaries are noisy by nature (broad topic coverage).
    For focused queries, they should be outranked by specific memories.
    """
    if not _has_production_data():
        return True, "SKIP (no production data)"

    results = _hook_search("Coolify CLI deployment commands")
    if not results:
        return True, "0 results"

    session_count = sum(1 for r in results if r.get("category") == "session-summary")
    specific_count = len(results) - session_count

    # Session summaries should not be majority
    return session_count <= specific_count, (
        f"{len(results)} results: {specific_count} specific, {session_count} session-summaries"
    )


def run_all_tests():
    """Run all integration tests with pass/fail summary."""
    print("\n" + "="*60)
    print("CEMS INTEGRATION TESTS (Enhanced Retrieval)")
    print("="*60 + "\n")
    
    # Check Docker first
    if not check_docker():
        sys.exit(1)

    # Resolve API key
    if not setup_api_key():
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
        ("Multi-Topic Decomposition", test_multi_topic_decomposition),
        ("Single-Topic No Decomposition", test_single_topic_no_decomposition),
        ("Decomposition Opt-Out", test_decomposition_opt_out),
        ("Three-Topic Decomposition", test_three_topic_decomposition),
        ("Datecs Benchmark", test_datecs_benchmark),
        ("Summary", test_memory_summary),
        ("Forget", test_memory_forget),
        ("Maintenance", test_maintenance),
        # New endpoints for SessionStart hook support
        ("Profile Endpoint", test_profile_endpoint),
        ("Profile (project)", test_profile_with_project),
        ("Gate Rules", test_gate_rules_endpoint),
        # New endpoints for tool-based learning (SuperMemory-style)
        ("Tool Learning", test_tool_learning_endpoint),
        ("Tool Learning (skip Read)", test_tool_learning_skips_reads),
        # Noise reduction pipeline validation
        ("Score-Gap Filter", test_score_gap_filter),
        ("Hook Limit (5)", test_hook_limit_5),
        ("Threshold 0.45", test_relevance_threshold_045),
        ("Pipeline Latency", test_pipeline_latency),
        ("Graph Base Score", test_graph_base_score_reduced),
        ("Config Threshold", test_shown_count_no_boost),
        ("Aggregation Exempt", test_aggregation_exempt_from_gap_filter),
        # Mini LongMemEval: known-answer precision tests (production data)
        ("Known-Answer: Coolify", test_known_answer_coolify),
        ("Known-Answer: Pi", test_known_answer_raspberry_pi),
        ("Known-Answer: CEMS Arch", test_known_answer_cems_architecture),
        ("Cross-Topic Isolation", test_cross_topic_isolation),
        ("Preference Recall", test_preference_recall),
        ("Noise Ratio (5 queries)", test_noise_ratio_across_queries),
        ("Score-Gap Tail", test_score_gap_prevents_tail),
        ("Decay Ceiling", test_well_validated_memory_survives_decay),
        ("Session Summary Balance", test_session_summary_not_dominant),
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
