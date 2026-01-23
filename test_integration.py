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
API_KEY = "cems_ak_89a65317e256bf43ab5641bfe929809386b1e8c57b0c2d8e"
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
    """Test POST /api/memory/search endpoint."""
    try:
        result = call_api("POST", "/api/memory/search", {
            "query": "Python preferences",
            "limit": 5,
            "scope": "personal",
        })
        
        if result.get("success"):
            results = result.get("results", [])
            mode = result.get("mode", "unknown")
            return True, f"found {len(results)} results (mode: {mode})"
        return False, f"failed: {result.get('error', 'unknown error')}"
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


def run_all_tests():
    """Run all integration tests with pass/fail summary."""
    print("\n" + "="*60)
    print("CEMS INTEGRATION TESTS")
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
