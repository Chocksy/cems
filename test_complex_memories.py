#!/usr/bin/env python3
"""Complex memory test suite for CEMS enhanced retrieval.

This test suite validates that CEMS can find semantically related memories
even when there are NO keyword matches. These tests verify:
1. HyDE (Hypothetical Document Embeddings) effectiveness
2. LLM re-ranking for relevance
3. Graph traversal for related memories
4. Category summary matching

Each test memory is carefully designed to have NO keyword overlap with its
target query, requiring reasoning to connect them.
"""

import json
import time
import requests
from dataclasses import dataclass

# Configuration
BASE_URL = "http://localhost:8765"
API_KEY = "cems_ak_31996bd6f47b4c11f72a7a911e914496d746164d253bd319"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


@dataclass
class ComplexMemoryTest:
    """A test case with memory and query that require reasoning to match."""
    name: str
    memory_content: str
    memory_category: str
    memory_tags: list[str]
    query: str
    why_related: str  # Explanation of semantic relationship
    expected_found: bool  # Should this memory be found?


# =============================================================================
# TEST CASES: Memories with NO keyword overlap to queries
# =============================================================================

COMPLEX_TESTS = [
    # -------------------------------------------------------------------------
    # Test 1: Hardware troubleshooting via symptoms (NO product name mentioned)
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="fiscal_device_via_symptoms",
        memory_content=(
            "[WORKING_SOLUTION] When the receipt printer shows blinking lights and "
            "refuses to respond after power cycle, check if the communication port "
            "settings match: COM1, 9600 baud, 8N1. Romanian fiscal regulations require "
            "the device to be online before transactions. The driver installation from "
            "the vendor website (datecs.ro) includes a diagnostic tool that can reset "
            "the internal state machine."
        ),
        memory_category="hardware",
        memory_tags=["troubleshooting", "pos-systems", "romania"],
        query="datecs fp-700 printer Windows remote connection",
        why_related="Memory describes Datecs printer troubleshooting (from datecs.ro) without naming FP-700",
        expected_found=True,
    ),

    # -------------------------------------------------------------------------
    # Test 2: ERPNet connection via protocol details (NO erpnet mentioned)
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="erpnet_via_protocol",
        memory_content=(
            "[WORKING_SOLUTION] The open-source fiscal printer connector uses a "
            "specific handshake sequence: ENQ -> ACK -> STX data ETX -> checksum. "
            "The .NET library at github.com/nickel-bg/something-fp handles the "
            "protocol implementation. You need to call InitPrinter() before any "
            "fiscal operations, and CloseDay() at end of shift."
        ),
        memory_category="development",
        memory_tags=["pos-integration", "dotnet", "fiscal"],
        query="erpnet fp fiscal printer integration",
        why_related="Memory describes ErpNet.FP protocol (github nickel-bg) without naming it",
        expected_found=True,
    ),

    # -------------------------------------------------------------------------
    # Test 3: Remote Windows access via approach (NO ssh, rdp, or remote mentioned)
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="remote_windows_via_approach",
        memory_content=(
            "[WORKING_SOLUTION] To run scripts on an employee workstation without "
            "interrupting their work, use WinRM with PowerShell sessions. Enable it "
            "with 'winrm quickconfig'. Then from your machine: "
            "Enter-PSSession -ComputerName STORE-PC1 -Credential $cred. "
            "Scripts run in background using Start-Job. The employee sees nothing."
        ),
        memory_category="infrastructure",
        memory_tags=["automation", "windows", "powershell"],
        query="connect to Windows computer remotely run script background",
        why_related="Memory describes remote Windows scripting without keywords 'remote', 'SSH', 'connect'",
        expected_found=True,
    ),

    # -------------------------------------------------------------------------
    # Test 4: Printer driver via symptoms (NO driver mentioned)
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="printer_driver_via_behavior",
        memory_content=(
            "[WORKING_SOLUTION] DUDE software from the hardware manufacturer is "
            "required for proper communication. Without it, the device appears "
            "in Device Manager with a yellow warning. The setup executable from "
            "datecs.ro/download installs virtual COM ports that the fiscal "
            "software expects. After installation, reboot is mandatory."
        ),
        memory_category="hardware",
        memory_tags=["setup", "installation", "pos"],
        query="datecs printer driver installation Windows",
        why_related="Memory describes DUDE driver installation from datecs.ro without 'driver' keyword",
        expected_found=True,
    ),

    # -------------------------------------------------------------------------
    # Test 5: Semantic opposite - SHOULD NOT be found
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="unrelated_server_ssh",
        memory_content=(
            "[WORKING_SOLUTION] Use SSH to access Hetzner servers for project "
            "management. Connect with: ssh root@your-server.hetzner.com. "
            "Coolify dashboard is at https://coolify.your-domain.com."
        ),
        memory_category="infrastructure",
        memory_tags=["ssh", "deployment", "hetzner"],
        query="datecs fp-700 printer Windows remote connection",
        why_related="About SSH to servers - semantically unrelated to fiscal printers",
        expected_found=False,
    ),

    # -------------------------------------------------------------------------
    # Test 6: Python preference via code style (NO 'python' mentioned)
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="python_pref_via_style",
        memory_content=(
            "[PREFERENCE] Always use type hints in function signatures. "
            "Prefer dataclasses over dictionaries for structured data. "
            "Use f-strings not .format(). Black formatter with 100 char line length. "
            "pytest with fixtures, not unittest."
        ),
        memory_category="coding-style",
        memory_tags=["preferences", "code-quality"],
        query="What are my Python coding preferences?",
        why_related="Describes Python idioms (type hints, dataclasses, f-strings, Black, pytest) without saying 'Python'",
        expected_found=True,
    ),

    # -------------------------------------------------------------------------
    # Test 7: Graph relationship test - linked via entity
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="graph_link_docker",
        memory_content=(
            "[WORKING_SOLUTION] Container orchestration with docker-compose. "
            "Use depends_on for startup order. Health checks are critical. "
            "Volume mounts for persistent data. Networks for service isolation."
        ),
        memory_category="deployment",
        memory_tags=["containers", "devops"],
        query="How do I deploy services with Kubernetes?",
        why_related="Docker relates to Kubernetes (both container orchestration) - tests graph entity linking",
        expected_found=True,  # Should find via entity similarity
    ),

    # -------------------------------------------------------------------------
    # Test 8: MicroERP via business context (NO microerp mentioned)
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="microerp_via_context",
        memory_content=(
            "[WORKING_SOLUTION] The Romanian point-of-sale system used in retail "
            "stores requires specific ANAF compliance for fiscal receipts. "
            "Each transaction must be recorded with timestamp and unique ID. "
            "The software connects to hardware via serial port emulation."
        ),
        memory_category="business",
        memory_tags=["retail", "compliance", "romania"],
        query="MicroERP fiscal printer connection setup",
        why_related="Describes MicroERP context (Romanian POS, ANAF, retail) without naming it",
        expected_found=True,
    ),

    # -------------------------------------------------------------------------
    # Test 9: Debug strategy via approach
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="debug_via_approach",
        memory_content=(
            "[PREFERENCE] When something breaks, first check the logs. "
            "Add strategic print statements at function entry/exit. "
            "Validate assumptions with assertions. Use binary search "
            "to narrow down the problem area. Never assume, always verify."
        ),
        memory_category="debugging",
        memory_tags=["methodology", "troubleshooting"],
        query="How do I find and fix bugs in my code?",
        why_related="Describes debugging methodology without 'bug' or 'error' keywords",
        expected_found=True,
    ),

    # -------------------------------------------------------------------------
    # Test 10: Performance via symptoms
    # -------------------------------------------------------------------------
    ComplexMemoryTest(
        name="perf_via_symptoms",
        memory_content=(
            "[WORKING_SOLUTION] When API responses take too long, profile first. "
            "Check N+1 queries with EXPLAIN ANALYZE. Add database indexes on "
            "frequently filtered columns. Use connection pooling. Consider "
            "Redis caching for hot data. Async where IO-bound."
        ),
        memory_category="optimization",
        memory_tags=["backend", "database"],
        query="My application is slow, how to optimize?",
        why_related="Describes performance optimization without 'slow' or 'optimize' keywords",
        expected_found=True,
    ),
]


def add_memory(content: str, category: str, tags: list[str]) -> dict:
    """Add a memory via the API."""
    payload = {
        "content": content,
        "category": category,
        "tags": tags,
        "infer": True,  # Use LLM for fact extraction
    }
    response = requests.post(
        f"{BASE_URL}/api/memory/add",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    return response.json()


def search_memory(query: str, mode: str = "hybrid", enable_rerank: bool = True) -> dict:
    """Search memories with specified mode."""
    payload = {
        "query": query,
        "limit": 10,
        "mode": mode,
        "enable_rerank": enable_rerank,
        "enable_hyde": True,
    }
    response = requests.post(
        f"{BASE_URL}/api/memory/search",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    return response.json()


def search_raw(query: str) -> dict:
    """Search memories with raw vector search (no enhancements)."""
    payload = {
        "query": query,
        "limit": 10,
        "raw": True,
    }
    response = requests.post(
        f"{BASE_URL}/api/memory/search",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    return response.json()


def delete_memory(memory_id: str) -> dict:
    """Delete a memory."""
    response = requests.post(
        f"{BASE_URL}/api/memory/forget",
        headers=HEADERS,
        json={"memory_id": memory_id},
        timeout=10,
    )
    return response.json()


def run_complex_tests():
    """Run all complex memory tests and report results."""
    print("\n" + "=" * 70)
    print("CEMS COMPLEX MEMORY RETRIEVAL TESTS")
    print("Testing semantic reasoning, not keyword matching")
    print("=" * 70 + "\n")

    added_memories: list[tuple[str, str]] = []  # (test_name, memory_id)
    results = []

    # Step 1: Add all test memories
    print("STEP 1: Adding test memories...")
    print("-" * 50)
    
    for test in COMPLEX_TESTS:
        try:
            result = add_memory(test.memory_content, test.memory_category, test.memory_tags)
            if result.get("success"):
                mem_results = result.get("results", [])
                for mem in mem_results:
                    if mem.get("event") in ("ADD", "UPDATE") and mem.get("id"):
                        added_memories.append((test.name, mem["id"]))
                        print(f"  ✅ {test.name}: {mem['id'][:8]}...")
                        break
                else:
                    print(f"  ⚠️ {test.name}: Added but no ID returned")
            else:
                print(f"  ❌ {test.name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ❌ {test.name}: {e}")

    print(f"\nAdded {len(added_memories)} memories. Waiting 3s for indexing...")
    time.sleep(3)

    # Step 2: Run searches and compare modes
    print("\n" + "=" * 70)
    print("STEP 2: Running searches (comparing raw vs enhanced)")
    print("=" * 70)

    for test in COMPLEX_TESTS:
        print(f"\n--- Test: {test.name} ---")
        print(f"Query: '{test.query}'")
        print(f"Why related: {test.why_related}")
        print(f"Expected: {'SHOULD find' if test.expected_found else 'Should NOT find'}")
        
        # Raw search (old behavior)
        raw_result = search_raw(test.query)
        raw_found = False
        raw_contents = []
        for r in raw_result.get("results", []):
            content = r.get("content", "")[:60]
            raw_contents.append(content)
            if any(keyword in content.lower() for keyword in test.memory_content.lower().split()[:5]):
                raw_found = True
        
        # Enhanced search (new behavior)
        enhanced_result = search_memory(test.query, mode="hybrid", enable_rerank=True)
        enhanced_found = False
        enhanced_contents = []
        for r in enhanced_result.get("results", []):
            content = r.get("content", "")[:60]
            enhanced_contents.append(content)
            if any(keyword in content.lower() for keyword in test.memory_content.lower().split()[:5]):
                enhanced_found = True

        # Check if our specific memory was found
        raw_has_test_mem = any(
            test.memory_content[:30].lower() in (r.get("content", "") or "").lower()
            for r in raw_result.get("results", [])
        )
        enhanced_has_test_mem = any(
            test.memory_content[:30].lower() in (r.get("content", "") or "").lower()
            for r in enhanced_result.get("results", [])
        )

        # Determine pass/fail
        if test.expected_found:
            passed = enhanced_has_test_mem
            improvement = enhanced_has_test_mem and not raw_has_test_mem
        else:
            passed = not enhanced_has_test_mem
            improvement = raw_has_test_mem and not enhanced_has_test_mem

        results.append({
            "name": test.name,
            "expected": test.expected_found,
            "raw_found": raw_has_test_mem,
            "enhanced_found": enhanced_has_test_mem,
            "passed": passed,
            "improvement": improvement,
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        if improvement:
            status += " (IMPROVED!)"

        print(f"  Raw search: {len(raw_result.get('results', []))} results, test memory found: {raw_has_test_mem}")
        print(f"  Enhanced:   {len(enhanced_result.get('results', []))} results, test memory found: {enhanced_has_test_mem}")
        print(f"  Result: {status}")

        # Show what was found
        if enhanced_result.get("results"):
            print("  Enhanced found:")
            for r in enhanced_result.get("results", [])[:3]:
                print(f"    - [{r.get('category', 'N/A')}] {r.get('content', '')[:50]}...")

    # Step 3: Cleanup
    print("\n" + "=" * 70)
    print("STEP 3: Cleanup")
    print("=" * 70)
    
    for test_name, memory_id in added_memories:
        try:
            delete_memory(memory_id)
            print(f"  🗑️ Deleted: {test_name}")
        except Exception as e:
            print(f"  ⚠️ Failed to delete {test_name}: {e}")

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["passed"])
    improved = sum(1 for r in results if r["improvement"])
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"Improvements over raw: {improved}/{total}")

    print("\nDetailed results:")
    print("-" * 50)
    for r in results:
        status = "✅" if r["passed"] else "❌"
        imp = " 🚀" if r["improvement"] else ""
        print(f"  {status}{imp} {r['name']}: raw={r['raw_found']}, enhanced={r['enhanced_found']}")

    print("\n" + "=" * 70)

    return passed, total


if __name__ == "__main__":
    passed, total = run_complex_tests()
    exit(0 if passed >= total * 0.7 else 1)  # Pass if 70%+ succeed
