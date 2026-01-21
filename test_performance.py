#!/usr/bin/env python3
"""Performance and relevance test for CEMS memory system.

This script:
1. Ingests ~100-200 memories from RSpec testing guidelines
2. Tests search performance (should be fast with batch operations)
3. Verifies relevance of search results

Uses direct REST API instead of MCP protocol for simpler testing.
"""

import json
import subprocess
import time
from typing import Any

# Configuration - Direct REST API via docker exec
PYTHON_API = "http://localhost:8765"
API_KEY = "cems_ak_89a65317e256bf43ab5641bfe929809386b1e8c57b0c2d8e"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# RSpec Testing Guidelines Memories - broken down into individual, searchable facts
TESTING_MEMORIES = [
    # Core Guidelines
    ("patterns", "RSpec: ALWAYS use 'its', 'its_block', 'its_call', 'its_map' from rspec-its and moarspec for ALL examples"),
    ("patterns", "RSpec: NEVER use standalone descriptive 'it' blocks like 'it \"should do something\" do...end' - this is ABSOLUTELY FORBIDDEN"),
    ("patterns", "RSpec: MUST define a subject (named or implicit) within describe blocks"),
    ("patterns", "RSpec: Use 'is_expected.to' instead of explicit 'expect(subject)' - STRONGLY discouraged to use explicit expect()"),
    ("patterns", "RSpec: Each its* or it{} block MUST contain only ONE logical assertion"),
    ("patterns", "RSpec: Chaining expectations with .and is ALLOWED and encouraged: is_expected.to matcher1.and matcher2"),
    
    # Context Structure
    ("patterns", "RSpec: Context names MUST follow patterns: 'when ...', 'with ...', 'without ...'"),
    ("anti-patterns", "RSpec: BAD context name: 'user is admin' - GOOD: 'when user is admin'"),
    ("patterns", "RSpec: Keep context nesting level minimal for readability"),
    
    # Let Definitions
    ("patterns", "RSpec: Use 'let' for test data setup, not instance variables"),
    ("patterns", "RSpec: Use 'super().merge(...)' for context-specific overrides in let blocks"),
    ("anti-patterns", "RSpec: AVOID putting assertion logic or excessive stubbing in let blocks"),
    ("anti-patterns", "RSpec: NEVER use instance variables in specs (@user) - always use let"),
    ("anti-patterns", "RSpec: AVOID let! unless absolutely necessary - prefer lazy-loaded let"),
    ("anti-patterns", "RSpec: NO before blocks with multiple lines - use let for setup instead"),
    
    # Subject Definition
    ("patterns", "RSpec: Named subject example: subject(:action) { SomeApplicationAction.new(params) }"),
    ("patterns", "RSpec: Implicit subject within describe: describe '#process' do; subject { instance.process }; end"),
    
    # Helper/Matcher Usage
    ("patterns", "RSpec: ALWAYS investigate available helpers before writing manual assertions for jobs, mailers, validations"),
    ("patterns", "RSpec: Use ActiveJob::TestHelper for job testing - have_enqueued_job, perform_enqueued_jobs"),
    ("patterns", "RSpec: Use ActionMailer::TestHelper for mailer testing"),
    ("patterns", "RSpec: Use shoulda-matchers for validations: it { is_expected.to validate_presence_of(:name) }"),
    ("patterns", "RSpec: Use moarspec helpers: its_call, invoke, ret"),
    ("patterns", "RSpec: Check spec/support/matchers/ for custom matchers like call_action, change_reloaded"),
    
    # Job Testing
    ("patterns", "RSpec: Test job enqueue with: its_block { is_expected.to have_enqueued_job(MyJob).with(param: true) }"),
    ("anti-patterns", "RSpec: BAD job test: expect { MyJob.perform_later(id: 1) }.to change(MyJob.jobs, :size).by(1)"),
    ("patterns", "RSpec: Test job execution with perform_enqueued_jobs block"),
    ("patterns", "RSpec: Example job test: its_block { is_expected.to have_enqueued_job(MyJob).with(id: 1).on_queue('default') }"),
    
    # Validation Testing
    ("anti-patterns", "RSpec: BAD validation test: subject.name = ''; subject.valid?; expect(subject.errors[:name]).to include(...)"),
    ("patterns", "RSpec: GOOD validation test: it { is_expected.to validate_presence_of(:name) }"),
    ("patterns", "RSpec: Use shoulda-matchers gem for all validation specs"),
    
    # Action/Service Testing
    ("patterns", "RSpec: Use custom call_action matcher: its_block { is_expected.to call_action(ProcessingAction).with(order: order) }"),
    ("anti-patterns", "RSpec: BAD: expect(ProcessingAction).to receive(:call).with(order: order)"),
    ("patterns", "RSpec: Use moarspec invoke matcher: its_block { is_expected.to invoke(:notify).on(NotifierService).with(user) }"),
    
    # Database Persistence
    ("patterns", "RSpec: Use change_reloaded matcher: its_block { is_expected.to change_reloaded(instance, :status).from('pending').to('completed') }"),
    ("anti-patterns", "RSpec: BAD: subject.update_status('completed'); expect(subject.reload.status).to eq('completed')"),
    
    # Matcher Usage
    ("patterns", "RSpec: Use semantic matchers over raw eq/== for complex objects"),
    ("patterns", "RSpec: Use have_attributes for object comparison: its(:user) { is_expected.to have_attributes(name: 'Test') }"),
    ("patterns", "RSpec: Use match_array for array comparison: its(:users) { is_expected.to match_array([user1, user2]) }"),
    ("anti-patterns", "RSpec: BAD: its(:user) { is_expected.to eq(User.new(name: 'Test', email: 'test@example.com')) }"),
    ("patterns", "RSpec: Use be_, be_successful, be > 0 for semantic assertions"),
    
    # Factory Usage
    ("patterns", "RSpec: Use traits for different states: create(:user, :admin), create(:project, :archived)"),
    ("anti-patterns", "RSpec: BAD: create(:admin_user) - separate factory for state"),
    ("patterns", "RSpec: Use build vs create when persistence not needed to minimize DB interactions"),
    ("patterns", "RSpec: Use build_list(:user, 3) when persistence not needed"),
    ("patterns", "RSpec: Keep factories minimal - use traits for additional attributes"),
    
    # Shared Context/Examples
    ("patterns", "RSpec: Extract shared setup into shared_context, not before blocks"),
    ("patterns", "RSpec: Use shared_examples ONLY for truly reusable specs - avoid overuse"),
    ("patterns", "RSpec: Prefer explicit contexts over shared examples when possible"),
    ("patterns", "RSpec: shared_context 'with admin user' do; let(:current_user) { create(:user, :admin) }; end"),
    
    # Test Coverage
    ("patterns", "RSpec: Test edge cases and error conditions - not just happy paths"),
    ("anti-patterns", "RSpec: NO testing of private methods - only public interface"),
    ("anti-patterns", "RSpec: NO stubbing of the system under test - only external dependencies"),
    ("patterns", "RSpec: Test error conditions: context 'when service fails' do; before { allow(ExternalService).to... }"),
    
    # Describe Blocks
    ("patterns", "RSpec: Use describe for methods: describe '#method_name' or describe 'response body'"),
    ("patterns", "RSpec: Define subject within describe blocks if needed"),
    ("patterns", "RSpec: Example: describe '#process' do; subject { service.process(params) }; context 'with valid params'..."),
    
    # More its* Examples
    ("patterns", "RSpec: its(:status) { is_expected.to eq(200) }"),
    ("patterns", "RSpec: its(:name) { is_expected.to start_with('Test') }"),
    ("patterns", "RSpec: its(:count) { is_expected.to be > 0 }"),
    ("patterns", "RSpec: its_call(:arg) { is_expected.to ret true } - moarspec ret matcher"),
    
    # Chaining Examples
    ("patterns", "RSpec: Chained assertion: is_expected.to change { Model.count }.by(1).and change { OtherModel.count }.by(-1)"),
    ("patterns", "RSpec: Chained with invoke: is_expected.to change { user.name }.to('New Name').and invoke(:some_method).on(instance)"),
    
    # Common Anti-patterns with Explanations
    ("anti-patterns", "RSpec: FORBIDDEN - Multiple assertions in one block: its(:data) { is_expected.to include(key1: 'value1'); is_expected.to include(key2: 'value2') }"),
    ("anti-patterns", "RSpec: FORBIDDEN - Using explicit expect: its(:value) { expect(subject).to eq(5) }"),
    ("anti-patterns", "RSpec: FORBIDDEN - Dynamic spec generation with .each"),
    ("anti-patterns", "RSpec: FORBIDDEN - Deep context nesting"),
    ("anti-patterns", "RSpec: FORBIDDEN - Over-stubbing or expectations in let definitions"),
    
    # Response Testing
    ("patterns", "RSpec: Response testing: describe 'response' do; subject(:response) { get('/some/url') }; its(:status) { is_expected.to eq 200 }"),
    ("patterns", "RSpec: Response body parsing: describe '.body' do; subject { JSON.parse(response.body, symbolize_names: true) }"),
    
    # Error Testing
    ("patterns", "RSpec: Test raises error: its_block { is_expected.to raise_error(ValidationError) }"),
    ("patterns", "RSpec: Test specific error: its_block { is_expected.to raise_error(MyFailingJob::SpecificError) }"),
    
    # Permissions Testing
    ("patterns", "RSpec: Permission context: context 'with manager permissions' do; let(:current_user_role) { :manager }"),
    ("patterns", "RSpec: Permission assertion: its(:permissions) { is_expected.to include(:manage_organization) }"),
    ("patterns", "RSpec: Negative permission: its(:permissions) { is_expected.not_to include(:manage_organization) }"),
    
    # Additional Variations
    ("patterns", "When writing RSpec tests, always start by checking if there are existing helpers or matchers for common patterns"),
    ("patterns", "For testing background jobs in RSpec, use have_enqueued_job matcher from ActiveJob::TestHelper"),
    ("patterns", "In RSpec, context blocks should describe scenarios using 'when', 'with', or 'without' prefix"),
    ("anti-patterns", "Never write 'it \"should do X\"' in RSpec - use its* helpers instead"),
    ("anti-patterns", "Avoid using @instance_variables in RSpec specs - they make tests harder to understand"),
    ("patterns", "RSpec subjects should be defined at the beginning of describe/context blocks"),
    ("patterns", "Use let(:foo) { super().merge(...) } to override parent let definitions in nested contexts"),
    ("patterns", "For model validations in RSpec, prefer shoulda-matchers over manual validation tests"),
    ("patterns", "Test service objects by testing their public interface, not private methods"),
    ("patterns", "In RSpec request specs, use its(:status) to test HTTP status codes"),
    ("patterns", "When testing mailers in RSpec, use ActionMailer::TestHelper methods"),
    ("patterns", "For database change assertions, prefer change_reloaded custom matcher over manual reload"),
    ("anti-patterns", "Don't use multiple expect statements in a single it block - split into multiple its* calls"),
    ("patterns", "RSpec describe blocks should focus on a single method or logical grouping"),
    ("patterns", "Use factory traits instead of creating separate factories for different object states"),
    ("anti-patterns", "Avoid stubbing the class under test - only stub external dependencies"),
    ("patterns", "Test error conditions with its_block { is_expected.to raise_error(ErrorClass) }"),
    ("patterns", "For complex object assertions, use have_attributes matcher instead of eq"),
    ("patterns", "Use match_array when testing collections where order doesn't matter"),
    ("patterns", "shared_examples should only be used for truly identical behavior across contexts"),
    ("anti-patterns", "Deep context nesting (>3 levels) makes tests hard to read - refactor"),
    ("patterns", "Use build(:factory) instead of create(:factory) when you don't need database persistence"),
    ("patterns", "let blocks should be simple - avoid complex logic or assertions inside them"),
    ("patterns", "Context 'when user is authenticated' should use 'when' prefix for conditional scenarios"),
    ("patterns", "Context 'with valid params' should use 'with' prefix for input variations"),
    ("patterns", "Context 'without required field' should use 'without' prefix for absence scenarios"),
    
    # More specific examples
    ("patterns", "RSpec job execution test: perform_enqueued_jobs { is_expected.to raise_error(MyJob::SpecificError) }"),
    ("patterns", "RSpec mailer delivery count: is_expected.to change { ActionMailer::Base.deliveries.count }.by(1)"),
    ("patterns", "RSpec chained mailer and change: is_expected.to change { user.name }.to('New Name').and change { ActionMailer::Base.deliveries.count }.by(1)"),
    
    # Project-specific
    ("patterns", "Check spec/support/matchers/ directory for project-specific custom matchers"),
    ("patterns", "Custom matchers like call_action should be used within its_block"),
    ("patterns", "moarspec provides its_call, invoke, ret helpers for method call testing"),
    ("patterns", "rspec-its gem provides its, its_block, its_map helpers"),
    
    # Style preferences
    ("preferences", "RSpec style: Prefer one-liner its* syntax over multi-line it blocks"),
    ("preferences", "RSpec style: Group related assertions using .and chaining"),
    ("preferences", "RSpec style: Use named subjects for clarity: subject(:result) { ... }"),
    ("preferences", "RSpec style: Keep spec files organized: describe Class, describe '#method', context 'scenario'"),
    
    # Additional guidelines
    ("guidelines", "Always use subject when the describe block is testing a specific method return value"),
    ("guidelines", "Named subject improves test readability: subject(:api_response) { get('/api/users') }"),
    ("guidelines", "Use its_block for testing side effects like database changes or job enqueues"),
    ("guidelines", "Use its for testing simple attributes or method return values"),
    ("guidelines", "Prefer explicit setup in let blocks over implicit setup in before blocks"),
]


def call_api(method: str, endpoint: str, data: dict | None = None) -> dict:
    """Call the REST API directly via docker exec."""
    curl_cmd = [
        "docker", "exec", "cems-server",
        "curl", "-s", "-X", method,
        f"http://localhost:8765{endpoint}",
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


def ingest_memories(use_infer: bool = False):
    """Ingest all testing memories.
    
    Args:
        use_infer: If True, use LLM for fact extraction (slow, ~5-10s per memory).
                   If False, store raw content directly (fast, ~0.1-0.2s per memory).
    """
    mode = "with LLM inference" if use_infer else "direct import (fast)"
    print(f"\n{'='*60}")
    print(f"Ingesting {len(TESTING_MEMORIES)} memories {mode}...")
    print(f"{'='*60}\n")
    
    success_count = 0
    error_count = 0
    total_time = 0
    
    for i, (category, content) in enumerate(TESTING_MEMORIES, 1):
        try:
            start = time.time()
            result = call_api("POST", "/api/memory/add", {
                "content": content,
                "category": category,
                "infer": use_infer,  # Fast mode when False
            })
            elapsed = time.time() - start
            total_time += elapsed
            
            if result.get("success"):
                success_count += 1
            else:
                error_count += 1
                print(f"  Error on memory {i}: {result.get('error', 'unknown')}")
            
            if i % 20 == 0:
                avg = total_time / i
                print(f"  Progress: {i}/{len(TESTING_MEMORIES)} (avg: {avg:.2f}s/memory)")
        except Exception as e:
            error_count += 1
            print(f"  Error on memory {i}: {e}")
    
    print(f"\n✅ Ingested: {success_count}/{len(TESTING_MEMORIES)}")
    print(f"❌ Errors: {error_count}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⏱️  Avg per memory: {total_time/len(TESTING_MEMORIES):.2f}s")
    
    return success_count, error_count


def test_search_performance(query: str, expected_keywords: list[str]) -> dict:
    """Test search and verify relevance."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        result = call_api("POST", "/api/memory/search", {
            "query": query,
            "limit": 10,
        })
        elapsed = time.time() - start
        
        results = result.get("results", [])
        
        print(f"⏱️  Search time: {elapsed:.3f}s")
        print(f"📊 Results: {len(results)}")
        
        # Check relevance
        relevant_count = 0
        for r in results[:5]:
            memory_content = r.get("content", "").lower()
            is_relevant = any(kw.lower() in memory_content for kw in expected_keywords)
            if is_relevant:
                relevant_count += 1
            
            status = "✓" if is_relevant else "✗"
            score = r.get("score", 0)
            short_content = r.get("content", "")[:80] + "..."
            print(f"  {status} [{score:.3f}] {short_content}")
        
        relevance_pct = (relevant_count / min(5, len(results)) * 100) if results else 0
        print(f"\n🎯 Relevance: {relevant_count}/5 ({relevance_pct:.0f}%)")
        
        return {
            "query": query,
            "time": elapsed,
            "count": len(results),
            "relevance": relevance_pct,
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"query": query, "error": str(e)}


def run_tests():
    """Run all performance and relevance tests."""
    test_queries = [
        ("how is testing implemented in this project", ["rspec", "test", "spec", "its"]),
        ("write tests for this controller", ["rspec", "test", "its", "describe", "subject"]),
        ("how to write RSpec tests", ["rspec", "its", "describe", "context", "subject"]),
        ("what are the testing guidelines", ["rspec", "guidelines", "patterns", "forbidden"]),
        ("how to test background jobs", ["job", "enqueued", "have_enqueued_job", "perform"]),
        ("how to test validations", ["validate", "shoulda", "validation"]),
        ("what is forbidden in RSpec", ["forbidden", "never", "avoid", "bad"]),
        ("how to use its_block", ["its_block", "is_expected", "change"]),
        ("testing service objects", ["service", "action", "call_action", "invoke"]),
        ("RSpec context naming", ["context", "when", "with", "without"]),
        ("how to use let in RSpec", ["let", "super", "merge"]),
        ("factory best practices", ["factory", "trait", "build", "create"]),
        ("shared examples usage", ["shared", "examples", "context"]),
        ("testing error conditions", ["error", "raise_error", "exception"]),
        ("matchers for complex objects", ["have_attributes", "match_array", "semantic"]),
    ]
    
    print("\n" + "="*60)
    print("CEMS PERFORMANCE & RELEVANCE TEST")
    print("="*60)
    
    # Step 1: Ingest memories using fast direct import mode (infer=False)
    # This skips LLM fact extraction but is ~50x faster
    success, errors = ingest_memories(use_infer=False)
    
    if errors > len(TESTING_MEMORIES) / 2:
        print("\n❌ Too many errors during ingestion. Aborting tests.")
        return
    
    # Wait a moment for indexing
    print("\nWaiting 2s for indexing...")
    time.sleep(2)
    
    # Step 2: Run search tests
    print("\n" + "="*60)
    print("SEARCH PERFORMANCE TESTS")
    print("="*60)
    
    results = []
    for query, keywords in test_queries:
        result = test_search_performance(query, keywords)
        results.append(result)
        time.sleep(0.5)  # Small delay between queries
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in results if "error" not in r]
    avg_time = sum(r["time"] for r in successful) / len(successful) if successful else 0
    avg_relevance = sum(r["relevance"] for r in successful) / len(successful) if successful else 0
    
    print(f"\n📊 Total queries: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"⏱️  Avg search time: {avg_time:.3f}s")
    print(f"🎯 Avg relevance: {avg_relevance:.0f}%")
    
    if avg_time > 1.0:
        print("\n⚠️  WARNING: Search times are slow (>1s). Check for N+1 queries.")
    else:
        print("\n✅ Search performance is good (<1s).")
    
    if avg_relevance < 60:
        print("⚠️  WARNING: Relevance is low (<60%). Check embedding quality.")
    else:
        print("✅ Relevance is acceptable (≥60%).")


if __name__ == "__main__":
    run_tests()
