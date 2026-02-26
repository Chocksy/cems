# User Profile Injection Implementation Plan

## Codex-Investigator Review Summary (2026-02-04)

Key findings from agent review:
1. **Plan is over-engineered for MVP** - Full profile tables and extraction is premature
2. **Simpler approach: Profile Probe** - Do a quick "preference search" at query time
3. **No new infrastructure needed** - Can test with existing search functionality
4. **Validate first** - Manual test before any code changes

### Recommended Simplified MVP

```python
# For preference queries, add a "profile probe":
profile_queries = ["I use I prefer my favorite tools style"]
profile_results = search(profile_queries, limit=5)
profile_context = extract_key_phrases(profile_results)
# Inject profile_context into HyDE prompt
```

### Revised Implementation Order

1. **Phase 0**: Manual validation (inject profile into prompt, test manually)
2. **Phase 1**: Profile probe MVP (no new tables, ~2-4 hours)
3. **Phase 2**: Measure impact on LongMemEval
4. **Phase 3**: Full infrastructure (only if MVP succeeds)

---

## Original Plan (kept for reference)

## Problem Statement

Our single-session-preference recall is at 66.7% (12/18). The remaining 6 failures all share a common pattern: they require **pre-existing knowledge of user preferences** that isn't being retrieved because the query terms don't overlap with the stored preferences.

Examples of failures:
- "recommend a cocktail" → needs to know "user took mixology class, likes Hendrick's gin"
- "furniture arrangement tips" → needs to know "user likes mid-century modern, wants walnut dresser"
- "battery life tips" → needs to know "user bought a portable power bank"

## Supermemory's Approach

From their article, they implement:

1. **User Profile at Session Start**: Inject static + episodic content before any query
2. **Hybrid Memory**: Not just RAG, but fact extraction + temporal tracking + profile building
3. **Automatic Capture**: Store all conversation turns, not just tool calls

Their LongMemEval score: **81.6%** (we're at **88.0%** overall, but they may be better on preference queries)

## Proposed Solution: Profile-Augmented Retrieval

### Architecture

```
CURRENT FLOW:
  Query → Synthesis → HyDE → Retrieval → Results

PROPOSED FLOW:
  Session Start → Fetch User Profile → Cache in Memory
  Query + Profile Context → Enhanced Synthesis → HyDE → Retrieval → Results
```

### Key Components

#### 1. User Profile Schema

Add a structured profile that summarizes user preferences:

```python
class UserProfile:
    user_id: str
    # Static preferences (rarely change)
    coding_style: list[str]  # ["prefer functions over classes", "use TypeScript"]
    tools: list[str]  # ["Adobe Premiere Pro", "Sony A7R IV", "Hendrick's gin"]
    interests: list[str]  # ["mid-century modern design", "mixology", "stand-up comedy"]
    role: str  # "founder", "student", "engineer"

    # Episodic/current context
    current_goals: list[str]  # ["migrate to new Postgres provider", "reduce costs"]
    recent_topics: list[str]  # ["debugging auth bug", "power bank purchase"]

    # Metadata
    last_updated: datetime
    version: int
```

#### 2. Profile Extraction

When memories are added, extract preference signals:

```python
PREFERENCE_PATTERNS = [
    "I use", "I prefer", "my favorite", "I really like",
    "I'm into", "I work with", "I recently bought",
    "I took a class", "I'm learning", "my style is"
]

def extract_preferences(content: str) -> list[str]:
    """Extract preference statements from memory content."""
    # Use LLM or pattern matching to identify preferences
    pass
```

#### 3. Profile Injection Points

**Option A: Query-Time Injection (Recommended for first implementation)**
- Before query synthesis, fetch top N profile facts
- Append to synthesis prompt: "User context: likes mid-century modern, bought power bank"
- Lower latency impact, targeted injection

**Option B: Session-Start Injection (Supermemory approach)**
- On session init, fetch full profile
- Inject into system prompt or context window
- Higher coverage but uses context tokens

**Option C: Hybrid Injection**
- Session start: inject static preferences
- Query time: inject relevant episodic context

### Implementation Phases

#### Phase 1: Profile Table + Extraction (Foundation)

1. Add `user_profiles` table:
   ```sql
   CREATE TABLE user_profiles (
       user_id TEXT PRIMARY KEY,
       profile_data JSONB NOT NULL,
       embedding vector(768),  -- For profile similarity search
       updated_at TIMESTAMP DEFAULT NOW()
   );
   ```

2. Add preference extraction to memory write pipeline
3. Periodic profile aggregation job (or on-demand)

#### Phase 2: Query-Time Profile Injection

1. Before synthesis, fetch user profile
2. Inject relevant preferences into synthesis prompt
3. Inject into HyDE prompt for better hypothetical generation

#### Phase 3: Profile-Aware Retrieval

1. Add profile embedding to search (low weight)
2. Boost memories that match profile topics
3. Add profile context to reranker (if enabled)

#### Phase 4: Session-Start Injection (Optional)

1. Add `/profile` endpoint to fetch user profile
2. Claude Code plugin injects at session start
3. Track profile staleness and refresh

## Validation Plan

### Test Cases

1. **Cocktail query**: "recommend a cocktail for a get-together"
   - Expected: Should find mixology class memory
   - Validation: Check if "mixology" or "Hendrick's" in retrieved results

2. **Furniture query**: "tips for rearranging bedroom furniture"
   - Expected: Should find mid-century modern preference
   - Validation: Check if "mid-century" in retrieved results

3. **Battery query**: "battery life tips for my phone"
   - Expected: Should find power bank purchase
   - Validation: Check if "power bank" in retrieved results

### Metrics

- single-session-preference Recall@5: Target 80%+ (from 66.7%)
- Overall Recall@5: Maintain 88%+
- Latency: <100ms overhead for profile fetch

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Profile staleness | Add `updated_at`, periodic refresh |
| Profile extraction errors | Use LLM for extraction, human review |
| Context bloat | Limit profile to top N preferences |
| Latency overhead | Cache profile in memory, async refresh |
| Privacy concerns | Profile is per-user, encrypted at rest |

## Questions for Review

1. Should profile extraction happen synchronously on write, or async batch?
2. How many profile facts to inject at query time? (5? 10? 20?)
3. Should we inject full profile or only query-relevant facts?
4. How to handle profile conflicts/updates (old vs new preferences)?
5. Should profile be visible to user for editing?

## Files to Create/Modify

| File | Change | Priority |
|------|--------|----------|
| `src/cems/models.py` | Add `UserProfile` model | High |
| `src/cems/db/profile.py` | Profile storage/retrieval | High |
| `src/cems/profile.py` | Profile extraction logic | High |
| `src/cems/retrieval.py` | Inject profile into synthesis/HyDE | High |
| `src/cems/memory/write.py` | Extract preferences on write | Medium |
| `scripts/migrate_profiles.sql` | DB migration | High |
| `tests/test_profile.py` | Profile extraction tests | Medium |

## Success Criteria

- [ ] Profile extraction identifies preferences from memory content
- [ ] Profile injection improves synthesis for preference queries
- [ ] single-session-preference Recall@5 reaches 80%+
- [ ] No regression on other query types
- [ ] Latency overhead <100ms
