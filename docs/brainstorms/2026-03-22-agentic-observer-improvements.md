---
date: 2026-03-22
topic: agentic-observer-improvements
---

# Agentic Observer Improvements

## What We're Building

Adapt the observer's session summary extraction to categorize facts by ASMR-style types, and add relevance-based sorting to agentic context loading.

Two changes:

1. **Categorized extraction**: Each fact gets a category prefix (personal_info, preferences, events, temporal, updates, decisions) so search agents can reason about fact types
2. **Relevance-sorted context**: When loading memories for agentic search, sort by relevance feedback (relevant_count - noise_count) before filling the context budget

## Why This Approach

Our current flat facts got 98.4% on LongMemEval-S — great but the weakest areas are temporal-reasoning (96.2%) and preferences (93.3%). Categorized facts help the temporal navigator and inference engine agents reason more effectively about what type of information each fact represents.

## Key Decisions

- **Backwards compatible**: The output format stays the same (facts list + context). Facts just get a `[CATEGORY]` prefix. No schema changes needed.
- **6 fact categories** matching ASMR: `[PERSONAL]`, `[PREFERENCE]`, `[EVENT]`, `[TEMPORAL]`, `[UPDATE]`, `[DECISION]`
- **Relevance sorting**: Use `relevant_count - noise_count` as a signal, with `shown_count` as tiebreaker. High-relevance memories fill context first regardless of age.
- **Context budget**: 700K chars cap, filled in priority order: project → profile → high-relevance → recent

## Next Steps
→ Implement the prompt change + relevance sorting
