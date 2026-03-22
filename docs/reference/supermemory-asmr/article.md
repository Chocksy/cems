# We Broke the Frontier in Agent Memory: Introducing ~99% SOTA Memory System

**Source:** https://x.com/DhravyaShah/status/2035517012647272689
**Author:** Dhravya Shah (@DhravyaShah), founder of Supermemory
**Date:** March 22, 2026
**GitHub:** https://github.com/supermemoryai

## Key Claim

Supermemory achieved **~99% on LongMemEval_s** benchmark using a new technique called **ASMR (Agentic Search and Memory Retrieval)**.

## Images

- `cover.jpg` — Banner: "#1 in agent memory. Broke the frontier. Introducing ASMR" (~99% vs others barely 90%)
- `architecture-diagram.jpg` — Full ASMR pipeline: Data Ingestion → Knowledge Store → Active Search Orchestration → Decision Forest & Consensus
- `results-table.png` — Detailed comparison table by question category across systems
- `results-chart.jpg` — Bar chart visualization of results across all categories

---

## Full Article Text

Agent memory may now be essentially solved. Within years, billions of agents will be personalized per user, continuously learning from interactions.

Supermemory previously published research showing ~85% on LongMemEval-s, leading all publicly benchmarked systems. Their new result: **~99% on LongMemEval_s**.

> **Important caveat:** "This is not in our main production Supermemory engine (yet). Rather, this blog covers a new, highly experimental agentic flow."

### Introduction

LongMemEval is a rigorous benchmark simulating real production chaos: 115k+ token histories, contradictory info, multi-session events, and time-based reasoning. Most systems fail at retrieval, not reasoning — specifically, distinguishing stale facts from newer corrections.

### Architecture: ASMR (Agentic Search and Memory Retrieval)

Key properties:
- Simple to implement
- Requires no Vector Database or embeddings
- Can run completely in-memory, even embedded in robots

#### 1. Parallel Ingestion via Observer Agents

Three parallel reader agents (Gemini 2.0 Flash) process raw sessions concurrently, extracting knowledge across six categories:
- Personal Information
- Preferences
- Events
- Temporal Data
- Updates
- Assistant Info

The 115k+ token conversation histories are distributed round-robin across observers (sessions 1,4,7... | 2,5,8... | 3,6,9...).

Each observer extracts structured findings stored as pure structured storage — **no embeddings**.

#### 2. Active Agentic Retrieval

Three parallel search agents reason over stored findings:
- **Agent 1 (Direct Seeker):** Exact match retrieval, literal fact extraction, recent-first prioritization
- **Agent 2 (Inference Engine):** Related context discovery, implication analysis, supporting evidence retrieval
- **Agent 3 (Temporal Navigator):** Timeline reconstruction, duration calculation, state change tracking

#### 3. Answering Ensembles

**Run 1 — 8-Variant Ensemble: 98.60% accuracy**
- 8 specialized prompt variants run in parallel (Precise Counter, Time Specialist, Context Deep Dive, etc.)
- If any variant arrives at the correct answer, it counts
- This is the "oracle" best-of-8 approach

**Run 2 — 12-Variant Decision Forest: 97.20% accuracy**
- 12 agents (GPT-4o-mini) answer independently
- An Aggregator LLM (also GPT-4o-mini) synthesizes via:
  - Majority voting mechanism
  - Domain-weighted trust scores
  - Conflict resolution
- This is the "democratic consensus" approach — more realistic for production

### Results by Category

| Category | 8-Variant Ensemble | 12-Variant Forest | Initial | Mastra | EmergenceMem | Zep |
|----------|-------------------|-------------------|---------|--------|--------------|-----|
| Knowledge Update | 100.00% | 100.00% | 89.74% | 83.33% | 83.33% | 100.00% |
| Single-session Assistant | 100.00% | 96.15% | 88.46% | 80.77% | 84.62% | 73.08% |
| Single-session User | 98.50% | 95.57% | 95.70% | - | - | - |
| Temporal Reasoning | 98.00% | 81.98% | 95.50% | 87% | 62.40% | - |
| Multi-session | 96.99% | 99.99% | 76.00% | - | - | 71.60% |
| Single-session Preferences | 98.67% | 98.67% | 76.00% | - | - | - |
| **OVERALL** | **98.60%** | **97.20%** | **85.20%** | **94.87%** | **80.06%** | **71.20%** |

### Key Learnings

1. **Agentic retrieval beats vector search** — eliminating semantic similarity traps around temporal updates. No embeddings needed.
2. **Parallel processing is critical** — dedicated agents improve speed and granularity vs single-agent approaches.
3. **Specialization beats generalization** — specialist agents (fact hunter, context weaver, timeline tracker) outperform a single master prompt.

### What's Next

The full experimental code will be open-sourced. A public build-out planned for early April 2026, with releases tracked at github.com/supermemoryai.

> *"Agent memory is now (probably) a solved problem?"*

---

## Relevance to CEMS

This is directly relevant to our system. Key parallels and differences:

### What we already have
- **Observer agents** extracting knowledge from sessions (our observer system V2)
- **Category-based extraction** (we use 28 canonical categories)
- **LongMemEval benchmark** (we score 98% Recall@5 on 50 questions)

### What they do differently (worth investigating)
- **No vector DB / no embeddings** — pure structured text storage with agentic retrieval
- **Parallel search agents** with specialized roles (Direct Seeker, Inference Engine, Temporal Navigator)
- **Ensemble answering** — multiple prompt variants + aggregation for higher accuracy
- **Round-robin session distribution** across parallel observers

### Ideas to explore
1. Add specialized search agents (temporal, inference, direct) instead of relying solely on vector similarity
2. Ensemble answering for high-stakes recall questions
3. Test our system on the full LongMemEval_s (not just 50 questions) to compare properly
4. Consider whether pure structured storage could complement/replace our vector approach
