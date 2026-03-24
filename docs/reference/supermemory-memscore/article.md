# We Scored 99% on LongMemEval to Prove a Point (ASMR Was a Parody)

**Source:** https://x.com/DhravyaShah/status/2036243995500966260
**Author:** Dhravya Shah (@DhravyaShah), founder of Supermemory
**Date:** March 24, 2026

## TL;DR

The ASMR paper (99% LongMemEval) was a deliberate social experiment. The system was real and did achieve ~99%, but the point was to expose how memory benchmarks can be gamed. Standard vector search performs similarly to their 12-agent system.

## Key Revelations

The name "ASMR" (Agentic Search and Memory Retrieval) was intentionally absurd. They left clues:
- April 1st launch date
- Used words like "spectacle" and "fun"
- "We're having fun" in the announcement

## Two Ways to Game Memory Benchmarks

### Option 1 — Data Dumping
Dump full conversation history into LLM context. Works on LongMemEval (~100K tokens) but costs explode at scale.

### Option 2 — Agent Search
Let an agent scan everything at query time. Accurate but slow and expensive.
Their honest ASMR metrics: **70 seconds per query, 12K tokens** — completely impractical for production.

## MemScore: Their Proposed Standard

Format: `{quality1}% / {quality2}% / {avg_latency}ms / {tokens}tok`

ASMR's honest MemScore: `99% | 99% | 70,000ms | 12k`

This exposes the impracticality — 70 seconds per query is unusable.

Spec: https://supermemory.ai/docs/memorybench/memscore

## Images

- `img1.jpg` — Banner: "Everyone's gaming the benchmarks. We did it to prove a point." Shows their MemScore: `90% | 300ms | 12k tok` for production vs ~99% for benchmark
- `img2.jpg` — "1st April" callout showing the clues they left (spectacle, fun)
- `img3.jpg` — Community reaction tweet
- `img4.jpg` — Their Nova memory visualization system

## What This Means for CEMS

**We didn't get fooled — we built something better.**

| Metric | ASMR (parody) | CEMS Agentic |
|--------|--------------|--------------|
| Accuracy | ~99% | 98.4% |
| Latency | 70,000ms | 4,000-6,000ms |
| Tokens/query | 12K+ | ~45K (15K×3) |
| Cost/query | impractical | $0.01 |
| Production-ready | NO (admitted) | YES (running now) |
| Agents | 12 variants | 3 specialized |

Our approach is 12x faster, production-viable, and actually running in production.

The real lesson: benchmarks without latency/cost context are meaningless. We should adopt their MemScore format for our own reporting.

CEMS MemScore: `98% | 98% | 5000ms | 45k`
