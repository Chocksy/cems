"""Empirical Qwen3-32B vs Gemini-2.5-Flash-Lite cost/latency test.

Uses real prompts from lint.py + consolidation.py to compare:
- Tokens in/out
- Per-call cost at current OpenRouter pricing
- Latency
- Output quality/correctness
"""
import json
import os
import time
import httpx

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"

PRICING = {
    "qwen/qwen3-32b":               {"in": 0.080e-6, "out": 0.240e-6},
    "google/gemini-2.5-flash-lite": {"in": 0.100e-6, "out": 0.400e-6},
}

# --- Prompt 1: Lint contradiction check (short, expects yes/no) ---
CONTRADICTION_PROMPT = """Do these two memories contradict each other? Answer ONLY "yes" or "no".

Memory A:
User prefers dark mode for all dashboards. Confirmed on 2026-02-10 when they manually set theme=dark in settings and stated "I always use dark mode, hurts my eyes otherwise."

Memory B:
User switched to light mode on 2026-03-18 for the wiki dashboard specifically, saying "the light theme is easier to read long entity pages, even though I use dark mode elsewhere."

Contradiction means they make incompatible claims about the same topic.
Similar or overlapping information is NOT a contradiction.
Answer:"""

# --- Prompt 2: Consolidation-style dedup classification (medium length) ---
CONSOLIDATION_PROMPT = """Are these two memories about the same topic and safe to merge? Answer with JSON: {"merge": true/false, "reason": "..."}.

Memory A (category: deployment):
Deployed CEMS v0.12.3 to Coolify on Hetzner via Tailscale on 2026-03-20. Required manual migration of memory_conflicts table via psql because run_migrations() was missing the entry. Fixed by adding migration to database.py.

Memory B (category: deployment):
Incident 2026-03-20: wiki dashboard 500 error on Coolify/Hetzner production — root cause memory_conflicts table absent because scripts/migrate_conflicts.sql was never executed by Docker. Fix committed in run_migrations() with schema_migrations tracking table.

Answer:"""


def call_model(model: str, prompt: str, max_tokens: int = 200) -> dict:
    """Call OpenRouter, return dict with usage, latency, response text."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    r = httpx.post(URL, headers=headers, json=payload, timeout=60.0)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    text = data["choices"][0]["message"]["content"]
    return {
        "text": text,
        "latency_s": elapsed,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "gen_id": data.get("id"),
    }


def cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING[model]
    return in_tok * p["in"] + out_tok * p["out"]


def run_test(name: str, prompt: str, max_tokens: int, runs: int = 3):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    for model in ("qwen/qwen3-32b", "google/gemini-2.5-flash-lite"):
        results = []
        for i in range(runs):
            try:
                r = call_model(model, prompt, max_tokens=max_tokens)
                results.append(r)
            except Exception as e:
                print(f"  [{model}] run {i+1} FAILED: {e}")
        if not results:
            continue
        avg_in = sum(r["input_tokens"] for r in results) / len(results)
        avg_out = sum(r["output_tokens"] for r in results) / len(results)
        avg_lat = sum(r["latency_s"] for r in results) / len(results)
        avg_cost = sum(cost(model, r["input_tokens"], r["output_tokens"]) for r in results) / len(results)
        print(f"\n  {model}:")
        print(f"    avg input tokens:  {avg_in:.0f}")
        print(f"    avg output tokens: {avg_out:.0f}")
        print(f"    avg latency:       {avg_lat*1000:.0f}ms")
        print(f"    avg cost/call:     ${avg_cost*1000:.4f}/1000 calls  (${avg_cost*1e6:.2f}/M calls)")
        sample = (results[0]['text'] or '').strip()[:150]
        print(f"    sample output:     {sample!r}")


if __name__ == "__main__":
    # Qwen3 has a thinking mode that eats tokens before producing visible output.
    # Raise max_tokens so both models can complete. Also add /no_think to try to
    # suppress Qwen's reasoning phase (matches prod config in client.py).
    run_test("LINT — Contradiction check (yes/no)", CONTRADICTION_PROMPT + " /no_think", max_tokens=50, runs=3)
    run_test("CONSOLIDATION — Dedup classification (JSON)", CONSOLIDATION_PROMPT + " /no_think", max_tokens=200, runs=3)
