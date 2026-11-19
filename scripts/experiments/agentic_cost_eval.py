"""Agentic search cost/quality eval: 3 agents + RRF vs one merged call, across cheap models.

Pulls the real Chocksy/cems memory dump and entity index from the production API
(read-only list endpoints), replays real logged user prompts through each variant,
grades the returned top-3 with an LLM judge and prints a cost/quality table.

Usage:
    uv run python scripts/experiments/agentic_cost_eval.py --smoke
    uv run python scripts/experiments/agentic_cost_eval.py

Artifacts (dump, queries, raw results, grades, report) go to SCRATCH below.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from cems.agentic.rrf import reciprocal_rank_fusion
from cems.agentic.search import (
    ENTITY_PICKER_SYSTEM,
    ENTITY_PICKER_USER_PROMPT,
    EXCLUDED_MEMORY_CATEGORIES,
    MAX_CONTEXT_CHARS,
    MEMORIES_SYSTEM_PROMPT,
    MERGED_ROLE_INSTRUCTIONS,
    PROFILE_CATEGORIES,
    RECENT_DAYS,
    SEARCH_USER_PROMPT,
    _format_entities_for_picker,
    _format_memories_for_agents,
    _parse_agent_response,
)
from cems.lib.json_parsing import parse_json_dict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hooks"))
from cems_user_prompts_submit import _clean_search_query, is_confirmatory  # noqa: E402
from utils.credentials import resolve_credentials  # noqa: E402

SCRATCH = Path("/Volumes/External/caches/scratch/cems-cost-eval")
PROJECT = "Chocksy/cems"
OR_URL = "https://openrouter.ai/api/v1"

BASELINE_MODEL = "google/gemini-2.5-flash-lite"
CANDIDATES = [
    "~deepseek/deepseek-flash-latest",
    "tencent/hy4-flash",
    "qwen/qwen3.7-flash",
    "openai/gpt-5-nano",
    "~z-ai/glm-flash-latest",
]
JUDGE_MODEL = "anthropic/claude-haiku-5.5"

# Extra request fields for candidate models only (baseline requests stay byte-for-byte
# production). The smoke run with defaults showed why: the deepseek alias was spread over
# 7 providers in 5 searches (near-zero cache hits, up to $0.013 per call) and spent the
# whole max_tokens budget on reasoning (finish=length, empty answer).
CANDIDATE_ROUTING = {"provider": {"sort": "price"}}
CANDIDATE_REASONING = {
    "openai/gpt-5-nano": {"reasoning": {"effort": "minimal"}},
}
DEFAULT_CANDIDATE_REASONING = {"reasoning": {"enabled": False}}


def extra_body(model: str) -> dict:
    if model not in CANDIDATES:
        return {}
    return {**CANDIDATE_ROUTING, **CANDIDATE_REASONING.get(model, DEFAULT_CANDIDATE_REASONING)}

# Same values production hardcodes in _run_single_agent / _run_entity_picker
MEMORY_MAX_TOKENS = 1000
ENTITY_MAX_TOKENS = 500
TEMPERATURE = 0.1
CALL_TIMEOUT = 30.0
PROD_BUDGET_S = 10.0

SPEND_CAP = 6.0
SEARCHES_PER_DAY = 250
N_QUERIES = 40
N_NO_ANSWER = 8

# Legacy 3-agent shape (production before the merged call), kept so V0/V7 stay runnable
AGENT_SYSTEM_PROMPTS = {
    "direct_seeker": (
        "You are a Direct Seeker — a memory retrieval specialist focused on EXACT MATCHES.\n\n"
        "Your strategy:\n"
        "- Look for memories that EXPLICITLY mention the specific entities, names, values, "
        "or facts asked about in the question.\n"
        "- Match concrete details: names, numbers, dates, places, products, tools, versions.\n"
        "- Prefer memories with LITERAL matches over memories with vague topical overlap.\n"
        "- Ignore memories that are merely in the same topic area but lack the specific answer.\n\n"
        "You are precision-focused. Return fewer highly-relevant memories rather than many "
        "loosely-related ones."
    ),
    "inference_engine": (
        "You are an Inference Engine — a memory retrieval specialist focused on IMPLICIT CONNECTIONS.\n\n"
        "Your strategy:\n"
        "- Look for memories where the answer can be INFERRED even if not explicitly stated.\n"
        "- Connect dots across related memories.\n"
        "- Consider social and contextual cues: relationships, roles, preferences expressed indirectly.\n"
        "- Look for memories that provide CONTEXT needed to answer the question.\n"
        "- For preference questions, find memories where the user demonstrated a preference "
        "through actions or choices, not just explicit statements.\n\n"
        "You are recall-focused. Cast a wider net to catch memories that a literal matcher would miss."
    ),
    "temporal_navigator": (
        "You are a Temporal Navigator — a memory retrieval specialist focused on TIME and SEQUENCE.\n\n"
        "Your strategy:\n"
        "- Pay close attention to dates and chronological ordering.\n"
        "- For \"most recent\" or \"latest\" questions, find the LAST memory that mentions the topic.\n"
        "- For knowledge-update questions (where info changed), find BOTH old and new, "
        "ranking the most recent first.\n"
        "- Track state changes: if the user switched from X to Y, the memory mentioning Y "
        "is more relevant.\n\n"
        "You are temporally-focused. Your unique value is understanding WHEN things happened."
    ),
}

# Hand-written queries no stored memory should answer (no-answer probes)
NO_ANSWER_QUERIES = [
    "how many grams of salt should go into a 1kg sourdough loaf",
    "what is the best way to repot a fiddle leaf fig without killing it",
    "explain the offside rule in ice hockey",
    "what are the main differences between baroque and classical music",
    "how do I remove a red wine stain from a wool carpet",
    "at what temperature does water boil at 3000 meters altitude",
    "recommend a beginner watercolor brush set",
    "how do emperor penguins keep their eggs warm",
]


# ---------------------------------------------------------------------------
# Spend ledger (persists across runs so the cap covers the whole eval)
# ---------------------------------------------------------------------------

class Spend:
    def __init__(self, path: Path, cap: float):
        self.path = path
        self.cap = cap
        self.total = json.loads(path.read_text())["total"] if path.exists() else 0.0
        self.run = 0.0

    def add(self, cost: float) -> None:
        self.total += cost
        self.run += cost
        self.path.write_text(json.dumps({"total": self.total}))
        if self.total > self.cap:
            raise SystemExit(
                f"ABORT: cumulative OpenRouter spend ${self.total:.4f} passed the ${self.cap:.2f} cap"
            )

    def check(self) -> None:
        if self.total >= self.cap:
            raise SystemExit(f"ABORT: spend ${self.total:.4f} already at the ${self.cap:.2f} cap")


SPEND: Spend
OR_KEY = ""


def key_usage() -> float | None:
    """Total spend on this OpenRouter key (includes any other traffic using the key)."""
    try:
        r = httpx.get(f"{OR_URL}/key", headers={"Authorization": f"Bearer {OR_KEY}"}, timeout=30)
        return float(r.json()["data"]["usage"])
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return None


def _load_openrouter_key() -> str:
    for line in (REPO / ".env").read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip("'\"")
    raise SystemExit("OPENROUTER_API_KEY not found in .env")


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------

def call_llm(model: str, system: str | None, user: str, max_tokens: int,
             cache_system: bool, temperature: float = TEMPERATURE) -> dict:
    """One chat completion with usage accounting. Never raises on provider errors."""
    SPEND.check()
    messages: list[dict] = []
    if system and cache_system:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        })
    elif system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "usage": {"include": True},
        **extra_body(model),
    }
    rec: dict = {"model": model, "text": "", "error": None, "cost": 0.0, "retries": 0}
    # Transient infra errors (upstream 429, 5xx, DNS) are retried so they do not count
    # against a model; production does not retry, so retries are recorded and reported.
    for attempt in range(4):
        rec["retries"] = attempt
        status, data, rec["latency"], err = _post_once(payload)
        rec["error"] = err
        if err and (status in (429, 500, 502, 503, 529) or status == 0) and attempt < 3:
            time.sleep(2 ** (attempt + 1))
            continue
        break
    if data:
        usage = data.get("usage") or {}
        ptd = usage.get("prompt_tokens_details") or {}
        ctd = usage.get("completion_tokens_details") or {}
        rec.update({
            "cost": float(usage.get("cost") or 0.0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": ptd.get("cached_tokens", 0) or 0,
            "cache_write_tokens": ptd.get("cache_write_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": ctd.get("reasoning_tokens", 0) or 0,
            "provider": data.get("provider"),
            "served_model": data.get("model"),
        })
        choices = data.get("choices") or []
        if choices:
            rec["text"] = (choices[0].get("message") or {}).get("content") or ""
            rec["finish"] = choices[0].get("finish_reason")
    SPEND.add(rec["cost"])
    return rec


def _post_once(payload: dict) -> tuple[int, dict | None, float, str | None]:
    """POST with a hard total deadline (OpenRouter keeps idle connections alive with
    whitespace, so httpx's per-read timeout alone never fires). Status 0 = transport error."""
    t0 = time.perf_counter()
    try:
        with httpx.stream(
            "POST", f"{OR_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OR_KEY}",
                "HTTP-Referer": "https://github.com/cems",
                "X-Title": "CEMS cost eval",
            },
            json=payload, timeout=CALL_TIMEOUT,
        ) as r:
            body = bytearray()
            for chunk in r.iter_bytes():
                body.extend(chunk)
                if time.perf_counter() - t0 > CALL_TIMEOUT:
                    return r.status_code, None, time.perf_counter() - t0, "deadline: >30s total"
        latency = time.perf_counter() - t0
        data = json.loads(body)
        if r.status_code != 200 or "error" in data:
            code = (data.get("error") or {}).get("code") if isinstance(data, dict) else None
            status = code if isinstance(code, int) else r.status_code
            return status, data, latency, f"HTTP {status}: {str(data.get('error', data))[:300]}"
        return r.status_code, data, latency, None
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return 0, None, time.perf_counter() - t0, f"{type(e).__name__}: {e}"[:300]


# ---------------------------------------------------------------------------
# Data: production memories + entity pages (read-only list endpoints)
# ---------------------------------------------------------------------------

def _cems_get(path: str, params: dict) -> dict:
    creds = resolve_credentials(str(REPO))
    r = httpx.get(
        f"{creds['CEMS_API_URL']}{path}",
        params=params,
        headers={"Authorization": f"Bearer {creds['CEMS_API_KEY']}"},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()


def _list_docs(max_docs: int, stop_before: datetime | None = None, **filters) -> list[dict]:
    """Page /api/memory/list (newest first). Browse mode has no side effects."""
    out: list[dict] = []
    offset = 0
    while len(out) < max_docs:
        page = _cems_get("/api/memory/list", {"limit": 200, "offset": offset, **filters})
        rows = page.get("results", [])
        if not rows:
            break
        for row in rows:
            if stop_before and _parse_dt(row["created_at"]) < stop_before:
                return out
            out.append(row)
            if len(out) >= max_docs:
                break
        offset += len(rows)
    return out


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=UTC)
    return datetime.fromisoformat(value)


def _approx_relevance(doc: dict) -> float:
    # Production uses (relevant - noise) + shown*0.01; the list API only exposes shown_count.
    return (doc.get("shown_count") or 0) * 0.01


def pull_data() -> dict:
    """Rebuild the 3-bucket context the way _load_context_memories does, via the API."""
    seen: set[str] = set()
    chosen: list[dict] = []
    total_chars = 0

    def add(docs: list[dict]) -> int:
        nonlocal total_chars
        n = 0
        for d in docs:
            if d["id"] in seen:
                continue
            if total_chars + len(d.get("content") or "") > MAX_CONTEXT_CHARS:
                break
            seen.add(d["id"])
            chosen.append(d)
            total_chars += len(d.get("content") or "")
            n += 1
        return n

    project_docs = _list_docs(1000, source_ref_prefix=f"project:{PROJECT}")
    project_docs = [d for d in project_docs if d.get("category") not in EXCLUDED_MEMORY_CATEGORIES]
    project_docs.sort(key=_approx_relevance, reverse=True)
    b1 = add(project_docs)

    before = len(chosen)
    for cat in PROFILE_CATEGORIES:
        add(_list_docs(200, category=cat))
    b2 = len(chosen) - before

    cutoff = datetime.now(UTC) - timedelta(days=RECENT_DAYS)
    recent = []
    for d in _list_docs(500, stop_before=cutoff):
        src = d.get("source_ref") or ""
        cat = d.get("category") or ""
        if cat in EXCLUDED_MEMORY_CATEGORIES:
            continue
        if src and f"project:{PROJECT}".lower() not in src.lower() and cat not in PROFILE_CATEGORIES:
            continue
        recent.append(d)
    recent.sort(key=_approx_relevance, reverse=True)
    b3 = add(recent)

    chosen.sort(key=lambda m: (str(m.get("created_at") or ""), str(m.get("id", ""))))

    index = _cems_get("/api/wiki/index", {"project": PROJECT, "limit": 200}).get("index", [])
    pages = _list_docs(1000, category="entity-page", source_ref_prefix=f"project:{PROJECT}")
    return {
        "pulled_at": datetime.now(UTC).isoformat(),
        "buckets": {"project": b1, "profile": b2, "recent": b3},
        "memories": chosen,
        "entities": index,
        "entity_full": {p["id"]: p["content"] for p in pages},
    }


def load_data(fresh: bool) -> dict:
    path = SCRATCH / "dump.json"
    if path.exists() and not fresh:
        return json.loads(path.read_text())
    data = pull_data()
    path.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Queries: real logged prompts from the cems project + no-answer probes
# ---------------------------------------------------------------------------

def _raw_logged_prompts() -> list[tuple[str, str]]:
    """(prompt, source) pairs from hook logs, cems cwd only, oldest first."""
    out: list[tuple[str, str]] = []
    debug_log = Path.home() / ".claude/hooks/logs/user_prompts_submit_debug.log"
    if debug_log.exists():
        text = debug_log.read_text(errors="replace")
        for m in re.finditer(r"prompt=(.*?) \(len=\d+\), cwd='([^']*)'", text):
            if "/cems" not in m.group(2).lower():
                continue
            try:
                out.append((ast.literal_eval(m.group(1)), "hooks debug log"))
            except (ValueError, SyntaxError):
                continue
    for f in sorted((Path.home() / ".claude/hooks/logs/verbose").glob("*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") == "UserPromptSubmit" and "/cems" in (d.get("cwd") or "").lower():
                out.append((d.get("prompt") or "", "hooks verbose log"))
    for proj in ("-Users-razvan-Development-cems", "-Volumes-External-Development-cems"):
        idx = Path.home() / ".claude/projects" / proj / "sessions-index.json"
        if idx.exists():
            for e in json.loads(idx.read_text()).get("entries", []):
                out.append(((e.get("firstPrompt") or "").rstrip("…"), "claude sessions index"))
    return out


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def collect_queries() -> list[dict]:
    path = SCRATCH / "queries.json"
    if path.exists():
        return json.loads(path.read_text())
    n_real = N_QUERIES - N_NO_ANSWER
    real: list[dict] = []
    for prompt, source in _raw_logged_prompts():
        p = prompt.strip()
        # Same skips as the hook, plus system-injected notifications and test pings
        if not p or p.startswith(("/", "<")) or is_confirmatory(p) or len(p) < 15:
            continue
        q = _clean_search_query(p)[:2000]
        if len(q) < 15 or "No prompt" in q:
            continue
        w = _words(q)
        if any(len(w & _words(r["q"])) / max(len(w | _words(r["q"])), 1) >= 0.6 for r in real):
            continue
        real.append({"q": q, "source": source, "no_answer": False})
        if len(real) >= n_real:
            break
    probes = [{"q": q, "source": "hand-written no-answer", "no_answer": True}
              for q in NO_ANSWER_QUERIES]
    # Interleave probes (never first, so the cold call is a real query)
    queries: list[dict] = []
    step = max(len(real) // len(probes), 1)
    for i, r in enumerate(real):
        queries.append(r)
        if (i + 1) % step == 0 and probes:
            queries.append(probes.pop(0))
    queries.extend(probes)
    path.write_text(json.dumps(queries, indent=1))
    return queries


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def make_variants(best_candidate: str | None) -> dict[str, dict]:
    v = {
        "V0": {"model": BASELINE_MODEL, "shape": "multi", "label": "baseline 3 agents + RRF"},
        "V1": {"model": BASELINE_MODEL, "shape": "merged", "label": "merged 1 call"},
    }
    for i, m in enumerate(CANDIDATES, start=2):
        v[f"V{i}"] = {"model": m, "shape": "merged", "label": "merged 1 call"}
    if best_candidate:
        v["V7"] = {"model": best_candidate, "shape": "multi", "label": "3 agents + RRF"}
    return v


class Context:
    def __init__(self, data: dict):
        self.memories = data["memories"]
        for m in self.memories:
            m["created_at"] = _parse_dt(m["created_at"]) if isinstance(m["created_at"], str) else m["created_at"]
        self.memories_text = _format_memories_for_agents(self.memories)
        self.mem_short = {str(m["id"])[:8]: str(m["id"]) for m in self.memories}
        self.mem_text = {str(m["id"]): m["content"] for m in self.memories}
        self.entities = data["entities"]
        self.entities_text = _format_entities_for_picker(self.entities)
        self.ent_short = {e["id"][:8]: e["id"] for e in self.entities}
        self.ent_text = {
            e["id"]: data["entity_full"].get(e["id"]) or f"{e['title']}\n{e['summary']}"
            for e in self.entities
        }


def _is_empty_json_list(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(```(?:json)?\s*)?\[\s*\]\s*(```)?\s*", text or ""))


def run_search(ctx: Context, query: str, model: str, shape: str, nonce: str) -> dict:
    # Nonce line busts any cache left by earlier variants/runs so call #1 is truly cold;
    # the memory dump itself is byte-identical across variants.
    system = f"[eval session {nonce}]\n" + MEMORIES_SYSTEM_PROMPT.format(
        formatted_memories=ctx.memories_text)
    project_context = f"\nCURRENT PROJECT: {PROJECT}\n"
    roles = list(AGENT_SYSTEM_PROMPTS) if shape == "multi" else ["merged"]
    jobs = []
    for role in roles:
        instructions = AGENT_SYSTEM_PROMPTS.get(role, MERGED_ROLE_INSTRUCTIONS)
        user = SEARCH_USER_PROMPT.format(
            role_instructions=instructions, question=query, project_context=project_context)
        jobs.append((role, system, user, MEMORY_MAX_TOKENS, True))
    ent_user = ENTITY_PICKER_USER_PROMPT.format(
        question=query, n=len(ctx.entities), formatted_entities=ctx.entities_text,
        project_context=project_context)
    jobs.append(("entity_picker", ENTITY_PICKER_SYSTEM, ent_user, ENTITY_MAX_TOKENS, False))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(call_llm, model, s, u, mt, c) for _, s, u, mt, c in jobs]
        recs = [f.result() for f in futures]
    wall = time.perf_counter() - t0

    rankings: list[list[str]] = []
    mem_ok = False
    top_ent: list[str] = []
    for (role, *_), rec in zip(jobs, recs):
        rec["role"] = role
        valid = set(ctx.ent_short) if role == "entity_picker" else set(ctx.mem_short)
        parsed = _parse_agent_response(rec["text"], valid) if rec["text"] else []
        # Bad = error, empty completion, or no valid id and not an explicit "[]"
        rec["bad"] = bool(rec["error"] or not rec["text"].strip()
                          or (not parsed and not _is_empty_json_list(rec["text"])))
        if role == "entity_picker":
            top_ent = [ctx.ent_short[s] for s in parsed[:3]]
        else:
            mem_ok = mem_ok or not rec["bad"]
            if parsed:
                rankings.append(parsed)
    if shape == "multi":
        merged = reciprocal_rank_fusion(rankings)[:3] if rankings else []
    else:
        merged = rankings[0][:3] if rankings else []
    return {
        "top_mem": [ctx.mem_short[s] for s in merged],
        "top_ent": top_ent,
        "calls": recs,
        "wall": wall,
        "cost": sum(r["cost"] for r in recs),
        # Search-level failure: no memory call gave a usable answer (production degrades)
        "failed": not mem_ok,
        "bad_mem_calls": sum(r["bad"] for r in recs[:-1]),
        "n_mem_calls": len(recs) - 1,
        "ent_failed": recs[-1]["bad"],
        "over_budget": any(r.get("latency", 0) > PROD_BUDGET_S for r in recs),
    }


def run_variant(ctx: Context, name: str, spec: dict, queries: list[dict], tag: str,
                rerun: bool) -> dict:
    path = SCRATCH / f"{tag}results_{name}.json"
    if path.exists() and not rerun:
        cached = json.loads(path.read_text())
        if cached["model"] == spec["model"] and len(cached["searches"]) == len(queries):
            print(f"  {name}: reusing {path.name}")
            return cached
    nonce = uuid.uuid4().hex[:12]
    out = {**spec, "name": name, "nonce": nonce, "searches": []}
    for i, q in enumerate(queries):
        s = run_search(ctx, q["q"], spec["model"], spec["shape"], nonce)
        s["qi"] = i
        out["searches"].append(s)
        prov = sorted({str(c.get("provider")) for c in s["calls"]})
        print(f"  {name} q{i:02d} ${s['cost']:.5f} {s['wall']:.1f}s mem={len(s['top_mem'])} "
              f"ent={len(s['top_ent'])} fail={s['failed']} prov={prov} "
              f"total=${SPEND.total:.3f}", flush=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    return out


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You grade search results for a developer's memory system.
The QUESTION is what the developer typed into their coding assistant. Each ITEM is a stored
memory (or knowledge page) that the search returned for it. Grade each item:
0 = irrelevant to the question
1 = related topic, but would not really help answer or act on the question
2 = directly useful for answering or acting on the question

QUESTION: {question}

{items}

Return ONLY a JSON object mapping each item label to its grade, e.g. {{"I1": 2, "I2": 0}}."""

JUDGE_ITEM_CHARS = 8000
JUDGE_BATCH = 10


def judge(ctx: Context, queries: list[dict], results: list[dict]) -> dict:
    path = SCRATCH / "grades.json"
    grades: dict = json.loads(path.read_text()) if path.exists() else {}
    todo: dict[int, list[tuple[str, str]]] = {}
    for res in results:
        for s in res["searches"]:
            qi = s["qi"]
            for kind, ids in (("mem", s["top_mem"]), ("ent", s["top_ent"])):
                for full in ids:
                    key = grade_key(queries[qi]["q"], kind, full)
                    if key not in grades and (kind, full) not in todo.setdefault(qi, []):
                        todo[qi].append((kind, full))
    n_pairs = sum(len(v) for v in todo.values())
    print(f"Judging {n_pairs} new pairs with {JUDGE_MODEL}")
    log = (SCRATCH / "judge_log.jsonl").open("a")
    for qi, items in todo.items():
        q = queries[qi]["q"]
        for attempt in range(2):  # one retry for pairs the judge left ungraded
            pending = [it for it in items if grade_key(q, *it) not in grades]
            for start in range(0, len(pending), JUDGE_BATCH):
                batch = pending[start:start + JUDGE_BATCH]
                blocks = []
                for j, (kind, full) in enumerate(batch, start=1):
                    text = (ctx.mem_text if kind == "mem" else ctx.ent_text).get(full, "")
                    label = "memory" if kind == "mem" else "knowledge page"
                    blocks.append(f"--- ITEM I{j} ({label}) ---\n{text[:JUDGE_ITEM_CHARS]}")
                rec = call_llm(JUDGE_MODEL, None, JUDGE_PROMPT.format(
                    question=q, items="\n\n".join(blocks)), 3000, False, 0.0)
                log.write(json.dumps({"qi": qi, "attempt": attempt, "items": batch,
                                      **{k: rec.get(k) for k in ("text", "finish", "error",
                                                                 "cost", "completion_tokens")}})
                          + "\n")
                parsed = parse_json_dict(rec["text"], log_errors=False)
                for j, (kind, full) in enumerate(batch, start=1):
                    g = parsed.get(f"I{j}")
                    if isinstance(g, int | float) and 0 <= g <= 2:
                        grades[grade_key(q, kind, full)] = int(g)
        path.write_text(json.dumps(grades))
    log.close()
    return grades


def grade_key(query: str, kind: str, full_id: str) -> str:
    return f"{kind}|{hashlib.sha1(query.encode()).hexdigest()[:12]}|{full_id}"


# ---------------------------------------------------------------------------
# Scoring + report
# ---------------------------------------------------------------------------

def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(int(round(p * (len(xs) - 1))), len(xs) - 1)]


def score(res: dict, queries: list[dict], grades: dict, base: dict | None) -> dict:
    mem_means, hit2, ent_means, noans_ok, overlaps = [], [], [], [], []
    ungraded = 0
    for s in res["searches"]:
        q = queries[s["qi"]]
        mg = [grades.get(grade_key(q["q"], "mem", m)) for m in s["top_mem"]]
        eg = [grades.get(grade_key(q["q"], "ent", e)) for e in s["top_ent"]]
        ungraded += sum(g is None for g in mg + eg)
        mg = [g or 0 for g in mg]
        eg = [g or 0 for g in eg]
        if q["no_answer"]:
            noans_ok.append(all(g == 0 for g in mg))
        else:
            mem_means.append(sum(mg + [0] * (3 - len(mg))) / 3)
            hit2.append(any(g == 2 for g in mg))
            ent_means.append(sum(eg + [0] * (3 - len(eg))) / 3)
        if base:
            b = set(base["searches"][s["qi"]]["top_mem"])
            a = set(s["top_mem"])
            overlaps.append(len(a & b) / max(len(a), len(b), 1))
    ss = res["searches"]
    # Cold = first search whose calls all completed (an aborted call reports no cost);
    # warm = mean of the searches after it.
    first = next((i for i, s in enumerate(ss) if not any(c["error"] for c in s["calls"])), 0)
    cold = ss[first]["cost"]
    rest = ss[first + 1:]
    warm = statistics.mean(s["cost"] for s in rest) if rest else cold
    walls = [s["wall"] for s in ss]

    def monthly(f: float) -> float:
        return SEARCHES_PER_DAY * 30 * (f * cold + (1 - f) * warm)

    providers = sorted({c["provider"] for s in ss for c in s["calls"] if c.get("provider")})
    return {
        "mem": statistics.mean(mem_means) if mem_means else 0.0,
        "hit2": statistics.mean(hit2) if hit2 else 0.0,
        "ent": statistics.mean(ent_means) if ent_means else 0.0,
        "noans": statistics.mean(noans_ok) if noans_ok else None,
        "fail": statistics.mean(s["failed"] for s in ss),
        "bad_calls": (sum(s.get("bad_mem_calls", 0) for s in ss)
                      / max(sum(s.get("n_mem_calls", 1) for s in ss), 1)),
        "ent_fail": statistics.mean(s["ent_failed"] for s in ss),
        "over10": statistics.mean(s["over_budget"] for s in ss),
        "p50": _pct(walls, 0.5),
        "p95": _pct(walls, 0.95),
        "warm": warm,
        "cold": cold,
        "m60": monthly(0.6),
        "m20": monthly(0.2),
        "overlap": statistics.mean(overlaps) if overlaps else None,
        "ungraded": ungraded,
        "retried": sum(1 for s in ss for c in s["calls"] if c.get("retries")),
        "aborted": sum(1 for s in ss for c in s["calls"] if (c["error"] or "").startswith("deadline")),
        "cold_index": first,
        "n_calls": sum(len(s["calls"]) for s in ss),
        "providers": providers,
        "cached_share_warm": _cached_share(ss[1:]),
    }


def _cached_share(searches: list[dict]) -> float:
    prompt = sum(c.get("prompt_tokens") or 0 for s in searches for c in s["calls"]
                 if c["role"] != "entity_picker")
    cached = sum(c.get("cached_tokens") or 0 for s in searches for c in s["calls"]
                 if c["role"] != "entity_picker")
    return cached / prompt if prompt else 0.0


def render(rows: list[tuple[str, dict, dict]], n_q: int, meta: dict) -> str:
    head = ("| Variant | Model | Shape | Mem grade (0-2) | >=1 grade-2 | Entity grade | "
            "No-answer OK | Mem fail | Bad mem calls | Ent fail | >10s | p50 s | p95 s | Warm $/search | "
            "Cold $/search | $/mo @60% cold | $/mo @20% cold | Overlap w/ V0 | "
            "Cached share (warm) |")
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for name, spec, m in rows:
        noans = "n/a" if m["noans"] is None else f"{m['noans']:.0%}"
        overlap = "-" if m["overlap"] is None else f"{m['overlap']:.0%}"
        lines.append(
            f"| {name} | {spec['model']} | {spec['shape']} | {m['mem']:.2f} | {m['hit2']:.0%} | "
            f"{m['ent']:.2f} | {noans} | {m['fail']:.0%} | {m['bad_calls']:.0%} | "
            f"{m['ent_fail']:.0%} | "
            f"{m['over10']:.0%} | {m['p50']:.1f} | {m['p95']:.1f} | ${m['warm']:.5f} | "
            f"${m['cold']:.5f} | ${m['m60']:.2f} | ${m['m20']:.2f} | {overlap} | "
            f"{m['cached_share_warm']:.0%} |"
        )
    notes = [
        "",
        f"Queries: {n_q} ({meta['n_noans']} no-answer probes). Dump: {meta['n_mem']} memories, "
        f"{meta['dump_chars']:,} chars; {meta['n_ent']} entity pages. Judge: {JUDGE_MODEL}.",
        "Mem/entity grade: mean over answerable queries, top-3 slots, missing slot = 0.",
        "No-answer OK: share of probes where every returned memory graded 0 (or none returned).",
        "Mem fail: search where no memory call gave a usable answer. Bad mem calls: share of "
        "memory calls with an error, empty completion, or no valid id and no explicit [] "
        "(invalid/hallucinated ids, truncated output). "
        ">10s: any call in the search over the 10s production budget.",
        "Candidates (not Gemini) are sent with provider sort=price and reasoning off "
        "(gpt-5-nano: effort minimal). Gemini requests are production-identical.",
        "Cold $/search: first fully completed search of the variant (fresh cache, nonce-busted). "
        "Warm: mean of the searches after it. "
        f"Monthly: {SEARCHES_PER_DAY}/day x 30.",
        "",
        "Providers: " + "; ".join(f"{n}={','.join(m['providers'])}" for n, _, m in rows),
        "Ungraded pairs (judge gave no grade, counted as 0): "
        + ", ".join(f"{n}={m['ungraded']}" for n, _, m in rows),
        "Calls retried after a transient 429/5xx/network error (production would not retry): "
        + ", ".join(f"{n}={m['retried']}/{m['n_calls']}" for n, _, m in rows),
        "Calls aborted at the 30s deadline (cost not reported by OpenRouter; counted as failed): "
        + ", ".join(f"{n}={m['aborted']}" for n, _, m in rows),
        "Search used as the cold sample (0 = first): "
        + ", ".join(f"{n}=q{m['cold_index']}" for n, _, m in rows),
    ]
    return "\n".join(lines + notes)


def estimate(ctx: Context, variants: dict, n_q: int) -> float:
    prices = {m["id"]: m.get("pricing", {}) for m in
              httpx.get(f"{OR_URL}/models", timeout=30).json()["data"]}
    missing = [v["model"] for v in variants.values() if v["model"] not in prices]
    if missing:
        raise SystemExit(f"Model ids not on OpenRouter: {missing}. Look up the current ids.")
    dump_tok = len(ctx.memories_text) / 3.6
    ent_tok = len(ctx.entities_text) / 3.6
    total = 0.0
    for name, v in variants.items():
        p = prices[v["model"]]
        pin = float(p.get("prompt", 0))
        pread = float(p.get("input_cache_read", pin))
        pwrite = float(p.get("input_cache_write", 0))
        calls = 3 if v["shape"] == "multi" else 1
        # Worst case for implicit-cache models: assume half the warm calls miss
        miss = 0.0 if v["model"] == BASELINE_MODEL else 0.5
        warm_call = dump_tok * (miss * pin + (1 - miss) * pread)
        cold_call = dump_tok * (pin + pwrite)
        cost = calls * (cold_call + (n_q - 1) * warm_call) + n_q * ent_tok * pin
        print(f"  estimate {name} {v['model']}: ${cost:.3f}")
        total += cost
    total += n_q * len(variants) * 0.002  # judge
    return total


def main() -> None:
    global SPEND, OR_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="5 queries: V0, V1, one candidate")
    ap.add_argument("--smoke-model", default=CANDIDATES[0])
    ap.add_argument("--variants", help="comma list, e.g. V0,V3 (default: all)")
    ap.add_argument("--fresh-data", action="store_true", help="re-pull the dump from the API")
    ap.add_argument("--rerun", action="store_true", help="ignore cached variant results")
    ap.add_argument("--cap", type=float, default=SPEND_CAP)
    args = ap.parse_args()

    if not Path("/Volumes/External").is_mount():
        raise SystemExit("/Volumes/External is not mounted; refusing to write artifacts elsewhere")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OR_KEY = _load_openrouter_key()
    SPEND = Spend(SCRATCH / "spend.json", args.cap)

    data = load_data(args.fresh_data)
    ctx = Context(data)
    queries = collect_queries()
    real = sum(not q["no_answer"] for q in queries)
    print(f"Dump: {len(ctx.memories)} memories ({len(ctx.memories_text):,} chars, buckets "
          f"{data['buckets']}), {len(ctx.entities)} entity pages. Queries: {len(queries)} "
          f"({real} real, sources {sorted({q['source'] for q in queries})})")

    tag = ""
    if args.smoke:
        queries = queries[:5]
        tag = "smoke_"
        all_v = make_variants(None)
        cand = next(k for k, v in all_v.items() if v["model"] == args.smoke_model)
        variants = {k: all_v[k] for k in ("V0", "V1", cand)}
    else:
        variants = make_variants(None)
    if args.variants:
        wanted = args.variants.split(",")
        variants = {k: v for k, v in variants.items() if k in wanted}

    est = estimate(ctx, variants, len(queries))
    print(f"Estimated cost for this run: ${est:.2f}; spent so far ${SPEND.total:.3f}; "
          f"cap ${SPEND.cap:.2f}")
    if SPEND.total + est > SPEND.cap:
        raise SystemExit("ABORT: the estimate would pass the spend cap")

    usage_start = key_usage()
    results: dict[str, dict] = {}
    for name, spec in variants.items():
        print(f"Running {name}: {spec['model']} ({spec['shape']})")
        results[name] = run_variant(ctx, name, spec, queries, tag, args.rerun)

    grades = judge(ctx, queries, list(results.values()))

    def scored() -> list[tuple[str, dict, dict]]:
        base = results.get("V0")
        return [(n, variants[n], score(results[n], queries, grades, base)) for n in results]

    rows = scored()
    if not args.smoke and not args.variants:
        cands = [(n, s, m) for n, s, m in rows if s["model"] in CANDIDATES and m["fail"] <= 0.2]
        if cands:
            best = max(cands, key=lambda r: (r[2]["hit2"], r[2]["mem"]))
            print(f"Best merged candidate: {best[0]} {best[1]['model']} -> running V7 (3 agents)")
            variants = make_variants(best[1]["model"])
            results["V7"] = run_variant(ctx, "V7", variants["V7"], queries, tag, args.rerun)
            grades = judge(ctx, queries, list(results.values()))
            rows = scored()

    meta = {"n_noans": sum(q["no_answer"] for q in queries), "n_mem": len(ctx.memories),
            "dump_chars": len(ctx.memories_text), "n_ent": len(ctx.entities)}
    report = render(rows, len(queries), meta)
    report += f"\n\nOpenRouter spend (usage.cost sum): this run ${SPEND.run:.4f}, eval total ${SPEND.total:.4f}\n"
    usage_end = key_usage()
    if usage_start is not None and usage_end is not None:
        report += (f"Key usage delta during this run: ${usage_end - usage_start:.4f} "
                   "(includes aborted calls and any other traffic on the same key)\n")
    print("\n" + report)
    out = SCRATCH / f"{tag}report.md"
    out.write_text(f"# Agentic search cost eval ({datetime.now(UTC):%Y-%m-%d %H:%M} UTC)\n\n"
                   + report)
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
