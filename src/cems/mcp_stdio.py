"""
CEMS MCP Server — stdio transport.

A FastMCP-based server that exposes CEMS memory tools over stdio,
suitable for use as a Claude Code / Cursor MCP server.

Entry point: cems-mcp (defined in pyproject.toml)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

from cems.shared.credentials import resolve_credentials

logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _get_config() -> tuple[str, str, str, str]:
    """Return (api_url, api_key, search_mode, team_id) using the shared credential resolver.
    Precedence: env vars > per-project .cems/credentials > global ~/.cems/credentials.
    """
    creds = resolve_credentials(os.getcwd())
    api_url = creds.get("CEMS_API_URL", "")
    api_key = creds.get("CEMS_API_KEY", "")
    search_mode = creds.get("CEMS_SEARCH_MODE", "")
    team_id = creds.get("CEMS_TEAM_ID", "")
    return api_url.rstrip("/"), api_key, search_mode, team_id


API_URL, API_KEY, SEARCH_MODE, TEAM_ID = _get_config()

_NOT_CONFIGURED_MSG = (
    "CEMS is not configured for this project. "
    "Run: cems setup --project"
)

# ---------------------------------------------------------------------------
# Session auto-detection (for search context enrichment)
# ---------------------------------------------------------------------------

from cems.shared.session_detect import detect_session_id

# Cache at module init — stdio MCP = 1 session per process
_SESSION_ID = detect_session_id()

# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no extra deps)
# ---------------------------------------------------------------------------

def _request(method: str, path: str, body: dict | None = None) -> dict:
    """Make an authenticated request to the CEMS API. Returns parsed JSON."""
    url = f"{API_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("User-Agent", "CEMS-MCP/1.0")
    if TEAM_ID:
        req.add_header("X-Team-Id", TEAM_ID)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _fetch_profile() -> str:
    """Fetch the user profile to use as server instructions."""
    try:
        result = _request("GET", "/api/memory/profile")
        if result.get("success") and result.get("context"):
            return result["context"]
    except Exception as e:
        logger.debug(f"Failed to fetch profile: {e}")
    return ""


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

instructions = _fetch_profile() if API_URL and API_KEY else ""

mcp = FastMCP(
    "cems",
    instructions=instructions or None,
)

# ---- Tools ----------------------------------------------------------------

def _format_agentic_for_mcp(data: dict) -> str:
    """Format agentic results as structured text for MCP tool response."""
    parts = []
    entities = data.get("entities", [])
    memories = data.get("memories", data.get("results", []))[:3]

    if entities:
        parts.append("KNOWLEDGE TOPICS matching your query:\n")
        for i, e in enumerate(entities, 1):
            sources = f" ({e['sources']} sources)" if e.get("sources") else ""
            parts.append(f"{i}. {e['title']}{sources}")
            if e.get("summary"):
                parts.append(f"   {e['summary']}")
            short_id = e.get("id", "")[:8]
            parts.append(f"   \u2192 /recall {short_id} for full details")
        parts.append("")

    if memories:
        parts.append("RELEVANT MEMORIES:")
        for i, m in enumerate(memories, 1):
            category = m.get("category", "general")
            content = m.get("content", "")
            mid = (m.get("memory_id") or m.get("id", ""))[:8]
            score = m.get("score", 0.0)
            parts.append(f"{i}. [{category}] (score: {score:.2f}) {content} (id: {mid})")

    if not entities and not memories:
        return json.dumps(data)

    n_entities = len(entities)
    n_memories = len(memories)
    parts.append(
        f"\n--- Agentic: {n_entities} topics, {n_memories} memories "
        f"from {data.get('total_candidates', 0)} candidates ---"
    )
    if entities:
        parts.append("Use /recall <id> to read full topic pages.")

    return "\n".join(parts)


@mcp.tool()
def memory_search(
    query: str,
    scope: str = "both",
    max_results: int = 10,
    max_tokens: int = 4000,
    enable_graph: bool = True,
    enable_query_synthesis: bool = True,
    raw: bool = False,
    project: str | None = None,
) -> str:
    """Search memories using unified retrieval pipeline: query synthesis, vector+graph search, relevance filtering, temporal ranking, and token budgeting."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    payload: dict = {
        "query": query,
        "scope": scope,
        "limit": max_results,  # API expects 'limit'
        "max_tokens": max_tokens,
        "enable_graph": enable_graph,
        "enable_query_synthesis": enable_query_synthesis,
        "raw": raw,
    }
    if project:
        payload["project"] = project
    # Pass session_id for search context enrichment (auto-detected at startup)
    if _SESSION_ID:
        payload["session_id"] = _SESSION_ID
    # Pass search mode resolved at startup (project creds > env > global creds)
    search_mode = SEARCH_MODE
    if search_mode:
        payload["mode"] = search_mode
    result = _request("POST", "/api/memory/search", payload)
    # Format entity-first for agentic mode
    if result.get("entities") or result.get("mode") == "agentic":
        return _format_agentic_for_mcp(result)
    return json.dumps(result)


@mcp.tool()
def memory_add(
    content: str,
    scope: str = "personal",
    category: str = "general",
    tags: list[str] | None = None,
    infer: bool = True,
    source_ref: str | None = None,
) -> str:
    """Store a memory. Set infer=false for bulk imports (faster)."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    payload: dict = {
        "content": content,
        "scope": scope,
        "category": category,
        "tags": tags or [],
        "infer": infer,
    }
    if source_ref:
        payload["source_ref"] = source_ref
    return json.dumps(_request("POST", "/api/memory/add", payload))


@mcp.tool()
def memory_forget(
    memory_id: str,
    hard_delete: bool = False,
) -> str:
    """Delete or archive a memory."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("POST", "/api/memory/forget", {
        "memory_id": memory_id,
        "hard_delete": hard_delete,
    }))


@mcp.tool()
def memory_update(
    memory_id: str,
    content: str,
) -> str:
    """Update an existing memory's content."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("POST", "/api/memory/update", {
        "memory_id": memory_id,
        "content": content,
    }))


@mcp.tool()
def memory_maintenance(
    job_type: str = "consolidation",
) -> str:
    """Run memory maintenance jobs (consolidation, distillation, summarization, reindex, all)."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("POST", "/api/memory/maintenance", {
        "job_type": job_type,
    }))


@mcp.tool()
def memory_pin(
    memory_id: str,
    pin: bool = True,
) -> str:
    """Pin or unpin a memory. Pinned memories are protected from distillation, consolidation, and noise pruning."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("POST", "/api/memory/pin", {
        "memory_id": memory_id,
        "pin": pin,
    }))


@mcp.tool()
def memory_get(
    memory_id: str,
) -> str:
    """Get full document content by ID. Returns content_detailed (original full text) when available, otherwise content."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("GET", f"/api/memory/get?id={memory_id}"))


# ---- Resources ------------------------------------------------------------

@mcp.resource("memory://status")
def memory_status() -> str:
    """Current status of the memory system."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("GET", "/api/memory/status"), indent=2)


@mcp.resource("memory://personal/summary")
def memory_personal_summary() -> str:
    """Summary of personal memories."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("GET", "/api/memory/summary/personal"), indent=2)


@mcp.resource("memory://shared/summary")
def memory_shared_summary() -> str:
    """Summary of shared team memories."""
    if not API_URL:
        return _NOT_CONFIGURED_MSG
    return json.dumps(_request("GET", "/api/memory/summary/shared"), indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
