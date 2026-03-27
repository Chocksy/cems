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

def _get_config() -> tuple[str, str]:
    """Return (api_url, api_key) using the shared credential resolver.
    Precedence: env vars > per-project .cems/credentials > global ~/.cems/credentials.
    """
    creds = resolve_credentials(os.getcwd())
    api_url = creds.get("CEMS_API_URL", "")
    api_key = creds.get("CEMS_API_KEY", "")
    return api_url.rstrip("/"), api_key


API_URL, API_KEY = _get_config()

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
    # Pass search mode from credentials or env (CEMS_SEARCH_MODE=agentic)
    search_mode = os.getenv("CEMS_SEARCH_MODE") or resolve_credentials(os.getcwd()).get("CEMS_SEARCH_MODE", "")
    if search_mode:
        payload["mode"] = search_mode
    return json.dumps(_request("POST", "/api/memory/search", payload))


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
    return json.dumps(_request("POST", "/api/memory/update", {
        "memory_id": memory_id,
        "content": content,
    }))


@mcp.tool()
def memory_maintenance(
    job_type: str = "consolidation",
) -> str:
    """Run memory maintenance jobs (consolidation, distillation, summarization, reindex, all)."""
    return json.dumps(_request("POST", "/api/memory/maintenance", {
        "job_type": job_type,
    }))


@mcp.tool()
def memory_pin(
    memory_id: str,
    pin: bool = True,
) -> str:
    """Pin or unpin a memory. Pinned memories are protected from distillation, consolidation, and noise pruning."""
    return json.dumps(_request("POST", "/api/memory/pin", {
        "memory_id": memory_id,
        "pin": pin,
    }))


@mcp.tool()
def memory_get(
    memory_id: str,
) -> str:
    """Get full document content by ID. Returns content_detailed (original full text) when available, otherwise content."""
    return json.dumps(_request("GET", f"/api/memory/get?id={memory_id}"))


# ---- Resources ------------------------------------------------------------

@mcp.resource("memory://status")
def memory_status() -> str:
    """Current status of the memory system."""
    return json.dumps(_request("GET", "/api/memory/status"), indent=2)


@mcp.resource("memory://personal/summary")
def memory_personal_summary() -> str:
    """Summary of personal memories."""
    return json.dumps(_request("GET", "/api/memory/summary/personal"), indent=2)


@mcp.resource("memory://shared/summary")
def memory_shared_summary() -> str:
    """Summary of shared team memories."""
    return json.dumps(_request("GET", "/api/memory/summary/shared"), indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
