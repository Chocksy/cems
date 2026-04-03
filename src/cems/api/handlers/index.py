"""Index API handlers.

REST API endpoints for repository indexing:
- POST /api/index/repo - Index a git repository
- POST /api/index/path - Index a local path (server-side only)
- GET /api/index/patterns - List available index patterns
"""

import asyncio
import logging
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory

logger = logging.getLogger(__name__)


# Allowlisted git hosting domains — eliminates SSRF entirely.
# Private/self-hosted repos should use the CLI (`cems index repo`), which
# runs locally and has no SSRF concern.
ALLOWED_GIT_HOSTS = frozenset({
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "dev.azure.com",
    "ssh.dev.azure.com",
    "codeberg.org",
})


def validate_repo_url(url: str) -> None:
    """Validate that a repo URL points to an allowlisted public git host.

    Using a domain allowlist instead of DNS-based checks because:
    - DNS resolution checks are vulnerable to TOCTOU rebinding attacks
    - IP pinning breaks TLS certificate validation
    - Allowlisting known hosts is the only approach that is both correct and simple

    For private/self-hosted repos, use the CLI: `cems index repo <url>`

    Raises:
        ValueError with user-safe message on failure
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise ValueError("Invalid URL")

    if parsed.scheme != "https":
        raise ValueError("Only HTTPS git URLs are allowed")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")

    if hostname not in ALLOWED_GIT_HOSTS:
        raise ValueError(
            f"Only public git hosts are allowed over HTTP API "
            f"({', '.join(sorted(ALLOWED_GIT_HOSTS))}). "
            f"For private repos, use the CLI: cems index repo <url>"
        )


async def api_index_repo(request: Request):
    """Index a git repository by cloning and extracting knowledge.

    POST /api/index/repo
    Body: {
        "repo_url": "https://github.com/org/repo",
        "branch": "main",          (optional, default "main")
        "scope": "shared",         (optional, default "shared")
        "patterns": ["rspec_conventions", "readme_docs"]  (optional, all if omitted)
    }
    """
    try:
        body = await request.json()
        repo_url = body.get("repo_url")
        if not repo_url:
            return JSONResponse({"error": "repo_url is required"}, status_code=400)

        # SSRF prevention: validate URL, reject private IPs
        try:
            validate_repo_url(repo_url)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        branch = body.get("branch", "main")
        scope = body.get("scope", "shared")
        patterns = body.get("patterns")

        memory = get_memory()

        from cems.indexer import RepositoryIndexer

        indexer = RepositoryIndexer(memory)

        # Run sync indexer in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(
            indexer.index_git_repo,
            repo_url=repo_url,
            branch=branch,
            scope=scope,
            patterns=patterns,
        )

        return JSONResponse({
            "success": True,
            "result": result,
        })
    except RuntimeError as e:
        logger.error(f"API index_repo error: {e}")
        return JSONResponse({"error": "Indexing failed"}, status_code=400)
    except Exception as e:
        logger.error(f"API index_repo error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def api_index_path(request: Request):
    """Index a local directory path on the server.

    POST /api/index/path
    Body: {
        "path": "/path/to/repo",
        "scope": "shared",         (optional, default "shared")
        "patterns": ["readme_docs"]  (optional, all if omitted)
    }

    SECURITY: This endpoint is disabled over HTTP to prevent arbitrary
    filesystem reads. Use the CLI (`cems index path`) for local indexing.
    """
    return JSONResponse(
        {"error": "Path indexing is disabled over HTTP. Use the CLI: cems index path /path/to/dir"},
        status_code=403,
    )


async def api_index_patterns(request: Request):
    """List available index patterns.

    GET /api/index/patterns
    """
    try:
        from cems.indexer import RepositoryIndexer

        memory = get_memory()
        indexer = RepositoryIndexer(memory)
        patterns = indexer.list_patterns()

        return JSONResponse({
            "success": True,
            "patterns": patterns,
        })
    except Exception as e:
        logger.error(f"API index_patterns error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)
