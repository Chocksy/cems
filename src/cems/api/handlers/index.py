"""Index API handlers.

REST API endpoints for repository indexing:
- POST /api/index/repo - Index a git repository
- POST /api/index/path - Index a local path (server-side only)
- GET /api/index/patterns - List available index patterns
"""

import asyncio
import logging
import re

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory

logger = logging.getLogger(__name__)

# Only allow HTTPS git URLs (prevent SSRF via file://, ext::, internal IPs)
_SAFE_REPO_URL = re.compile(r"^https://[a-zA-Z0-9._-]+[a-zA-Z0-9]/")


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

        # SSRF prevention: only allow HTTPS URLs
        if not _SAFE_REPO_URL.match(repo_url):
            return JSONResponse(
                {"error": "Only HTTPS git URLs are allowed (e.g. https://github.com/org/repo)"},
                status_code=400,
            )

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
