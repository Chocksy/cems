"""Index API handlers.

REST API endpoints for repository indexing:
- POST /api/index/repo - Index a git repository
- POST /api/index/path - Index a local path (server-side only)
- GET /api/index/patterns - List available index patterns
"""

import asyncio
import logging
from functools import partial

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory

logger = logging.getLogger(__name__)


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
        return JSONResponse({"error": str(e)}, status_code=400)
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
    """
    try:
        body = await request.json()
        path = body.get("path")
        if not path:
            return JSONResponse({"error": "path is required"}, status_code=400)

        scope = body.get("scope", "shared")
        patterns = body.get("patterns")

        memory = get_memory()

        from cems.indexer import RepositoryIndexer

        indexer = RepositoryIndexer(memory)

        # Run sync indexer in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(
            indexer.index_local_path,
            repo_path=path,
            scope=scope,
            patterns=patterns,
        )

        return JSONResponse({
            "success": True,
            "result": result,
        })
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"API index_path error: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


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
