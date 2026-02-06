"""HTTP client for CEMS API.

This module provides HTTP client classes for interacting with the CEMS server.
Used by the CLI and other clients that need to communicate with a remote CEMS instance.
"""

import os
from typing import Any, Literal

import httpx

from cems.config import CEMSConfig


class CEMSClientError(Exception):
    """Base exception for CEMS client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CEMSAuthError(CEMSClientError):
    """Authentication error (401/403)."""

    pass


class CEMSConnectionError(CEMSClientError):
    """Connection error (server unreachable)."""

    pass


class CEMSClient:
    """HTTP client for CEMS memory operations.

    This client wraps the CEMS REST API for memory operations.
    Authentication is via API key (Bearer token).

    Example:
        client = CEMSClient(api_url="https://cems.example.com", api_key="cems_ak_...")
        client.add("Remember this fact", category="general")
        results = client.search("fact")
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        team_id: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize CEMS client.

        Args:
            api_url: CEMS server URL. Falls back to CEMS_API_URL env var.
            api_key: API key for authentication. Falls back to CEMS_API_KEY env var.
            team_id: Optional team ID for shared memory operations.
            timeout: Request timeout in seconds.

        Raises:
            CEMSClientError: If api_url or api_key is not provided.
        """
        self.api_url = (api_url or os.environ.get("CEMS_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("CEMS_API_KEY", "")
        self.team_id = team_id or os.environ.get("CEMS_TEAM_ID")
        self.timeout = timeout

        if not self.api_url:
            raise CEMSClientError(
                "CEMS_API_URL is required. Set it via environment variable or --api-url flag."
            )
        if not self.api_key:
            raise CEMSClientError(
                "CEMS_API_KEY is required. Set it via environment variable or --api-key flag."
            )

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.team_id:
            self._headers["X-Team-ID"] = self.team_id

    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the CEMS API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., /api/memory/add)
            json: Request body as dict
            params: Query parameters

        Returns:
            Response JSON as dict

        Raises:
            CEMSAuthError: On 401/403 responses
            CEMSConnectionError: On connection failures
            CEMSClientError: On other errors
        """
        url = f"{self.api_url}{endpoint}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=self._headers,
                    json=json,
                    params=params,
                )

                if response.status_code == 401:
                    raise CEMSAuthError("Invalid API key", status_code=401)
                if response.status_code == 403:
                    raise CEMSAuthError("Access denied", status_code=403)
                if response.status_code >= 400:
                    try:
                        error = response.json().get("error", response.text)
                    except Exception:
                        error = response.text
                    raise CEMSClientError(
                        f"API error: {error}", status_code=response.status_code
                    )

                return response.json()

        except httpx.ConnectError as e:
            raise CEMSConnectionError(f"Cannot connect to {self.api_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise CEMSConnectionError(f"Request timed out: {e}") from e
        except CEMSClientError:
            raise
        except Exception as e:
            raise CEMSClientError(f"Request failed: {e}") from e

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def add(
        self,
        content: str,
        category: str = "general",
        scope: Literal["personal", "shared"] = "personal",
        tags: list[str] | None = None,
        source_ref: str | None = None,
        pinned: bool = False,
        pin_reason: str | None = None,
    ) -> dict[str, Any]:
        """Add a memory.

        Args:
            content: Content to remember
            category: Category for organization
            scope: "personal" or "shared"
            tags: Optional tags
            source_ref: Project reference (e.g., "project:org/repo")
            pinned: If True, memory is pinned and never auto-pruned
            pin_reason: Reason for pinning

        Returns:
            API response with result
        """
        payload: dict[str, Any] = {
            "content": content,
            "category": category,
            "scope": scope,
            "tags": tags or [],
        }
        if source_ref:
            payload["source_ref"] = source_ref
        if pinned:
            payload["pinned"] = pinned
        if pin_reason:
            payload["pin_reason"] = pin_reason

        return self._request("POST", "/api/memory/add", json=payload)

    def search(
        self,
        query: str,
        limit: int = 10,
        scope: Literal["personal", "shared", "both"] = "both",
        max_tokens: int = 2000,
        enable_graph: bool = True,
        enable_query_synthesis: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Search memories using unified retrieval pipeline.

        The unified pipeline implements 5 stages:
        1. Query synthesis (LLM expands query for better retrieval)
        2. Candidate retrieval (vector + graph search)
        3. Relevance filtering (threshold-based)
        4. Temporal ranking (time decay + priority)
        5. Token-budgeted assembly

        Args:
            query: Search query
            limit: Maximum results (default 10)
            scope: Which scope to search
            max_tokens: Token budget for results
            enable_graph: Include graph traversal
            enable_query_synthesis: Use LLM query expansion
            raw: Debug mode - bypass filtering to see all results

        Returns:
            Full search result including results, tokens_used, etc.
        """
        return self._request(
            "POST",
            "/api/memory/search",
            json={
                "query": query,
                "limit": limit,
                "scope": scope,
                "max_tokens": max_tokens,
                "enable_graph": enable_graph,
                "enable_query_synthesis": enable_query_synthesis,
                "raw": raw,
            },
        )

    def delete(self, memory_id: str, hard: bool = False) -> dict[str, Any]:
        """Delete or archive a memory.

        Args:
            memory_id: Memory ID to delete
            hard: If True, permanently delete. If False, archive.

        Returns:
            API response
        """
        return self._request(
            "POST",
            "/api/memory/forget",
            json={"memory_id": memory_id, "hard_delete": hard},
        )

    def update(self, memory_id: str, content: str) -> dict[str, Any]:
        """Update a memory's content.

        Args:
            memory_id: Memory ID to update
            content: New content

        Returns:
            API response
        """
        return self._request(
            "POST",
            "/api/memory/update",
            json={"memory_id": memory_id, "content": content},
        )

    def status(self) -> dict[str, Any]:
        """Get system status.

        Returns:
            Status information dict
        """
        return self._request("GET", "/api/memory/status")

    def summary(
        self, scope: Literal["personal", "shared"] = "personal"
    ) -> dict[str, Any]:
        """Get memory summary for a scope.

        Args:
            scope: "personal" or "shared"

        Returns:
            Summary with total count and categories
        """
        endpoint = f"/api/memory/summary/{scope}"
        return self._request("GET", endpoint)

    def maintenance(
        self, job_type: Literal["consolidation", "summarization", "reindex", "all"]
    ) -> dict[str, Any]:
        """Run a maintenance job.

        Args:
            job_type: Type of maintenance to run

        Returns:
            Job results
        """
        return self._request(
            "POST",
            "/api/memory/maintenance",
            json={"job_type": job_type},
        )

    def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Health status
        """
        return self._request("GET", "/health")

    # =========================================================================
    # Index Operations
    # =========================================================================

    def index_repo(
        self,
        repo_url: str,
        branch: str = "main",
        scope: Literal["personal", "shared"] = "shared",
        patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Index a git repository.

        Clones the repo, extracts knowledge using pattern-based extractors,
        and stores results as pinned memories.

        Args:
            repo_url: Git repository URL
            branch: Branch to index (default "main")
            scope: Memory scope for extracted knowledge
            patterns: Specific pattern names to use (default: all)

        Returns:
            Indexing results with files_scanned, memories_created, etc.
        """
        payload: dict[str, Any] = {
            "repo_url": repo_url,
            "branch": branch,
            "scope": scope,
        }
        if patterns:
            payload["patterns"] = patterns

        return self._request("POST", "/api/index/repo", json=payload)

    def index_path(
        self,
        path: str,
        scope: Literal["personal", "shared"] = "shared",
        patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Index a local directory path (server-side).

        The path must be accessible from the CEMS server.

        Args:
            path: Local directory path on the server
            scope: Memory scope for extracted knowledge
            patterns: Specific pattern names to use (default: all)

        Returns:
            Indexing results with files_scanned, memories_created, etc.
        """
        payload: dict[str, Any] = {
            "path": path,
            "scope": scope,
        }
        if patterns:
            payload["patterns"] = patterns

        return self._request("POST", "/api/index/path", json=payload)

    def list_index_patterns(self) -> dict[str, Any]:
        """List available index patterns.

        Returns:
            Dict with patterns list containing name, description, file_patterns, etc.
        """
        return self._request("GET", "/api/index/patterns")


class CEMSAdminClient:
    """HTTP client for CEMS admin operations.

    This client wraps the CEMS Admin API for user and team management.
    Authentication is via admin key (CEMS_ADMIN_KEY) or admin user API key.

    Example:
        client = CEMSAdminClient(api_url="https://cems.example.com", admin_key="...")
        result = client.create_user("john", email="john@example.com")
        print(f"API Key: {result['api_key']}")
    """

    def __init__(
        self,
        api_url: str | None = None,
        admin_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize admin client.

        Args:
            api_url: CEMS server URL. Falls back to CEMS_API_URL env var.
            admin_key: Admin key or admin user API key. Falls back to CEMS_ADMIN_KEY,
                       then CEMS_API_KEY env vars.
            timeout: Request timeout in seconds.

        Raises:
            CEMSClientError: If api_url or admin_key is not provided.
        """
        self.api_url = (api_url or os.environ.get("CEMS_API_URL", "")).rstrip("/")
        self.admin_key = admin_key or os.environ.get(
            "CEMS_ADMIN_KEY", os.environ.get("CEMS_API_KEY", "")
        )
        self.timeout = timeout

        if not self.api_url:
            raise CEMSClientError(
                "CEMS_API_URL is required. Set it via environment variable or --api-url flag."
            )
        if not self.admin_key:
            raise CEMSClientError(
                "Admin key is required. Set CEMS_ADMIN_KEY or use an admin user's API key."
            )

        self._headers = {
            "Authorization": f"Bearer {self.admin_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the admin API."""
        url = f"{self.api_url}{endpoint}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=self._headers,
                    json=json,
                    params=params,
                )

                if response.status_code == 401:
                    raise CEMSAuthError("Invalid admin key", status_code=401)
                if response.status_code == 403:
                    raise CEMSAuthError(
                        "Access denied - admin privileges required", status_code=403
                    )
                if response.status_code == 503:
                    try:
                        error = response.json().get("error", "Service unavailable")
                    except Exception:
                        error = "Service unavailable"
                    raise CEMSClientError(error, status_code=503)
                if response.status_code >= 400:
                    try:
                        error = response.json().get("error", response.text)
                    except Exception:
                        error = response.text
                    raise CEMSClientError(
                        f"API error: {error}", status_code=response.status_code
                    )

                return response.json()

        except httpx.ConnectError as e:
            raise CEMSConnectionError(f"Cannot connect to {self.api_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise CEMSConnectionError(f"Request timed out: {e}") from e
        except CEMSClientError:
            raise
        except Exception as e:
            raise CEMSClientError(f"Request failed: {e}") from e

    # =========================================================================
    # Admin Info
    # =========================================================================

    def info(self) -> dict[str, Any]:
        """Get admin API status.

        Returns:
            Admin API status info
        """
        return self._request("GET", "/admin")

    # =========================================================================
    # User Management
    # =========================================================================

    def list_users(
        self, include_inactive: bool = False, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        """List all users.

        Args:
            include_inactive: Include inactive users
            limit: Maximum users to return
            offset: Pagination offset

        Returns:
            Dict with users list and count
        """
        params = {
            "include_inactive": "true" if include_inactive else "false",
            "limit": str(limit),
            "offset": str(offset),
        }
        return self._request("GET", "/admin/users", params=params)

    def create_user(
        self,
        username: str,
        email: str | None = None,
        is_admin: bool = False,
        settings: dict | None = None,
    ) -> dict[str, Any]:
        """Create a new user.

        Args:
            username: Unique username
            email: Optional email
            is_admin: Whether user is admin
            settings: Optional user settings

        Returns:
            Dict with user info and API key (shown only once!)
        """
        data: dict[str, Any] = {"username": username, "is_admin": is_admin}
        if email:
            data["email"] = email
        if settings:
            data["settings"] = settings

        return self._request("POST", "/admin/users", json=data)

    def get_user(self, user_id_or_username: str) -> dict[str, Any]:
        """Get user details.

        Args:
            user_id_or_username: User ID (UUID) or username

        Returns:
            User details dict
        """
        return self._request("GET", f"/admin/users/{user_id_or_username}")

    def update_user(
        self,
        user_id: str,
        email: str | None = None,
        is_active: bool | None = None,
        is_admin: bool | None = None,
        settings: dict | None = None,
    ) -> dict[str, Any]:
        """Update user fields.

        Args:
            user_id: User ID (UUID)
            email: New email
            is_active: New active status
            is_admin: New admin status
            settings: New settings

        Returns:
            Updated user info
        """
        data: dict[str, Any] = {}
        if email is not None:
            data["email"] = email
        if is_active is not None:
            data["is_active"] = is_active
        if is_admin is not None:
            data["is_admin"] = is_admin
        if settings is not None:
            data["settings"] = settings

        return self._request("PATCH", f"/admin/users/{user_id}", json=data)

    def delete_user(self, user_id: str) -> dict[str, Any]:
        """Delete a user.

        Args:
            user_id: User ID (UUID)

        Returns:
            Deletion confirmation
        """
        return self._request("DELETE", f"/admin/users/{user_id}")

    def reset_api_key(self, user_id: str) -> dict[str, Any]:
        """Reset a user's API key.

        Args:
            user_id: User ID (UUID)

        Returns:
            Dict with new API key (shown only once!)
        """
        return self._request("POST", f"/admin/users/{user_id}/reset-key")

    # =========================================================================
    # Team Management
    # =========================================================================

    def list_teams(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """List all teams.

        Args:
            limit: Maximum teams to return
            offset: Pagination offset

        Returns:
            Dict with teams list and count
        """
        params = {"limit": str(limit), "offset": str(offset)}
        return self._request("GET", "/admin/teams", params=params)

    def create_team(
        self,
        name: str,
        company_id: str,
        settings: dict | None = None,
    ) -> dict[str, Any]:
        """Create a new team.

        Args:
            name: Unique team name
            company_id: Company identifier
            settings: Optional team settings

        Returns:
            Created team info
        """
        data: dict[str, Any] = {"name": name, "company_id": company_id}
        if settings:
            data["settings"] = settings

        return self._request("POST", "/admin/teams", json=data)

    def get_team(self, team_id_or_name: str) -> dict[str, Any]:
        """Get team details with members.

        Args:
            team_id_or_name: Team ID (UUID) or name

        Returns:
            Team details with member list
        """
        return self._request("GET", f"/admin/teams/{team_id_or_name}")

    def delete_team(self, team_id: str) -> dict[str, Any]:
        """Delete a team.

        Args:
            team_id: Team ID (UUID)

        Returns:
            Deletion confirmation
        """
        return self._request("DELETE", f"/admin/teams/{team_id}")

    def add_team_member(
        self, team_id: str, user_id: str, role: str = "member"
    ) -> dict[str, Any]:
        """Add a user to a team.

        Args:
            team_id: Team ID (UUID)
            user_id: User ID (UUID) to add
            role: Role ('admin', 'member', 'viewer')

        Returns:
            Membership info
        """
        return self._request(
            "POST",
            f"/admin/teams/{team_id}/members",
            json={"user_id": user_id, "role": role},
        )

    def remove_team_member(self, team_id: str, user_id: str) -> dict[str, Any]:
        """Remove a user from a team.

        Args:
            team_id: Team ID (UUID)
            user_id: User ID (UUID) to remove

        Returns:
            Removal confirmation
        """
        return self._request("DELETE", f"/admin/teams/{team_id}/members/{user_id}")
