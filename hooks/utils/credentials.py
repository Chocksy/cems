#!/usr/bin/env python3
"""CEMS credentials loader with per-project support.

Resolution order:
  1. Environment variables (CEMS_API_URL, CEMS_API_KEY) — highest priority
  2. .cems/credentials found by walking up from CWD — project override
  3. ~/.cems/credentials — global fallback

Project credentials: place a .cems/credentials file in your project root.
Same dotenv format as the global file. Walk-up stops before $HOME to avoid
conflating the global ~/.cems/credentials with a project credential file.

Credentials file format (dotenv):
    CEMS_API_URL=https://cems.example.com
    CEMS_API_KEY=cems_ak_...
"""

from __future__ import annotations

import os
from pathlib import Path

_HOME = str(Path.home())
_DEFAULT_CREDENTIALS_PATH = str(Path.home() / ".cems" / "credentials")

# Cache keyed by resolved file path (supports per-project resolution)
_cache: dict[str, dict[str, str]] = {}


def _get_credentials_path() -> Path:
    """Get credentials file path (supports CEMS_CREDENTIALS_FILE override for testing)."""
    return Path(os.getenv("CEMS_CREDENTIALS_FILE", _DEFAULT_CREDENTIALS_PATH))


def _parse_credentials_file(path: str) -> dict[str, str]:
    """Parse a credentials file as key=value pairs."""
    if _cache and path in _cache:
        return _cache[path]
    result = {}
    try:
        p = Path(path)
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and value:
                        result[key] = value
    except OSError:
        pass
    if _cache is not None:
        _cache[path] = result
    return result


def _find_project_credentials(cwd: str) -> str | None:
    """Walk up from CWD looking for .cems/credentials. Stops before $HOME."""
    try:
        path = Path(cwd).resolve()
    except (OSError, ValueError):
        return None
    while str(path) != _HOME and path != path.parent:
        project_creds = path / ".cems" / "credentials"
        if project_creds.is_file():
            return str(project_creds)
        path = path.parent
    return None


def resolve_credentials(cwd: str | None = None) -> dict[str, str]:
    """Resolve CEMS credentials with full precedence chain.

    1. Environment variables (always win — useful for CI, testing)
    2. Per-project .cems/credentials (walk up from CWD, stop before $HOME)
    3. Global ~/.cems/credentials (fallback)

    Returns dict with all keys found (CEMS_API_URL, CEMS_API_KEY, etc.)
    """
    # 1. Check env vars — if both URL and key are set, use them
    env_url = os.environ.get("CEMS_API_URL", "")
    env_key = os.environ.get("CEMS_API_KEY", "")
    if env_url and env_key:
        result = {"CEMS_API_URL": env_url, "CEMS_API_KEY": env_key}
        # Also pick up other env vars
        for k in ("CEMS_TEAM_ID", "CEMS_SEARCH_MODE"):
            v = os.environ.get(k, "")
            if v:
                result[k] = v
        return result

    # 2. Walk up from CWD looking for project .cems/credentials
    if cwd:
        project_path = _find_project_credentials(cwd)
        if project_path:
            return _parse_credentials_file(project_path)

    # 3. Global fallback
    return _parse_credentials_file(str(_get_credentials_path()))


class CEMSClient:
    """Lightweight CEMS API client resolved from credentials.

    Encapsulates URL + auth so hooks don't pass api_url/api_key everywhere.
    Created once in main() via CEMSClient.from_cwd(cwd).
    """

    def __init__(self, api_url: str, api_key: str):
        self.url = api_url
        self.key = api_key

    @classmethod
    def from_cwd(cls, cwd: str | None = None) -> CEMSClient | None:
        """Resolve credentials from CWD and return a client, or None if unconfigured."""
        creds = resolve_credentials(cwd)
        url = creds.get("CEMS_API_URL", "")
        key = creds.get("CEMS_API_KEY", "")
        if url and key:
            return cls(url, key)
        return None

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    def get(self, path: str, **kwargs):
        """HTTP GET to CEMS API. Returns httpx.Response."""
        import httpx
        kwargs.setdefault("timeout", 5.0)
        return httpx.get(
            f"{self.url}{path}",
            headers={"Authorization": f"Bearer {self.key}"},
            **kwargs,
        )

    def post(self, path: str, **kwargs):
        """HTTP POST to CEMS API. Returns httpx.Response."""
        import httpx
        kwargs.setdefault("timeout", 5.0)
        return httpx.post(
            f"{self.url}{path}",
            headers={"Authorization": f"Bearer {self.key}"},
            **kwargs,
        )


def get_cems_url(cwd: str | None = None) -> str:
    """Get CEMS API URL with optional CWD-based project resolution."""
    return resolve_credentials(cwd).get("CEMS_API_URL", "")


def get_cems_key(cwd: str | None = None) -> str:
    """Get CEMS API key with optional CWD-based project resolution."""
    return resolve_credentials(cwd).get("CEMS_API_KEY", "")


def get_credentials_env(cwd: str | None = None) -> dict[str, str]:
    """Get a dict of CEMS env vars suitable for subprocess.Popen(env=...).

    Merges current os.environ with resolved credentials.
    Env vars already set take priority (won't be overridden).
    """
    env = dict(os.environ)
    creds = resolve_credentials(cwd)
    for key in ("CEMS_API_URL", "CEMS_API_KEY"):
        if not env.get(key) and key in creds:
            env[key] = creds[key]
    return env
