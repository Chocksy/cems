#!/usr/bin/env python3
"""CEMS credentials loader.

Loads CEMS_API_URL and CEMS_API_KEY from environment variables first,
falling back to ~/.cems/credentials file. This allows team members to
configure CEMS once via `install.sh` without needing shell exports.

Credentials file format (dotenv):
    CEMS_API_URL=https://cems.chocksy.com
    CEMS_API_KEY=cems_ak_...
"""

import os
from pathlib import Path

_DEFAULT_CREDENTIALS_PATH = str(Path.home() / ".cems" / "credentials")

# Module-level cache (loaded once per process)
_cache: dict[str, str] | None = None


def _get_credentials_path() -> Path:
    """Get credentials file path (supports CEMS_CREDENTIALS_FILE override for testing)."""
    return Path(os.getenv("CEMS_CREDENTIALS_FILE", _DEFAULT_CREDENTIALS_PATH))


def _load_credentials_file() -> dict[str, str]:
    """Parse ~/.cems/credentials as key=value pairs."""
    creds_file = _get_credentials_path()
    result = {}
    try:
        if creds_file.exists():
            for line in creds_file.read_text().splitlines():
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
    return result


def _get_cached() -> dict[str, str]:
    """Get cached credentials (loads file once)."""
    global _cache
    if _cache is None:
        _cache = _load_credentials_file()
    return _cache


def get_cems_url() -> str:
    """Get CEMS API URL. Env var takes priority, then credentials file."""
    env_val = os.getenv("CEMS_API_URL", "")
    if env_val:
        return env_val
    return _get_cached().get("CEMS_API_URL", "")


def get_cems_key() -> str:
    """Get CEMS API key. Env var takes priority, then credentials file."""
    env_val = os.getenv("CEMS_API_KEY", "")
    if env_val:
        return env_val
    return _get_cached().get("CEMS_API_KEY", "")


def get_credentials_env() -> dict[str, str]:
    """Get a dict of CEMS env vars suitable for subprocess.Popen(env=...).

    Merges current os.environ with credentials file values.
    Env vars already set take priority (won't be overridden).
    """
    env = dict(os.environ)
    creds = _get_cached()
    for key in ("CEMS_API_URL", "CEMS_API_KEY"):
        if not env.get(key) and key in creds:
            env[key] = creds[key]
    return env
