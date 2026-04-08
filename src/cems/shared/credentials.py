"""Shared CEMS credential resolution primitives.

Pure stdlib — no httpx or heavy dependencies. Used by:
- src/cems/observer/daemon.py (CredentialResolver)
- src/cems/commands/env.py (cems env)

NOT used by hooks/ (they run standalone via uv and cannot import from cems).
hooks/utils/credentials.py maintains its own copy of these primitives.
"""

from __future__ import annotations

import os
from pathlib import Path

_HOME = str(Path.home().resolve())  # Resolve symlinks for consistent walk-up comparison
_DEFAULT_CREDENTIALS_PATH = str(Path.home() / ".cems" / "credentials")


def parse_credentials_file(path: str) -> dict[str, str]:
    """Parse a dotenv-style credentials file as key=value pairs.

    Skips empty lines, comments (#), and lines without =.
    Strips surrounding quotes from values.
    """
    result: dict[str, str] = {}
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
    return result


def find_project_credentials(cwd: str) -> str | None:
    """Walk up from CWD looking for .cems/credentials. Stops before $HOME.

    Returns the file path if found, None otherwise. Resolves symlinks.
    """
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

    1. Per-project .cems/credentials (walk up from CWD, stop before $HOME)
    2. Environment variables (both CEMS_API_URL and CEMS_API_KEY must be set)
    3. Global ~/.cems/credentials (fallback)

    Returns dict with all keys found (CEMS_API_URL, CEMS_API_KEY, etc.)
    """
    # 1. Walk up from CWD looking for project .cems/credentials
    if cwd:
        project_path = find_project_credentials(cwd)
        if project_path:
            return parse_credentials_file(project_path)

    # 2. Check env vars — require BOTH URL and key
    env_url = os.environ.get("CEMS_API_URL", "")
    env_key = os.environ.get("CEMS_API_KEY", "")
    if env_url and env_key:
        result = {"CEMS_API_URL": env_url, "CEMS_API_KEY": env_key}
        for k in ("CEMS_TEAM_ID", "CEMS_SEARCH_MODE"):
            v = os.environ.get(k, "")
            if v:
                result[k] = v
        return result

    # 3. Global fallback
    global_path = os.getenv("CEMS_CREDENTIALS_FILE", _DEFAULT_CREDENTIALS_PATH)
    return parse_credentials_file(global_path)
