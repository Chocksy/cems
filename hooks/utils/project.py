"""Project identification utilities for CEMS hooks."""

import re
import subprocess


def get_project_id(cwd: str) -> str | None:
    """Extract project ID from git remote (e.g., 'org/repo').

    Parses the git remote origin URL to extract the org/repo identifier.
    Works with both SSH and HTTPS formats.

    Args:
        cwd: Current working directory (project root)

    Returns:
        Project ID like 'org/repo' or None if not a git repo
    """
    if not cwd:
        return None

    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # SSH: git@github.com:org/repo.git -> org/repo
            if url.startswith("git@"):
                match = re.search(r":(.+?)(?:\.git)?$", url)
            else:
                # HTTPS: https://github.com/org/repo.git -> org/repo
                match = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
            if match:
                return match.group(1).removesuffix('.git')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
