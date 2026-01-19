"""Repository indexer - scans codebases and extracts knowledge into CEMS."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from cems.indexer.extractors import extract_knowledge
from cems.indexer.patterns import IndexPattern, get_default_patterns, match_files
from cems.models import MemoryMetadata, MemoryScope

if TYPE_CHECKING:
    from cems.memory import CEMSMemory

logger = logging.getLogger(__name__)


class RepositoryIndexer:
    """Index repositories and extract knowledge into CEMS memory.

    This indexer scans codebases and extracts:
    - RSpec/testing conventions
    - Architecture decisions
    - Code style guidelines
    - Documentation
    - API specifications

    Extracted knowledge is stored as PINNED memories that won't decay.
    """

    def __init__(
        self,
        memory: "CEMSMemory",
        patterns: list[IndexPattern] | None = None,
    ):
        """Initialize the indexer.

        Args:
            memory: CEMSMemory instance to store extracted knowledge
            patterns: Custom patterns to use (defaults to built-in patterns)
        """
        self.memory = memory
        self.patterns = patterns or get_default_patterns()

    def index_local_path(
        self,
        repo_path: str | Path,
        scope: str = "shared",
        patterns: list[str] | None = None,
    ) -> dict:
        """Index a local repository path.

        Args:
            repo_path: Path to the repository
            scope: Memory scope ("personal" or "shared")
            patterns: Optional list of pattern names to use (uses all if None)

        Returns:
            Dict with indexing results
        """
        repo_path = Path(repo_path).resolve()
        if not repo_path.is_dir():
            raise ValueError(f"Path does not exist or is not a directory: {repo_path}")

        # Filter patterns if specific ones requested
        active_patterns = self.patterns
        if patterns:
            active_patterns = [p for p in self.patterns if p.name in patterns]

        results = {
            "repo_path": str(repo_path),
            "files_scanned": 0,
            "knowledge_extracted": 0,
            "memories_created": 0,
            "patterns_used": [p.name for p in active_patterns],
            "errors": [],
        }

        for pattern in active_patterns:
            logger.info(f"Scanning for pattern: {pattern.name}")
            matched_files = match_files(repo_path, pattern)
            results["files_scanned"] += len(matched_files)

            for file_path in matched_files:
                try:
                    knowledge_items = extract_knowledge(file_path, pattern.extract_type)
                    results["knowledge_extracted"] += len(knowledge_items)

                    for item in knowledge_items:
                        # Store as pinned memory
                        mem_result = self.memory.add(
                            content=item.content,
                            scope=scope,  # type: ignore
                            category=item.category,
                            source="indexer",
                            tags=item.tags or [],
                        )

                        # Pin the memory
                        if mem_result and "results" in mem_result:
                            for r in mem_result["results"]:
                                if r.get("id") and r.get("event") in ("ADD", "UPDATE"):
                                    memory_id = r["id"]
                                    # Update metadata to pin it
                                    metadata = self.memory.get_metadata(memory_id)
                                    if metadata:
                                        metadata.pinned = True
                                        metadata.pin_reason = f"Indexed from {file_path.name}"
                                        metadata.pin_category = pattern.pin_category
                                        metadata.source = "indexer"
                                        metadata.source_ref = f"repo:{repo_path.name}:{file_path.relative_to(repo_path)}"
                                        metadata.priority = pattern.priority
                                        self.memory.metadata_store.save_metadata(metadata)
                                    results["memories_created"] += 1

                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e}"
                    logger.warning(error_msg)
                    results["errors"].append(error_msg)

        return results

    def index_git_repo(
        self,
        repo_url: str,
        branch: str = "main",
        scope: str = "shared",
        patterns: list[str] | None = None,
    ) -> dict:
        """Clone and index a git repository.

        Args:
            repo_url: Git repository URL
            branch: Branch to clone
            scope: Memory scope
            patterns: Optional list of pattern names to use

        Returns:
            Dict with indexing results
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clone the repository
            logger.info(f"Cloning {repo_url} ({branch}) to {tmpdir}")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmpdir],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone repository: {e.stderr}")

            # Index the cloned repo
            results = self.index_local_path(tmpdir, scope=scope, patterns=patterns)
            results["repo_url"] = repo_url
            results["branch"] = branch
            return results

    def list_patterns(self) -> list[dict]:
        """List available index patterns.

        Returns:
            List of pattern info dicts
        """
        return [
            {
                "name": p.name,
                "description": p.description,
                "file_patterns": p.file_patterns,
                "extract_type": p.extract_type,
                "pin_category": p.pin_category,
            }
            for p in self.patterns
        ]


def create_indexer(memory: "CEMSMemory") -> RepositoryIndexer:
    """Create a repository indexer.

    Args:
        memory: CEMSMemory instance

    Returns:
        Configured RepositoryIndexer
    """
    return RepositoryIndexer(memory)
