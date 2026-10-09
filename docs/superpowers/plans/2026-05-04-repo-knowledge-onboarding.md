# Repo Knowledge Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `cems index repo` and `cems index path` create project-scoped, protected repo knowledge and optionally build entity pages so a new user can point CEMS at a repository and get useful recall topics.

**Architecture:** Keep the current `memory_documents` architecture: indexed repo knowledge remains normal memory documents, and entity pages remain `memory_documents` with `category='entity-page'`. Add metadata at indexing time (`source_ref`, file tags, `pinned` tag), make local path indexing write through the authenticated HTTP client, then optionally run relation building and compilation as the post-index knowledge build. Avoid a new entity table or a separate index storage path.

**Tech Stack:** Python 3.11+, Click, Starlette, PostgreSQL/pgvector, existing CEMS REST API, existing APScheduler maintenance jobs, pytest.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/cems/indexer/indexer.py` | Modify | Preserve existing pattern scanner, add project/source metadata, support any writer with an `add(...)` method |
| `src/cems/indexer/writers.py` | Create | Adapter that lets the local CLI index files and write each extracted item through `CEMSClient.add(...)` |
| `src/cems/api/handlers/index.py` | Modify | Accept `build_entities`; run relation builder and compilation after remote repo indexing when requested |
| `src/cems/client.py` | Modify | Add `build_entities` to index calls; widen maintenance job contract and allow sweep params |
| `src/cems/commands/index.py` | Modify | Make `cems index path` run locally, add `--build-entities`, display post-index job results |
| `src/cems/commands/maintenance.py` | Modify | Expose all maintenance job types that the API already supports |
| `src/cems/mcp_stdio.py` | Modify | Expose all maintenance job types in the stdio MCP tool docstring/behavior |
| `mcp-wrapper/src/index.ts` | Modify | Expose all maintenance job types in the Streamable HTTP MCP wrapper schema |
| `tests/test_indexer.py` | Create | Unit coverage for project source refs, pinned tags, file tags, unknown pattern reporting, and local path indexing behavior |
| `tests/test_server.py` | Modify | API coverage for `build_entities` invoking relations and compilation |
| `tests/test_client.py` | Create | Client payload coverage for `build_entities` and maintenance params |
| `tests/test_rule_commands.py` or `tests/test_index_commands.py` | Create/modify | Click command coverage for local `cems index path` and maintenance job choices |

---

### Task 1: Make Indexed Knowledge Project-Scoped and Protected

**Files:**
- Create: `tests/test_indexer.py`
- Modify: `src/cems/indexer/indexer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_indexer.py`:

```python
from pathlib import Path

from cems.indexer.indexer import RepositoryIndexer


class RecordingWriter:
    def __init__(self):
        self.calls = []

    def add(self, content, scope="personal", category="general", source=None, tags=None, source_ref=None):
        self.calls.append({
            "content": content,
            "scope": scope,
            "category": category,
            "source": source,
            "tags": tags or [],
            "source_ref": source_ref,
        })
        return {"results": [{"id": f"doc-{len(self.calls)}", "event": "ADD"}]}


def test_index_local_path_writes_project_metadata_and_pinned_tags(tmp_path):
    repo = tmp_path / "widget"
    repo.mkdir()
    (repo / "README.md").write_text("# Widget\n\nWidget setup instructions and conventions.\n")

    writer = RecordingWriter()
    result = RepositoryIndexer(writer).index_local_path(
        repo,
        scope="shared",
        project="Acme/widget",
        patterns=["readme_docs"],
    )

    assert result["files_scanned"] == 1
    assert result["knowledge_extracted"] >= 1
    assert result["memories_created"] >= 1
    assert result["source_ref"] == "project:Acme/widget"

    first = writer.calls[0]
    assert first["source"] == "indexer"
    assert first["source_ref"] == "project:Acme/widget"
    assert first["scope"] == "shared"
    assert "pinned" in first["tags"]
    assert "indexed" in first["tags"]
    assert "pattern:readme_docs" in first["tags"]
    assert "pin-category:documentation" in first["tags"]
    assert "file:README.md" in first["tags"]


def test_index_local_path_reports_unknown_patterns(tmp_path):
    repo = tmp_path / "widget"
    repo.mkdir()
    (repo / "README.md").write_text("# Widget\n\nWidget setup instructions and conventions.\n")

    writer = RecordingWriter()
    result = RepositoryIndexer(writer).index_local_path(
        repo,
        project="Acme/widget",
        patterns=["not_a_pattern"],
    )

    assert result["files_scanned"] == 0
    assert result["knowledge_extracted"] == 0
    assert result["memories_created"] == 0
    assert result["errors"] == ["Unknown index pattern: not_a_pattern"]
    assert writer.calls == []


def test_index_git_repo_derives_project_from_https_url(monkeypatch, tmp_path):
    cloned_repo = tmp_path / "clone"
    cloned_repo.mkdir()
    (cloned_repo / "README.md").write_text("# Widget\n\nWidget setup instructions and conventions.\n")

    class TempDir:
        def __enter__(self):
            return str(cloned_repo)

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = []

    def fake_run(args, check, capture_output, text):
        calls.append(args)
        return object()

    monkeypatch.setattr("cems.indexer.indexer.tempfile.TemporaryDirectory", lambda: TempDir())
    monkeypatch.setattr("cems.indexer.indexer.subprocess.run", fake_run)

    writer = RecordingWriter()
    result = RepositoryIndexer(writer).index_git_repo(
        repo_url="https://github.com/Acme/widget.git",
        branch="main",
        patterns=["readme_docs"],
    )

    assert calls[0] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        "https://github.com/Acme/widget.git",
        str(cloned_repo),
    ]
    assert result["source_ref"] == "project:Acme/widget"
    assert writer.calls[0]["source_ref"] == "project:Acme/widget"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
pytest tests/test_indexer.py -v
```

Expected: fail with `TypeError` for unexpected `project` keyword and/or missing `source_ref`/tag assertions.

- [ ] **Step 3: Update `src/cems/indexer/indexer.py`**

Replace the file with this implementation:

```python
"""Repository indexer - scans codebases and extracts knowledge into CEMS."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from cems.indexer.extractors import extract_knowledge
from cems.indexer.patterns import IndexPattern, get_default_patterns, match_files

logger = logging.getLogger(__name__)


class MemoryWriter(Protocol):
    """Minimal write interface used by the repository indexer."""

    def add(
        self,
        content: str,
        scope: str = "personal",
        category: str = "general",
        source: str | None = None,
        tags: list[str] | None = None,
        source_ref: str | None = None,
    ) -> dict:
        """Store one extracted knowledge item."""


def project_from_repo_url(repo_url: str) -> str | None:
    """Extract org/repo from common HTTPS/SSH git URLs."""
    cleaned = repo_url.strip().removesuffix(".git")

    https_match = re.search(r"https://[^/]+/([^/]+/[^/]+)$", cleaned)
    if https_match:
        return https_match.group(1)

    ssh_match = re.search(r"git@[^:]+:([^/]+/[^/]+)$", cleaned)
    if ssh_match:
        return ssh_match.group(1)

    return None


def project_from_local_path(repo_path: Path) -> str | None:
    """Read git origin from a local checkout and return org/repo when possible."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    return project_from_repo_url(result.stdout.strip())


def source_ref_for_project(project: str | None) -> str | None:
    """Return the canonical project-scoped source_ref."""
    if not project:
        return None
    return f"project:{project}"


def _safe_file_tag(repo_path: Path, file_path: Path) -> str:
    """Create a compact file tag for indexed memories."""
    try:
        rel = file_path.relative_to(repo_path).as_posix()
    except ValueError:
        rel = file_path.name
    return f"file:{rel}"[:200]


def _index_tags(item_tags: list[str] | None, pattern: IndexPattern, repo_path: Path, file_path: Path) -> list[str]:
    """Build tags that mark indexed knowledge as protected and traceable."""
    tags = list(item_tags or [])
    tags.extend([
        "indexed",
        "pinned",
        f"pattern:{pattern.name}",
        f"pin-category:{pattern.pin_category}",
        _safe_file_tag(repo_path, file_path),
    ])

    deduped = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped[:50]


class RepositoryIndexer:
    """Index repositories and extract knowledge into CEMS memory.

    Extracted knowledge is stored as project-scoped memory documents tagged with
    ``pinned`` so maintenance jobs treat repo knowledge as durable onboarding context.
    """

    def __init__(
        self,
        memory: MemoryWriter,
        patterns: list[IndexPattern] | None = None,
    ):
        self.memory = memory
        self.patterns = patterns or get_default_patterns()

    def _active_patterns(self, pattern_names: list[str] | None, errors: list[str]) -> list[IndexPattern]:
        if not pattern_names:
            return self.patterns

        by_name = {pattern.name: pattern for pattern in self.patterns}
        active = []
        for name in pattern_names:
            pattern = by_name.get(name)
            if pattern:
                active.append(pattern)
            else:
                errors.append(f"Unknown index pattern: {name}")
        return active

    def index_local_path(
        self,
        repo_path: str | Path,
        scope: str = "shared",
        patterns: list[str] | None = None,
        project: str | None = None,
        source_ref: str | None = None,
    ) -> dict:
        """Index a local repository path."""
        repo_path = Path(repo_path).resolve()
        if not repo_path.is_dir():
            raise ValueError(f"Path does not exist or is not a directory: {repo_path}")

        resolved_project = project or project_from_local_path(repo_path)
        resolved_source_ref = source_ref or source_ref_for_project(resolved_project)

        results = {
            "repo_path": str(repo_path),
            "project": resolved_project,
            "source_ref": resolved_source_ref,
            "files_scanned": 0,
            "knowledge_extracted": 0,
            "memories_created": 0,
            "patterns_used": [],
            "errors": [],
        }

        active_patterns = self._active_patterns(patterns, results["errors"])
        results["patterns_used"] = [pattern.name for pattern in active_patterns]

        for pattern in active_patterns:
            logger.info("Scanning for pattern: %s", pattern.name)
            matched_files = match_files(repo_path, pattern)
            results["files_scanned"] += len(matched_files)

            for file_path in matched_files:
                try:
                    knowledge_items = extract_knowledge(file_path, pattern.extract_type)
                    results["knowledge_extracted"] += len(knowledge_items)

                    for item in knowledge_items:
                        mem_result = self.memory.add(
                            content=item.content,
                            scope=scope,
                            category=item.category,
                            source="indexer",
                            tags=_index_tags(item.tags, pattern, repo_path, file_path),
                            source_ref=resolved_source_ref,
                        )

                        if mem_result and "results" in mem_result:
                            for row in mem_result["results"]:
                                if row.get("id") and row.get("event") in ("ADD", "UPDATE"):
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
        project: str | None = None,
    ) -> dict:
        """Clone and index a git repository."""
        resolved_project = project or project_from_repo_url(repo_url)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Cloning %s (%s) to %s", repo_url, branch, tmpdir)
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmpdir],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone repository: {e.stderr}") from e

            results = self.index_local_path(
                tmpdir,
                scope=scope,
                patterns=patterns,
                project=resolved_project,
            )
            results["repo_url"] = repo_url
            results["branch"] = branch
            return results

    def list_patterns(self) -> list[dict]:
        """List available index patterns."""
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


def create_indexer(memory: MemoryWriter) -> RepositoryIndexer:
    """Create a repository indexer."""
    return RepositoryIndexer(memory)
```

- [ ] **Step 4: Run the indexer tests**

Run:

```bash
pytest tests/test_indexer.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cems/indexer/indexer.py tests/test_indexer.py
git commit -m "$(cat <<'EOF'
Improve repo indexer metadata

Indexed repository knowledge now carries project source refs, file tags, and pinned tags so it behaves as durable onboarding context.
EOF
)"
```

---

### Task 2: Make `cems index path` Work Locally Against the Remote Server

**Files:**
- Create: `src/cems/indexer/writers.py`
- Create: `tests/test_index_commands.py`
- Modify: `src/cems/commands/index.py`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_index_commands.py`:

```python
from click.testing import CliRunner

import cems.commands.index as index_module


class FakeClient:
    def __init__(self):
        self.add_calls = []

    def add(self, content, category="general", scope=None, tags=None, source_ref=None):
        self.add_calls.append({
            "content": content,
            "category": category,
            "scope": scope,
            "tags": tags or [],
            "source_ref": source_ref,
        })
        return {"results": [{"id": "doc-1", "event": "ADD"}]}

    def maintenance(self, job_type, **kwargs):
        return {"success": True, "job_type": job_type, "results": {"ok": True}}


def test_index_path_runs_local_indexer_and_writes_through_client(monkeypatch, tmp_path):
    repo = tmp_path / "widget"
    repo.mkdir()
    (repo / "README.md").write_text("# Widget\n\nWidget setup instructions and conventions.\n")

    fake_client = FakeClient()
    monkeypatch.setattr(index_module, "get_client", lambda ctx: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        index_module.index,
        ["path", str(repo), "--scope", "shared", "--patterns", "readme_docs"],
        obj={"api_url": "https://cems.example.com", "api_key": "key", "verbose": False},
    )

    assert result.exit_code == 0
    assert "Indexing complete" in result.output
    assert fake_client.add_calls
    assert fake_client.add_calls[0]["scope"] == "shared"
    assert "pinned" in fake_client.add_calls[0]["tags"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_index_commands.py::test_index_path_runs_local_indexer_and_writes_through_client -v
```

Expected: fail because `index_path` still calls the disabled `/api/index/path` endpoint.

- [ ] **Step 3: Add an HTTP writer adapter**

Create `src/cems/indexer/writers.py`:

```python
"""Writer adapters for repository indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cems.client import CEMSClient


class CEMSClientMemoryWriter:
    """Adapter that lets RepositoryIndexer write through the HTTP client."""

    def __init__(self, client: "CEMSClient"):
        self.client = client

    def add(
        self,
        content: str,
        scope: str = "personal",
        category: str = "general",
        source: str | None = None,
        tags: list[str] | None = None,
        source_ref: str | None = None,
    ) -> dict:
        """Store one indexed item through the REST API.

        The REST memory API does not expose a `source` field today, so the
        source is represented through tags and source_ref.
        """
        normalized_tags = list(tags or [])
        if source:
            normalized_tags.append(f"source:{source}")

        return self.client.add(
            content=content,
            category=category,
            scope=scope,
            tags=normalized_tags,
            source_ref=source_ref,
        )
```

- [ ] **Step 4: Modify `src/cems/commands/index.py`**

Update imports:

```python
import json

import click
from rich.table import Table

from cems.cli_utils import console, get_client, handle_error
from cems.client import CEMSClientError
from cems.indexer import RepositoryIndexer
from cems.indexer.writers import CEMSClientMemoryWriter
```

Replace the body of `index_path` with:

```python
def index_path(
    ctx: click.Context,
    path: str,
    scope: str,
    patterns: tuple,
) -> None:
    """Index a local directory path from this machine.

    Extracted knowledge is written to the configured CEMS server through the
    authenticated REST API.
    """
    try:
        client = get_client(ctx)
        writer = CEMSClientMemoryWriter(client)
        indexer = RepositoryIndexer(writer)

        console.print(f"[cyan]Indexing {path}...[/cyan]")
        with console.status("Extracting knowledge..."):
            r = indexer.index_local_path(
                path,
                scope=scope,
                patterns=list(patterns) if patterns else None,
            )

        console.print("[green]Indexing complete![/green]")
        console.print(f"  Files scanned: {r.get('files_scanned', 0)}")
        console.print(f"  Knowledge extracted: {r.get('knowledge_extracted', 0)}")
        console.print(f"  Memories created: {r.get('memories_created', 0)}")
        console.print(f"  Patterns used: {', '.join(r.get('patterns_used', []))}")
        if r.get("source_ref"):
            console.print(f"  Source ref: {r['source_ref']}")

        errors = r.get("errors", [])
        if errors:
            console.print(f"\n[yellow]Warnings ({len(errors)}):[/yellow]")
            for err in errors[:5]:
                console.print(f"  [dim]{err}[/dim]")

        if ctx.obj["verbose"]:
            console.print(json.dumps({"success": True, "result": r}, indent=2, default=str))

    except (CEMSClientError, ValueError) as e:
        handle_error(e)
```

- [ ] **Step 5: Run the CLI test**

Run:

```bash
pytest tests/test_index_commands.py::test_index_path_runs_local_indexer_and_writes_through_client -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cems/indexer/writers.py src/cems/commands/index.py tests/test_index_commands.py
git commit -m "$(cat <<'EOF'
Make local path indexing write through the client

`cems index path` now indexes local files on the developer machine and stores extracted knowledge through the configured CEMS server.
EOF
)"
```

---

### Task 3: Add Explicit Entity Building After Indexing

**Files:**
- Modify: `src/cems/client.py`
- Modify: `src/cems/api/handlers/index.py`
- Modify: `src/cems/commands/index.py`
- Modify: `tests/test_server.py`
- Modify: `tests/test_index_commands.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write failing client tests**

Create `tests/test_client.py`:

```python
from cems.client import CEMSClient


class RecordingClient(CEMSClient):
    def __init__(self):
        super().__init__(api_url="https://cems.example.com", api_key="key")
        self.requests = []

    def _request(self, method, endpoint, **kwargs):
        self.requests.append((method, endpoint, kwargs))
        return {"success": True}


def test_index_repo_sends_build_entities_flag():
    client = RecordingClient()

    client.index_repo(
        "https://github.com/Acme/widget",
        branch="main",
        scope="shared",
        patterns=["readme_docs"],
        build_entities=True,
    )

    method, endpoint, kwargs = client.requests[0]
    assert method == "POST"
    assert endpoint == "/api/index/repo"
    assert kwargs["json"]["build_entities"] is True


def test_maintenance_accepts_relation_and_compilation_params():
    client = RecordingClient()

    client.maintenance("relations", limit=200, full_sweep=True)
    client.maintenance("compilation", limit=30)

    assert client.requests[0][2]["json"] == {
        "job_type": "relations",
        "full_sweep": True,
        "limit": 200,
    }
    assert client.requests[1][2]["json"] == {
        "job_type": "compilation",
        "limit": 30,
    }
```

- [ ] **Step 2: Run the client tests to verify they fail**

Run:

```bash
pytest tests/test_client.py -v
```

Expected: fail because `index_repo` does not accept `build_entities` and `maintenance` does not accept params.

- [ ] **Step 3: Update `CEMSClient`**

In `src/cems/client.py`, update the imports if needed:

```python
from typing import Any, Literal
```

Replace the maintenance method with:

```python
    def maintenance(
        self,
        job_type: Literal[
            "consolidation",
            "distillation",
            "summarization",
            "reindex",
            "reflect",
            "relations",
            "compilation",
            "orphan_assigner",
            "lint",
            "all",
        ],
        *,
        full_sweep: bool | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Run a maintenance job."""
        payload: dict[str, Any] = {"job_type": job_type}
        if full_sweep is not None:
            payload["full_sweep"] = full_sweep
        if limit is not None:
            payload["limit"] = limit
        if offset is not None:
            payload["offset"] = offset

        return self._request("POST", "/api/memory/maintenance", json=payload)
```

Replace the `index_repo` signature and payload construction with:

```python
    def index_repo(
        self,
        repo_url: str,
        branch: str = "main",
        scope: Literal["personal", "shared"] = "shared",
        patterns: list[str] | None = None,
        build_entities: bool = False,
    ) -> dict[str, Any]:
        """Index a git repository."""
        payload: dict[str, Any] = {
            "repo_url": repo_url,
            "branch": branch,
            "scope": scope,
            "build_entities": build_entities,
        }
        if patterns:
            payload["patterns"] = patterns

        return self._request("POST", "/api/index/repo", json=payload)
```

- [ ] **Step 4: Write failing API test for `build_entities`**

Append to `tests/test_server.py`:

```python
class TestIndexAPI:
    @patch("cems.db.database.is_database_initialized", return_value=True)
    @patch("cems.db.database.get_database")
    @patch("cems.api.handlers.index.RepositoryIndexer")
    @patch("cems.api.handlers.index.RelationBuilderJob")
    @patch("cems.api.handlers.index.CompilationJob")
    @patch("cems.api.handlers.index.get_memory")
    def test_index_repo_build_entities_runs_relations_then_compilation(
        self,
        mock_get_memory,
        mock_compilation_job,
        mock_relation_job,
        mock_indexer_cls,
        mock_db,
        mock_is_db,
        mock_memory,
        mock_user,
    ):
        mock_get_memory.return_value = mock_memory
        mock_indexer = mock_indexer_cls.return_value
        mock_indexer.index_git_repo.return_value = {
            "files_scanned": 1,
            "knowledge_extracted": 1,
            "memories_created": 1,
            "patterns_used": ["readme_docs"],
            "errors": [],
        }
        mock_relation_job.return_value.run_async = AsyncMock(return_value={"docs_processed": 1})
        mock_compilation_job.return_value.run_async = AsyncMock(return_value={"pages_created": 1})

        mock_session = MagicMock()
        mock_user_service = MagicMock()
        mock_user_service.get_user_by_api_key.return_value = mock_user
        mock_db.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("cems.admin.services.UserService", return_value=mock_user_service):
            from cems.server import create_http_app

            app = create_http_app()
            client = TestClient(app)
            response = client.post(
                "/api/index/repo",
                json={
                    "repo_url": "https://github.com/Acme/widget",
                    "branch": "main",
                    "patterns": ["readme_docs"],
                    "build_entities": True,
                },
                headers={"Authorization": "Bearer test-api-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["knowledge_build"]["relations"] == {"docs_processed": 1}
        assert data["result"]["knowledge_build"]["compilation"] == {"pages_created": 1}
```

- [ ] **Step 5: Run the API test to verify it fails**

Run:

```bash
pytest tests/test_server.py::TestIndexAPI::test_index_repo_build_entities_runs_relations_then_compilation -v
```

Expected: fail because `api_index_repo` does not import or run maintenance jobs.

- [ ] **Step 6: Update `src/cems/api/handlers/index.py` imports**

Replace the imports at the top with:

```python
import asyncio
import logging
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import JSONResponse

from cems.api.deps import get_memory
from cems.indexer import RepositoryIndexer
from cems.maintenance.compilation import CompilationJob
from cems.maintenance.relation_builder import RelationBuilderJob
```

Remove the inline `from cems.indexer import RepositoryIndexer` imports from `api_index_repo` and `api_index_patterns`.

- [ ] **Step 7: Update `api_index_repo` to run the post-index build**

Inside `api_index_repo`, read the new flag after `patterns`:

```python
        build_entities = bool(body.get("build_entities", False))
```

After the `asyncio.to_thread(...)` call and before the JSON response, add:

```python
        if build_entities:
            relations = await RelationBuilderJob(memory).run_async(limit=200)
            compilation = await CompilationJob(memory).run_async(limit=30)
            result["knowledge_build"] = {
                "relations": relations,
                "compilation": compilation,
            }
```

- [ ] **Step 8: Add CLI flags and local post-index build**

In `src/cems/commands/index.py`, add this option to both `index_repo` and `index_path`:

```python
@click.option(
    "--build-entities/--no-build-entities",
    default=False,
    help="After indexing, run relation building and entity compilation.",
)
```

Update `index_repo` signature:

```python
def index_repo(
    ctx: click.Context,
    repo_url: str,
    branch: str,
    scope: str,
    patterns: tuple,
    build_entities: bool,
) -> None:
```

Pass it to the client:

```python
            result = client.index_repo(
                repo_url,
                branch=branch,
                scope=scope,
                patterns=list(patterns) if patterns else None,
                build_entities=build_entities,
            )
```

Update `index_path` signature:

```python
def index_path(
    ctx: click.Context,
    path: str,
    scope: str,
    patterns: tuple,
    build_entities: bool,
) -> None:
```

After printing indexing warnings in `index_path`, add:

```python
        if build_entities:
            console.print("\n[cyan]Building relations and entity pages...[/cyan]")
            relations = client.maintenance("relations", limit=200)
            compilation = client.maintenance("compilation", limit=30)
            console.print("[green]Knowledge build complete![/green]")
            if ctx.obj["verbose"]:
                console.print(json.dumps({
                    "relations": relations,
                    "compilation": compilation,
                }, indent=2, default=str))
```

- [ ] **Step 9: Run focused tests**

Run:

```bash
pytest tests/test_client.py tests/test_server.py::TestIndexAPI::test_index_repo_build_entities_runs_relations_then_compilation tests/test_index_commands.py -v
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/cems/client.py src/cems/api/handlers/index.py src/cems/commands/index.py tests/test_client.py tests/test_server.py tests/test_index_commands.py
git commit -m "$(cat <<'EOF'
Add optional entity build after repo indexing

Indexing can now explicitly run relation building and entity compilation so repository imports become navigable knowledge topics.
EOF
)"
```

---

### Task 4: Align Maintenance Job Surfaces Across CLI and MCP

**Files:**
- Modify: `src/cems/commands/maintenance.py`
- Modify: `src/cems/mcp_stdio.py`
- Modify: `mcp-wrapper/src/index.ts`
- Modify: `tests/test_index_commands.py`

- [ ] **Step 1: Add failing CLI maintenance test**

Append to `tests/test_index_commands.py`:

```python
import cems.commands.maintenance as maintenance_module


def test_maintenance_cli_accepts_knowledge_engine_jobs(monkeypatch):
    calls = []

    class FakeClient:
        def maintenance(self, job_type, **kwargs):
            calls.append((job_type, kwargs))
            return {"success": True, "results": {"ok": True}}

    monkeypatch.setattr(maintenance_module, "get_client", lambda ctx: FakeClient())

    runner = CliRunner()
    result = runner.invoke(
        maintenance_module.maintenance,
        ["run", "compilation", "--limit", "30"],
        obj={"api_url": "https://cems.example.com", "api_key": "key", "verbose": False},
    )

    assert result.exit_code == 0
    assert calls == [("compilation", {"limit": 30})]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_index_commands.py::test_maintenance_cli_accepts_knowledge_engine_jobs -v
```

Expected: fail because `compilation` is not an accepted Click choice and `--limit` does not exist.

- [ ] **Step 3: Update `src/cems/commands/maintenance.py`**

Replace the command decorator and function with:

```python
@maintenance.command("run")
@click.argument(
    "job_type",
    type=click.Choice([
        "consolidation",
        "distillation",
        "summarization",
        "reindex",
        "reflect",
        "relations",
        "compilation",
        "orphan_assigner",
        "lint",
        "all",
    ]),
)
@click.option("--full-sweep", is_flag=True, help="Force supported jobs to process already-seen items.")
@click.option("--limit", type=int, help="Optional job-specific batch limit.")
@click.option("--offset", type=int, help="Optional job-specific offset.")
@click.pass_context
def run_maintenance(
    ctx: click.Context,
    job_type: str,
    full_sweep: bool,
    limit: int | None,
    offset: int | None,
) -> None:
    """Run a maintenance job immediately."""
    try:
        client = get_client(ctx)

        console.print(f"[cyan]Running {job_type}...[/cyan]")
        with console.status("Running maintenance..."):
            result = client.maintenance(
                job_type,  # type: ignore[arg-type]
                full_sweep=full_sweep if full_sweep else None,
                limit=limit,
                offset=offset,
            )

        if result.get("success"):
            console.print(f"[green]{job_type} completed[/green]")
            if ctx.obj["verbose"]:
                console.print(json.dumps(result, indent=2, default=str))
        else:
            console.print(f"[yellow]{job_type} may have failed[/yellow]")
            console.print(json.dumps(result, indent=2, default=str))

    except CEMSClientError as e:
        handle_error(e)
```

- [ ] **Step 4: Update Python stdio MCP maintenance docstring**

In `src/cems/mcp_stdio.py`, replace `memory_maintenance` with:

```python
@mcp.tool()
def memory_maintenance(
    job_type: str = "consolidation",
    full_sweep: bool = False,
    limit: int | None = None,
    offset: int | None = None,
) -> str:
    """Run maintenance jobs.

    Valid job_type values: consolidation, distillation, summarization, reindex,
    reflect, relations, compilation, orphan_assigner, lint, all.
    """
    if not API_URL:
        return _NOT_CONFIGURED_MSG

    payload: dict = {"job_type": job_type}
    if full_sweep:
        payload["full_sweep"] = True
    if limit is not None:
        payload["limit"] = limit
    if offset is not None:
        payload["offset"] = offset

    return json.dumps(_request("POST", "/api/memory/maintenance", payload))
```

- [ ] **Step 5: Update Node MCP wrapper schema**

In `mcp-wrapper/src/index.ts`, replace the `memory_maintenance` `inputSchema` with:

```typescript
        inputSchema: {
          job_type: z
            .enum([
              "consolidation",
              "distillation",
              "summarization",
              "reindex",
              "reflect",
              "relations",
              "compilation",
              "orphan_assigner",
              "lint",
              "all",
            ])
            .default("consolidation")
            .describe("Type of maintenance"),
          full_sweep: z.boolean().default(false).describe("Force supported jobs to process already-seen items"),
          limit: z.number().optional().describe("Optional job-specific batch limit"),
          offset: z.number().optional().describe("Optional job-specific offset"),
        },
```

The existing handler already forwards `args`, so no handler body change is needed.

- [ ] **Step 6: Run focused tests and TypeScript check**

Run:

```bash
pytest tests/test_index_commands.py::test_maintenance_cli_accepts_knowledge_engine_jobs tests/test_client.py -v
```

Expected: pass.

Run:

```bash
npm --prefix mcp-wrapper test
```

Expected: pass if the wrapper has tests; if no test script exists, npm reports the missing script.

- [ ] **Step 7: Commit**

```bash
git add src/cems/commands/maintenance.py src/cems/mcp_stdio.py mcp-wrapper/src/index.ts tests/test_index_commands.py
git commit -m "$(cat <<'EOF'
Expose knowledge maintenance jobs across clients

CLI and MCP maintenance surfaces now match the server API so agents can run relations and compilation when explicitly requested.
EOF
)"
```

---

### Task 5: Document and Verify the Repo Onboarding Flow

**Files:**
- Modify: `README.md`
- Modify: `docs/CLIENT.md` if it exists
- Modify: `docs/API.md` if it exists

- [ ] **Step 1: Check which docs exist**

Run:

```bash
ls docs
```

Expected: docs directory is listed. Only modify `docs/CLIENT.md` and `docs/API.md` if they exist.

- [ ] **Step 2: Update README indexing examples**

In `README.md`, add this under the client usage or documentation section:

```markdown
### Index a Repository

Point CEMS at a repository to store durable project knowledge:

```bash
cems index repo https://github.com/org/repo --build-entities
```

For private or local repositories, run indexing from a checkout:

```bash
cems index path /path/to/repo --build-entities
```

Indexed knowledge is stored with `source_ref=project:org/repo` and tagged as
`pinned`, so it can be recalled by project and protected from routine cleanup.
The `--build-entities` flag runs relation building and entity compilation after
indexing, making the imported repo navigable through knowledge topics.
```
```

- [ ] **Step 3: Update client docs if present**

If `docs/CLIENT.md` exists, add:

```markdown
## Repository Indexing

Use `cems index repo` for public HTTPS repositories:

```bash
cems index repo https://github.com/org/repo --patterns readme_docs --build-entities
```

Use `cems index path` for local or private checkouts:

```bash
cems index path . --build-entities
```

`--build-entities` is explicit because it runs maintenance jobs that may call
LLMs. Omit it when you only want to store extracted memories.
```
```

- [ ] **Step 4: Update API docs if present**

If `docs/API.md` exists, add `build_entities` to `POST /api/index/repo`:

```markdown
### POST /api/index/repo

Indexes a public HTTPS git repository.

```json
{
  "repo_url": "https://github.com/org/repo",
  "branch": "main",
  "scope": "shared",
  "patterns": ["readme_docs"],
  "build_entities": true
}
```

When `build_entities` is true, the server runs relation building and entity
compilation after indexing and returns a `knowledge_build` object in the result.
```
```

- [ ] **Step 5: Run verification**

Run:

```bash
pytest tests/test_indexer.py tests/test_index_commands.py tests/test_client.py tests/test_server.py::TestIndexAPI -v
```

Expected: all selected tests pass.

Run:

```bash
ruff check src/cems/indexer src/cems/commands/index.py src/cems/commands/maintenance.py src/cems/api/handlers/index.py src/cems/client.py tests/test_indexer.py tests/test_index_commands.py tests/test_client.py
```

Expected: no lint errors.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/CLIENT.md docs/API.md
git commit -m "$(cat <<'EOF'
Document repo knowledge onboarding

Add repo and path indexing examples with explicit entity building so new users can bootstrap project knowledge.
EOF
)"
```

If either docs file does not exist, omit it from `git add`.

---

## Self-Review

**Spec coverage:** The plan covers the observed missing pieces: project-scoped repo knowledge, pinned indexed memories, local private repo indexing, optional entity creation, maintenance surface alignment, and docs for new users.

**Placeholder scan:** No `TBD`, `TODO`, "implement later", or unexpanded edge-case instructions remain. Code snippets define the concrete functions, flags, tests, and commands needed for each task.

**Type consistency:** `RepositoryIndexer.index_local_path(..., project=None, source_ref=None)` is introduced before use. `CEMSClient.maintenance(..., full_sweep=None, limit=None, offset=None)` is introduced before CLI/MCP use. `build_entities` is consistently named in CLI, client, and API.

**Out of scope for this plan:** Cleaning all inline imports, resolving the 543 open conflicts, and diagnosing Tailscale/SSH are separate workstreams. They are important but independent from repo knowledge onboarding.

