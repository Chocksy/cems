# Private Mode: Provider Layer and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the CEMS memory pipeline run against any OpenAI-compatible model endpoint, and ship a compose profile plus installer that brings up a self-contained CEMS server with Ollama on a fresh VPS.

**Architecture:** One provider layer: `CEMSConfig` gains `llm_base_url`, `llm_api_key`, `embedding_base_url`, `embedding_api_key`; the LLM client and both embedding clients read them. OpenRouter remains the default and OpenRouter-only extras (attribution headers, provider routing, `dimensions`) are sent only when the host is OpenRouter. The llama.cpp backend is deleted. The pgvector column is created at the configured dimension and a mismatch aborts startup. Private mode is a compose profile (`ollama` + `ollama-pull`) plus two preset env files; no mode flag exists in code.

**Tech Stack:** Python 3.12, pydantic-settings, `openai` SDK, `httpx`, PostgreSQL 16 + pgvector, Docker Compose profiles, Ollama, cloud-init, bash.

**Spec:** `docs/superpowers/specs/2026-09-22-private-mode-and-enterprise-positioning-design.md` (Parts 1 and 2). Part 3 (site) gets its own plan after this one ships.

## Global Constraints

- All new config fields use env prefix `CEMS_` (existing `SettingsConfigDict(env_prefix="CEMS_")`).
- `OPENROUTER_API_KEY` keeps working unchanged as the fallback key. Existing deployments must start with no `.env` change.
- Default embedding dimension stays 1536. Default LLM base URL stays `https://openrouter.ai/api/v1`.
- No `private_mode` flag in code. Private mode is compose profile `private` plus `deploy/.env.private-cpu.example` or `deploy/.env.private-gpu.example`.
- CPU preset models: `gemma4:e4b` (chat) and `embeddinggemma` (embeddings, 768 dims).
- Ollama requires a non-empty API key and ignores it; presets set `CEMS_LLM_API_KEY=ollama`.
- Do not build or push Docker images by hand. CI builds on version tags.
- Tests run with `uv run pytest`. Lint with `uv run ruff check src tests`.
- Commit messages: conventional prefix, no em dashes.

---

## File Map

| File | Responsibility |
|---|---|
| `src/cems/config.py` | New provider fields, `is_openrouter_host()` helper. Remove `embedding_backend`, `llamacpp_*`. |
| `src/cems/llm/client.py` | `OpenRouterClient` uses config base URL and key; OpenRouter extras conditional. |
| `src/cems/embedding.py` | Both embedding clients use config URL and key; `dimensions` conditional. |
| `src/cems/memory/core.py` | Single embedder code path. |
| `src/cems/llamacpp_server.py` | Deleted. |
| `src/cems/db/database.py` | Vector column at configured dimension; `check_embedding_dimension()`. |
| `src/cems/agentic/search.py` + `src/cems/api/handlers/memory.py` | `enable_agentic_search` flag. |
| `src/cems/admin/routes.py` | Health check uses configured endpoint. |
| `deploy/docker-compose.yml` | `private` profile: `ollama`, `ollama-pull`; env passthrough. |
| `deploy/.env.private-cpu.example`, `deploy/.env.private-gpu.example` | Presets. |
| `install-server.sh` | Server installer. |
| `deploy/cloud-init/{aws,hetzner,digitalocean}.yaml` | Boot scripts. |
| `docs/DEPLOYMENT.md`, `docs/CLIENT.md`, `README.md` | Docs. |
| `tests/test_config.py`, `tests/test_llm.py`, `tests/test_embedding.py`, `tests/test_database_dimension.py` | Tests. |

---

### Task 1: Provider config fields

**Files:**
- Modify: `src/cems/config.py:78-125` (model settings, embedding backend, llama.cpp settings)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `CEMSConfig.llm_base_url: str`, `CEMSConfig.llm_api_key: str | None`, `CEMSConfig.embedding_base_url: str | None`, `CEMSConfig.embedding_api_key: str | None`, `CEMSConfig.enable_agentic_search: bool`, methods `resolved_llm_api_key() -> str | None`, `resolved_embedding_base_url() -> str`, `resolved_embedding_api_key() -> str | None`, module function `is_openrouter_host(url: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
import os
from unittest.mock import patch

from cems.config import CEMSConfig, is_openrouter_host


class TestProviderConfig:
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-x"}, clear=False)
    def test_defaults_point_at_openrouter(self):
        cfg = CEMSConfig()
        assert cfg.llm_base_url == "https://openrouter.ai/api/v1"
        assert cfg.resolved_llm_api_key() == "sk-or-x"
        assert cfg.resolved_embedding_base_url() == "https://openrouter.ai/api/v1"
        assert cfg.resolved_embedding_api_key() == "sk-or-x"
        assert cfg.embedding_dimension == 1536
        assert cfg.enable_agentic_search is True

    @patch.dict(
        os.environ,
        {
            "CEMS_LLM_BASE_URL": "http://ollama:11434/v1",
            "CEMS_LLM_API_KEY": "ollama",
            "CEMS_EMBEDDING_DIMENSION": "768",
        },
        clear=False,
    )
    def test_custom_endpoint_falls_through_to_embeddings(self):
        cfg = CEMSConfig()
        assert cfg.llm_base_url == "http://ollama:11434/v1"
        assert cfg.resolved_llm_api_key() == "ollama"
        assert cfg.resolved_embedding_base_url() == "http://ollama:11434/v1"
        assert cfg.resolved_embedding_api_key() == "ollama"
        assert cfg.embedding_dimension == 768

    @patch.dict(
        os.environ,
        {
            "CEMS_LLM_BASE_URL": "http://ollama:11434/v1",
            "CEMS_LLM_API_KEY": "ollama",
            "CEMS_EMBEDDING_BASE_URL": "https://api.openai.com/v1",
            "CEMS_EMBEDDING_API_KEY": "sk-openai",
        },
        clear=False,
    )
    def test_embedding_endpoint_overrides_independently(self):
        cfg = CEMSConfig()
        assert cfg.resolved_embedding_base_url() == "https://api.openai.com/v1"
        assert cfg.resolved_embedding_api_key() == "sk-openai"

    def test_llamacpp_fields_are_gone(self):
        assert not hasattr(CEMSConfig(), "embedding_backend")
        assert not hasattr(CEMSConfig(), "llamacpp_base_url")

    def test_is_openrouter_host(self):
        assert is_openrouter_host("https://openrouter.ai/api/v1")
        assert not is_openrouter_host("http://ollama:11434/v1")
        assert not is_openrouter_host("https://api.openai.com/v1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::TestProviderConfig -v`
Expected: FAIL, `ImportError: cannot import name 'is_openrouter_host'`.

- [ ] **Step 3: Implement**

In `src/cems/config.py`, replace the block from the `# Model names use OpenRouter format` comment through the end of the llama.cpp settings (currently `embedding_model`, `llm_model`, `embedding_backend`, `embedding_dimension`, all `llamacpp_*` fields) with:

```python
    # =========================================================================
    # Model Provider Settings
    # =========================================================================
    # Any OpenAI-compatible endpoint works: OpenRouter (default), Ollama,
    # vLLM, LiteLLM, Azure OpenAI. Embeddings default to the LLM endpoint.
    llm_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible base URL for chat completions",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for the LLM endpoint. Falls back to OPENROUTER_API_KEY",
    )
    llm_model: str = Field(
        default="qwen/qwen3-32b",
        description="Model for maintenance ops (provider/model on OpenRouter, plain name elsewhere)",
    )
    embedding_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL for embeddings. Defaults to llm_base_url",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for the embeddings endpoint. Defaults to the LLM key",
    )
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Vector dimension. Must match the embedding model and the database column",
    )
    enable_agentic_search: bool = Field(
        default=True,
        description="Allow mode=agentic search (needs a long-context model)",
    )

    def resolved_llm_api_key(self) -> str | None:
        import os

        return self.llm_api_key or os.getenv("OPENROUTER_API_KEY")

    def resolved_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    def resolved_embedding_api_key(self) -> str | None:
        return self.embedding_api_key or self.resolved_llm_api_key()
```

Add at module level, below the class:

```python
def is_openrouter_host(url: str) -> bool:
    """True when the endpoint is OpenRouter (enables OpenRouter-only extras)."""
    from urllib.parse import urlparse

    return urlparse(url).hostname == "openrouter.ai"
```

Also update the module docstring's "Environment Variables for Server" list to add `CEMS_LLM_BASE_URL`, `CEMS_LLM_API_KEY`, `CEMS_EMBEDDING_BASE_URL`, `CEMS_EMBEDDING_API_KEY`, `CEMS_EMBEDDING_DIMENSION`, `CEMS_ENABLE_AGENTIC_SEARCH`. Remove the `Literal` import if nothing else uses it (check with `grep -n Literal src/cems/config.py`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS. If other tests in the file referenced `embedding_backend` or `llamacpp_*`, delete those assertions.

- [ ] **Step 5: Commit**

```bash
git add src/cems/config.py tests/test_config.py
git commit -m "feat(config): generic OpenAI-compatible provider settings, drop llama.cpp fields"
```

---

### Task 2: LLM client uses configured endpoint

**Files:**
- Modify: `src/cems/llm/client.py:1-100` (constants, `__init__`), `src/cems/llm/client.py:150-160` (`fast_route` block)
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `CEMSConfig.llm_base_url`, `CEMSConfig.resolved_llm_api_key()`, `is_openrouter_host()` from Task 1.
- Produces: `OpenRouterClient(api_key=None, model=None, base_url=None, ...)`; attribute `client.base_url: str`; attribute `client.is_openrouter: bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm.py` inside `TestOpenRouterClient`:

```python
    @patch.dict(os.environ, {"CEMS_LLM_BASE_URL": "http://ollama:11434/v1", "CEMS_LLM_API_KEY": "ollama"})
    @patch("cems.llm.client.OpenAI")
    def test_custom_base_url_no_openrouter_extras(self, mock_openai_class):
        from cems.llm import OpenRouterClient

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"), finish_reason="stop")]
        )

        client = OpenRouterClient(model="gemma4:e4b")

        init_kwargs = mock_openai_class.call_args[1]
        assert init_kwargs["base_url"] == "http://ollama:11434/v1"
        assert init_kwargs["api_key"] == "ollama"
        assert "default_headers" not in init_kwargs
        assert client.is_openrouter is False

        client.complete("hi")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" not in call_kwargs
        assert call_kwargs["model"] == "gemma4:e4b"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("cems.llm.client.OpenAI")
    def test_openrouter_keeps_extras(self, mock_openai_class):
        from cems.llm import OpenRouterClient

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"), finish_reason="stop")]
        )

        client = OpenRouterClient()
        assert client.is_openrouter is True
        assert mock_openai_class.call_args[1]["default_headers"]["X-Title"] == "CEMS Memory Server"

        client.complete("hi")
        assert "extra_body" in mock_client.chat.completions.create.call_args[1]

    @patch.dict(os.environ, {"CEMS_LLM_BASE_URL": "http://ollama:11434/v1"}, clear=True)
    def test_custom_base_url_still_requires_key(self):
        from cems.llm import OpenRouterClient

        with pytest.raises(ValueError, match="API key required"):
            OpenRouterClient()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -k "custom_base_url or keeps_extras" -v`
Expected: FAIL on `assert init_kwargs["base_url"] == "http://ollama:11434/v1"` and on `is_openrouter` attribute.

- [ ] **Step 3: Implement**

In `src/cems/llm/client.py`:

Replace the imports and `__init__` body:

```python
import logging
import os

from openai import OpenAI

from cems.config import CEMSConfig, is_openrouter_host

logger = logging.getLogger(__name__)

# Kept for backwards-compatible imports; the live value comes from CEMSConfig.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

```python
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        base_url: str | None = None,
    ):
        cfg = CEMSConfig()
        self.base_url = base_url or cfg.llm_base_url
        self.is_openrouter = is_openrouter_host(self.base_url)

        self.api_key = api_key or cfg.resolved_llm_api_key()
        if not self.api_key:
            raise ValueError(
                "LLM API key required. Set CEMS_LLM_API_KEY (or OPENROUTER_API_KEY "
                "when using OpenRouter), or pass api_key."
            )

        self.model = self._resolve_model(model or os.getenv("CEMS_LLM_MODEL") or cfg.llm_model)
        self.site_url = site_url or os.getenv("CEMS_OPENROUTER_SITE_URL", "https://github.com/cems")
        self.site_name = site_name or os.getenv("CEMS_OPENROUTER_SITE_NAME", "CEMS Memory Server")

        client_kwargs: dict = {"base_url": self.base_url, "api_key": self.api_key}
        if self.is_openrouter:
            client_kwargs["default_headers"] = {
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            }
        self._client = OpenAI(**client_kwargs)
```

In `complete()`, change the routing block to:

```python
        if fast_route and self.is_openrouter:
            kwargs["extra_body"] = {
                "provider": {
                    "order": FAST_PROVIDERS,
                    "allow_fallbacks": True,
                }
            }
```

In `_resolve_model`, the `if model is None` branch is now unreachable when config is present but keep it. Note the existing `test_client_requires_api_key` matches `"OpenRouter API key required"`; change its `match=` to `"API key required"`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cems/llm/client.py tests/test_llm.py
git commit -m "feat(llm): read base URL and key from config, gate OpenRouter extras on host"
```

---

### Task 3: Embedding clients use configured endpoint

**Files:**
- Modify: `src/cems/embedding.py` (constants, `EmbeddingClient.__init__`, `_call_api`, `AsyncEmbeddingClient.__init__`, `_get_client`, its `_call_api`)
- Create: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `CEMSConfig.resolved_embedding_base_url()`, `resolved_embedding_api_key()`, `is_openrouter_host()`.
- Produces: `EmbeddingClient(api_key=None, model=None, dimensions=None, base_url=None)`; same for `AsyncEmbeddingClient`; attribute `embeddings_url: str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embedding.py`:

```python
"""Tests for embedding clients against configurable endpoints."""

import os
from unittest.mock import MagicMock, patch

import pytest


def _fake_response(n: int, dim: int = 3):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"index": i, "embedding": [0.1] * dim} for i in range(n)]}
    return resp


class TestEmbeddingClient:
    @patch.dict(os.environ, {"CEMS_LLM_BASE_URL": "http://ollama:11434/v1", "CEMS_LLM_API_KEY": "ollama"})
    @patch("cems.embedding.httpx.Client")
    def test_custom_endpoint_omits_dimensions_and_attribution(self, mock_client_class):
        from cems.embedding import EmbeddingClient

        http = MagicMock()
        http.post.return_value = _fake_response(1)
        mock_client_class.return_value = http

        client = EmbeddingClient(model="embeddinggemma", dimensions=768)
        client.embed("hello")

        assert client.embeddings_url == "http://ollama:11434/v1/embeddings"
        headers = mock_client_class.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer ollama"
        assert "HTTP-Referer" not in headers
        url, kwargs = http.post.call_args[0][0], http.post.call_args[1]
        assert url == "http://ollama:11434/v1/embeddings"
        assert "dimensions" not in kwargs["json"]
        assert kwargs["json"]["model"] == "embeddinggemma"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or"})
    @patch("cems.embedding.httpx.Client")
    def test_openrouter_sends_dimensions(self, mock_client_class):
        from cems.embedding import EmbeddingClient

        http = MagicMock()
        http.post.return_value = _fake_response(1)
        mock_client_class.return_value = http

        client = EmbeddingClient(dimensions=1536)
        client.embed("hello")

        assert client.embeddings_url == "https://openrouter.ai/api/v1/embeddings"
        assert http.post.call_args[1]["json"]["dimensions"] == 1536
        assert mock_client_class.call_args[1]["headers"]["HTTP-Referer"] == "https://github.com/cems"

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_key(self):
        from cems.embedding import EmbeddingClient

        with pytest.raises(ValueError, match="API key required"):
            EmbeddingClient()


class TestAsyncEmbeddingClient:
    @patch.dict(os.environ, {"CEMS_LLM_BASE_URL": "http://ollama:11434/v1", "CEMS_LLM_API_KEY": "ollama"})
    @pytest.mark.asyncio
    async def test_custom_endpoint_url(self):
        from cems.embedding import AsyncEmbeddingClient

        client = AsyncEmbeddingClient(model="embeddinggemma")
        assert client.embeddings_url == "http://ollama:11434/v1/embeddings"
        assert client.is_openrouter is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: FAIL on `embeddings_url` attribute.

- [ ] **Step 3: Implement**

In `src/cems/embedding.py`:

Replace the module docstring's env var list with `CEMS_EMBEDDING_BASE_URL`, `CEMS_EMBEDDING_API_KEY`, `CEMS_EMBEDDING_MODEL`, `CEMS_EMBEDDING_DIMENSION`, and the fallback `OPENROUTER_API_KEY`. Remove the `CEMS_EMBEDDING_BACKEND` line.

Add import: `from cems.config import CEMSConfig, is_openrouter_host`.

Keep `OPENROUTER_EMBEDDINGS_URL` constant for imports but stop using it.

Add a shared helper above the classes:

```python
def _resolve_endpoint(api_key: str | None, base_url: str | None) -> tuple[str, str, bool]:
    """Return (embeddings_url, api_key, is_openrouter) from args and config."""
    cfg = CEMSConfig()
    resolved_base = (base_url or cfg.resolved_embedding_base_url()).rstrip("/")
    key = api_key or cfg.resolved_embedding_api_key()
    if not key:
        raise ValueError(
            "Embedding API key required. Set CEMS_EMBEDDING_API_KEY, CEMS_LLM_API_KEY "
            "or OPENROUTER_API_KEY, or pass api_key."
        )
    return f"{resolved_base}/embeddings", key, is_openrouter_host(resolved_base)


def _headers(api_key: str, is_openrouter: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://github.com/cems"
        headers["X-Title"] = "CEMS Memory Server"
    return headers
```

`EmbeddingClient.__init__` becomes:

```python
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ):
        self.embeddings_url, self.api_key, self.is_openrouter = _resolve_endpoint(api_key, base_url)
        self.model = model or os.getenv("CEMS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.dimensions = dimensions
        self._client = httpx.Client(timeout=30.0, headers=_headers(self.api_key, self.is_openrouter))
```

In `EmbeddingClient._call_api`, change:

```python
        if self.dimensions and self.is_openrouter:
            payload["dimensions"] = self.dimensions
```

and post to `self.embeddings_url` instead of `OPENROUTER_EMBEDDINGS_URL`.

Apply the same three changes to `AsyncEmbeddingClient`: `__init__` calls `_resolve_endpoint`, `_get_client` builds `httpx.AsyncClient(timeout=30.0, headers=_headers(self.api_key, self.is_openrouter))`, and its `_call_api` posts to `self.embeddings_url` with the conditional `dimensions`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_embedding.py tests/test_llm.py -v`
Expected: PASS. If `pytest.mark.asyncio` is unknown, check `pyproject.toml` for `asyncio_mode = "auto"` and drop the marker.

- [ ] **Step 5: Commit**

```bash
git add src/cems/embedding.py tests/test_embedding.py
git commit -m "feat(embedding): configurable endpoint, OpenRouter-only extras gated on host"
```

---

### Task 4: Remove the llama.cpp backend

**Files:**
- Delete: `src/cems/llamacpp_server.py`
- Modify: `src/cems/memory/core.py:25-35, 70-130`
- Modify: `deploy/docker-compose.yml:42-43` (remove `CEMS_EMBEDDING_BACKEND`, keep `CEMS_EMBEDDING_DIMENSION`)
- Test: existing suites

**Interfaces:**
- Consumes: `EmbeddingClient`, `AsyncEmbeddingClient` from Task 3.
- Produces: `CEMSMemory._ensure_initialized()` and `_ensure_initialized_async()` with one embedder path.

- [ ] **Step 1: Find every reference**

Run: `grep -rn "llamacpp\|embedding_backend\|EMBEDDING_BACKEND" src tests docs deploy README.md --include="*" | grep -v __pycache__`
Expected: hits in `core.py`, `embedding.py` docstring (already fixed), `docker-compose.yml`, `docs/DEPLOYMENT.md` (handled in Task 9), maybe tests.

- [ ] **Step 2: Delete the module and simplify core**

```bash
git rm src/cems/llamacpp_server.py
```

In `src/cems/memory/core.py` remove the `AsyncLlamaCppEmbeddingClient` import under `TYPE_CHECKING` and the runtime import. Replace the embedder section of `_ensure_initialized` with:

```python
        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.config.embedding_model)
            logger.info(
                f"[MEMORY] Embeddings via {self._embedder.embeddings_url} "
                f"({self.config.embedding_dimension}-dim)"
            )
```

Replace the embedder section of `_ensure_initialized_async` with:

```python
        from cems.embedding import AsyncEmbeddingClient, EmbeddingClient

        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.config.embedding_model)
        if self._async_embedder is None:
            self._async_embedder = AsyncEmbeddingClient(model=self.config.embedding_model)
        logger.info(
            f"[MEMORY] Embeddings via {self._async_embedder.embeddings_url} "
            f"({self.config.embedding_dimension}-dim)"
        )
```

Remove the docstring note about llamacpp requiring async. In `deploy/docker-compose.yml` delete the `CEMS_EMBEDDING_BACKEND: openrouter` line.

- [ ] **Step 3: Run the full suite and lint**

Run: `uv run ruff check src tests && uv run pytest -q`
Expected: PASS. Fix any test that patched `cems.llamacpp_server` by deleting it.

- [ ] **Step 4: Commit**

```bash
git add -A src/cems/memory/core.py src/cems/llamacpp_server.py deploy/docker-compose.yml tests
git commit -m "refactor: remove llama.cpp embedding backend, one embedder path"
```

---

### Task 5: Vector column follows configured dimension, mismatch aborts

**Files:**
- Modify: `src/cems/db/database.py:166-230` (`run_migrations`, `core_memory_tables_v1`)
- Create: `tests/test_database_dimension.py`

**Interfaces:**
- Consumes: `CEMSConfig.embedding_dimension`.
- Produces: `check_embedding_dimension(actual: int | None, expected: int) -> None` (raises `RuntimeError`), `get_embedding_column_dimension(db) -> int | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_database_dimension.py`:

```python
import pytest

from cems.db.database import check_embedding_dimension


def test_matching_dimension_passes():
    check_embedding_dimension(actual=1536, expected=1536)


def test_missing_table_passes():
    check_embedding_dimension(actual=None, expected=768)


def test_mismatch_raises_with_guidance():
    with pytest.raises(RuntimeError) as exc:
        check_embedding_dimension(actual=1536, expected=768)
    msg = str(exc.value)
    assert "database has 1536, config has 768" in msg
    assert "docs/DEPLOYMENT.md#private-mode" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_database_dimension.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

In `src/cems/db/database.py` add near the top (after imports):

```python
def check_embedding_dimension(actual: int | None, expected: int) -> None:
    """Abort startup when the pgvector column does not match config.

    actual is None when memory_chunks does not exist yet (fresh database).
    """
    if actual is None or actual == expected:
        return
    raise RuntimeError(
        f"Embedding dimension mismatch: database has {actual}, config has {expected}. "
        "Private mode needs a fresh database. See docs/DEPLOYMENT.md#private-mode."
    )


def get_embedding_column_dimension(db) -> int | None:
    """Read vector(N) from memory_chunks.embedding, or None if the table is missing."""
    from sqlalchemy import text

    with db.engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'memory_chunks' AND a.attname = 'embedding'
                """
            )
        ).fetchone()
    return int(row[0]) if row and row[0] is not None and row[0] > 0 else None
```

Check how `db` exposes its engine: `grep -n "engine" src/cems/db/database.py | head`. Use that attribute name.

In `run_migrations()`, before building `migrations`:

```python
    from cems.config import CEMSConfig

    dim = CEMSConfig().embedding_dimension
    check_embedding_dimension(get_embedding_column_dimension(db), dim)
```

Change the `core_memory_tables_v1` SQL string to an f-string and replace `embedding vector(1536) NOT NULL` with `embedding vector({dim}) NOT NULL`. Grep the same file and `scripts/*.sql` for other `1536` literals; any inside `run_migrations` also use `{dim}`. Doubled braces are not needed since the SQL has no `{}` otherwise; confirm with `grep -n "{" src/cems/db/database.py` around the string.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_database_dimension.py tests/test_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cems/db/database.py tests/test_database_dimension.py
git commit -m "feat(db): create vector column at configured dimension, abort on mismatch"
```

---

### Task 6: Agentic search flag and health check endpoint

**Files:**
- Modify: `src/cems/api/handlers/memory.py:436-441`
- Modify: `src/cems/admin/routes.py:340-375`
- Test: `tests/test_server.py` (add one case), `tests/test_admin.py` (adjust key names)

**Interfaces:**
- Consumes: `CEMSConfig.enable_agentic_search`, `OpenRouterClient` from Task 2, `EmbeddingClient` from Task 3.
- Produces: search API returns HTTP 400 `{"error": "agentic search disabled on this server"}` when the flag is off; admin health JSON keys `llm` and `embeddings`.

- [ ] **Step 1: Write the failing test**

Find the existing search test in `tests/test_server.py` that posts with `mode` (`grep -n "mode" tests/test_server.py | head`). Add next to it, copying its client fixture and auth header pattern:

```python
def test_agentic_search_disabled_returns_400(client, auth_headers, monkeypatch):
    monkeypatch.setenv("CEMS_ENABLE_AGENTIC_SEARCH", "false")
    resp = client.post(
        "/api/memory/search",
        json={"query": "anything", "mode": "agentic"},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "agentic search disabled" in resp.json()["error"]
```

Adjust the route path and fixture names to match the file's existing search test.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_server.py -k agentic_search_disabled -v`
Expected: FAIL (200 or 500 instead of 400).

- [ ] **Step 3: Implement**

In `src/cems/api/handlers/memory.py`, at the `if mode == "agentic":` branch:

```python
        if mode == "agentic":
            if not memory.config.enable_agentic_search:
                return JSONResponse(
                    {"error": "agentic search disabled on this server (CEMS_ENABLE_AGENTIC_SEARCH=false)"},
                    status_code=400,
                )
            from cems.agentic.search import agentic_search_async
```

In `src/cems/admin/routes.py` health check, replace the OpenRouter block:

```python
    from cems.config import CEMSConfig
    from cems.embedding import EmbeddingClient
    from cems.llm.client import OpenRouterClient

    cfg = CEMSConfig()
    results = {}

    try:
        llm = OpenRouterClient()
        text_out = llm.complete("Reply with the single word OK", max_tokens=5, fast_route=False)
        results["llm"] = {"ok": True, "endpoint": llm.base_url, "model": llm.model, "response": text_out}
    except Exception as e:
        results["llm"] = {"ok": False, "endpoint": cfg.llm_base_url, "error": str(e)}

    try:
        emb = EmbeddingClient()
        vec = emb.embed("health check")
        results["embeddings"] = {
            "ok": len(vec) == cfg.embedding_dimension,
            "endpoint": emb.embeddings_url,
            "model": emb.model,
            "dimension": len(vec),
            "expected_dimension": cfg.embedding_dimension,
        }
    except Exception as e:
        results["embeddings"] = {"ok": False, "endpoint": cfg.resolved_embedding_base_url(), "error": str(e)}
```

Keep whatever the function returns after `results` is built. Update any assertion in `tests/test_admin.py` that looks for `openrouter_llm` to `llm`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_server.py tests/test_admin.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cems/api/handlers/memory.py src/cems/admin/routes.py tests/test_server.py tests/test_admin.py
git commit -m "feat: agentic search flag and provider-agnostic health check"
```

---

### Task 7: Compose profile and presets

**Files:**
- Modify: `deploy/docker-compose.yml`
- Create: `deploy/.env.private-cpu.example`, `deploy/.env.private-gpu.example`
- Modify: `deploy/.env.example`

**Interfaces:**
- Consumes: env vars from Tasks 1 and 6.
- Produces: `docker compose --profile private up -d` starts `postgres`, `ollama`, `ollama-pull`, `cems-server`.

- [ ] **Step 1: Edit the compose file**

In `deploy/docker-compose.yml`, replace the `cems-server` environment block's model lines with passthrough:

```yaml
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}
      CEMS_LLM_BASE_URL: ${CEMS_LLM_BASE_URL:-https://openrouter.ai/api/v1}
      CEMS_LLM_API_KEY: ${CEMS_LLM_API_KEY:-}
      CEMS_LLM_MODEL: ${CEMS_LLM_MODEL:-qwen/qwen3-32b}
      CEMS_AGENTIC_MODEL: ${CEMS_AGENTIC_MODEL:-google/gemini-2.5-flash-lite}
      CEMS_EMBEDDING_BASE_URL: ${CEMS_EMBEDDING_BASE_URL:-}
      CEMS_EMBEDDING_API_KEY: ${CEMS_EMBEDDING_API_KEY:-}
      CEMS_EMBEDDING_MODEL: ${CEMS_EMBEDDING_MODEL:-openai/text-embedding-3-small}
      CEMS_EMBEDDING_DIMENSION: ${CEMS_EMBEDDING_DIMENSION:-1536}
      CEMS_ENABLE_QUERY_SYNTHESIS: ${CEMS_ENABLE_QUERY_SYNTHESIS:-false}
      CEMS_ENABLE_PREFERENCE_SYNTHESIS: ${CEMS_ENABLE_PREFERENCE_SYNTHESIS:-true}
      CEMS_ENABLE_QUERY_DECOMPOSITION: ${CEMS_ENABLE_QUERY_DECOMPOSITION:-true}
      CEMS_ENABLE_AGENTIC_SEARCH: ${CEMS_ENABLE_AGENTIC_SEARCH:-true}
```

Empty-string env values: pydantic-settings treats `CEMS_EMBEDDING_BASE_URL=""` as empty string, not `None`. In `CEMSConfig.resolved_embedding_base_url()` and `resolved_embedding_api_key()` the `or` chain already treats `""` as unset, and `resolved_llm_api_key()` does too. Confirm with a one-line test added to `tests/test_config.py`:

```python
    @patch.dict(os.environ, {"CEMS_EMBEDDING_BASE_URL": "", "CEMS_LLM_API_KEY": "", "OPENROUTER_API_KEY": "k"})
    def test_empty_env_values_mean_unset(self):
        cfg = CEMSConfig()
        assert cfg.resolved_embedding_base_url() == "https://openrouter.ai/api/v1"
        assert cfg.resolved_llm_api_key() == "k"
```

Add the profile services after `cems-server`:

```yaml
  ollama:
    image: ollama/ollama:latest
    container_name: cems-ollama
    profiles: ["private"]
    volumes:
      - ollama_data:/root/.ollama
    expose:
      - "11434"
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 15s
      timeout: 10s
      retries: 10
    restart: unless-stopped

  ollama-pull:
    image: ollama/ollama:latest
    container_name: cems-ollama-pull
    profiles: ["private"]
    environment:
      OLLAMA_HOST: http://ollama:11434
    entrypoint: ["/bin/sh", "-c"]
    command:
      - >
        ollama pull ${CEMS_LLM_MODEL:-gemma4:e4b} &&
        ollama pull ${CEMS_EMBEDDING_MODEL:-embeddinggemma} &&
        [ -z "${CEMS_AGENTIC_MODEL_PULL:-}" ] || ollama pull ${CEMS_AGENTIC_MODEL_PULL}
    depends_on:
      ollama:
        condition: service_healthy
    restart: "no"
```

Add `ollama_data:` under `volumes:`.

Change `cems-server.depends_on` to:

```yaml
    depends_on:
      postgres:
        condition: service_healthy
      ollama-pull:
        condition: service_completed_successfully
        required: false
```

`required: false` needs Compose v2.20+. It makes the dependency a no-op when the profile is inactive.

Add a GPU override file `deploy/docker-compose.gpu.yml`:

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

- [ ] **Step 2: Write the presets**

`deploy/.env.private-cpu.example`:

```
# CEMS private mode, CPU preset. Everything runs on this box.
# Needs about 8 GB RAM free for the two models.
POSTGRES_PASSWORD=change_me
CEMS_ADMIN_KEY=change_me

CEMS_LLM_BASE_URL=http://ollama:11434/v1
CEMS_LLM_API_KEY=ollama
CEMS_LLM_MODEL=gemma4:e4b
CEMS_EMBEDDING_MODEL=embeddinggemma
CEMS_EMBEDDING_DIMENSION=768

# Recall-time LLM features off: too slow on CPU. Background jobs stay on.
CEMS_ENABLE_QUERY_SYNTHESIS=false
CEMS_ENABLE_PREFERENCE_SYNTHESIS=false
CEMS_ENABLE_QUERY_DECOMPOSITION=false
CEMS_ENABLE_AGENTIC_SEARCH=false
```

`deploy/.env.private-gpu.example`:

```
# CEMS private mode, GPU preset. One NVIDIA card with 20 GB or more.
# Start with: docker compose --profile private -f docker-compose.yml -f docker-compose.gpu.yml up -d
POSTGRES_PASSWORD=change_me
CEMS_ADMIN_KEY=change_me

CEMS_LLM_BASE_URL=http://ollama:11434/v1
CEMS_LLM_API_KEY=ollama
CEMS_LLM_MODEL=__GPU_CHAT_MODEL__
CEMS_AGENTIC_MODEL=__GPU_CHAT_MODEL__
CEMS_AGENTIC_MODEL_PULL=__GPU_CHAT_MODEL__
CEMS_EMBEDDING_MODEL=embeddinggemma
CEMS_EMBEDDING_DIMENSION=768

# All recall-time features on.
CEMS_ENABLE_QUERY_SYNTHESIS=true
CEMS_ENABLE_PREFERENCE_SYNTHESIS=true
CEMS_ENABLE_QUERY_DECOMPOSITION=true
CEMS_ENABLE_AGENTIC_SEARCH=true
```

Replace `__GPU_CHAT_MODEL__` in the same step using this rule: run `ollama search` or browse `https://ollama.com/library`, pick the newest instruction-tuned open model whose default tag is 32B parameters or fewer, supports tool calling, and lists a context window of at least 128k. Run `ollama show <model>` to confirm the context length. Write the chosen name into the file and into the docs table in Task 9. Do not leave the placeholder.

Update `deploy/.env.example` to document the new optional variables with the OpenRouter defaults, one comment line each.

- [ ] **Step 3: Validate the compose file**

Run: `cd deploy && docker compose config -q && docker compose --profile private config -q && docker compose --profile private -f docker-compose.yml -f docker-compose.gpu.yml config -q`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add deploy/docker-compose.yml deploy/docker-compose.gpu.yml deploy/.env.example deploy/.env.private-cpu.example deploy/.env.private-gpu.example tests/test_config.py
git commit -m "feat(deploy): private compose profile with Ollama, CPU and GPU presets"
```

---

### Task 8: Server installer and cloud-init

**Files:**
- Create: `install-server.sh`
- Create: `deploy/cloud-init/aws.yaml`, `deploy/cloud-init/hetzner.yaml`, `deploy/cloud-init/digitalocean.yaml`
- Create: `tests/test_install_server.py`

**Interfaces:**
- Consumes: compose file and presets from Task 7.
- Produces: `install-server.sh [--private|--private-gpu] [--openrouter-key KEY] [--yes] [--dir /opt/cems]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_install_server.py`:

```python
"""Smoke tests for install-server.sh: no Docker needed, uses --dry-run."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install-server.sh"


def run(*args):
    return subprocess.run(["bash", str(SCRIPT), "--dry-run", *args], capture_output=True, text=True)


def test_script_exists_and_is_bash():
    assert SCRIPT.exists()
    assert SCRIPT.read_text().startswith("#!/usr/bin/env bash")


def test_default_mode_needs_openrouter_key():
    r = run("--yes")
    assert r.returncode != 0
    assert "OPENROUTER_API_KEY" in r.stderr + r.stdout


def test_private_cpu_selects_preset_and_profile():
    r = run("--private", "--yes")
    assert r.returncode == 0, r.stderr
    assert ".env.private-cpu.example" in r.stdout
    assert "--profile private" in r.stdout


def test_private_gpu_adds_override_file():
    r = run("--private-gpu", "--yes")
    assert r.returncode == 0, r.stderr
    assert ".env.private-gpu.example" in r.stdout
    assert "docker-compose.gpu.yml" in r.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_install_server.py -v`
Expected: FAIL, script missing.

- [ ] **Step 3: Write the installer**

Create `install-server.sh`:

```bash
#!/usr/bin/env bash
# CEMS server installer.
#
#   curl -fsSL https://getcems.com/install-server.sh | bash -s -- --private --yes
#
# Flags:
#   --private          Private mode, CPU preset (Ollama on this box)
#   --private-gpu      Private mode, GPU preset (needs NVIDIA container toolkit)
#   --openrouter-key K Default mode key (or set OPENROUTER_API_KEY)
#   --dir PATH         Install directory (default /opt/cems)
#   --yes              No prompts
#   --dry-run          Print what would run, change nothing
set -euo pipefail

RAW="https://raw.githubusercontent.com/chocksy/cems/main"
DIR="/opt/cems"
MODE="default"
YES=0
DRY=0
KEY="${OPENROUTER_API_KEY:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --private) MODE="cpu" ;;
    --private-gpu) MODE="gpu" ;;
    --openrouter-key) KEY="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --yes) YES=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }

if [ "$MODE" = "default" ] && [ -z "$KEY" ]; then
  echo "Default mode needs an OpenRouter key: pass --openrouter-key or set OPENROUTER_API_KEY. Or use --private." >&2
  exit 1
fi

case "$MODE" in
  cpu) PRESET=".env.private-cpu.example"; PROFILE="--profile private"; FILES="-f docker-compose.yml" ;;
  gpu) PRESET=".env.private-gpu.example"; PROFILE="--profile private"; FILES="-f docker-compose.yml -f docker-compose.gpu.yml" ;;
  *)   PRESET=".env.example";             PROFILE="";                  FILES="-f docker-compose.yml" ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  say "Installing Docker"
  run sh -c "curl -fsSL https://get.docker.com | sh"
fi
if ! docker compose version >/dev/null 2>&1 && [ "$DRY" = 0 ]; then
  echo "docker compose plugin missing. Install docker-compose-plugin and re-run." >&2
  exit 1
fi

say "Preparing $DIR"
run mkdir -p "$DIR/deploy"
run sh -c "curl -fsSL $RAW/deploy/docker-compose.yml -o $DIR/deploy/docker-compose.yml"
[ "$MODE" = gpu ] && run sh -c "curl -fsSL $RAW/deploy/docker-compose.gpu.yml -o $DIR/deploy/docker-compose.gpu.yml"

if [ -f "$DIR/deploy/.env" ]; then
  say "Keeping existing $DIR/deploy/.env"
else
  say "Writing .env from $PRESET"
  run sh -c "curl -fsSL $RAW/deploy/$PRESET -o $DIR/deploy/.env"
  PG=$(openssl rand -hex 16); ADMIN="cems_admin_$(openssl rand -hex 16)"
  run sed -i.bak "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$PG/; s/^CEMS_ADMIN_KEY=.*/CEMS_ADMIN_KEY=$ADMIN/" "$DIR/deploy/.env"
  if [ "$MODE" = default ]; then
    run sed -i.bak "s|^OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=$KEY|" "$DIR/deploy/.env"
  fi
  run rm -f "$DIR/deploy/.env.bak"
fi

say "Starting CEMS"
run sh -c "cd $DIR/deploy && docker compose $PROFILE $FILES up -d"

if [ "$DRY" = 0 ]; then
  say "Waiting for /health"
  for _ in $(seq 1 60); do
    curl -fsS http://localhost:8765/health >/dev/null 2>&1 && break
    sleep 5
  done
  curl -fsS http://localhost:8765/health >/dev/null || { echo "Server did not become healthy. Check: docker compose -f $DIR/deploy/docker-compose.yml logs" >&2; exit 1; }
  ADMIN_KEY=$(grep '^CEMS_ADMIN_KEY=' "$DIR/deploy/.env" | cut -d= -f2)
  say "CEMS is up on port 8765"
  echo "Admin key (shown once, stored in $DIR/deploy/.env): $ADMIN_KEY"
  echo "Next: cems admin --admin-key \$ADMIN_KEY users create <name>"
fi
```

Note the private-mode `.env` presets carry no `OPENROUTER_API_KEY` line; the compose default `${OPENROUTER_API_KEY:-}` handles it. Model pull on first boot can take several minutes; the 60 by 5 second health loop covers about five minutes. If a CPU box takes longer in Task 10, raise the loop count and record it.

Make it executable: `chmod +x install-server.sh`.

- [ ] **Step 4: Write the cloud-init files**

`deploy/cloud-init/hetzner.yaml`:

```yaml
#cloud-config
# Hetzner Cloud: paste into "Cloud config" when creating a server (Ubuntu 24.04).
# Or: hcloud server create --user-data-from-file hetzner.yaml ...
package_update: true
packages: [curl, openssl]
runcmd:
  - curl -fsSL https://getcems.com/install-server.sh -o /root/install-server.sh
  - bash /root/install-server.sh --private --yes
```

`deploy/cloud-init/digitalocean.yaml`: same content, comment reads `# DigitalOcean: paste into "User data" under Advanced Options (Ubuntu 24.04). Or: doctl compute droplet create --user-data-file digitalocean.yaml ...`.

`deploy/cloud-init/aws.yaml`: same content, comment reads `# AWS EC2: paste into "User data" under Advanced details (Ubuntu 24.04 AMI). Or: aws ec2 run-instances --user-data file://aws.yaml ...`. Add `- ufw allow 8765/tcp` as a first `runcmd` line only in the Hetzner and DigitalOcean files; AWS uses security groups.

- [ ] **Step 5: Run tests and shellcheck**

Run: `uv run pytest tests/test_install_server.py -v && (command -v shellcheck >/dev/null && shellcheck install-server.sh || echo "shellcheck not installed, skipped")`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add install-server.sh deploy/cloud-init tests/test_install_server.py
git commit -m "feat(deploy): install-server.sh with private mode and cloud-init snippets"
```

---

### Task 9: Documentation

**Files:**
- Modify: `docs/DEPLOYMENT.md` (env table at "Environment Variables", new "Private mode" section before "Kubernetes")
- Modify: `docs/CLIENT.md` (new section "Running your agent privately")
- Modify: `README.md` (Quick Start note and doc table row)

**Interfaces:**
- Consumes: everything shipped in Tasks 1 to 8.

- [ ] **Step 1: Update the env var table in `docs/DEPLOYMENT.md`**

Replace the `CEMS_EMBEDDING_BACKEND` row and add rows:

```markdown
| `CEMS_LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat endpoint (Ollama, vLLM, LiteLLM, Azure OpenAI) |
| `CEMS_LLM_API_KEY` | `OPENROUTER_API_KEY` | Key for the LLM endpoint. Ollama ignores it but needs a value |
| `CEMS_EMBEDDING_BASE_URL` | same as LLM | Separate embeddings endpoint if needed |
| `CEMS_EMBEDDING_API_KEY` | same as LLM | Key for the embeddings endpoint |
| `CEMS_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `CEMS_EMBEDDING_DIMENSION` | `1536` | Must match the model. Fixed at first boot; see Private mode |
| `CEMS_ENABLE_AGENTIC_SEARCH` | `true` | Allow `mode=agentic` search. Needs a long-context model |
```

Move `OPENROUTER_API_KEY` from "Required" to "Optional" with description "Required only when using OpenRouter (the default)".

- [ ] **Step 2: Add the "Private mode" section**

Insert before `## Kubernetes`:

````markdown
## Private mode

Private mode runs the whole memory pipeline on models you control. Memories always live in your PostgreSQL. Private mode moves the model calls too.

### What leaves your network

| | Default (OpenRouter) | Private cloud (Bedrock, Azure, Vertex via a gateway) | Private mode (Ollama) |
|---|---|---|---|
| Memories, embeddings at rest | Your Postgres | Your Postgres | Your Postgres |
| Extraction, consolidation, lint | OpenRouter, then the model vendor | Your cloud account | This box |
| Embedding calls | OpenRouter (OpenAI model) | Your cloud account | This box |
| Recall-time query synthesis, agentic search | OpenRouter | Your cloud account | This box (GPU preset) or off (CPU preset) |
| Your coding agent (Claude Code, Cursor, Codex) | Its vendor | Its vendor, or your cloud if the agent supports it | Its vendor, or a local model if the agent supports it |

CEMS does not change what your coding agent sends to its vendor. See [Running your agent privately](CLIENT.md#running-your-agent-privately).

### Three commands

```bash
curl -fsSL https://getcems.com/install-server.sh -o install-server.sh
bash install-server.sh --private --yes        # or --private-gpu
cems admin --admin-key <printed key> users create alice
```

The first boot pulls `gemma4:e4b` and `embeddinggemma` (about 6 GB). The health wait covers that.

### Cloud-init (boot a ready server)

Paste one of these into the user-data field when creating the VM. The box installs Docker, starts CEMS in private mode, and is ready on port 8765.

- [AWS EC2](../deploy/cloud-init/aws.yaml)
- [Hetzner Cloud](../deploy/cloud-init/hetzner.yaml)
- [DigitalOcean](../deploy/cloud-init/digitalocean.yaml)

Read the admin key afterwards from `/opt/cems/deploy/.env`.

### Hardware and cost

| Tier | Example box | What runs | Approx. monthly |
|---|---|---|---|
| CPU | Hetzner CPX31 (4 vCPU, 8 GB) or CX32 (16 GB) | Extraction, consolidation, embeddings. Plain hybrid recall | fill from provider pricing page |
| GPU, single card | Hetzner GEX44 (RTX 4000 SFF, 20 GB) or AWS g6.xlarge | Everything on, `<GPU model from preset>`, 128k context | fill from provider pricing page |
| GPU, large | AWS p4d or Hetzner GEX131 | Everything on, 1M-context model | fill from provider pricing page |

Fill the price column from the providers' current pricing pages on the day this ships, and put the date next to the table.

### Limits

- The embedding dimension is fixed when the database is first created. Switching from OpenRouter (1536) to Ollama (768) needs a fresh database; the server refuses to start otherwise with `Embedding dimension mismatch`.
- Ollama downloads models from the internet on first boot. After that the box needs no outbound access for CEMS to work.
- CPU preset turns off query synthesis, query decomposition and agentic search. Set the `CEMS_ENABLE_*` variables to `true` to turn them back on if the box can take it.

### Bring your own endpoint

Any OpenAI-compatible server works without the Ollama profile. Set `CEMS_LLM_BASE_URL`, `CEMS_LLM_API_KEY`, `CEMS_LLM_MODEL`, `CEMS_EMBEDDING_MODEL` and `CEMS_EMBEDDING_DIMENSION` in `.env` and run `docker compose up -d`. For Bedrock or Azure OpenAI put a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) in front and point CEMS at it.
````

Delete any remaining llama.cpp mentions in the file (`grep -n llama docs/DEPLOYMENT.md`).

- [ ] **Step 3: Add the agent section to `docs/CLIENT.md`**

Append:

```markdown
## Running your agent privately

CEMS keeps memories on your server. Whether your coding agent sends code to its vendor depends on the agent. Verified from each vendor's docs on 2026-09-22:

| Agent | CEMS integration | Open-weight / local model | Bedrock, Azure, Vertex |
|---|---|---|---|
| Claude Code | Tested (hooks + MCP) | No | Yes |
| Cursor | Tested (MCP) | No | Bedrock and Azure keys only |
| Codex CLI | Tested (MCP) | Yes (`--oss`, Ollama) | Bedrock, Azure |
| Goose | Tested (MCP) | Yes | Yes |
| OpenCode | Tested (MCP) | Yes | Yes |
| Aider, Cline, Continue, Roo Code, Kilo Code | MCP, untested | Yes | Yes |

Links: [Claude Code](https://code.claude.com/docs/en/third-party-integrations), [Codex](https://learn.chatgpt.com/docs/config-file/config-advanced), [Cursor](https://cursor.com/help/models-and-usage/api-keys), [Goose](https://goose-docs.ai/docs/getting-started/providers/), [OpenCode](https://opencode.ai/docs/providers/), [Aider](https://aider.chat/docs/llms.html), [Cline](https://docs.cline.bot/provider-config/openai-compatible), [Continue](https://docs.continue.dev/customize/model-providers/overview), [Roo Code](https://docs.roocode.com/providers), [Kilo Code](https://kilo.ai/docs/ai-providers).
```

OpenCode moves to "Tested" only after Task 10 step 4 passes. Until then write "MCP, untested".

- [ ] **Step 4: README**

In `README.md` Quick Start, after the deploy block add:

```markdown
> Want nothing to leave your network? `bash install-server.sh --private --yes` runs the whole pipeline on local models. See [Private mode](docs/DEPLOYMENT.md#private-mode).
```

Replace the OpenRouter-only `.env` example comment with `OPENROUTER_API_KEY=sk-or-your-key   # or use --private, see docs`.

- [ ] **Step 5: Commit**

```bash
git add docs/DEPLOYMENT.md docs/CLIENT.md README.md
git commit -m "docs: private mode, provider variables, agent privacy table"
```

---

### Task 10: Live verification

**Files:**
- Modify: `docs/DEPLOYMENT.md` (record results in the hardware table and Limits)

**Interfaces:**
- Consumes: everything above, pushed to `main` so the raw URLs resolve. The `getcems.com/install-server.sh` redirect (Part 3 plan) is not live yet; use the raw GitHub URL in the cloud-init `runcmd` for this test and switch to the short URL when the site ships.

- [ ] **Step 1: CPU box**

Create a Hetzner CX32 (Ubuntu 24.04) with `deploy/cloud-init/hetzner.yaml` as user data (raw URL substituted). Then:

```bash
ssh root@<ip> 'until curl -fsS localhost:8765/health; do sleep 10; done; grep CEMS_ADMIN_KEY /opt/cems/deploy/.env'
ssh root@<ip> 'cd /opt/cems/deploy && docker compose --profile private ps'
```

Expected: all four services up, `ollama-pull` exited 0. Time from boot to healthy goes into the docs.

- [ ] **Step 2: Round trip**

From your machine:

```bash
export CEMS_API_URL=http://<ip>:8765
cems admin --admin-key <key> users create alice   # note the API key
export CEMS_API_KEY=<alice key>
cems memory add "We deploy with Coolify on Hetzner" --category infra
cems memory search "how do we deploy"
curl -s -H "Authorization: Bearer <admin key>" $CEMS_API_URL/admin/health | python3 -m json.tool
```

Expected: the search returns the memory; health shows `llm.ok` and `embeddings.ok` true with `dimension: 768`.

- [ ] **Step 3: Egress check**

```bash
ssh root@<ip> 'iptables -I OUTPUT -o eth0 -p tcp --dport 443 -j REJECT; iptables -I OUTPUT -o eth0 -p tcp --dport 80 -j REJECT'
cems memory add "second memory with no internet"
cems memory search "second memory"
ssh root@<ip> 'iptables -D OUTPUT -o eth0 -p tcp --dport 443 -j REJECT; iptables -D OUTPUT -o eth0 -p tcp --dport 80 -j REJECT'
```

Expected: add and search work with outbound HTTP blocked.

- [ ] **Step 4: OpenCode**

On your machine, install OpenCode, configure an `ollama` provider per https://opencode.ai/docs/providers/ pointing at `http://<ip>:11434/v1` (open port 11434 temporarily in ufw for this test, close after), and add the CEMS MCP server using `examples/mcp_config.json` as the template. Ask it to "remember that we use Coolify" then in a new session "how do we deploy?". Expected: it calls the CEMS store and recall tools. Flip OpenCode to "Tested" in `docs/CLIENT.md`.

- [ ] **Step 5: GPU box**

Create a Hetzner GEX44 (or any box with an NVIDIA card and the container toolkit), run `bash install-server.sh --private-gpu --yes`, repeat step 2, then:

```bash
cems memory search "how do we deploy" --mode agentic
```

Expected: results returned through the GPU model. Record model name, `ollama show` context length, boot-to-healthy time and the provider's list price in the hardware table.

- [ ] **Step 6: Tear down and commit the numbers**

Delete both servers in the provider console. Fill the price column and dates in `docs/DEPLOYMENT.md`.

```bash
git add docs/DEPLOYMENT.md docs/CLIENT.md
git commit -m "docs: private mode verified on Hetzner CPU and GPU, prices and timings"
```

---

## Self-review

- Spec Part 1 config fields: Task 1. LLM client: Task 2. Embeddings: Task 3. Single core path and llama.cpp removal: Task 4. Schema and mismatch: Task 5. Agentic flag and health check: Task 6. Tests listed in spec: Tasks 1, 2, 3, 5, 6.
- Spec Part 2 compose and presets: Task 7. Installer and cloud-init: Task 8. Docs: Task 9. Verification on CPU, egress, OpenCode, GPU: Task 10.
- Spec Part 3 (site) is deliberately not in this plan. Also out of this plan: the `getcems.com/install-server.sh` redirect, which lives in the site repo's `public/_redirects`.
- Names used consistently: `llm_base_url`, `resolved_llm_api_key()`, `resolved_embedding_base_url()`, `resolved_embedding_api_key()`, `is_openrouter_host()`, `embeddings_url`, `is_openrouter`, `check_embedding_dimension()`, `get_embedding_column_dimension()`, `enable_agentic_search`.
- Remaining conditional: the GPU chat model name is chosen by rule in Task 7 step 2 and recorded in Task 10.
