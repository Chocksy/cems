# Plan: Remove Ollama (Container + Embeddings) and Align with QMD-Style Retrieval (Non-Ollama)

**Goal**: Remove Ollama container usage and Ollama embedding dependency, revert to OpenRouter embeddings (or existing provider) while preserving QMD-style retrieval improvements (RRF, lexical stream, strong-signal skip, position-aware rerank blending). This plan is written so another agent can implement directly.

---

## 0) Constraints & Ground Truth

- QMD **does not** use Ollama at runtime. It uses **node-llama-cpp** with local GGUF models for embeddings and reranking.
- Our current CEMS changes added an **Ollama container**, an **Ollama embedding client**, and **config knobs** that default to OpenRouter but imply Ollama usage. This violates our architecture rule of keeping files small and adds infra complexity.
- Request: **drop the Ollama container** and remove its embedding path.

---

## 1) Remove Ollama Infrastructure

### Remove the Docker service
**File**: `docker-compose.yml`

Remove the `ollama:` service and `ollama_data:` volume.

**Example patch:**
```yaml
# delete this block
ollama:
  image: ollama/ollama
  container_name: cems-ollama
  ports:
    - "11434:11434"
  volumes:
    - ollama_data:/root/.ollama
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
    interval: 30s
    timeout: 10s
    retries: 3

# delete this volume
ollama_data:
```

---

## 2) Remove Ollama Embedding Client (Local Embeddings)

### Delete the Ollama embedding class and helpers
**File**: `src/cems/embedding.py`

Remove:
- `AsyncOllamaEmbeddingClient`
- `_ollama_client` singleton
- `get_async_ollama_embedding_client()`
- `DEFAULT_OLLAMA_URL`, `DEFAULT_OLLAMA_MODEL`, `DEFAULT_OLLAMA_DIM`
- References in module docstring that say Ollama is default/fast

**Example snippet to delete (exact blocks):**
```py
# DEFAULT_OLLAMA_URL / DEFAULT_OLLAMA_MODEL / DEFAULT_OLLAMA_DIM
# class AsyncOllamaEmbeddingClient
# _ollama_client singleton
# get_async_ollama_embedding_client()
```

### Keep OpenRouter embedding clients only
Ensure `EmbeddingClient` and `AsyncEmbeddingClient` remain as the only embedding backends.

---

## 3) Remove Ollama Embedding Backend Switch

### Simplify embedder initialization
**File**: `src/cems/memory/core.py`

Replace the Ollama/OpenRouter conditional with a single OpenRouter async embedder.

**Before (example)**
```py
if self._async_embedder is None:
    embedding_backend = getattr(self.config, 'embedding_backend', 'ollama')
    if embedding_backend == 'ollama':
        self._async_embedder = AsyncOllamaEmbeddingClient(...)
    else:
        self._async_embedder = AsyncEmbeddingClient(model=self.config.embedding_model)
```

**After (example)**
```py
if self._async_embedder is None:
    self._async_embedder = AsyncEmbeddingClient(model=self.config.embedding_model)
```

Also remove `AsyncOllamaEmbeddingClient` import from this file.

---

## 4) Remove Ollama Embedding Config Knobs

**File**: `src/cems/config.py`

Delete these fields entirely:
- `ollama_url` (if only used for embeddings; keep only if reranker still uses it)
- `embedding_backend`
- `ollama_embedding_model`
- `ollama_embedding_dim`

If reranking still uses Ollama (see section 5), keep `ollama_url` and `reranker_model`. Otherwise, remove them too.

---

## 5) Decide on Reranker Backend (Non-Ollama Options)

You asked to drop the Ollama container because it "kills everything". If we drop Ollama **entirely**, then:

- Set `reranker_backend` default to `"llm"`
- Remove Qwen/Ollama reranker code paths

### Option A: Keep reranking via OpenRouter (no Ollama)
**Files**: `src/cems/retrieval.py`, `src/cems/memory/retrieval.py`, `src/cems/config.py`

Implementation steps:
- Delete `rerank_with_qwen` and `rerank_with_qwen_sync` functions.
- Remove `ollama` dependency from `pyproject.toml`.
- In memory retrieval, remove the `reranker_backend == "qwen"` branch.

**Example branch removal:**
```py
# remove this block
if reranker_backend == "qwen":
    candidates = rerank_with_qwen_sync(...)
elif reranker_backend == "llm" and client:
    candidates = rerank_with_llm(...)
```

Replace with:
```py
if reranker_backend == "llm" and client:
    candidates = rerank_with_llm(...)
```

### Option B: Keep reranking but change backend (non-Ollama)
If you want local reranking without Ollama, it will require a separate integration (e.g., llama.cpp bindings or another local inference stack). That is **not** in the current codebase.

---

## 6) QMD-Style Local Models (llama.cpp parity)

QMD runs **local GGUF models** via **node-llama-cpp**. To match that approach in CEMS (Python), you need a llama.cpp-based runtime.

### Recommended approach (parity with QMD)
- **Embedding model**: `embeddinggemma-300M` GGUF (same family QMD uses)
- **Reranker model**: `Qwen3-Reranker-0.6B` GGUF
- **Query expansion**: a small GGUF model fine-tuned for expansion (QMD uses a qmd-specific expansion model)

### Implementation options
1. **llama-cpp-python** (direct Python bindings)
   - Pros: in-process, fast, no extra service
   - Cons: needs native build; API surface can change
2. **llama.cpp server** (run as a local HTTP service)
   - Pros: decoupled, language-agnostic
   - Cons: extra process to manage

### Minimal design (backend interface)
Introduce explicit backend interfaces and keep QMD parity isolated in new files.

**New interface definitions (example):**
```py
# src/cems/embedding_backends/base.py
class EmbeddingBackend:
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

# src/cems/rerank_backends/base.py
class RerankBackend:
    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return a list of scores aligned to docs."""
        raise NotImplementedError
```

**llama.cpp-backed implementations (skeleton only):**
```py
# src/cems/embedding_backends/llamacpp.py
from .base import EmbeddingBackend

class LlamaCppEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # load model via llama-cpp-python or llama.cpp client

    async def embed(self, text: str) -> list[float]:
        # call the llama.cpp embedding API here
        ...

# src/cems/rerank_backends/llamacpp.py
from .base import RerankBackend

class LlamaCppRerankBackend(RerankBackend):
    def __init__(self, model_path: str):
        self.model_path = model_path

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        # call the llama.cpp reranking API here
        ...
```

### Wiring (example)
```py
# src/cems/config.py
reranker_backend: str = "llm"  # "llm" | "llamacpp" | "disabled"
embedding_backend: str = "openrouter"  # "openrouter" | "llamacpp"

# src/cems/memory/core.py
if self._async_embedder is None:
    if self.config.embedding_backend == "llamacpp":
        self._async_embedder = LlamaCppEmbeddingBackend(model_path=...)
    else:
        self._async_embedder = AsyncEmbeddingClient(...)

# src/cems/memory/retrieval.py
if reranker_backend == "llamacpp":
    candidates = await rerank_with_llamacpp(...)
elif reranker_backend == "llm":
    candidates = rerank_with_llm(...)
```

### Required operational change
If you switch to local GGUF embeddings, **you must re-embed** existing data if dimensions differ from OpenRouter (this is mandatory for vector index consistency).

### Summary
This gives QMD parity without Ollama. It is a separate integration that should live behind clean backend interfaces to avoid bloating core files.

---

## 7) Ensure File Size Rules Are Respected

Your architecture plan defines "large files" as >400 lines. After changes, ensure:

- `src/cems/embedding.py` ≤ 400
- `src/cems/retrieval.py` ≤ 400 (currently 800+)
- `src/cems/memory/retrieval.py` ≤ 400

If needed, split:

### Suggested split for `src/cems/retrieval.py`
- `src/cems/retrieval/rerank.py` (reranking functions)
- `src/cems/retrieval/rrf.py` (RRF utilities)
- `src/cems/retrieval/normalization.py` (score normalization helpers)

Update imports accordingly. This is not optional if you want to stay within your own plan.

---

## 8) Tests to Add/Update (Minimum)

Add or update tests to prove:

- The async embedder always uses OpenRouter and does not reference Ollama.
- No Ollama references remain in config or memory initialization.

**Example test** (pseudo):
```py
from cems.memory.core import CEMSMemory
from cems.config import CEMSConfig
from cems.embedding import AsyncEmbeddingClient

async def test_async_embedder_is_openrouter():
    mem = CEMSMemory(CEMSConfig())
    await mem._ensure_initialized_async()
    assert isinstance(mem._async_embedder, AsyncEmbeddingClient)
```

---

## 9) Cleanup Dependencies

**File**: `pyproject.toml`

Remove:
- `ollama>=0.4.0`

Run `uv lock` after removing if you update the lockfile.

---

## 10) Final Checklist

- [ ] `docker-compose.yml` has no Ollama service/volume
- [ ] `src/cems/embedding.py` has no Ollama client
- [ ] `src/cems/memory/core.py` uses only `AsyncEmbeddingClient`
- [ ] `src/cems/config.py` has no Ollama embedding config
- [ ] `pyproject.toml` has no `ollama` dependency
- [ ] Tests pass
- [ ] File size limits respected

---

## Notes for the implementing agent

- **Do not** keep partial Ollama features. Either keep Ollama as a first-class supported component with proper infra, or remove it completely.
- QMD’s local pipeline is via node-llama-cpp, not Ollama. If you want local embeddings or rerank without Ollama, you need a different runtime integration.
