# Private mode and enterprise positioning

Date: 2026-09-22
Status: approved 2026-09-22
Repos: Chocksy/cems (code, docs), getcems.com (site)

## Problem

Enterprise buyers say they do not want to hand their data to Anthropic or OpenAI. CEMS today stores memories in the customer's PostgreSQL but sends every extraction, consolidation, query synthesis and embedding call to OpenRouter. The site FAQ claims nothing is sent to third parties. That claim is false in the default setup. There is no supported way to run the memory pipeline on models the customer controls.

## Answer

The brain lives in your Postgres. The models are a plug. CEMS gets a generic OpenAI-compatible provider layer so the pipeline can run against Ollama, vLLM, LiteLLM, Azure OpenAI or Bedrock through a gateway. A "private mode" preset ships Ollama in the compose file so a VPS comes up self-contained. The site is rebuilt around this positioning with Hubstaff as proof.

What CEMS does not fix, and the site says so: the coding agent itself still talks to its vendor. Claude Code and Cursor only go private through Bedrock, Azure or Vertex. Codex, Goose, OpenCode, Aider, Cline, Continue and Roo Code can run open-weight models.

## Decisions

| Topic | Decision |
|---|---|
| Runtime for local models | Ollama, in a compose profile |
| Provider abstraction | Generic OpenAI-compatible base URL + key. OpenRouter stays default |
| llama.cpp backend | Removed. It was a benchmarking aid |
| Embedding dimension | Column created at configured dimension. Mismatch with an existing table is a hard error. No re-embed job |
| Delivery | `install-server.sh --private` plus three cloud-init snippets (AWS, Hetzner, DigitalOcean). No marketplace listings, no Terraform |
| Recall-time LLM in private mode | Nothing is hard-disabled. Two presets: CPU preset turns recall-time features off, GPU preset keeps everything on with larger models |
| Default private models | CPU preset: `gemma4:e4b` chat, `embeddinggemma` embeddings (768 dims), about 7 GB RAM. GPU preset: a long-context open model chosen at implementation time from the Ollama library, same embeddings |
| Site | One homepage, plain neutral design, ocean theme retired |
| Proof | "Runs at Hubstaff across 50 engineers since July 2026" plus existing benchmark numbers |
| Enterprise offer | "We install it in your cloud. You pay for your servers. One-time setup fee." Lead capture is a mailto link |
| Agent list | Tested: Claude Code, Cursor, Codex, Goose, OpenCode. Listed untested: Aider, Cline, Continue, Roo Code |
| Name | "Private mode". Flags `--private` and `--private-gpu`, presets `.env.private-cpu` and `.env.private-gpu` |
| Order of work | Code, then docs, then site |

## Part 1: provider layer (Chocksy/cems)

### Config

New fields in `src/cems/config.py`, all with env prefix `CEMS_`:

| Field | Default | Purpose |
|---|---|---|
| `llm_base_url` | `https://openrouter.ai/api/v1` | Chat completions endpoint |
| `llm_api_key` | falls back to `OPENROUTER_API_KEY` | Key for chat endpoint |
| `embedding_base_url` | same as `llm_base_url` | Embeddings endpoint |
| `embedding_api_key` | same as `llm_api_key` | Key for embeddings endpoint |
| `embedding_dimension` | 1536 | Already exists. Now enforced |

`embedding_backend` and every `llamacpp_*` field are removed. `src/cems/llamacpp_server.py` is deleted.

Ollama ignores the API key but requires one to be present. The preset sets `CEMS_LLM_API_KEY=ollama`.

### LLM client

`src/cems/llm/client.py`: `OpenRouterClient` reads `base_url` and `api_key` from config instead of the constants. The OpenRouter attribution headers are sent only when the base URL host is `openrouter.ai`. The `fast_route` provider preference is sent only in the same case. `_resolve_model` keeps its mapping table; an unknown name passes through unchanged so `gemma4:e4b` works.

`src/cems/agentic/search.py` builds its client through `get_client()` so it follows the same base URL. It gets an `CEMS_ENABLE_AGENTIC_SEARCH` flag (default true) so the CPU preset can turn it off. No mode flag decides this; the preset does.

`src/cems/admin/routes.py` health check: replace the hard-coded OpenRouter URL and model with the configured ones. Report `llm` and `embeddings` instead of `openrouter_llm`.

### Embeddings

`src/cems/embedding.py`: `EmbeddingClient` and `AsyncEmbeddingClient` post to `{embedding_base_url}/embeddings` with the configured key. The `dimensions` parameter is sent only when the host is `openrouter.ai` or `api.openai.com`; Ollama rejects it.

`src/cems/memory/core.py`: drop the backend branch. One code path.

### Schema

`src/cems/db/database.py`: the `memory_chunks.embedding` column is created as `vector({embedding_dimension})`. On startup, read the existing column's `atttypmod` from `pg_attribute`. If it does not match config, raise with:

```
Embedding dimension mismatch: database has 1536, config has 768.
Private mode needs a fresh database. See docs/DEPLOYMENT.md#private-mode.
```

Any other `vector(1536)` literals in migrations or indexes use the same config value.

### No mode flag

There is no `private_mode` switch in code. Private mode is a compose profile plus a preset `.env`. The CPU preset sets `CEMS_ENABLE_QUERY_SYNTHESIS`, `CEMS_ENABLE_PREFERENCE_SYNTHESIS`, `CEMS_ENABLE_QUERY_DECOMPOSITION` and `CEMS_ENABLE_AGENTIC_SEARCH` to false. The GPU preset leaves them at their defaults and points `CEMS_LLM_MODEL` and `CEMS_AGENTIC_MODEL` at a long-context model.

### Tests

- Unit: `OpenRouterClient` with a non-OpenRouter base URL sends no attribution headers and no provider preference.
- Unit: embedding client omits `dimensions` for a non-OpenAI host.
- Unit: dimension mismatch raises the documented error; matching dimension passes.
- Existing suites keep passing after llama.cpp removal.

## Part 2: private mode deployment

### Compose

`deploy/docker-compose.yml` gains a profile named `private`:

- `ollama`: image `ollama/ollama`, volume `ollama_data`, port 11434 internal only.
- `ollama-pull`: one-shot container that runs `ollama pull gemma4:e4b && ollama pull embeddinggemma` against the `ollama` service, then exits. `cems-server` depends on it completing.

`cems-server` reads `CEMS_LLM_BASE_URL`, `CEMS_LLM_API_KEY`, `CEMS_EMBEDDING_BASE_URL`, `CEMS_EMBEDDING_DIMENSION`, `CEMS_LLM_MODEL`, `CEMS_EMBEDDING_MODEL` and the `CEMS_ENABLE_*` flags from `.env` with the current OpenRouter values as fallbacks.

### Presets

`deploy/.env.private-cpu.example`:

```
POSTGRES_PASSWORD=change_me
CEMS_ADMIN_KEY=change_me
CEMS_LLM_BASE_URL=http://ollama:11434/v1
CEMS_LLM_API_KEY=ollama
CEMS_LLM_MODEL=gemma4:e4b
CEMS_EMBEDDING_MODEL=embeddinggemma
CEMS_EMBEDDING_DIMENSION=768
CEMS_ENABLE_QUERY_SYNTHESIS=false
CEMS_ENABLE_PREFERENCE_SYNTHESIS=false
CEMS_ENABLE_QUERY_DECOMPOSITION=false
CEMS_ENABLE_AGENTIC_SEARCH=false
```

`deploy/.env.private-gpu.example`: same base URL and embeddings, `CEMS_LLM_MODEL` and `CEMS_AGENTIC_MODEL` set to a long-context open model, all feature flags at defaults. The compose profile adds the NVIDIA device reservation to the `ollama` service when `OLLAMA_GPU=1`.

The install script picks the preset from `--private` (CPU) or `--private-gpu`.

### Hardware and cost table

Lives in `docs/DEPLOYMENT.md` and in the site's enterprise section. Filled at implementation time with current list prices, then rechecked before the site ships.

| Tier | Example box | What runs | Approx. monthly |
|---|---|---|---|
| CPU | Hetzner CPX31 or CX32, 8 to 16 GB | Extraction, consolidation, embeddings. Plain hybrid recall | tens of euros |
| GPU, single card | Hetzner GEX44 (RTX 4000, 20 GB) or an AWS g6 | Everything on, mid-size model, 128k context | low hundreds |
| GPU, large | AWS p4/p5 or Hetzner GEX131 | Everything on, 1M-context model | high hundreds and up |

The table answers "how much would it cost" before anyone asks. The CPU row is the default for the three-command install.

### `install-server.sh`

New file at repo root, served at `getcems.com/install-server.sh`. Flags: `--private`, `--private-gpu`, `--openrouter-key <key>`, `--yes`.

1. Install Docker and the compose plugin if missing (Debian, Ubuntu, Fedora via the official convenience script).
2. Create `/opt/cems`, download `deploy/docker-compose.yml`.
3. Write `.env` from the matching example, generating `POSTGRES_PASSWORD` and `CEMS_ADMIN_KEY` with `openssl rand`.
4. `docker compose --profile private up -d` or `docker compose up -d`.
5. Wait for `/health`, print the admin key once and the next command (`cems admin users create`).

Idempotent: re-running with an existing `.env` keeps it.

### Cloud-init

`deploy/cloud-init/{aws,hetzner,digitalocean}.yaml`. All three are the same `#cloud-config` with a `runcmd` that curls the script with `--private --yes`. Kept as three files so the docs can link each provider directly and so provider-specific lines can be added later without a template layer.

### Docs

- `docs/DEPLOYMENT.md`: new "Private mode" section. Data-flow table with three modes. Dimension limit. GPU upgrade. Cloud-init snippets. Remove llama.cpp section.
- `README.md`: one paragraph and a link.
- `docs/CLIENT.md`: "Running your agent privately" with the agent table and vendor doc links.

### Verification

- Fresh Hetzner CX32 (8 GB) or CPX31 (16 GB): cloud-init boots, `/health` green, `cems memory add` and `cems memory search` round-trip through Ollama. Note the RAM tier that worked in the docs.
- One GPU box with the GPU preset: agentic search returns results through Ollama. Record the model, context length and price used in the hardware table.
- Egress check: with `docker network` set to `internal` after model pull, the server still serves add and search.
- OpenCode configured against Ollama uses the CEMS MCP server for one recall and one store.

## Part 3: site (getcems.com)

Astro plus Tailwind stays. Ocean assets and `OceanFloorBg.astro` are removed. Dark neutral palette, one accent colour, no illustrations beyond a data-flow diagram and one product screenshot.

Sections in order:

1. **Hero**. Headline: "Your engineering team's memory. Portable across every AI agent. Running on your servers." Sub: what CEMS remembers and that it works with Claude Code, Cursor, Codex, Goose and any MCP client. Two buttons: install (developer) and "Run it on your servers" (anchor to section 7).
2. **Switching story**. Same brain in Claude Code today, Codex tomorrow, a local model next year. Three-panel graphic, short copy. This is the personal-project hook.
3. **What leaves your network**. Table, three columns: Default (OpenRouter), Private cloud (Bedrock, Azure, Vertex through a gateway), Fully local (Ollama). Rows: memories, model calls, embeddings, the agent itself. Honest about the agent row.
4. **How it works**. Three steps, existing copy tightened.
5. **Works with**. Agent table with two columns: memory integration (tested / MCP) and private-mode path (open-weight / enterprise cloud / none).
6. **Proof**. "Runs at Hubstaff across 50 engineers since July 2026." Benchmark numbers beside it.
7. **Install**. Three tabs: solo developer one-liner, team server compose, private mode `install-server.sh --private` with the cloud-init note.
8. **Enterprise**. "We install it in your cloud. You pay for your servers. One-time setup fee." Mailto button with prefilled subject.
9. **FAQ**. Rewritten. Data question answer: "Memories never leave your Postgres. Model calls go to OpenRouter by default, or to your own models in private mode." Add: "Does my coding agent still send code to its vendor?" and "Can I switch agents and keep the memory?"

Meta title and description updated to the new positioning. JSON-LD FAQ regenerated from the new list.

### Verification

- `npm run build` passes.
- Lighthouse accessibility 90 or above on the built page.
- Every claim in sections 3, 5 and 7 maps to a verified item in Part 2.

## Out of scope

- Multi-tenant SaaS.
- Re-embedding an existing OpenRouter install into private mode.
- Slack, Notion or Confluence ingest.
- Pointing Claude Code or Cursor at local models.
- Marketplace images and Terraform.
- A separate `/enterprise` page.
