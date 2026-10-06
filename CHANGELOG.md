# Changelog

## 0.14.2

Reliability release for `/api/memory/search`. A slow LLM provider used to stall searches for 20 to 36 seconds and block other requests on the same worker. Now the optional LLM steps have a time limit, and search falls back to the original query when they run out of time.

### Behavior changes

- **The request flags now really turn LLM steps off.** A request with `enable_query_synthesis=false` gets no query synthesis, and a request with `enable_hyde=false` gets no HyDE. This holds for temporal, preference and aggregation queries too, which previously forced these steps. The HTTP API already defaults both flags to `false`, so default API searches now skip LLM rewriting. Clients that want enriched search must send `true`. Server settings work as before when the request allows the step: `CEMS_ENABLE_FORCED_SYNTHESIS` (default `true`) still forces expansion past `CEMS_ENABLE_QUERY_SYNTHESIS=false`, and the CPU preset still turns it off.
- **Agentic search reports failures.** If no agent completes, `mode=agentic` returns an HTTP 500 error instead of a successful "no memories found". Causes include provider errors, timeouts, no free capacity and empty completions. If some agents succeed, the response includes `partial: true` and `degraded_agents`. An agent that returns an explicit `[]` still means no match.
- Aggregation detection now matches whole phrases only. For example, "call the CEMS system" no longer matches "all the".

### Time limits and fallback

- The optional LLM steps in a search (auto-mode intent, decomposition, synthesis, HyDE) share a total time limit of **2s** per request. They run outside the event loop. Each provider call has a 2s timeout, makes no retries and returns at most 512 tokens.
- If a step times out, fails, has no API key or finds no free capacity, search continues with the original query. Relevance filtering, project ranking, scope and metadata still apply. The logs record the stage, reason and elapsed time for each skipped or failed step. The HTTP response shape for non-agentic search does not change.
- Each process runs at most **4** optional-step LLM calls at once, with no queue. If a request times out or is cancelled while its call is still running, that call keeps its slot until it finishes, so stalled calls cannot pile up.
- Agentic search uses its own process-wide pool of 8 calls and keeps its 10s limit per agent. When the pool is full, requests fail immediately instead of waiting.
- Embedding and database errors still fail the search. They are logged as `Search transport failure`.

### Scope

- The 2s limit covers only the optional LLM steps, not the whole request. Embedding and database time are not included.
- The limits apply per process.
- Maintenance and background jobs keep using the shared client with the SDK's default timeout and retries.
- All of this works with any OpenAI-compatible endpoint. The bounded client keeps the configured base URL, key and model, and still sends OpenRouter-only headers and routing only to openrouter.ai.

### Configuration

New settings. Invalid values (zero, negative, NaN or infinity) fail when the config loads.

| Variable | Default | Meaning |
|---|---|---|
| `CEMS_RETRIEVAL_LLM_BUDGET_SECONDS` | `2.0` | Total time for the optional LLM steps in one search |
| `CEMS_RETRIEVAL_LLM_TIMEOUT_SECONDS` | `2.0` | Provider timeout per call, with no retries |
| `CEMS_RETRIEVAL_LLM_MAX_TOKENS` | `512` | Maximum response tokens for each rewrite |
| `CEMS_RETRIEVAL_LLM_MAX_CONCURRENCY` | `4` | Maximum concurrent optional-step LLM calls per process |

`deploy/docker-compose.yml` passes the budget and timeout settings to the server. On a local GPU model (Ollama), raise them if synthesis keeps timing out.
