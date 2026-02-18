+++
name = "cems-before-llm"
description = "Inject CEMS memories and profile into LLM context"
events = ["BeforeLLMCall"]
command = "./handler.sh"
timeout = 10

[requires]
bins = ["curl", "jq"]
env = ["CEMS_API_URL", "CEMS_API_KEY"]
+++

# CEMS Before LLM Hook

Searches CEMS for relevant memories based on the latest user message and
injects them as a system message into the messages array before the LLM call.

On the first iteration of a session, also fetches the user's CEMS profile
(preferences, guidelines) and prepends it.

## How it works

1. Reads `BeforeLLMCall` payload from stdin (has `messages[]`, `iteration`, etc.)
2. Extracts the last user message content
3. POSTs to CEMS `/api/memory/search` with that text
4. If iteration == 1, also GETs `/api/memory/profile`
5. Prepends a system message with memory context
6. Outputs `{"action":"modify","data":{...}}` with the modified messages

## Environment Variables

- `CEMS_API_URL` — CEMS server URL (e.g., `https://cems.example.com:8765`)
- `CEMS_API_KEY` — CEMS API key for Bearer auth
