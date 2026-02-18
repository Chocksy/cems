+++
name = "cems-after-tool"
description = "Send tool usage data to CEMS for incremental learning"
events = ["AfterToolCall"]
command = "./handler.sh"
timeout = 5

[requires]
bins = ["curl", "jq"]
env = ["CEMS_API_URL", "CEMS_API_KEY"]
+++

# CEMS After Tool Hook

Sends tool call results to CEMS for incremental learning extraction.
Only fires for significant tools (Edit, Write, Bash with meaningful output).
Skips read-only tools (Read, Glob, Grep, etc.).

## Environment Variables

- `CEMS_API_URL` — CEMS server URL
- `CEMS_API_KEY` — CEMS API key for Bearer auth
