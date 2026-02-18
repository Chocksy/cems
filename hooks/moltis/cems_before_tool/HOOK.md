+++
name = "cems-before-tool"
description = "Check CEMS gate rules before tool execution"
events = ["BeforeToolCall"]
command = "./handler.sh"
timeout = 5

[requires]
bins = ["curl", "jq"]
env = ["CEMS_API_URL", "CEMS_API_KEY"]
+++

# CEMS Before Tool Hook

Checks CEMS gate rules before allowing a tool call. Can block dangerous
operations (e.g., destructive commands, production deploys) based on
user-defined rules stored in CEMS.

## How it works

1. Reads `BeforeToolCall` payload from stdin (`tool_name`, `arguments`)
2. Fetches gate rules from CEMS `/api/memory/gate-rules`
3. Matches tool name and arguments against rules
4. Exit 0 = allow, Exit 1 + stderr reason = block

## Environment Variables

- `CEMS_API_URL` — CEMS server URL
- `CEMS_API_KEY` — CEMS API key for Bearer auth
