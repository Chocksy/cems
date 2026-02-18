+++
name = "cems-session-end"
description = "Send session transcript to CEMS for summarization"
events = ["AgentEnd"]
command = "./handler.sh"
timeout = 15

[requires]
bins = ["curl", "jq"]
env = ["CEMS_API_URL", "CEMS_API_KEY"]
+++

# CEMS Session End Hook

Reads the Moltis session JSONL file and sends the transcript to CEMS
for summarization and long-term memory extraction.

Uses `AgentEnd` instead of `SessionEnd` because `AgentEnd` carries
useful metadata (`text`, `iterations`, `tool_calls`) while `SessionEnd`
only has `session_key`.

## How it works

1. Reads `AgentEnd` payload from stdin (`session_key`, `text`, etc.)
2. Finds the session JSONL file on disk (under `~/.moltis/agents/*/sessions/`)
3. Formats the JSONL into readable transcript text
4. POSTs to CEMS `/api/session/summarize` with `mode=finalize`

## Environment Variables

- `CEMS_API_URL` — CEMS server URL
- `CEMS_API_KEY` — CEMS API key for Bearer auth
- `MOLTIS_DATA_DIR` — Override Moltis data dir (default: `~/.moltis`)
