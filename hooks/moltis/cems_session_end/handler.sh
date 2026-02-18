#!/usr/bin/env bash
# CEMS Session End Hook — Transcript summarization for Moltis
#
# Receives AgentEnd payload on stdin:
#   {session_key, text, iterations, tool_calls}
#
# Reads the session JSONL from disk, formats it, and POSTs to CEMS.
# Exit 0 always (read-only event, nothing to modify).

set -uo pipefail

# Read full stdin
INPUT=$(cat)

SESSION_KEY=$(echo "$INPUT" | jq -r '.session_key // ""')

if [ -z "$SESSION_KEY" ]; then
  exit 0
fi

# Sanitize session_key for filename (colons → underscores)
SAFE_KEY=$(echo "$SESSION_KEY" | tr ':' '_')

# Find the session JSONL file
MOLTIS_DIR="${MOLTIS_DATA_DIR:-${HOME}/.moltis}"
SESSION_FILE=""

# Search under agents/*/sessions/ for the file
for f in "${MOLTIS_DIR}"/agents/*/sessions/"${SAFE_KEY}.jsonl"; do
  if [ -f "$f" ]; then
    SESSION_FILE="$f"
    break
  fi
done

if [ -z "$SESSION_FILE" ] || [ ! -f "$SESSION_FILE" ]; then
  # No session file found — nothing to summarize
  exit 0
fi

# Format JSONL into readable transcript text
# Extract role + content from each line, skip system/notice messages
TRANSCRIPT=$(jq -r '
  select(.role != "system" and .role != "notice") |
  if .role == "user" then
    "User: " + (
      if (.content | type) == "array" then
        [.content[] | select(.type == "text") | .text] | join("\n")
      else
        (.content // "")
      end
    )
  elif .role == "assistant" then
    "Assistant: " + (.content // "(no content)") +
    if .tool_calls then
      "\n  [Tool calls: " + ([.tool_calls[].function.name // .tool_calls[].name // "unknown"] | join(", ")) + "]"
    else ""
    end
  elif .role == "tool_result" then
    "Tool (" + (.tool_name // "unknown") + "): " +
    if .success then "success" else "error" end +
    if .error then " — " + .error else "" end
  elif .role == "tool" then
    "Tool result: " + ((.content // "") | tostring | .[0:200])
  else
    .role + ": " + ((.content // "") | tostring | .[0:200])
  end
' "$SESSION_FILE" 2>/dev/null || echo "")

if [ -z "$TRANSCRIPT" ] || [ ${#TRANSCRIPT} -lt 50 ]; then
  # Transcript too short to be worth summarizing
  exit 0
fi

# Truncate to 50k chars (CEMS caps at this anyway)
TRANSCRIPT=$(echo "$TRANSCRIPT" | head -c 50000)

# Extract session ID (first 8 chars of key for tagging)
SESSION_ID=$(echo "$SESSION_KEY" | head -c 32)

# Get metadata from AgentEnd payload
ITERATIONS=$(echo "$INPUT" | jq -r '.iterations // 0')
TOOL_CALLS=$(echo "$INPUT" | jq -r '.tool_calls // 0')

# Build project context from session key (agent:xxx:main → xxx)
# This is best-effort; Moltis doesn't expose project/git info in hooks
PROJECT_CONTEXT="moltis session (${ITERATIONS} iterations, ${TOOL_CALLS} tool calls)"

# POST to CEMS
curl -sf --max-time 30 \
  -X POST \
  -H "Authorization: Bearer ${CEMS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg content "$TRANSCRIPT" \
    --arg session_id "$SESSION_ID" \
    --arg project_context "$PROJECT_CONTEXT" \
    '{
      "content": $content,
      "session_id": $session_id,
      "project_context": $project_context,
      "mode": "finalize",
      "epoch": 0
    }')" \
  "${CEMS_API_URL}/api/session/summarize" >/dev/null 2>&1 || true

exit 0
