#!/usr/bin/env bash
# CEMS After Tool Hook — Incremental tool learning for Moltis
#
# Receives AfterToolCall payload on stdin:
#   {session_key, tool_name, success, result}
#
# POSTs to CEMS /api/tool/learning for significant tool completions.
# Exit 0 always (read-only event).

set -uo pipefail

INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""')
SESSION_KEY=$(echo "$INPUT" | jq -r '.session_key // ""')
SUCCESS=$(echo "$INPUT" | jq -r '.success // true')

# Skip read-only / non-learnable tools
case "$TOOL_NAME" in
  Read|Glob|Grep|LS|WebFetch|WebSearch|memory_search|memory_get)
    exit 0
    ;;
esac

# Skip if no tool name
if [ -z "$TOOL_NAME" ]; then
  exit 0
fi

# Extract result summary (truncate to keep payload small)
RESULT_SUMMARY=$(echo "$INPUT" | jq -r '
  .result // null |
  if . == null then ""
  elif type == "object" then tostring | .[0:500]
  elif type == "string" then .[0:500]
  else tostring | .[0:500]
  end
' 2>/dev/null || echo "")

# Build and send tool learning payload (fire-and-forget)
curl -sf --max-time 5 \
  -X POST \
  -H "Authorization: Bearer ${CEMS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg tool_name "$TOOL_NAME" \
    --arg tool_output "$RESULT_SUMMARY" \
    --arg session_id "$SESSION_KEY" \
    --argjson success "$SUCCESS" \
    '{
      "tool_name": $tool_name,
      "tool_output": $tool_output,
      "session_id": $session_id,
      "context_snippet": ("Moltis agent used " + $tool_name + " — success: " + ($success | tostring))
    }')" \
  "${CEMS_API_URL}/api/tool/learning" >/dev/null 2>&1 || true

exit 0
