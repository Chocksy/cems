#!/usr/bin/env bash
# CEMS Before LLM Hook — Memory injection for Moltis
#
# Receives BeforeLLMCall payload on stdin:
#   {session_key, provider, model, messages[], tool_count, iteration}
#
# Outputs modified payload to inject CEMS memories as system message.
# Exit 0 with no stdout = continue unchanged (on error/no results).

set -uo pipefail
# Note: no 'set -e' — we handle errors explicitly to avoid tripping
# Moltis circuit breaker (disables hook after 3 consecutive failures)

# Read full stdin into variable
INPUT=$(cat)

# Extract fields
ITERATION=$(echo "$INPUT" | jq -r '.iteration // 1')
MESSAGES=$(echo "$INPUT" | jq -c '.messages // []')

# Extract last user message content for search query
LAST_USER_MSG=$(echo "$MESSAGES" | jq -r '
  [.[] | select(.role == "user")] | last |
  if .content | type == "array" then
    [.content[] | select(.type == "text") | .text] | join(" ")
  else
    .content // ""
  end
')

# Skip if no user message or too short
if [ -z "$LAST_USER_MSG" ] || [ ${#LAST_USER_MSG} -lt 10 ]; then
  exit 0
fi

CONTEXT_PARTS=""

# --- 1. Fetch profile on first iteration ---
if [ "$ITERATION" = "1" ]; then
  PROFILE=$(curl -sf --max-time 5 \
    -H "Authorization: Bearer ${CEMS_API_KEY}" \
    "${CEMS_API_URL}/api/memory/profile" 2>/dev/null || echo "")

  if [ -n "$PROFILE" ]; then
    PROFILE_TEXT=$(echo "$PROFILE" | jq -r '.context // empty' 2>/dev/null || echo "")
    if [ -n "$PROFILE_TEXT" ]; then
      CONTEXT_PARTS="${CONTEXT_PARTS}

<cems-profile>
${PROFILE_TEXT}
</cems-profile>"
    fi
  fi
fi

# --- 2. Search CEMS for relevant memories ---
# Truncate query to 200 chars to avoid noisy embeddings
SEARCH_QUERY=$(echo "$LAST_USER_MSG" | head -c 200)

SEARCH_RESULT=$(curl -sf --max-time 8 \
  -X POST \
  -H "Authorization: Bearer ${CEMS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg q "$SEARCH_QUERY" '{"query": $q, "scope": "both"}')" \
  "${CEMS_API_URL}/api/memory/search" 2>/dev/null || echo "")

if [ -n "$SEARCH_RESULT" ]; then
  # Check if we got results
  RESULT_COUNT=$(echo "$SEARCH_RESULT" | jq -r '.results | length // 0' 2>/dev/null || echo "0")

  if [ "$RESULT_COUNT" -gt 0 ]; then
    # Format results
    MEMORIES=$(echo "$SEARCH_RESULT" | jq -r '
      .results | to_entries | map(
        "\(.key + 1). [\(.value.category // "general")] \(.value.content // .value.memory // "")"
      ) | join("\n")
    ' 2>/dev/null || echo "")

    if [ -n "$MEMORIES" ]; then
      CONTEXT_PARTS="${CONTEXT_PARTS}

<memory-recall>
Relevant memories for this conversation:

${MEMORIES}
</memory-recall>"

      # Fire-and-forget: log shown memories
      MEMORY_IDS=$(echo "$SEARCH_RESULT" | jq -c '[.results[].memory_id // .results[].id // empty]' 2>/dev/null || echo "[]")
      if [ "$MEMORY_IDS" != "[]" ] && [ "$MEMORY_IDS" != "null" ]; then
        curl -sf --max-time 2 \
          -X POST \
          -H "Authorization: Bearer ${CEMS_API_KEY}" \
          -H "Content-Type: application/json" \
          -d "{\"memory_ids\": ${MEMORY_IDS}}" \
          "${CEMS_API_URL}/api/memory/log-shown" >/dev/null 2>&1 &
      fi
    fi
  fi
fi

# --- 3. If we have context to inject, modify the messages ---
if [ -z "$CONTEXT_PARTS" ]; then
  # No context to inject — continue unchanged
  exit 0
fi

# Build the system message to prepend
SYSTEM_MSG=$(jq -n --arg content "$CONTEXT_PARTS" '{
  "role": "system",
  "content": $content
}')

# Prepend the system message to the messages array
MODIFIED_MESSAGES=$(echo "$MESSAGES" | jq --argjson sys "$SYSTEM_MSG" '[$sys] + .')

# Output the modification payload
# We pass back all original fields plus the modified messages
echo "$INPUT" | jq --argjson msgs "$MODIFIED_MESSAGES" '{
  "action": "modify",
  "data": (. + {"messages": $msgs})
}'
