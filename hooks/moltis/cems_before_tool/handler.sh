#!/usr/bin/env bash
# CEMS Before Tool Hook — Gate rules for Moltis
#
# Receives BeforeToolCall payload on stdin:
#   {session_key, tool_name, arguments}
#
# Checks CEMS gate rules. Exit 0 = allow, Exit 1 = block (stderr = reason).

set -uo pipefail

# Add CEMS bin dir to PATH (jq lives on the persistent volume)
export PATH="${HOME}/.moltis/bin:${PATH}"

INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""')
ARGUMENTS=$(echo "$INPUT" | jq -c '.arguments // {}')

if [ -z "$TOOL_NAME" ]; then
  exit 0
fi

# Cache gate rules locally (refreshed every 5 minutes)
CACHE_DIR="${HOME}/.cems/cache/gate_rules"
CACHE_FILE="${CACHE_DIR}/moltis.json"
mkdir -p "$CACHE_DIR"

# Refresh cache if stale (>300 seconds old)
REFRESH=false
if [ ! -f "$CACHE_FILE" ]; then
  REFRESH=true
else
  # Check age — use stat in a portable way
  if [ "$(uname)" = "Darwin" ]; then
    CACHE_AGE=$(( $(date +%s) - $(stat -f %m "$CACHE_FILE" 2>/dev/null || echo 0) ))
  else
    CACHE_AGE=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0) ))
  fi
  if [ "$CACHE_AGE" -gt 300 ]; then
    REFRESH=true
  fi
fi

if [ "$REFRESH" = true ]; then
  RULES=$(curl -sf --max-time 5 \
    -H "Authorization: Bearer ${CEMS_API_KEY}" \
    "${CEMS_API_URL}/api/memory/gate-rules" 2>/dev/null || echo "")

  if [ -n "$RULES" ]; then
    # Extract and cache the rules array
    echo "$RULES" | jq -c '.rules // []' > "$CACHE_FILE" 2>/dev/null || true
  fi
fi

# If no cache file exists, allow everything
if [ ! -f "$CACHE_FILE" ]; then
  exit 0
fi

# Check rules against current tool call
# Rules format: [{"content": "Bash: coolify deploy — Block production deploys", "tags": ["block"], ...}]
RULES_JSON=$(cat "$CACHE_FILE" 2>/dev/null || echo "[]")

if [ "$RULES_JSON" = "[]" ] || [ -z "$RULES_JSON" ]; then
  exit 0
fi

# Parse each rule and check for matches
# Rule content format: "ToolName: pattern — reason"
# Uses 'as $cap' binding because capture() output loses context in nested selects
BLOCK_REASON=$(echo "$RULES_JSON" | jq -r --arg tool "$TOOL_NAME" --arg args "$ARGUMENTS" '
  .[] |
  (.content // "") |
  capture("^(?<rule_tool>\\w+):\\s*(?<pattern>.+?)\\s*(?:—|–|\\s-\\s)\\s*(?<reason>.+)$") // null |
  select(. != null) |
  . as $cap |
  select($cap.rule_tool | ascii_downcase == ($tool | ascii_downcase)) |
  select($args | ascii_downcase | contains($cap.pattern | ascii_downcase)) |
  $cap.reason
' 2>/dev/null | head -1)

if [ -n "$BLOCK_REASON" ]; then
  echo "CEMS gate rule: ${BLOCK_REASON}" >&2
  exit 1
fi

exit 0
