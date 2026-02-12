#!/bin/bash
# Sync canonical hook sources → package data directory
# Run after modifying hooks/ to keep package data in sync.
#
# Note: Skills and Cursor hooks live only in src/cems/data/ now.
# Edit them there directly.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/src/cems/data"

echo "Syncing hooks → package data..."

# Claude hooks
cp "$REPO_ROOT"/hooks/cems_session_start.py \
   "$REPO_ROOT"/hooks/cems_user_prompts_submit.py \
   "$REPO_ROOT"/hooks/cems_post_tool_use.py \
   "$REPO_ROOT"/hooks/cems_stop.py \
   "$REPO_ROOT"/hooks/cems_pre_tool_use.py \
   "$REPO_ROOT"/hooks/cems_pre_compact.py \
   "$DATA_DIR/claude/hooks/"

cp "$REPO_ROOT"/hooks/utils/constants.py \
   "$REPO_ROOT"/hooks/utils/credentials.py \
   "$REPO_ROOT"/hooks/utils/hook_logger.py \
   "$REPO_ROOT"/hooks/utils/observer_manager.py \
   "$REPO_ROOT"/hooks/utils/transcript.py \
   "$DATA_DIR/claude/hooks/utils/"

echo "Done. Package data synced."
