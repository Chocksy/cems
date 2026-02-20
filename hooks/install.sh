#!/bin/bash
# Install CEMS hooks from source tree (for developers).
#
# For non-dev users, use: cems setup --claude
# Or the remote installer: curl -fsSL https://getcems.com/install.sh | bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$HOME/.claude/hooks"

mkdir -p "$TARGET_DIR/utils"

echo "Installing CEMS hooks to $TARGET_DIR..."

for f in cems_session_start.py cems_user_prompts_submit.py cems_post_tool_use.py cems_stop.py cems_pre_tool_use.py cems_pre_compact.py; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$TARGET_DIR/$f"
        chmod +x "$TARGET_DIR/$f"
    fi
done

for f in constants.py transcript.py hook_logger.py observer_manager.py credentials.py; do
    if [ -f "$SCRIPT_DIR/utils/$f" ]; then
        cp "$SCRIPT_DIR/utils/$f" "$TARGET_DIR/utils/$f"
    fi
done

echo "Done. Hooks installed to $TARGET_DIR"
