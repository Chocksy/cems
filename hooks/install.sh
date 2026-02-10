#!/bin/bash
# Install CEMS hooks to ~/.claude/hooks/
# Run from repo root: ./hooks/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$HOME/.claude/hooks"

mkdir -p "$TARGET_DIR/utils"

echo "Installing CEMS hooks to $TARGET_DIR..."

# Copy hook scripts
for f in cems_session_start.py user_prompts_submit.py cems_post_tool_use.py stop.py pre_tool_use.py pre_compact.py; do
    cp "$SCRIPT_DIR/$f" "$TARGET_DIR/$f"
    chmod +x "$TARGET_DIR/$f"
    echo "  Installed $f"
done

# Copy utils
for f in constants.py transcript.py hook_logger.py; do
    cp "$SCRIPT_DIR/utils/$f" "$TARGET_DIR/utils/$f"
    echo "  Installed utils/$f"
done

echo "Done. All hooks installed."
