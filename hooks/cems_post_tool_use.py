#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS PostToolUse Hook — DISABLED

Tool learning was superseded by the observer daemon, which captures session
learnings holistically from the full session transcript.

The daemon provides better quality learnings with less noise:
- Sees the full session context, not just individual tool calls
- Extracts final decisions, not intermediate debugging steps
- Creates 5-20x fewer memories per session
- Works across all harnesses (Claude Code, Cursor, Codex, Goose)

This file is kept as a no-op for backwards compatibility with users
who haven't run `cems update` yet. Their settings.json may still
reference this script — it exits cleanly instead of erroring.

The `cems setup` migration (`_migrate_removed_hooks`) removes the
PostToolUse entry from settings.json on next update.
"""

import sys

sys.exit(0)
