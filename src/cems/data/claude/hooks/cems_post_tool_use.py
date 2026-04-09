#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
CEMS PostToolUse Hook — DISABLED

Tool learning was superseded by the observer daemon, which captures session
learnings holistically from the full session transcript.

This file is kept as a no-op for backwards compatibility.
See hooks/cems_post_tool_use.py for full explanation.
"""

import sys

sys.exit(0)
