#!/usr/bin/env python3
"""Constants and utilities for Claude hooks."""

import os
from pathlib import Path


def ensure_session_log_dir(session_id: str = "default") -> Path:
    """
    Ensure session log directory exists and return its path.
    
    Args:
        session_id: Session identifier for organizing logs
        
    Returns:
        Path: Path to the session log directory
    """
    # Use current working directory as base
    base_dir = Path.cwd() / "logs" / "sessions" / session_id
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_current_session_id() -> str:
    """
    Get current session ID from environment or use default.
    
    Returns:
        str: Session identifier
    """
    return os.getenv('CLAUDE_SESSION_ID', 'default')