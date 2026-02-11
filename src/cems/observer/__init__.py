"""CEMS Observer Daemon.

Standalone daemon that watches Claude Code session transcripts and
produces high-level observations via the CEMS API.

Usage:
    python -m cems.observer          # Run daemon (polls every 30s)
    python -m cems.observer --once   # Run once and exit (for cron/testing)
"""
