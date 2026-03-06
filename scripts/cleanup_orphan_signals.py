#!/usr/bin/env python3
"""One-time cleanup of orphaned observer signal files.

Signals accumulate when:
1. The daemon marks a session done (staleness) before the stop signal arrives
2. State files get cleaned up (7-day TTL) but signal files are left behind

This script removes all signal files that are either:
- Have no matching active state file (already cleaned up)
- Match a state file that is already is_done=True

Usage:
    python scripts/cleanup_orphan_signals.py         # Dry run
    python scripts/cleanup_orphan_signals.py --apply  # Actually delete
"""

import json
import sys
from pathlib import Path

SIGNALS_DIR = Path.home() / ".cems" / "observer" / "signals"
STATE_DIR = Path.home() / ".cems" / "observer"


def main():
    apply = "--apply" in sys.argv

    if not SIGNALS_DIR.exists():
        print("No signals directory found.")
        return

    signal_files = list(SIGNALS_DIR.glob("*.json"))
    print(f"Found {len(signal_files)} signal files")

    to_remove = []
    for sf in signal_files:
        session_id = sf.stem
        state_file = STATE_DIR / f"{session_id}.json"

        reason = None
        if not state_file.exists():
            reason = "no state file (already cleaned up)"
        else:
            try:
                state = json.loads(state_file.read_text())
                if state.get("is_done", False):
                    reason = "state is_done=True"
            except (json.JSONDecodeError, OSError):
                reason = "corrupt state file"

        if reason:
            signal_data = "?"
            try:
                signal_data = json.loads(sf.read_text()).get("type", "?")
            except (json.JSONDecodeError, OSError):
                pass
            to_remove.append((sf, reason, signal_data))

    print(f"\nWill remove {len(to_remove)} orphaned signals:")
    for sf, reason, sig_type in to_remove:
        print(f"  {sf.stem[:12]}... ({sig_type}) - {reason}")

    if apply:
        removed = 0
        for sf, _, _ in to_remove:
            try:
                sf.unlink()
                removed += 1
            except OSError as e:
                print(f"  Failed to remove {sf}: {e}")
        print(f"\nRemoved {removed} signal files.")
    else:
        print(f"\nDry run. Pass --apply to delete.")


if __name__ == "__main__":
    main()
