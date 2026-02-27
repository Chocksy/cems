"""JSONL indexer for hook events — seek-based incremental refresh.

Reads hook_events.jsonl once on startup, builds in-memory session index.
Subsequent calls only read from byte_offset to EOF (append-only file).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

LOG_DIR = Path.home() / ".claude" / "hooks" / "logs"
EVENTS_FILE = LOG_DIR / "hook_events.jsonl"
VERBOSE_DIR = LOG_DIR / "verbose"
OBSERVER_DIR = Path.home() / ".cems" / "observer"
GATE_CACHE_DIR = Path.home() / ".cems" / "cache" / "gate_rules"


@dataclass
class SessionInfo:
    session_id: str
    first_ts: str = ""
    last_ts: str = ""
    project: str = ""
    events: list[dict] = field(default_factory=list)
    prompt_count: int = 0
    retrieval_count: int = 0
    gate_triggers: int = 0
    tool_count: int = 0
    source: str = ""  # startup, clear, resume
    injected_context: str = ""  # SessionStart output (profile + gate rules)


class EventIndex:
    """In-memory index of hook_events.jsonl with seek-based refresh."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionInfo] = {}
        self.retrievals: list[dict] = []
        self._byte_offset: int = 0
        self._event_count: int = 0

    def refresh(self) -> None:
        """Read new entries from hook_events.jsonl since last offset."""
        if not EVENTS_FILE.exists():
            return

        file_size = EVENTS_FILE.stat().st_size
        if file_size < self._byte_offset:
            # File was truncated/rotated — re-index from scratch
            self.sessions.clear()
            self.retrievals.clear()
            self._byte_offset = 0
            self._event_count = 0
        elif file_size == self._byte_offset:
            return  # No new data

        with open(EVENTS_FILE, "r") as f:
            f.seek(self._byte_offset)
            for line in f:
                self._process_line(line.strip())
            self._byte_offset = f.tell()

    def _process_line(self, line: str) -> None:
        if not line:
            return
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return

        self._event_count += 1
        event = entry.get("event", "")
        sid = entry.get("session_id", "")
        ts = entry.get("ts", "")

        if not sid:
            sid = "(no-session)"

        # Get or create session
        if sid not in self.sessions:
            self.sessions[sid] = SessionInfo(session_id=sid, first_ts=ts)

        session = self.sessions[sid]
        session.last_ts = ts

        # Store lean event reference
        session.events.append({
            "ts": ts,
            "event": event,
            "tool": entry.get("tool", ""),
            "extra": {k: v for k, v in entry.items()
                      if k not in ("ts", "event", "session_id", "verbose", "tool")},
        })

        # Update stats
        if event == "SessionStart":
            session.source = entry.get("source", "")
        elif event == "SessionStartOutput":
            # Store the injected context preview for display
            session.injected_context = entry.get("context_preview", "")
        elif event == "UserPromptSubmit":
            session.prompt_count += 1
        elif event == "MemoryRetrieval":
            session.retrieval_count += 1
            self.retrievals.append({
                "ts": ts,
                "session_id": sid,
                "query": entry.get("query", ""),
                "result_count": entry.get("result_count", 0),
                "avg_score": entry.get("avg_score", 0.0),
                "top_score": entry.get("top_score", 0.0),
                "details": entry.get("details", []),
            })
        elif event in ("Stop", "PreCompact"):
            cwd = entry.get("cwd", "")
            if cwd and not session.project:
                session.project = _project_from_cwd(cwd)
        elif event == "GateTriggered":
            session.gate_triggers += 1
        elif event == "UserPromptSubmitOutput":
            pass  # Stats only — output is in verbose log
        elif event == "PreToolUse":
            session.tool_count += 1

    def get_sessions(self, limit: int = 50) -> list[dict]:
        """Return recent sessions sorted by last_ts descending."""
        self.refresh()
        sessions = sorted(
            self.sessions.values(),
            key=lambda s: s.last_ts,
            reverse=True,
        )[:limit]
        return [
            {
                "session_id": s.session_id,
                "first_ts": s.first_ts,
                "last_ts": s.last_ts,
                "project": s.project,
                "prompt_count": s.prompt_count,
                "retrieval_count": s.retrieval_count,
                "gate_triggers": s.gate_triggers,
                "tool_count": s.tool_count,
                "event_count": len(s.events),
                "source": s.source,
            }
            for s in sessions
        ]

    def get_session_detail(self, sid: str, offset: int = 0, limit: int = 200) -> dict | None:
        """Return session detail with paginated events."""
        self.refresh()
        session = self.sessions.get(sid)
        if not session:
            return None

        total = len(session.events)
        # Reverse: newest events first
        reversed_events = list(reversed(session.events))
        paginated = reversed_events[offset:offset + limit]

        return {
            "session_id": session.session_id,
            "first_ts": session.first_ts,
            "last_ts": session.last_ts,
            "project": session.project,
            "source": session.source,
            "prompt_count": session.prompt_count,
            "retrieval_count": session.retrieval_count,
            "gate_triggers": session.gate_triggers,
            "tool_count": session.tool_count,
            "injected_context": session.injected_context,
            "events": paginated,
            "total_events": total,
            "offset": offset,
        }

    def get_session_verbose(self, sid: str) -> list[dict]:
        """Read verbose per-session log for detailed payloads."""
        if "/" in sid or "\\" in sid or ".." in sid:
            return []
        verbose_file = VERBOSE_DIR / f"{sid}.jsonl"
        if not verbose_file.exists():
            return []

        entries = []
        try:
            for line in verbose_file.read_text().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
        return entries

    def get_retrievals(self, limit: int = 50) -> list[dict]:
        """Return recent memory retrievals."""
        self.refresh()
        recent = list(self.retrievals[-limit:])
        recent.reverse()
        return recent

    # --- Observer methods ---

    def get_observer_sessions(self, limit: int = 100) -> list[dict]:
        """Read all observer state files and return sorted by last activity."""
        if not OBSERVER_DIR.exists():
            return []

        sessions = []
        for state_file in OBSERVER_DIR.glob("*.json"):
            if state_file.name == "daemon.pid":
                continue
            try:
                data = json.loads(state_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            last_activity = max(
                data.get("last_observed_at", 0),
                data.get("last_finalized_at", 0),
                data.get("session_started", 0),
            )
            sessions.append({
                "session_id": data.get("session_id", state_file.stem),
                "tool": data.get("tool", "") or "unknown",
                "project_id": data.get("project_id", ""),
                "source_ref": data.get("source_ref", ""),
                "observation_count": data.get("observation_count", 0),
                "epoch": data.get("epoch", 0),
                "is_done": data.get("is_done", False),
                "session_started": data.get("session_started", 0),
                "last_observed_at": data.get("last_observed_at", 0),
                "last_finalized_at": data.get("last_finalized_at", 0),
                "last_growth_seen_at": data.get("last_growth_seen_at", 0),
                "last_observed_bytes": data.get("last_observed_bytes", 0),
                "last_activity": last_activity,
            })

        sessions.sort(key=lambda s: s["last_activity"], reverse=True)
        return sessions[:limit]

    def get_observer_session_detail(self, sid: str) -> dict | None:
        """Return full state dict + last 50 daemon log lines for a session."""
        if "/" in sid or "\\" in sid or ".." in sid:
            return None

        state_file = OBSERVER_DIR / f"{sid}.json"
        if not state_file.exists():
            return None

        try:
            data = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        data.setdefault("tool", "unknown")
        if not data["tool"]:
            data["tool"] = "unknown"

        # Grep daemon log for matching lines
        log_lines: list[str] = []
        daemon_log = OBSERVER_DIR / "daemon.log"
        short_id = sid[:8]
        if daemon_log.exists():
            try:
                for line in daemon_log.read_text().splitlines():
                    if short_id in line:
                        log_lines.append(line)
                log_lines = log_lines[-50:]
            except OSError:
                pass

        data["daemon_log"] = log_lines
        return data

    def get_observer_stats(self) -> dict:
        """Aggregate observer stats: totals, by-tool breakdown, pending signals."""
        sessions = self.get_observer_sessions(limit=9999)

        by_tool: dict[str, int] = {}
        active = 0
        done = 0
        total_observations = 0

        for s in sessions:
            tool = s.get("tool", "unknown") or "unknown"
            by_tool[tool] = by_tool.get(tool, 0) + 1
            if s.get("is_done"):
                done += 1
            else:
                active += 1
            total_observations += s.get("observation_count", 0)

        # Count pending signals
        signals_dir = OBSERVER_DIR / "signals"
        pending_signals = _count_files(signals_dir, "*.json") if signals_dir.exists() else 0

        return {
            "total": len(sessions),
            "active": active,
            "done": done,
            "by_tool": by_tool,
            "total_observations": total_observations,
            "pending_signals": pending_signals,
        }

    def get_status(self) -> dict:
        """Return system status info — file sizes, daemon, caches."""
        self.refresh()
        status: dict = {
            "events_file": str(EVENTS_FILE),
            "events_file_size": _file_size(EVENTS_FILE),
            "events_count": self._event_count,
            "session_count": len(self.sessions),
            "retrieval_count": len(self.retrievals),
            "verbose_dir": str(VERBOSE_DIR),
            "verbose_files": _count_files(VERBOSE_DIR, "*.jsonl"),
        }

        # Observer daemon
        pid_file = OBSERVER_DIR / "daemon.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                status["daemon_pid"] = pid
                status["daemon_running"] = _pid_alive(pid)
            except (ValueError, OSError):
                status["daemon_running"] = False
        else:
            status["daemon_running"] = False

        # Observer state files
        status["observer_sessions"] = _count_files(OBSERVER_DIR, "*.json")

        # Gate rules cache
        status["gate_rules"] = _list_gate_caches()

        # Daemon log tail
        daemon_log = OBSERVER_DIR / "daemon.log"
        if daemon_log.exists():
            try:
                lines = daemon_log.read_text().splitlines()
                status["daemon_log_tail"] = lines[-10:]
            except OSError:
                status["daemon_log_tail"] = []

        return status


def _project_from_cwd(cwd: str) -> str:
    """Extract project name from cwd path like /Users/x/Development/foo."""
    parts = Path(cwd).parts
    # Look for Development/ or similar parent
    for i, part in enumerate(parts):
        if part.lower() in ("development", "projects", "repos", "src", "code"):
            if i + 1 < len(parts):
                return parts[i + 1]
    # Fallback: last directory component
    return parts[-1] if parts else ""


def _file_size(path: Path) -> str:
    """Human-readable file size."""
    if not path.exists():
        return "0"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def _pid_alive(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _list_gate_caches() -> list[dict]:
    """List gate rule cache files with their rule counts."""
    if not GATE_CACHE_DIR.exists():
        return []
    caches = []
    for f in GATE_CACHE_DIR.glob("*.json"):
        try:
            rules = json.loads(f.read_text())
            caches.append({
                "project": f.stem.replace("_", "/"),
                "rule_count": len(rules) if isinstance(rules, list) else 0,
            })
        except (json.JSONDecodeError, OSError):
            pass
    return caches
