"""Observer Daemon — multi-tool session learning engine.

Polls active sessions from Claude Code, Codex CLI, and Cursor IDE
using the adapter pattern. Handles signal-based lifecycle (compact/stop)
from hooks, incremental observations from file growth, and staleness
detection for sessions without hooks.

Usage:
    python -m cems.observer          # Run continuously (30s interval)
    python -m cems.observer --once   # Run one cycle and exit
"""

import json
import logging
import os
import os.path
import signal
import time
import urllib.error
import urllib.request

from cems.observer.adapters import SessionAdapter, SessionInfo, get_adapters
from cems.observer.signals import Signal, clear_signal, read_signal
from cems.observer.state import (
    ObservationState,
    cleanup_old_states,
    load_state,
    save_state,
    session_tag,
)

logger = logging.getLogger(__name__)

# Two-phase threshold: cheap raw-byte pre-filter + real extracted-text gate
MIN_RAW_DELTA_BYTES = 10_000   # Cheap pre-filter: skip tiny file changes
MIN_EXTRACTED_CHARS = 3_000    # Real gate: ~750 tokens, enough for meaningful summary

# Polling interval
POLL_INTERVAL = 30  # seconds

# State cleanup frequency (in poll cycles)
CLEANUP_EVERY_N_CYCLES = 100  # ~50 minutes

# Consecutive API failure backoff
MAX_CONSECUTIVE_FAILURES = 10
BACKOFF_INTERVAL = 300  # 5 minutes between cycles when backed off

# Staleness: no file growth for 5 minutes → auto-finalize
STALE_THRESHOLD = 300  # seconds


def send_summary(
    content: str,
    session_id: str,
    source_ref: str | None,
    project_context: str | None,
    mode: str,
    epoch: int,
    api_url: str,
    api_key: str,
) -> bool:
    """Send content to CEMS for session summary extraction.

    The server maintains one document per session epoch. Incremental calls
    update it, finalize calls replace it with a comprehensive final version.

    Args:
        content: Transcript text to summarize.
        session_id: Session identifier.
        source_ref: Project reference (e.g., "project:org/repo").
        project_context: Human-readable project context.
        mode: "incremental" or "finalize".
        epoch: Epoch number for document tagging.
        api_url: CEMS server URL.
        api_key: CEMS API key.

    Returns:
        True if the API call succeeded.
    """
    payload = {
        "content": content,
        "session_id": session_id,
        "mode": mode,
        "epoch": epoch,
        "session_tag": session_tag(session_id, epoch),
    }
    if project_context:
        payload["project_context"] = project_context
    if source_ref:
        payload["source_ref"] = source_ref

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/api/session/summarize",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status == 200:
                result = json.loads(response.read())
                title = result.get("title", "unknown")
                action = result.get("action", "unknown")
                tag = session_tag(session_id, epoch)
                logger.info(f"Session summary {action} [{tag}]: {title}")
                return True
            return False

    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(f"Auth failed ({e.code}): check CEMS_API_KEY") from e
        if e.code == 404:
            raise RuntimeError("Summarize endpoint not found (404): deploy CEMS update") from e
        logger.warning(f"Summary API call failed for {session_id[:8]}: {e}")
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"Summary API call failed for {session_id[:8]}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in summary call: {e}")
        return False


def _build_project_context(session: SessionInfo) -> str | None:
    """Build human-readable project context string from session metadata."""
    project_context = session.project_id
    if not project_context and session.cwd:
        project_context = os.path.basename(session.cwd.rstrip("/"))
    if project_context:
        if session.git_branch:
            project_context += f" ({session.git_branch})"
        if session.cwd:
            project_context += f" — {session.cwd}"
    return project_context


def check_staleness(state: ObservationState) -> bool:
    """Check if a session is stale (no file growth for STALE_THRESHOLD seconds).

    Only triggers if the session has been observed at least once (to avoid
    finalizing empty sessions).
    """
    if state.observation_count == 0:
        return False  # never observed, don't finalize empty
    if state.last_growth_seen_at == 0:
        return False  # first cycle for this state
    idle = time.time() - state.last_growth_seen_at
    return idle > STALE_THRESHOLD


def handle_signal(
    sig: Signal,
    session: SessionInfo,
    state: ObservationState,
    adapter: SessionAdapter,
    api_url: str,
    api_key: str,
) -> None:
    """Handle a lifecycle signal (compact or stop).

    - compact: finalize current epoch doc (if observed), bump epoch, continue watching
    - stop: finalize current epoch doc (if observed), mark session done

    Finalization is skipped when observation_count == 0 (no content to finalize).
    """
    logger.info(
        f"Signal '{sig.type}' for {session.session_id[:8]} "
        f"(tool={sig.tool}, epoch={state.epoch})"
    )

    # Finalize current epoch if there's content to finalize
    if state.observation_count > 0:
        adapter.enrich_metadata(session)
        # Thread watermark for SQLite-based adapters (e.g., Goose)
        session.extra.setdefault("last_observed_message_id", state.last_observed_message_id)
        content = adapter.extract_text(session, state.last_observed_bytes)
        project_context = _build_project_context(session)

        # Even if no new content, send a finalize to polish the existing summary
        success = send_summary(
            content=content or "(session ended)",
            session_id=session.session_id,
            source_ref=state.source_ref or session.source_ref,
            project_context=project_context,
            mode="finalize",
            epoch=state.epoch,
            api_url=api_url,
            api_key=api_key,
        )
        if not success:
            logger.warning(f"Finalize summary failed for {session.session_id[:8]}")
        state.last_finalized_at = time.time()

        # Update bytes pointer if we read new content
        if content:
            state.last_observed_bytes = session.file_size
            # Update watermark for SQLite-based adapters
            if "last_observed_message_id" in session.extra:
                state.last_observed_message_id = session.extra["last_observed_message_id"]

    if sig.type == "compact":
        state.epoch += 1
        logger.info(f"Epoch bumped to {state.epoch} for {session.session_id[:8]}")
    elif sig.type == "stop":
        state.is_done = True
        logger.info(f"Session {session.session_id[:8]} marked done")

    save_state(state)
    clear_signal(session.session_id)


def handle_finalize(
    session: SessionInfo,
    state: ObservationState,
    adapter: SessionAdapter,
    api_url: str,
    api_key: str,
    reason: str = "staleness",
) -> None:
    """Finalize a session (from staleness or other non-signal trigger)."""
    logger.info(
        f"Auto-finalize ({reason}) for {session.session_id[:8]} "
        f"(epoch={state.epoch})"
    )

    adapter.enrich_metadata(session)
    project_context = _build_project_context(session)

    success = send_summary(
        content="(session ended — auto-finalized)",
        session_id=session.session_id,
        source_ref=state.source_ref or session.source_ref,
        project_context=project_context,
        mode="finalize",
        epoch=state.epoch,
        api_url=api_url,
        api_key=api_key,
    )
    if not success:
        logger.warning(f"Auto-finalize summary failed for {session.session_id[:8]}")

    state.is_done = True
    state.last_finalized_at = time.time()
    save_state(state)


def process_session_growth(
    session: SessionInfo,
    state: ObservationState,
    adapter: SessionAdapter,
    api_url: str,
    api_key: str,
) -> bool:
    """Check for file growth and send incremental observation if threshold met.

    Returns True if an observation was actually sent to the API.
    Updates state.last_growth_seen_at internally for staleness tracking.
    """
    delta_bytes = session.file_size - state.last_observed_bytes
    grew = delta_bytes > 0

    if grew:
        state.last_growth_seen_at = time.time()

    # Phase 1: Cheap pre-filter on raw bytes
    if delta_bytes < MIN_RAW_DELTA_BYTES:
        if grew:
            save_state(state)  # persist last_growth_seen_at
        return False

    # Enrich session with project metadata
    adapter.enrich_metadata(session)

    # Thread watermark for SQLite-based adapters (e.g., Goose)
    session.extra.setdefault("last_observed_message_id", state.last_observed_message_id)

    # Phase 2: Extract text and gate on extracted length
    content = adapter.extract_text(session, state.last_observed_bytes)
    if not content or len(content) < MIN_EXTRACTED_CHARS:
        if grew:
            save_state(state)
        return False

    project_context = _build_project_context(session)

    success = send_summary(
        content=content,
        session_id=session.session_id,
        source_ref=session.source_ref,
        project_context=project_context,
        mode="incremental",
        epoch=state.epoch,
        api_url=api_url,
        api_key=api_key,
    )

    if success:
        state.project_id = session.project_id
        state.source_ref = session.source_ref
        state.last_observed_bytes = session.file_size
        # Update watermark for SQLite-based adapters
        if "last_observed_message_id" in session.extra:
            state.last_observed_message_id = session.extra["last_observed_message_id"]
        state.last_observed_at = time.time()
        state.observation_count += 1
        save_state(state)

    return success


def run_cycle(api_url: str, api_key: str) -> int:
    """Run one observation cycle across all adapters.

    For each adapter, discovers sessions and processes them:
    1. Check signals (compact/stop) — handle lifecycle events
    2. Check file growth — send incremental observations
    3. Check staleness — auto-finalize idle sessions

    Returns:
        Number of observations triggered.
    """
    adapters = get_adapters()
    observations_triggered = 0

    for adapter in adapters:
        sessions = adapter.discover_sessions(max_age_hours=2)
        if sessions:
            logger.debug(
                f"{adapter.tool_name}: {len(sessions)} sessions "
                f"({', '.join(s.session_id[:8] for s in sessions)})"
            )

        for session in sessions:
            try:
                state = load_state(session.session_id)

                # Backfill tool name for old state files
                if not state.tool:
                    state.tool = adapter.tool_name
                    save_state(state)

                # Skip completed sessions
                if state.is_done:
                    logger.debug(
                        f"Skipping done session {session.session_id[:8]} "
                        f"(tool={adapter.tool_name})"
                    )
                    continue

                # 1. Check signals (cheap file stat)
                sig = read_signal(session.session_id)
                if sig:
                    handle_signal(sig, session, state, adapter, api_url, api_key)
                    observations_triggered += 1
                    continue

                # 2. Check file growth → incremental observation
                grew = process_session_growth(
                    session, state, adapter, api_url, api_key,
                )
                if grew:
                    observations_triggered += 1

                # 3. Check staleness → auto-finalize
                # Reload state since process_session_growth may have saved
                state = load_state(session.session_id)
                if not state.is_done and check_staleness(state):
                    handle_finalize(
                        session, state, adapter, api_url, api_key,
                        reason="staleness",
                    )

            except RuntimeError:
                raise  # Auth/404 — propagate
            except Exception as e:
                logger.error(
                    f"Error processing {adapter.tool_name} session "
                    f"{session.session_id[:8]}: {e}"
                )
                continue

    return observations_triggered


def _handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    logger.info("SIGTERM received, shutting down...")
    raise SystemExit(0)


def run_daemon(api_url: str, api_key: str) -> None:
    """Run the observer daemon continuously.

    Polls every POLL_INTERVAL seconds, processes active sessions
    across all tool adapters, and periodically cleans up old state files.
    """
    signal.signal(signal.SIGTERM, _handle_sigterm)

    adapters = get_adapters()
    adapter_names = [a.tool_name for a in adapters]

    logger.info(f"Observer daemon started (PID {os.getpid()}, polling every {POLL_INTERVAL}s)")
    logger.info(f"Adapters: {', '.join(adapter_names)}")
    logger.info(f"CEMS API: {api_url}")
    logger.info(f"Thresholds: raw={MIN_RAW_DELTA_BYTES}B, extracted={MIN_EXTRACTED_CHARS}chars")
    logger.info(f"Staleness threshold: {STALE_THRESHOLD}s")

    cycle = 0
    consecutive_failures = 0

    while True:
        try:
            triggered = run_cycle(api_url, api_key)

            if triggered > 0:
                logger.info(f"Cycle {cycle}: {triggered} observation(s) triggered")
            consecutive_failures = 0

            cycle += 1
            if cycle % CLEANUP_EVERY_N_CYCLES == 0:
                cleanup_old_states(max_age_days=7)

        except KeyboardInterrupt:
            logger.info("Observer daemon stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in observer cycle: {e}")
            consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            if consecutive_failures == MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"{consecutive_failures} consecutive failures, "
                    f"backing off to {BACKOFF_INTERVAL}s interval"
                )
            time.sleep(BACKOFF_INTERVAL)
        else:
            time.sleep(POLL_INTERVAL)

    logger.info("Observer daemon stopped")
