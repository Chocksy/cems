"""Observer Daemon — watches sessions and produces observations.

Polls active Claude Code session files and sends new content to
the CEMS /api/session/observe endpoint for observation extraction.

Usage:
    python -m cems.observer          # Run continuously (30s interval)
    python -m cems.observer --once   # Run one cycle and exit
"""

import json
import logging
import os
import signal
import time
import urllib.error
import urllib.request

from cems.observer.session import (
    SessionInfo,
    discover_active_sessions,
    enrich_session_metadata,
    read_content_delta,
)
from cems.observer.state import (
    ObservationState,
    cleanup_old_states,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)

# Minimum new content (bytes) before triggering observation
OBSERVATION_THRESHOLD = 50_000  # ~12-15k tokens

# Polling interval
POLL_INTERVAL = 30  # seconds

# State cleanup frequency (in poll cycles)
CLEANUP_EVERY_N_CYCLES = 100  # ~50 minutes

# Consecutive API failure backoff
MAX_CONSECUTIVE_FAILURES = 10  # After this many, increase sleep interval
BACKOFF_INTERVAL = 300  # 5 minutes between cycles when backed off



def send_observation(
    content: str,
    session_id: str,
    source_ref: str | None,
    project_context: str,
    api_url: str,
    api_key: str,
) -> bool:
    """Send content to CEMS for observation extraction.

    Args:
        content: Transcript text to observe.
        session_id: Session identifier.
        source_ref: Project reference (e.g., "project:org/repo").
        project_context: Human-readable project context.
        api_url: CEMS server URL.
        api_key: CEMS API key.

    Returns:
        True if the API call succeeded.
    """
    payload = {
        "content": content,
        "session_id": session_id,
        "project_context": project_context,
    }
    if source_ref:
        payload["source_ref"] = source_ref

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/api/session/observe",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                result = json.loads(response.read())
                stored = result.get("observations_stored", 0)
                logger.info(f"Stored {stored} observations for session {session_id[:8]}")
                return True
            return False

    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            # Auth errors should propagate — no point retrying every 30s
            raise RuntimeError(f"Auth failed ({e.code}): check CEMS_API_KEY") from e
        if e.code == 404:
            raise RuntimeError("Observe endpoint not found (404): deploy CEMS update") from e
        logger.warning(f"Observation API call failed for {session_id[:8]}: {e}")
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"Observation API call failed for {session_id[:8]}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in observation call: {e}")
        return False


def process_session(session: SessionInfo, api_url: str, api_key: str) -> bool:
    """Process a single session: check for new content, send observation if threshold met.

    Args:
        session: Session to process.
        api_url: CEMS server URL.
        api_key: CEMS API key.

    Returns:
        True if an observation was triggered.
    """
    state = load_state(session.session_id)

    # Check if enough new content has accumulated
    delta_bytes = session.file_size - state.last_observed_bytes
    if delta_bytes < OBSERVATION_THRESHOLD:
        return False

    # Enrich session with project metadata
    enrich_session_metadata(session)

    # Read the new content
    content = read_content_delta(session, state.last_observed_bytes)
    if not content or len(content) < 200:
        return False

    # Build project context string
    project_context = session.project_id or "unknown project"
    if session.git_branch:
        project_context += f" ({session.git_branch})"
    if session.cwd:
        project_context += f" — {session.cwd}"

    # Send to CEMS
    success = send_observation(
        content=content,
        session_id=session.session_id,
        source_ref=session.source_ref,
        project_context=project_context,
        api_url=api_url,
        api_key=api_key,
    )

    if success:
        # Update state
        state.project_id = session.project_id
        state.source_ref = session.source_ref
        state.last_observed_bytes = session.file_size
        state.last_observed_at = time.time()
        state.observation_count += 1
        save_state(state)

    return success


def run_cycle(api_url: str, api_key: str) -> int:
    """Run one observation cycle: discover sessions, process new content.

    Args:
        api_url: CEMS server URL.
        api_key: CEMS API key.

    Returns:
        Number of observations triggered.
    """
    sessions = discover_active_sessions(max_age_hours=2)
    if not sessions:
        return 0

    observations_triggered = 0
    for session in sessions:
        try:
            if process_session(session, api_url, api_key):
                observations_triggered += 1
        except RuntimeError as e:
            # Auth/404 errors — propagate so daemon backs off
            logger.error(f"Error processing session {session.session_id[:8]}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing session {session.session_id[:8]}: {e}")
            continue

    return observations_triggered


def _handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful shutdown.

    Raises SystemExit to interrupt time.sleep() and exit immediately
    rather than waiting for the next cycle.
    """
    logger.info("SIGTERM received, shutting down...")
    raise SystemExit(0)


def run_daemon(api_url: str, api_key: str) -> None:
    """Run the observer daemon continuously.

    Polls every POLL_INTERVAL seconds, processes active sessions,
    and periodically cleans up old state files. Backs off on
    consecutive API failures to avoid hammering a broken server.

    Args:
        api_url: CEMS server URL.
        api_key: CEMS API key.
    """
    # Register SIGTERM handler for clean shutdown (e.g., kill <pid>)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    logger.info(f"Observer daemon started (PID {os.getpid()}, polling every {POLL_INTERVAL}s)")
    logger.info(f"CEMS API: {api_url}")
    logger.info(f"Observation threshold: {OBSERVATION_THRESHOLD} bytes")

    cycle = 0
    consecutive_failures = 0

    while True:
        try:
            triggered = run_cycle(api_url, api_key)

            if triggered > 0:
                logger.info(f"Cycle {cycle}: {triggered} observation(s) triggered")
                consecutive_failures = 0  # Reset on success
            elif triggered == 0:
                # No observations triggered but no errors — reset failures
                consecutive_failures = 0

            # Periodic cleanup
            cycle += 1
            if cycle % CLEANUP_EVERY_N_CYCLES == 0:
                cleanup_old_states(max_age_days=7)

        except KeyboardInterrupt:
            logger.info("Observer daemon stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in observer cycle: {e}")
            consecutive_failures += 1

        # Back off if too many consecutive failures
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
