"""Bounded off-loop execution for retrieval LLM calls (enrichment, agentic search).

The retrieval helpers (intent, decomposition, synthesis, HyDE) are synchronous
OpenAI SDK calls. Running them directly inside the async pipeline blocks the
server event loop; wrapping them in asyncio.wait_for(asyncio.to_thread(...))
alone is not enough because a timed-out thread keeps running, so stalls would
pile up unbounded worker threads.

EnrichmentRunner instead:
- runs calls on a dedicated pool with max_workers == capacity,
- admits a call only if a slot is free (no queueing; full -> "capacity"),
- holds the slot until the underlying call actually finishes, even when the
  awaiting request timed out or was cancelled,
- never swallows CancelledError.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

EnrichmentStatus = Literal["ok", "timeout", "failure", "capacity"]


@dataclass
class EnrichmentOutcome:
    status: EnrichmentStatus
    value: Any = None
    elapsed_ms: float = 0.0
    error: str | None = None  # exception type name only, never message/content


class EnrichmentRunner:
    """Run blocking callables off the event loop with a hard in-flight cap."""

    def __init__(self, max_in_flight: int, name: str = "enrich"):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        self.max_in_flight = max_in_flight
        self._in_flight = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_in_flight, thread_name_prefix=f"cems-{name}"
        )

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def _try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self.max_in_flight:
                return False
            self._in_flight += 1
            return True

    def _release(self, _future: Future | None = None) -> None:
        with self._lock:
            self._in_flight -= 1

    async def run(
        self, fn: Callable[..., Any], *args: Any, timeout: float, **kwargs: Any
    ) -> EnrichmentOutcome:
        """Run fn(*args, **kwargs) in the pool, waiting at most `timeout` seconds."""
        start = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - start) * 1000

        if not self._try_acquire():
            return EnrichmentOutcome("capacity", elapsed_ms=elapsed())

        try:
            cf = self._executor.submit(fn, *args, **kwargs)
        except BaseException:
            self._release()
            raise
        # Slot is freed only when the call itself ends (or is cancelled before
        # starting), never merely because the awaiting request gave up.
        cf.add_done_callback(self._release)

        try:
            # On timeout/cancellation wait_for cancels the wrapper future; the
            # wrapper then ignores the late result, so no exception is left
            # unretrieved on the event loop.
            value = await asyncio.wait_for(asyncio.wrap_future(cf), timeout=timeout)
        except TimeoutError:
            return EnrichmentOutcome("timeout", elapsed_ms=elapsed())
        except Exception as e:
            return EnrichmentOutcome("failure", elapsed_ms=elapsed(), error=type(e).__name__)
        return EnrichmentOutcome("ok", value=value, elapsed_ms=elapsed())


# Process-wide runners keyed by (pool name, capacity). Separate names keep
# long agentic calls from consuming the fast retrieval enrichment slots.
_runners: dict[tuple[str, int], EnrichmentRunner] = {}
_runners_lock = threading.Lock()


def get_enrichment_runner(max_in_flight: int, name: str = "retrieval") -> EnrichmentRunner:
    with _runners_lock:
        key = (name, max_in_flight)
        runner = _runners.get(key)
        if runner is None:
            runner = EnrichmentRunner(max_in_flight, name=name)
            _runners[key] = runner
        return runner
