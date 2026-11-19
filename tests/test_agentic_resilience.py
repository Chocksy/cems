"""Resilience tests for agentic_search_async with stalled/failing agents.

LLM and document store are local fakes; no network or database.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock

import httpx
import pytest

import cems.agentic.search as agentic
import cems.llm.client
import cems.memory.enrichment as enrichment_mod
from cems.agentic.search import AgenticSearchError, agentic_search_async

MEMORY_ID = "abc12345-0000-0000-0000-000000000000"
ENTITY = {"id": "ent12345-0000", "title": "Payroll", "summary": "approvals", "sources": "3"}


class StubAgentLLM:
    """Agent LLM stub; can stall, fail, or fail only the memory agent."""

    def __init__(self, response: str = '["abc12345"]'):
        self.response = response
        self.stalling = False
        self.stall = threading.Event()
        self.error: Exception | None = None
        self.fail_memory = False
        self.started = 0
        self._lock = threading.Lock()

    def complete(self, prompt, system=None, **kwargs):
        with self._lock:
            self.started += 1
        if self.stalling:
            self.stall.wait(timeout=10)
        if self.error:
            raise self.error
        if self.fail_memory and prompt.startswith(agentic.MERGED_ROLE_INSTRUCTIONS):
            raise RuntimeError("provider said: sk-or-SECRET")
        return self.response


def _store() -> AsyncMock:
    store = AsyncMock()
    store.get_all_documents = AsyncMock(return_value=[{
        "id": MEMORY_ID,
        "content": "Timesheet approvals live in the payroll repo",
        "category": "general",
        "source_ref": "project:alpha",
        "created_at": "2026-10-01",
        "scope": "personal",
        "tags": [],
    }])
    return store


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(enrichment_mod, "_runners", {})
    monkeypatch.setattr(cems.llm.client, "_client", None)
    monkeypatch.setattr(cems.llm.client, "_retrieval_clients", {})
    monkeypatch.setattr(agentic, "_load_entity_summaries", AsyncMock(return_value=[]))
    monkeypatch.setattr(agentic, "AGENT_TIMEOUT_SECONDS", 0.2)
    yield
    for runner in enrichment_mod._runners.values():
        runner._executor.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def llm(monkeypatch):
    stub = StubAgentLLM()
    monkeypatch.setattr(agentic, "get_client", lambda: stub)
    yield stub
    stub.stall.set()


def _search(query: str = "where are timesheet approvals?"):
    return agentic_search_async(
        document_store=_store(), user_id="user-1", query=query, project="alpha"
    )


def _agent_runner() -> enrichment_mod.EnrichmentRunner:
    return enrichment_mod.get_enrichment_runner(agentic.AGENT_MAX_IN_FLIGHT, name="agentic")


async def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


class TestStalledAgents:
    async def test_stalled_agents_outlive_request_without_blocking_loop(self, llm):
        """Old code: leaving `with ThreadPoolExecutor` waited for workers on the loop."""
        llm.stalling = True
        threading.Timer(1.0, llm.stall.set).start()  # agents finish long after the deadline

        gaps: list[float] = []
        stop = asyncio.Event()

        async def heartbeat():
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        hb = asyncio.create_task(heartbeat())
        start = time.perf_counter()
        with pytest.raises(AgenticSearchError, match="memory_seeker=timeout"):
            await _search()
        elapsed = time.perf_counter() - start

        # Ordinary async work still runs while the stalled agents hold their slots
        ordinary = await asyncio.wait_for(asyncio.sleep(0, result="ok"), timeout=0.1)
        stop.set()
        await hb

        assert elapsed < 0.6
        assert ordinary == "ok"
        assert max(gaps) < 0.2, f"event loop stalled for {max(gaps):.3f}s"
        runner = _agent_runner()
        assert runner.in_flight == 1  # still running after the request gave up
        await _wait_until(lambda: runner.in_flight == 0)

    async def test_total_provider_failure_is_an_error_not_empty_success(self, llm):
        llm.error = httpx.ConnectTimeout("connect")
        with pytest.raises(AgenticSearchError) as exc:
            await _search()
        assert "memory_seeker=failure" in str(exc.value)

    async def test_all_agents_complete_with_nothing_is_genuine_empty(self, llm):
        llm.response = "[]"
        result = await _search()
        assert result["count"] == 0
        assert result["partial"] is False
        assert result["degraded_agents"] == []


class TestPartialResults:
    async def test_partial_success_returns_rankings_with_degradation(self, llm, monkeypatch, caplog):
        monkeypatch.setattr(agentic, "_load_entity_summaries", AsyncMock(return_value=[ENTITY]))
        llm.response = '["ent12345"]'
        llm.fail_memory = True
        result = await _search()
        assert result["partial"] is True
        assert [d["agent"] for d in result["degraded_agents"]] == ["memory_seeker"]
        assert result["degraded_agents"][0]["reason"] == "failure"
        assert result["memories"] == []
        assert [e["title"] for e in result["entities"]] == ["Payroll"]
        assert "agent=memory_seeker reason=failure" in caplog.text
        assert "error=RuntimeError" in caplog.text
        assert "sk-or-SECRET" not in caplog.text


class TestCancellationAndCapacity:
    async def test_cancellation_propagates_and_slots_return_on_completion(self, llm, monkeypatch):
        llm.stalling = True
        monkeypatch.setattr(agentic, "AGENT_TIMEOUT_SECONDS", 5.0)
        task = asyncio.create_task(_search())
        await _wait_until(lambda: llm.started == 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        runner = _agent_runner()
        assert runner.in_flight == 1
        llm.stall.set()
        await _wait_until(lambda: runner.in_flight == 0)
        llm.stalling = False
        result = await _search()
        assert result["count"] == 1

    async def test_repeated_requests_cannot_grow_pools_or_work(self, llm, monkeypatch):
        monkeypatch.setattr(agentic, "AGENT_MAX_IN_FLIGHT", 1)
        llm.stalling = True
        threads_before = threading.active_count()

        outcomes = await asyncio.gather(*(_search() for _ in range(6)), return_exceptions=True)
        assert all(isinstance(o, AgenticSearchError) for o in outcomes)
        assert sum("capacity" in str(o) for o in outcomes) >= 5

        runner = _agent_runner()
        assert llm.started == 1
        assert runner.in_flight == 1
        assert runner._executor._work_queue.qsize() == 0
        assert len(runner._executor._threads) <= 1
        assert threading.active_count() - threads_before <= 1

        # Saturated requests fail fast instead of waiting out the budget
        start = time.perf_counter()
        with pytest.raises(AgenticSearchError, match="capacity"):
            await _search()
        assert time.perf_counter() - start < 0.1

        llm.stall.set()
        await _wait_until(lambda: runner.in_flight == 0)
        llm.stalling = False
        assert (await _search())["count"] == 1


class TestAgentClient:
    def test_agent_client_bounded_without_touching_shared_client(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        base = cems.llm.client.get_client()
        client = agentic.get_client()
        assert client is not base
        assert client._client.max_retries == 0
        budget = agentic.AGENT_TIMEOUT_SECONDS
        assert client._client.timeout == httpx.Timeout(budget, connect=min(budget, 1.0))
        assert client._max_tokens_cap is None
        assert base._client.max_retries == 2


class TestEmptyCompletions:
    """Empty provider output is an agent error; an explicit "[]" is a no-match."""

    @pytest.mark.parametrize("empty", ["", "   \n\t "])
    def test_helpers_raise_on_empty_completion(self, llm, empty):
        llm.response = empty
        with pytest.raises(agentic.AgentEmptyResponseError) as exc:
            agentic._run_single_agent("q", "mem text", 1, "m")
        assert str(exc.value) == "memory_seeker returned an empty completion"
        with pytest.raises(agentic.AgentEmptyResponseError):
            agentic._run_entity_picker("q", "entity text", 1, "m")

    def test_helpers_keep_explicit_empty_selection(self, llm):
        llm.response = "[]"
        assert agentic._run_single_agent("q", "mem text", 1, "m") == ("memory_seeker", "[]")
        assert agentic._run_entity_picker("q", "entity text", 1, "m") == ("entity_picker", "[]")

    @pytest.mark.parametrize("empty", ["", "  \n "])
    async def test_all_empty_completions_fail_instead_of_count_zero(self, llm, monkeypatch, caplog, empty):
        monkeypatch.setattr(agentic, "_load_entity_summaries", AsyncMock(return_value=[ENTITY]))
        llm.response = empty
        with pytest.raises(AgenticSearchError) as exc:
            await _search()
        message = str(exc.value)
        for role in ("entity_picker", "memory_seeker"):
            assert f"{role}=failure" in message
        assert llm.started == 2
        assert "error=AgentEmptyResponseError" in caplog.text

    async def test_empty_completion_alongside_ranking_is_partial(self, llm, monkeypatch):
        monkeypatch.setattr(agentic, "_load_entity_summaries", AsyncMock(return_value=[ENTITY]))

        def complete(prompt, system=None, **kwargs):
            return "" if system == agentic.ENTITY_PICKER_SYSTEM else '["abc12345"]'

        monkeypatch.setattr(llm, "complete", complete)
        result = await _search()
        assert result["partial"] is True
        assert [d["agent"] for d in result["degraded_agents"]] == ["entity_picker"]
        assert [d["reason"] for d in result["degraded_agents"]] == ["failure"]
        assert [m["memory_id"] for m in result["memories"]] == [MEMORY_ID]
