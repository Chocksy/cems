"""Resilience tests for retrieve_for_inference_async under slow/failed LLM enrichment.

All network/database access is faked: a minimal RetrievalMixin host supplies
canned search results, and the retrieval LLM client is a local stub.
"""

import asyncio
import gc
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

import cems.llm
import cems.llm.client
import cems.memory.enrichment as enrichment_mod
from cems.config import CEMSConfig
from cems.memory.enrichment import EnrichmentRunner
from cems.memory.retrieval import RetrievalMixin
from cems.models import MemoryMetadata, MemoryScope, SearchResult

TEMPORAL_Q = "did we talk about timesheet approvals before?"
PREFERENCE_Q = "can you recommend a good editor setup for me?"
AGGREGATION_Q = "how many deploys did we do in total?"
MEMORY_TEXT = "SECRET-MEMORY-BODY timesheet approvals are handled in the payroll repo"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _result(memory_id: str, score: float, source_ref: str, content: str = MEMORY_TEXT) -> SearchResult:
    return SearchResult(
        memory_id=memory_id,
        content=content,
        score=score,
        scope=MemoryScope.PERSONAL,
        metadata=MemoryMetadata(
            memory_id=memory_id,
            user_id="u1",
            scope=MemoryScope.PERSONAL,
            source_ref=source_ref,
        ),
    )


def _canned_results() -> list[SearchResult]:
    return [
        _result("beta-0000-0000", 0.72, "project:beta", "Beta team approves timesheets on Fridays"),
        _result("alpha-000-0000", 0.70, "project:alpha", MEMORY_TEXT),
        # below relevance threshold
        _result("noise-000-0000", 0.20, "project:alpha", "Unrelated note about lunch orders"),
        _result("noise-111-0000", 0.20, "project:alpha", "Unrelated note about parking"),
        _result("noise-222-0000", 0.20, "project:alpha", "Unrelated note about office plants"),
    ]


class FakeEmbedder:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return [[0.1, 0.2] for _ in texts]


class FakeMemory(RetrievalMixin):
    """Just enough of CEMSMemory for the retrieval pipeline; no DB."""

    def __init__(self, config: CEMSConfig, db_error: Exception | None = None, embed_error: Exception | None = None):
        self.config = config
        self._async_embedder = FakeEmbedder(embed_error)
        self.db_error = db_error
        self.vector_queries: list[str] = []

    async def _ensure_initialized_async(self) -> None:
        return None

    async def _search_raw_async(self, query, scope="both", category=None, limit=5, query_embedding=None):
        if self.db_error:
            raise self.db_error
        self.vector_queries.append(query)
        return _canned_results()

    async def _search_lexical_raw_async(self, query, scope="both", limit=5):
        if self.db_error:
            raise self.db_error
        return []

    async def get_related_memories_async(self, memory_id, limit=10):
        return []

    async def get_metadata_async(self, memory_id):
        return None


class StubLLM:
    """Retrieval LLM stub. Blocks while `stall` is clear when `stalling` is set."""

    def __init__(self, response: str = "term one\nterm two\nterm three", error: Exception | None = None):
        self.response = response
        self.error = error
        self.stalling = False
        self.stall = threading.Event()
        self.started = 0
        self.finished = 0
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def complete(self, prompt, **kwargs):
        with self._lock:
            self.started += 1
            self.prompts.append(prompt)
        try:
            if self.stalling:
                self.stall.wait(timeout=10)
            if self.error:
                raise self.error
            return self.response
        finally:
            with self._lock:
                self.finished += 1


def _config(**overrides) -> CEMSConfig:
    base = {
        "database_url": "postgresql://unused/unused",
        "retrieval_llm_budget_seconds": 2.0,
        "retrieval_llm_max_concurrency": 2,
    }
    base.update(overrides)
    return CEMSConfig(**base)


@pytest.fixture(autouse=True)
def fresh_runtime(monkeypatch):
    """Isolate the process-wide runner/client caches per test."""
    monkeypatch.setattr(enrichment_mod, "_runners", {})
    monkeypatch.setattr(cems.llm.client, "_client", None)
    monkeypatch.setattr(cems.llm.client, "_retrieval_clients", {})
    yield
    for runner in enrichment_mod._runners.values():
        runner._executor.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def llm(monkeypatch):
    stub = StubLLM()
    factory = MagicMock(return_value=stub)
    monkeypatch.setattr(cems.llm, "get_retrieval_client", factory)
    stub.factory = factory
    yield stub
    stub.stall.set()  # never leave worker threads blocked


async def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def _runner(config: CEMSConfig) -> EnrichmentRunner:
    return enrichment_mod.get_enrichment_runner(config.retrieval_llm_max_concurrency)


# ---------------------------------------------------------------------------
# 1. Opt-out flags are honoured
# ---------------------------------------------------------------------------


class TestOptOut:
    @pytest.mark.parametrize("query", [TEMPORAL_Q, PREFERENCE_Q, AGGREGATION_Q])
    async def test_opt_out_makes_no_llm_calls_and_keeps_ranking(self, query, monkeypatch):
        get_retrieval_client = MagicMock(side_effect=AssertionError("retrieval LLM client requested"))
        get_client = MagicMock(side_effect=AssertionError("shared LLM client requested"))
        monkeypatch.setattr(cems.llm, "get_retrieval_client", get_retrieval_client)
        monkeypatch.setattr(cems.llm.client, "get_client", get_client)

        memory = FakeMemory(_config())
        result = await memory.retrieve_for_inference_async(
            query,
            mode="vector",
            project="alpha",
            enable_query_synthesis=False,
            enable_hyde=False,
            enable_decomposition=False,
        )

        get_retrieval_client.assert_not_called()
        get_client.assert_not_called()
        assert result["queries_used"] == [query]
        assert memory.vector_queries == [query]  # no preference profile probe either
        assert result["degraded_enrichment"] == []
        ids = [r["memory_id"] for r in result["results"]]
        # Project boost puts the same-project memory first; noise is filtered out
        assert ids[0] == "alpha-000-0000"
        assert "noise-000-0000" not in ids
        assert result["results"][0]["source_ref"] == "project:alpha"
        assert result["results"][0]["scope"] == "personal"

    async def test_server_default_alone_cannot_block_forced_synthesis(self, llm):
        """Request opt-in + server default off still forces synthesis for temporal queries."""
        memory = FakeMemory(_config(enable_query_synthesis=False))
        result = await memory.retrieve_for_inference_async(
            TEMPORAL_Q, mode="vector", enable_query_synthesis=True, enable_hyde=False
        )
        assert llm.started == 1
        assert len(result["queries_used"]) == 4

    async def test_missing_api_key_degrades_instead_of_failing(self, monkeypatch):
        memory = FakeMemory(_config())
        result = await memory.retrieve_for_inference_async(
            TEMPORAL_Q, mode="vector", enable_query_synthesis=True
        )
        assert result["queries_used"] == [TEMPORAL_Q]
        assert result["degraded_enrichment"][0]["reason"] == "unavailable"
        assert result["results"]


# ---------------------------------------------------------------------------
# 2. Enrichment does not block the event loop
# ---------------------------------------------------------------------------


class TestNonBlocking:
    async def test_blocked_enrichment_keeps_loop_and_other_searches_live(self, llm):
        llm.stalling = True
        config = _config(retrieval_llm_budget_seconds=1.0)
        slow_memory = FakeMemory(config)
        fast_memory = FakeMemory(config)

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
        slow = asyncio.create_task(
            slow_memory.retrieve_for_inference_async(TEMPORAL_Q, mode="vector", enable_query_synthesis=True)
        )
        await _wait_until(lambda: llm.started == 1)

        fast_start = time.perf_counter()
        fast = await fast_memory.retrieve_for_inference_async(
            TEMPORAL_Q, mode="vector", enable_query_synthesis=False, enable_hyde=False
        )
        fast_elapsed = time.perf_counter() - fast_start

        assert not slow.done(), "slow request should still be waiting on enrichment"
        assert fast_elapsed < 0.3
        assert fast["results"]

        slow_result = await slow
        stop.set()
        await hb
        assert max(gaps) < 0.2, f"event loop stalled for {max(gaps):.3f}s"
        assert slow_result["queries_used"] == [TEMPORAL_Q]


# ---------------------------------------------------------------------------
# 3. Budget expiry falls back to the original query
# ---------------------------------------------------------------------------


class TestBudget:
    async def test_budget_expiry_returns_original_query_results(self, llm, caplog):
        llm.stalling = True
        memory = FakeMemory(_config(retrieval_llm_budget_seconds=0.2))
        caplog.set_level(logging.INFO, logger="cems.memory.retrieval")

        start = time.perf_counter()
        result = await memory.retrieve_for_inference_async(
            TEMPORAL_Q, mode="vector", project="alpha", enable_query_synthesis=True
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0
        assert result["queries_used"] == [TEMPORAL_Q]
        assert [r["memory_id"] for r in result["results"]][0] == "alpha-000-0000"
        assert result["degraded_enrichment"] == [
            {"stage": "synthesis", "reason": "timeout", "elapsed_ms": result["degraded_enrichment"][0]["elapsed_ms"]}
        ]
        degraded_logs = [r.getMessage() for r in caplog.records if "Enrichment degraded" in r.getMessage()]
        assert len(degraded_logs) == 1
        assert "stage=synthesis reason=timeout elapsed_ms=" in degraded_logs[0]
        assert "PIPELINE TOTAL" in caplog.text
        assert "degraded_enrichment=synthesis" in caplog.text

    async def test_budget_is_shared_across_stages(self, llm):
        """A stage that eats the budget leaves no time for later stages."""
        llm.stalling = True
        memory = FakeMemory(_config(retrieval_llm_budget_seconds=0.2))
        result = await memory.retrieve_for_inference_async(
            PREFERENCE_Q, mode="vector", enable_query_synthesis=True, enable_hyde=True
        )
        reasons = {d["stage"]: d["reason"] for d in result["degraded_enrichment"]}
        assert reasons == {"synthesis": "timeout", "hyde": "budget"}
        assert llm.started == 1
        assert result["queries_used"] == [PREFERENCE_Q]

    async def test_provider_failure_is_logged_without_content(self, llm, caplog):
        llm.error = RuntimeError("provider said: sk-or-SECRET")
        memory = FakeMemory(_config())
        caplog.set_level(logging.INFO, logger="cems.memory.retrieval")
        result = await memory.retrieve_for_inference_async(
            TEMPORAL_Q, mode="vector", enable_query_synthesis=True
        )
        assert result["queries_used"] == [TEMPORAL_Q]
        assert result["results"]
        assert result["degraded_enrichment"][0]["reason"] == "failure"
        line = next(r.getMessage() for r in caplog.records if "Enrichment degraded" in r.getMessage())
        assert "stage=synthesis reason=failure" in line
        assert "error=RuntimeError" in line
        assert "sk-or-SECRET" not in line
        assert "SECRET-MEMORY-BODY" not in caplog.text


# ---------------------------------------------------------------------------
# 4. Bounded capacity under stalls
# ---------------------------------------------------------------------------


class TestCapacity:
    async def test_stalled_calls_cannot_grow_unbounded(self, llm):
        llm.stalling = True
        config = _config(retrieval_llm_budget_seconds=0.3, retrieval_llm_max_concurrency=2)
        memory = FakeMemory(config)

        def request():
            return memory.retrieve_for_inference_async(TEMPORAL_Q, mode="vector", enable_query_synthesis=True)

        results = await asyncio.gather(*(request() for _ in range(6)))
        runner = _runner(config)

        reasons = sorted(r["degraded_enrichment"][0]["reason"] for r in results)
        assert reasons == ["capacity"] * 4 + ["timeout"] * 2
        assert all(r["queries_used"] == [TEMPORAL_Q] and r["results"] for r in results)
        # Timed-out calls still hold their slots: nothing new starts, nothing queues
        assert llm.started == 2
        assert runner.in_flight == 2
        assert runner._executor._work_queue.qsize() == 0
        assert len(runner._executor._threads) <= 2

        # Saturated requests fall back immediately rather than waiting the budget
        start = time.perf_counter()
        more = await asyncio.gather(*(request() for _ in range(10)))
        assert time.perf_counter() - start < 0.2
        assert all(r["degraded_enrichment"][0]["reason"] == "capacity" for r in more)
        assert llm.started == 2

        # Underlying completion restores capacity
        llm.stall.set()
        await _wait_until(lambda: runner.in_flight == 0)
        llm.stalling = False
        ok = await request()
        assert ok["degraded_enrichment"] == []
        assert len(ok["queries_used"]) == 4


# ---------------------------------------------------------------------------
# 5. Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_caller_cancellation_propagates_and_capacity_recovers(self, llm):
        llm.stalling = True
        config = _config(retrieval_llm_budget_seconds=5.0, retrieval_llm_max_concurrency=1)
        memory = FakeMemory(config)

        task = asyncio.create_task(
            memory.retrieve_for_inference_async(TEMPORAL_Q, mode="vector", enable_query_synthesis=True)
        )
        await _wait_until(lambda: llm.started == 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        runner = _runner(config)
        assert runner.in_flight == 1  # still held by the running call

        llm.stall.set()
        await _wait_until(lambda: runner.in_flight == 0)
        llm.stalling = False
        result = await memory.retrieve_for_inference_async(
            TEMPORAL_Q, mode="vector", enable_query_synthesis=True
        )
        assert result["degraded_enrichment"] == []
        assert len(result["queries_used"]) == 4


class TestEnrichmentRunner:
    async def test_cancel_while_running_holds_then_releases_slot(self):
        gate = threading.Event()
        runner = EnrichmentRunner(1)
        try:
            task = asyncio.create_task(runner.run(gate.wait, 10, timeout=5))
            await _wait_until(lambda: runner.in_flight == 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runner.in_flight == 1
            busy = await runner.run(lambda: "x", timeout=1)
            assert busy.status == "capacity"
            gate.set()
            await _wait_until(lambda: runner.in_flight == 0)
            ok = await runner.run(lambda: "x", timeout=1)
            assert (ok.status, ok.value) == ("ok", "x")
        finally:
            gate.set()
            runner._executor.shutdown(wait=True)

    async def test_exceptions_are_contained_and_never_unretrieved(self):
        loop = asyncio.get_running_loop()
        unhandled: list[dict] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, ctx: unhandled.append(ctx))
        gate = threading.Event()
        runner = EnrichmentRunner(2)

        def boom():
            raise ValueError("bad")

        def late_boom():
            gate.wait(5)
            raise ValueError("late")

        try:
            failed = await runner.run(boom, timeout=1)
            assert (failed.status, failed.error) == ("failure", "ValueError")
            timed_out = await runner.run(late_boom, timeout=0.05)
            assert timed_out.status == "timeout"
            gate.set()
            await _wait_until(lambda: runner.in_flight == 0)
            gc.collect()
            await asyncio.sleep(0.05)
            assert unhandled == []
        finally:
            gate.set()
            loop.set_exception_handler(previous)
            runner._executor.shutdown(wait=True)

    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            EnrichmentRunner(0)


# ---------------------------------------------------------------------------
# 6. Real search failures propagate
# ---------------------------------------------------------------------------


class TestSearchFailures:
    @pytest.mark.parametrize("synthesis", [False, True])
    async def test_embedding_failure_propagates(self, llm, caplog, synthesis):
        llm.error = RuntimeError("llm down")  # enrichment failing too must not mask it
        memory = FakeMemory(_config(), embed_error=httpx.ConnectTimeout("embed"))
        with pytest.raises(httpx.ConnectTimeout):
            await memory.retrieve_for_inference_async(
                TEMPORAL_Q, mode="vector", enable_query_synthesis=synthesis
            )
        assert "Search transport failure: stage=embedding" in caplog.text
        assert "PIPELINE TOTAL" not in caplog.text

    async def test_database_failure_propagates(self, llm, caplog):
        memory = FakeMemory(_config(), db_error=ConnectionError("db"))
        with pytest.raises(ConnectionError):
            await memory.retrieve_for_inference_async(
                "where is the payroll config", mode="vector", enable_query_synthesis=False
            )
        assert "Search transport failure: stage=database" in caplog.text


# ---------------------------------------------------------------------------
# 7. Enabled enrichment keeps rich behaviour
# ---------------------------------------------------------------------------


class TestEnabledEnrichment:
    async def test_synthesis_expansions_and_metadata_retained(self, llm):
        memory = FakeMemory(_config())
        result = await memory.retrieve_for_inference_async(
            TEMPORAL_Q, mode="vector", project="alpha", enable_query_synthesis=True, enable_hyde=False
        )
        assert result["queries_used"] == [TEMPORAL_Q, "term one", "term two", "term three"]
        assert memory._async_embedder.calls == [result["queries_used"]]
        assert result["degraded_enrichment"] == []
        top = result["results"][0]
        assert top["memory_id"] == "alpha-000-0000"
        assert top["source_ref"] == "project:alpha"
        assert result["total_candidates"] > 0
        llm.factory.assert_called_once_with(timeout=2.0, max_tokens_cap=512)

    async def test_preference_hyde_and_auto_intent(self, llm):
        llm.response = '{"primary_intent": "preference", "complexity": "complex", "domains": [], "entities": [], "requires_reasoning": true}'
        memory = FakeMemory(_config())
        result = await memory.retrieve_for_inference_async(
            PREFERENCE_Q, mode="auto", enable_query_synthesis=False, enable_hyde=True
        )
        assert result["intent"]["complexity"] == "complex"
        assert result["mode"] == "hybrid"
        # intent + HyDE ran; HyDE output appended as an extra query
        assert llm.started == 2
        assert len(result["queries_used"]) == 2
        # Preference HyDE ran the profile probe first
        assert memory.vector_queries[0].startswith("I use I prefer")

    async def test_auto_mode_intent_timeout_falls_back_to_vector(self, llm):
        llm.stalling = True
        memory = FakeMemory(_config(retrieval_llm_budget_seconds=0.2))
        result = await memory.retrieve_for_inference_async(
            "where is the payroll config", mode="auto", enable_query_synthesis=False, enable_hyde=True
        )
        assert result["mode"] == "vector"
        assert result["intent"] is None
        assert result["queries_used"] == ["where is the payroll config"]
        assert [d["stage"] for d in result["degraded_enrichment"]] == ["intent"]


# ---------------------------------------------------------------------------
# 8. Provider request settings
# ---------------------------------------------------------------------------


def _fake_completion(text: str = "ok"):
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


class TestRetrievalClientSettings:
    def test_retrieval_client_is_bounded_and_shared_client_untouched(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        base = cems.llm.client.get_client()
        base_timeout, base_retries = base._client.timeout, base._client.max_retries

        bounded = cems.llm.client.get_retrieval_client(timeout=2.0, max_tokens_cap=512)

        assert bounded is not base
        assert bounded._client.max_retries == 0
        assert bounded._client.timeout == httpx.Timeout(2.0, connect=1.0)
        # Shared maintenance client keeps SDK defaults
        assert base._client.timeout == base_timeout
        assert base._client.max_retries == base_retries == 2
        assert base._max_tokens_cap is None
        # Cached per limits, rebuilt if the shared client is replaced
        assert cems.llm.client.get_retrieval_client(timeout=2.0, max_tokens_cap=512) is bounded

    def test_token_cap_only_applies_to_retrieval_copy(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        base = cems.llm.client.get_client()
        bounded = cems.llm.client.get_retrieval_client(timeout=2.0, max_tokens_cap=512)

        with patch.object(base._client.chat.completions, "create", return_value=_fake_completion()) as base_create, \
                patch.object(bounded._client.chat.completions, "create", return_value=_fake_completion()) as bounded_create:
            base.complete("summarize")
            bounded.complete("rewrite")
            bounded.complete("decompose", max_tokens=256)

        assert base_create.call_args.kwargs["max_tokens"] == 4096
        assert [c.kwargs["max_tokens"] for c in bounded_create.call_args_list] == [512, 256]

    def test_bounded_copy_keeps_non_openrouter_endpoint(self, monkeypatch):
        """Local Ollama / any OpenAI-compatible endpoint: limits apply, no OpenRouter extras."""
        monkeypatch.setenv("CEMS_LLM_BASE_URL", "http://ollama:11434/v1")
        monkeypatch.setenv("CEMS_LLM_API_KEY", "ollama")
        monkeypatch.setenv("CEMS_LLM_MODEL", "qwen3:8b")
        base = cems.llm.client.get_client()
        bounded = cems.llm.client.get_retrieval_client(timeout=2.0, max_tokens_cap=512)

        assert bounded.is_openrouter is False
        assert bounded.base_url == base.base_url == "http://ollama:11434/v1"
        assert str(bounded._client.base_url).rstrip("/") == "http://ollama:11434/v1"
        assert bounded._client.api_key == "ollama"
        assert "X-Title" not in bounded._client.default_headers
        assert bounded._client.max_retries == 0
        assert bounded._client.timeout == httpx.Timeout(2.0, connect=1.0)
        assert base._client.max_retries == 2

        with patch.object(bounded._client.chat.completions, "create", return_value=_fake_completion()) as create:
            bounded.complete("rewrite", fast_route=True)
        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "qwen3:8b"
        assert kwargs["max_tokens"] == 512
        assert "extra_body" not in kwargs

    def test_bounded_copy_keeps_openrouter_extras(self, monkeypatch):
        monkeypatch.delenv("CEMS_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("CEMS_LLM_API_KEY", "test-key")
        bounded = cems.llm.client.get_retrieval_client(timeout=2.0, max_tokens_cap=512)

        assert bounded.is_openrouter is True
        assert bounded._client.default_headers["X-Title"] == bounded.site_name
        with patch.object(bounded._client.chat.completions, "create", return_value=_fake_completion()) as create:
            bounded.complete("rewrite", fast_route=True)
        assert "extra_body" in create.call_args.kwargs

    def test_default_budget_fits_gooseherd_deadline(self):
        config = CEMSConfig(database_url="postgresql://unused/unused")
        assert config.retrieval_llm_budget_seconds <= 2.0
        assert config.retrieval_llm_timeout_seconds <= config.retrieval_llm_budget_seconds
        assert config.retrieval_llm_max_tokens == 512
        assert config.retrieval_llm_max_concurrency >= 1


# ---------------------------------------------------------------------------
# 9. Query-type detection and config validation
# ---------------------------------------------------------------------------


class TestAggregationPhraseBoundaries:
    @pytest.mark.parametrize("query", [
        "did we talk about this before? call the CEMS system",
        "can you call the CEMS system to check timesheet approvals",
        "I totally forgot the deploy steps",
        "what is the recall threshold",
    ])
    def test_substrings_are_not_aggregation(self, query):
        from cems.retrieval import _is_aggregation_query
        assert not _is_aggregation_query(query)

    @pytest.mark.parametrize("query", [
        "How many camping trips did I take in total?",
        "list all the times we deployed",
        "what is the total spend on licences",
        "how often do we rotate keys",
        "How many different doctors did I visit?",
        "what changed throughout the migration",
        "All the different tools I tried",
    ])
    def test_whole_phrases_are_aggregation(self, query):
        from cems.retrieval import _is_aggregation_query
        assert _is_aggregation_query(query)


class TestRetrievalConfigValidation:
    @pytest.mark.parametrize("field,value", [
        ("retrieval_llm_budget_seconds", 0),
        ("retrieval_llm_budget_seconds", -1.0),
        ("retrieval_llm_budget_seconds", float("nan")),
        ("retrieval_llm_budget_seconds", float("inf")),
        ("retrieval_llm_timeout_seconds", 0),
        ("retrieval_llm_timeout_seconds", float("inf")),
        ("retrieval_llm_max_tokens", 0),
        ("retrieval_llm_max_concurrency", 0),
    ])
    def test_invalid_values_rejected(self, field, value):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _config(**{field: value})

    @pytest.mark.parametrize("env_value", ["nan", "0", "-2"])
    def test_invalid_env_fails_at_load(self, monkeypatch, env_value):
        from pydantic import ValidationError
        monkeypatch.setenv("CEMS_RETRIEVAL_LLM_BUDGET_SECONDS", env_value)
        with pytest.raises(ValidationError):
            CEMSConfig(database_url="postgresql://unused/unused")
