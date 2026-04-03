"""Tests for the DistillationJob — memory condensation to ~500 chars."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cems.config import CEMSConfig

TEST_UUID = "a6e153f9-41c5-4cbc-9a50-74160af381dd"


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_doc(
    doc_id: str,
    content: str,
    category: str = "general",
    content_detailed: str | None = None,
    shown_count: int = 1,
) -> dict:
    now = datetime.now(UTC)
    return {
        "id": doc_id,
        "content": content,
        "content_detailed": content_detailed,
        "category": category,
        "scope": "personal",
        "tags": [],
        "source": None,
        "source_ref": None,
        "created_at": now,
        "updated_at": now,
        "shown_count": shown_count,
        "last_shown_at": None,
        "relevant_count": 0,
        "noise_count": 0,
        "noise_snippet_count": 0,
    }


@pytest.fixture
def mock_memory():
    mock = MagicMock()
    mock.config = CEMSConfig(
        user_id=TEST_UUID,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    doc_store = AsyncMock()
    mock._ensure_document_store = AsyncMock(return_value=doc_store)
    mock._ensure_initialized_async = AsyncMock()
    mock._async_embedder = AsyncMock()
    mock._async_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    return mock, doc_store


class TestDistillationJob:
    """Tests for the DistillationJob."""

    def test_skips_short_content(self, mock_memory):
        """Memories with content <= 500 chars are not distilled."""
        from cems.maintenance.distillation import DistillationJob

        memory, doc_store = mock_memory
        docs = [_make_doc("short-1", "A" * 400)]  # Under threshold
        doc_store.get_all_documents = AsyncMock(return_value=docs)

        job = DistillationJob(memory)
        result = _run(job.run_async())

        assert result["candidates"] == 0
        assert result["distilled"] == 0
        doc_store.distill_document.assert_not_awaited()

    @patch("cems.maintenance.distillation.get_client")
    def test_distills_all_categories(self, mock_get_client, mock_memory):
        """All categories are eligible for distillation — only pinned tag protects."""
        from cems.maintenance.distillation import DistillationJob

        memory, doc_store = mock_memory
        condensed = "Terse summary."
        mock_client = MagicMock()
        mock_client.complete.return_value = condensed
        mock_get_client.return_value = mock_client

        docs = [
            _make_doc("gate-1", "A" * 1000, category="gate-rules"),
            _make_doc("pref-1", "B" * 1000, category="preferences"),
            _make_doc("session-1", "C" * 1000, category="session-summary"),
            _make_doc("pinned-1", "D" * 1000, category="general"),
        ]
        # Mark last one as pinned
        docs[3]["tags"] = ["pinned"]

        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.distill_document = AsyncMock(return_value=True)

        job = DistillationJob(memory)
        result = _run(job.run_async())

        # 3 candidates (gate-rules, preferences, session-summary), pinned skipped
        assert result["candidates"] == 3
        assert result["distilled"] == 3
        assert doc_store.distill_document.await_count == 3

    @patch("cems.maintenance.distillation.get_client")
    def test_distills_verbose_content(self, mock_get_client, mock_memory):
        """Verbose content is condensed and original preserved in content_detailed."""
        from cems.maintenance.distillation import DistillationJob

        memory, doc_store = mock_memory
        original = "A very long memory " * 50  # ~950 chars
        condensed = "Terse summary of the memory."

        mock_client = MagicMock()
        mock_client.complete.return_value = condensed
        mock_get_client.return_value = mock_client

        docs = [_make_doc("verbose-1", original)]
        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.distill_document = AsyncMock(return_value=True)

        job = DistillationJob(memory)
        result = _run(job.run_async())

        assert result["distilled"] == 1
        doc_store.distill_document.assert_awaited_once()

        # Verify the call preserves original in content_detailed
        call_kwargs = doc_store.distill_document.call_args[1]
        assert call_kwargs["content"] == condensed
        assert call_kwargs["content_detailed"] == original
        assert call_kwargs["user_id"] == TEST_UUID

    @patch("cems.maintenance.distillation.get_client")
    def test_preserves_existing_content_detailed(self, mock_get_client, mock_memory):
        """If content_detailed already exists, keep it (don't overwrite with condensed content)."""
        from cems.maintenance.distillation import DistillationJob

        memory, doc_store = mock_memory
        original_detailed = "The very first original content " * 30
        already_condensed = "Already condensed once but still over 500 chars. " * 15
        further_condensed = "Even more terse."

        mock_client = MagicMock()
        mock_client.complete.return_value = further_condensed
        mock_get_client.return_value = mock_client

        docs = [_make_doc("re-distill", already_condensed, content_detailed=original_detailed)]
        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.distill_document = AsyncMock(return_value=True)

        job = DistillationJob(memory)
        result = _run(job.run_async())

        assert result["distilled"] == 1
        call_kwargs = doc_store.distill_document.call_args[1]
        # Should keep the original content_detailed, not overwrite with already_condensed
        assert call_kwargs["content_detailed"] == original_detailed

    @patch("cems.maintenance.distillation.get_client")
    def test_skips_when_llm_doesnt_condense(self, mock_get_client, mock_memory):
        """If LLM returns same-length or longer text, skip."""
        from cems.maintenance.distillation import DistillationJob

        memory, doc_store = mock_memory
        content = "Some verbose content here." * 30  # No trailing space

        mock_client = MagicMock()
        mock_client.complete.return_value = content + " and more text"  # Longer
        mock_get_client.return_value = mock_client

        docs = [_make_doc("no-condense", content)]
        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.distill_document = AsyncMock(return_value=True)

        job = DistillationJob(memory)
        result = _run(job.run_async())

        assert result["skipped"] == 1
        assert result["distilled"] == 0
        doc_store.distill_document.assert_not_awaited()

    @patch("cems.maintenance.distillation.get_client")
    def test_handles_llm_error(self, mock_get_client, mock_memory):
        """LLM errors are counted and don't crash the job."""
        from cems.maintenance.distillation import DistillationJob

        memory, doc_store = mock_memory

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("LLM timeout")
        mock_get_client.return_value = mock_client

        docs = [_make_doc("error-1", "X" * 1000)]
        doc_store.get_all_documents = AsyncMock(return_value=docs)

        job = DistillationJob(memory)
        result = _run(job.run_async())

        assert result["errors"] == 1
        assert result["distilled"] == 0

    @patch("cems.maintenance.distillation.get_client")
    def test_caps_content_detailed_growth(self, mock_get_client, mock_memory):
        """content_detailed over DETAILED_CAP gets condensed."""
        from cems.maintenance.distillation import DETAILED_CAP, DistillationJob

        memory, doc_store = mock_memory
        huge_detailed = "X" * (DETAILED_CAP + 5000)
        condensed_detailed = "Condensed detailed content."
        condensed_content = "Terse summary."

        mock_client = MagicMock()
        # First call: _condense_detailed, second call: _distill
        mock_client.complete.side_effect = [condensed_detailed, condensed_content]
        mock_get_client.return_value = mock_client

        docs = [_make_doc("huge-detail", "Y" * 1000, content_detailed=huge_detailed)]
        doc_store.get_all_documents = AsyncMock(return_value=docs)
        doc_store.distill_document = AsyncMock(return_value=True)

        job = DistillationJob(memory)
        result = _run(job.run_async())

        assert result["distilled"] == 1
        call_kwargs = doc_store.distill_document.call_args[1]
        assert call_kwargs["content_detailed"] == condensed_detailed


class TestMemoryGetContentDetailed:
    """Test that memory_get returns content_detailed."""

    def test_memory_get_tool_exists(self):
        """mcp_stdio has memory_get tool."""
        from cems.mcp_stdio import memory_get
        assert callable(memory_get)

    def test_memory_get_calls_api(self):
        """memory_get calls /api/memory/get with the ID."""
        from cems.mcp_stdio import memory_get

        with patch("cems.mcp_stdio._request", return_value={"success": True, "document": {}}) as mock_req:
            memory_get("test-id")

        mock_req.assert_called_once_with("GET", "/api/memory/get?id=test-id")
