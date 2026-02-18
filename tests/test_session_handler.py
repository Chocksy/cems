"""Tests for session summarize API handler.

Regression tests for the double-append bug (v0.4.4) where incremental
session summaries grew exponentially because the handler pre-merged
content and then passed mode="append" to upsert_document_by_tag, which
appended again — doubling the document each cycle.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_memory():
    """Create a mock memory instance for session handler tests."""
    memory = MagicMock()
    memory.config = MagicMock()
    memory.config.user_id = "test-user-uuid"
    memory._ensure_initialized_async = AsyncMock()
    memory._async_embedder = MagicMock()
    memory._async_embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    doc_store = AsyncMock()
    memory._ensure_document_store = AsyncMock(return_value=doc_store)

    return memory, doc_store


def _make_request(body: dict):
    """Create a mock Starlette Request with the given JSON body."""
    request = AsyncMock()
    request.json = AsyncMock(return_value=body)
    return request


def _fake_summary(content: str = "Summary paragraph about the session."):
    """Return a minimal summary dict matching extract_session_summary output."""
    return {
        "content": content,
        "title": "Test Session",
        "priority": "medium",
        "tags": ["debugging"],
    }


def _fake_chunk(content: str):
    """Create a minimal Chunk object."""
    from cems.chunking import Chunk

    return Chunk(seq=0, pos=0, content=content, tokens=10, bytes=len(content))


class TestSessionSummarizeIncrementalMode:
    """Tests for incremental session summary upsert logic."""

    @patch("cems.api.handlers.session.get_memory")
    @patch("cems.llm.session_summary_extraction.extract_session_summary")
    @pytest.mark.asyncio
    async def test_incremental_uses_replace_mode_not_append(
        self, mock_extract, mock_get_memory, mock_memory
    ):
        """Regression: incremental mode must pass mode='replace' to upsert,
        NOT 'append', since the handler already pre-merges content."""
        memory, doc_store = mock_memory
        mock_get_memory.return_value = memory
        mock_extract.return_value = _fake_summary("New summary block.")

        # Simulate existing document in DB
        doc_store.find_document_by_tag = AsyncMock(return_value={
            "content": "Existing summary block.",
        })
        doc_store.upsert_document_by_tag = AsyncMock(return_value=("doc-123", "replaced"))

        with patch("cems.chunking.chunk_document", return_value=[_fake_chunk("x")]):
            from cems.api.handlers.session import api_session_summarize

            request = _make_request({
                "content": "some transcript text",
                "session_id": "test-session-abc",
                "mode": "incremental",
            })

            response = await api_session_summarize(request)

        # The critical assertion: mode must be "replace"
        call_kwargs = doc_store.upsert_document_by_tag.call_args
        assert call_kwargs.kwargs["mode"] == "replace", (
            "Incremental mode must use 'replace' — handler pre-merges content. "
            "Using 'append' causes exponential doubling (the v0.4.4 bug)."
        )

    @patch("cems.api.handlers.session.get_memory")
    @patch("cems.llm.session_summary_extraction.extract_session_summary")
    @pytest.mark.asyncio
    async def test_incremental_premerges_existing_and_new(
        self, mock_extract, mock_get_memory, mock_memory
    ):
        """Handler should concatenate existing + new content before upsert."""
        memory, doc_store = mock_memory
        mock_get_memory.return_value = memory
        mock_extract.return_value = _fake_summary("New block.")

        doc_store.find_document_by_tag = AsyncMock(return_value={
            "content": "Existing block.",
        })
        doc_store.upsert_document_by_tag = AsyncMock(return_value=("doc-123", "replaced"))

        with patch("cems.chunking.chunk_document", return_value=[_fake_chunk("x")]):
            from cems.api.handlers.session import api_session_summarize

            request = _make_request({
                "content": "transcript",
                "session_id": "test-session-abc",
                "mode": "incremental",
            })

            await api_session_summarize(request)

        # Content passed to upsert should be existing + separator + new
        call_kwargs = doc_store.upsert_document_by_tag.call_args
        upsert_content = call_kwargs.kwargs["content"]
        assert "Existing block." in upsert_content
        assert "New block." in upsert_content
        assert "\n\n---\n\n" in upsert_content

    @patch("cems.api.handlers.session.get_memory")
    @patch("cems.llm.session_summary_extraction.extract_session_summary")
    @pytest.mark.asyncio
    async def test_incremental_no_exponential_growth(
        self, mock_extract, mock_get_memory, mock_memory
    ):
        """Regression: simulate 5 incremental calls and verify linear growth.

        Before the fix, each call doubled the document. With the fix,
        each call adds exactly one new block.
        """
        memory, doc_store = mock_memory
        mock_get_memory.return_value = memory

        stored_content = ""

        def fake_find(**kwargs):
            if stored_content:
                return {"content": stored_content}
            return None

        async def fake_upsert(**kwargs):
            nonlocal stored_content
            stored_content = kwargs["content"]
            return ("doc-123", "replaced")

        doc_store.find_document_by_tag = AsyncMock(side_effect=fake_find)
        doc_store.upsert_document_by_tag = AsyncMock(side_effect=fake_upsert)

        block_sizes = []

        with patch("cems.chunking.chunk_document", return_value=[_fake_chunk("x")]):
            from cems.api.handlers.session import api_session_summarize

            for i in range(5):
                mock_extract.return_value = _fake_summary(f"Summary block {i}.")

                request = _make_request({
                    "content": f"transcript chunk {i}",
                    "session_id": "test-session-abc",
                    "mode": "incremental",
                })

                await api_session_summarize(request)
                block_sizes.append(len(stored_content))

        # Verify linear growth: each step should add roughly the same amount
        # With exponential doubling, block_sizes[4] would be ~16x block_sizes[0]
        # With linear growth, it should be ~5x
        growth_ratio = block_sizes[4] / block_sizes[0]
        assert growth_ratio < 10, (
            f"Content grew {growth_ratio:.0f}x over 5 calls — suggests exponential doubling. "
            f"Sizes: {block_sizes}"
        )

        # Also verify we have exactly 5 separator-delimited blocks
        block_count = stored_content.count("\n\n---\n\n") + 1
        assert block_count == 5, f"Expected 5 blocks, got {block_count}"

    @patch("cems.api.handlers.session.get_memory")
    @patch("cems.llm.session_summary_extraction.extract_session_summary")
    @pytest.mark.asyncio
    async def test_stored_summary_cap_at_100kb(
        self, mock_extract, mock_get_memory, mock_memory
    ):
        """Safety cap: stored content should not exceed ~100KB."""
        memory, doc_store = mock_memory
        mock_get_memory.return_value = memory

        # Simulate existing document that's already near the cap
        big_existing = ("X" * 5000 + "\n\n---\n\n") * 25  # ~130KB
        mock_extract.return_value = _fake_summary("One more block.")

        doc_store.find_document_by_tag = AsyncMock(return_value={
            "content": big_existing,
        })
        doc_store.upsert_document_by_tag = AsyncMock(return_value=("doc-123", "replaced"))

        with patch("cems.chunking.chunk_document", return_value=[_fake_chunk("x")]):
            from cems.api.handlers.session import api_session_summarize

            request = _make_request({
                "content": "transcript",
                "session_id": "test-session-abc",
                "mode": "incremental",
            })

            await api_session_summarize(request)

        call_kwargs = doc_store.upsert_document_by_tag.call_args
        upsert_content = call_kwargs.kwargs["content"]
        assert len(upsert_content) <= 100_000, (
            f"Stored content is {len(upsert_content)} chars — exceeds 100KB cap"
        )


class TestSessionSummarizeFinalizeMode:
    """Tests for finalize mode."""

    @patch("cems.api.handlers.session.get_memory")
    @patch("cems.llm.session_summary_extraction.extract_session_summary")
    @pytest.mark.asyncio
    async def test_finalize_uses_finalize_mode(
        self, mock_extract, mock_get_memory, mock_memory
    ):
        """Finalize mode should pass mode='finalize' to upsert."""
        memory, doc_store = mock_memory
        mock_get_memory.return_value = memory
        mock_extract.return_value = _fake_summary("Final summary.")

        doc_store.upsert_document_by_tag = AsyncMock(return_value=("doc-123", "finalized"))

        with patch("cems.chunking.chunk_document", return_value=[_fake_chunk("x")]):
            from cems.api.handlers.session import api_session_summarize

            request = _make_request({
                "content": "full transcript",
                "session_id": "test-session-abc",
                "mode": "finalize",
            })

            await api_session_summarize(request)

        call_kwargs = doc_store.upsert_document_by_tag.call_args
        assert call_kwargs.kwargs["mode"] == "finalize"


class TestSessionSummarizeFirstCall:
    """Tests for first incremental call (no existing document)."""

    @patch("cems.api.handlers.session.get_memory")
    @patch("cems.llm.session_summary_extraction.extract_session_summary")
    @pytest.mark.asyncio
    async def test_first_incremental_creates_document(
        self, mock_extract, mock_get_memory, mock_memory
    ):
        """First incremental call should create with just the new content."""
        memory, doc_store = mock_memory
        mock_get_memory.return_value = memory
        mock_extract.return_value = _fake_summary("First summary.")

        # No existing document
        doc_store.find_document_by_tag = AsyncMock(return_value=None)
        doc_store.upsert_document_by_tag = AsyncMock(return_value=("doc-123", "created"))

        with patch("cems.chunking.chunk_document", return_value=[_fake_chunk("x")]):
            from cems.api.handlers.session import api_session_summarize

            request = _make_request({
                "content": "transcript",
                "session_id": "test-session-abc",
                "mode": "incremental",
            })

            response = await api_session_summarize(request)

        call_kwargs = doc_store.upsert_document_by_tag.call_args
        assert call_kwargs.kwargs["content"] == "First summary."
        assert call_kwargs.kwargs["mode"] == "replace"

        data = response.body.decode()
        assert "doc-123" in data
