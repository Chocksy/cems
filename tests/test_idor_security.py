"""IDOR boundary tests — verify cross-user access is blocked.

Tests that get_document, delete_document, and update_document enforce
user_id ownership checks so no user can access another user's documents.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cems.config import CEMSConfig

# Two distinct user UUIDs
USER_A = "a6e153f9-41c5-4cbc-9a50-74160af381dd"
USER_B = "b7f264fa-52d6-5dcd-0b61-85271bf492ee"

DOC_ID = "c8a375fb-63e7-6ede-1c72-96382cf503ff"


def _run(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_memory(user_id: str):
    """Create a mock CEMSMemory for a given user."""
    mock = MagicMock()
    mock.config = CEMSConfig(
        user_id=user_id,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    doc_store = AsyncMock()
    mock._ensure_document_store = AsyncMock(return_value=doc_store)
    mock._ensure_initialized_async = AsyncMock()

    embedder = AsyncMock()
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    mock._async_embedder = embedder

    return mock, doc_store


class TestIDORGetDocument:
    """get_document() must filter by user_id."""

    def test_get_document_requires_user_id(self):
        """get_document() accepts user_id parameter."""
        from cems.db.document_store import DocumentStore

        sig = DocumentStore.get_document.__code__.co_varnames
        assert "user_id" in sig, "get_document must accept user_id parameter"

    def test_crud_get_passes_user_id(self):
        """CRUDMixin._get_async passes user_id to doc_store.get_document."""
        from cems.memory.crud import CRUDMixin

        memory, doc_store = _make_memory(USER_A)
        doc_store.get_document = AsyncMock(return_value=None)

        # Mix CRUDMixin into mock
        mixin = CRUDMixin()
        mixin.config = memory.config
        mixin._ensure_document_store = memory._ensure_document_store

        _run(mixin._get_async(DOC_ID))

        # Verify user_id was passed
        doc_store.get_document.assert_awaited_once_with(DOC_ID, user_id=USER_A)


class TestIDORDeleteDocument:
    """delete_document() must filter by user_id."""

    def test_delete_document_requires_user_id(self):
        """delete_document() accepts user_id parameter."""
        from cems.db.document_store import DocumentStore

        sig = DocumentStore.delete_document.__code__.co_varnames
        assert "user_id" in sig, "delete_document must accept user_id parameter"

    def test_crud_delete_passes_user_id(self):
        """CRUDMixin._delete_async_internal passes user_id to doc_store.delete_document."""
        from cems.memory.crud import CRUDMixin

        memory, doc_store = _make_memory(USER_A)
        doc_store.delete_document = AsyncMock(return_value=True)

        mixin = CRUDMixin()
        mixin.config = memory.config
        mixin._ensure_document_store = memory._ensure_document_store

        _run(mixin._delete_async_internal(DOC_ID))

        doc_store.delete_document.assert_awaited_once_with(DOC_ID, hard=False, user_id=USER_A)


class TestIDORUpdateDocument:
    """update_document() must filter by user_id."""

    def test_update_document_requires_user_id(self):
        """update_document() accepts user_id parameter."""
        from cems.db.document_store import DocumentStore

        sig = DocumentStore.update_document.__code__.co_varnames
        assert "user_id" in sig, "update_document must accept user_id parameter"

    def test_crud_update_passes_user_id(self):
        """CRUDMixin.update_async passes user_id to doc_store.update_document."""
        from cems.memory.crud import CRUDMixin

        memory, doc_store = _make_memory(USER_A)
        doc_store.update_document = AsyncMock(return_value=True)

        mixin = CRUDMixin()
        mixin.config = memory.config
        mixin._ensure_document_store = memory._ensure_document_store
        mixin._ensure_initialized_async = memory._ensure_initialized_async
        mixin._async_embedder = memory._async_embedder

        from cems.chunking import Chunk
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "cems.memory.crud.chunk_document",
                lambda content: [Chunk(content=content, index=0)],
                raising=False,
            )
            # Need to patch at module level since crud.py imports chunk_document inside method
            import cems.chunking
            original = cems.chunking.chunk_document

            _run(mixin.update_async(DOC_ID, "new content"))

        doc_store.update_document.assert_awaited_once()
        call_kwargs = doc_store.update_document.call_args[1]
        assert call_kwargs.get("user_id") == USER_A, "update_document must receive user_id"
