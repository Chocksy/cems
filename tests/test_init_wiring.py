"""Tests that verify init/wiring between components doesn't break.

These tests catch signature mismatches between callers and constructors —
e.g., _ensure_document_store() passing kwargs that DocumentStore.__init__
no longer accepts. Every test calls the REAL constructor (no mocks on init),
only mocking the DB connection layer.
"""

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from cems.config import CEMSConfig
from cems.db.document_store import DocumentStore
from cems.memory.core import CEMSMemory


class TestDocumentStoreWiring:
    """Verify _ensure_document_store() kwargs match DocumentStore.__init__ signature."""

    def test_ensure_document_store_kwargs_match_init(self):
        """Static check: every kwarg passed in _ensure_document_store exists in DocumentStore.__init__."""
        # Get DocumentStore.__init__ parameter names
        init_sig = inspect.signature(DocumentStore.__init__)
        init_params = set(init_sig.parameters.keys()) - {"self"}

        # Read the source of _ensure_document_store to extract kwargs
        import ast
        source = inspect.getsource(CEMSMemory._ensure_document_store)
        # Dedent so ast can parse it
        import textwrap
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Find the DocumentStore(...) call and extract keyword argument names
        called_kwargs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's calling DocumentStore
                func = node.func
                if isinstance(func, ast.Name) and func.id == "DocumentStore":
                    for kw in node.keywords:
                        if kw.arg:  # Skip **kwargs
                            called_kwargs.add(kw.arg)

        assert called_kwargs, "Should find at least one kwarg in DocumentStore() call"

        # Every kwarg passed must exist in __init__
        unknown = called_kwargs - init_params
        assert not unknown, (
            f"_ensure_document_store() passes kwargs not accepted by "
            f"DocumentStore.__init__: {unknown}. "
            f"Accepted params: {init_params}"
        )

    @pytest.mark.asyncio
    async def test_ensure_document_store_constructs_successfully(self):
        """Runtime check: actually call _ensure_document_store() with real constructor."""
        config = CEMSConfig(
            database_url="postgresql://test:test@localhost/test",
        )
        memory = CEMSMemory(config=config)

        # Mock only the DB connect — let the constructor run for real
        with patch.object(DocumentStore, "connect", new_callable=AsyncMock):
            doc_store = await memory._ensure_document_store()

        assert isinstance(doc_store, DocumentStore)
        assert doc_store.database_url == "postgresql://test:test@localhost/test"

    @pytest.mark.asyncio
    async def test_ensure_document_store_is_cached(self):
        """Second call returns the same instance (lazy singleton)."""
        config = CEMSConfig(
            database_url="postgresql://test:test@localhost/test",
        )
        memory = CEMSMemory(config=config)

        with patch.object(DocumentStore, "connect", new_callable=AsyncMock) as mock_connect:
            first = await memory._ensure_document_store()
            second = await memory._ensure_document_store()

        assert first is second
        # connect() should only be called once
        mock_connect.assert_awaited_once()
