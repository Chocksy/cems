"""Embedding dimension guard for the pgvector column."""

import pytest

from cems.db.database import check_embedding_dimension


def test_matching_dimension_passes():
    check_embedding_dimension(actual=1536, expected=1536)


def test_missing_table_passes():
    check_embedding_dimension(actual=None, expected=768)


def test_mismatch_raises_with_guidance():
    with pytest.raises(RuntimeError) as exc:
        check_embedding_dimension(actual=1536, expected=768)
    msg = str(exc.value)
    assert "database has 1536, config has 768" in msg
    assert "docs/DEPLOYMENT.md#private-mode" in msg
