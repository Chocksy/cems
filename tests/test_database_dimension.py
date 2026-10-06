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


def test_sync_engine_uses_psycopg2():
    # SQLAlchemy 2.1 maps bare postgresql:// to psycopg (v3), which the image does not ship.
    from cems.db.database import Database

    db = Database("postgresql+asyncpg://u:p@localhost:5432/cems")
    assert db.sync_engine.dialect.driver == "psycopg2"
