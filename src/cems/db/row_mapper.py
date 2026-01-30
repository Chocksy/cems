"""Row mapping utilities for CEMS database operations."""

from typing import Any

import asyncpg


def row_to_dict(
    row: asyncpg.Record,
    include_score: bool = False,
) -> dict[str, Any]:
    """Convert a database row to a dictionary.

    Args:
        row: Database row from asyncpg
        include_score: Include score field if present

    Returns:
        Memory dict with all standard fields
    """
    result = {
        "id": str(row["id"]),
        "content": row["content"],
        "user_id": str(row["user_id"]) if row["user_id"] else None,
        "team_id": str(row["team_id"]) if row["team_id"] else None,
        "scope": row["scope"],
        "category": row["category"],
        "tags": row["tags"],
        "source": row["source"],
        "source_ref": row["source_ref"],
        "priority": row["priority"],
        "pinned": row["pinned"],
        "pin_reason": row["pin_reason"],
        "pin_category": row.get("pin_category"),
        "archived": row["archived"],
        "access_count": row["access_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_accessed": row["last_accessed"],
        "expires_at": row["expires_at"],
    }

    if include_score and "score" in row.keys():
        result["score"] = row["score"]

    return result
