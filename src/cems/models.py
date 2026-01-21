"""Data models for CEMS."""

import sqlite3
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(UTC)


class MemoryScope(str, Enum):
    """Memory scope - personal or shared."""

    PERSONAL = "personal"
    SHARED = "shared"


class MemoryCategory(str, Enum):
    """Pre-defined memory categories."""

    PREFERENCES = "preferences"
    DECISIONS = "decisions"
    PATTERNS = "patterns"
    CONTEXT = "context"
    LEARNINGS = "learnings"
    GENERAL = "general"


class PinCategory(str, Enum):
    """Categories for pinned memories."""

    GUIDELINE = "guideline"  # Coding guidelines, style guides
    CONVENTION = "convention"  # Team conventions
    ARCHITECTURE = "architecture"  # Architecture decisions
    STANDARD = "standard"  # Industry standards
    DOCUMENTATION = "documentation"  # Important docs


class MemoryMetadata(BaseModel):
    """Extended metadata for a memory item."""

    model_config = ConfigDict(use_enum_values=True)

    memory_id: str = Field(description="Unique memory ID from Mem0")
    user_id: str = Field(description="User who owns this memory")
    scope: MemoryScope = Field(description="Personal or shared")
    category: str = Field(default="general", description="Memory category")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    last_accessed: datetime = Field(default_factory=_utcnow)
    access_count: int = Field(default=0, description="Number of times accessed")
    source: str | None = Field(default=None, description="Source of the memory")
    source_ref: str | None = Field(default=None, description="Reference (e.g., repo:file:line)")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")
    archived: bool = Field(default=False, description="Whether memory is archived")
    priority: float = Field(default=1.0, description="Priority weight for retrieval")
    # Pinned memory support - excluded from decay
    pinned: bool = Field(default=False, description="Pinned memories are never auto-pruned")
    pin_reason: str | None = Field(default=None, description="Why this memory is pinned")
    pin_category: str | None = Field(default=None, description="Pin category (guideline, convention, etc.)")
    expires_at: datetime | None = Field(default=None, description="When memory expires (None = never)")


class SearchResult(BaseModel):
    """Search result with metadata."""

    memory_id: str
    content: str
    score: float
    scope: MemoryScope
    metadata: MemoryMetadata | None = None


class CategorySummary(BaseModel):
    """Summary for a memory category."""

    category: str
    scope: MemoryScope
    summary: str
    item_count: int
    last_updated: datetime
    version: int = 1


class MetadataStore:
    """SQLite store for extended metadata."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    source TEXT,
                    source_ref TEXT,
                    tags TEXT,
                    archived INTEGER DEFAULT 0,
                    priority REAL DEFAULT 1.0,
                    pinned INTEGER DEFAULT 0,
                    pin_reason TEXT,
                    pin_category TEXT,
                    expires_at TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_memory_user
                    ON memory_metadata(user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_scope
                    ON memory_metadata(scope);
                CREATE INDEX IF NOT EXISTS idx_memory_category
                    ON memory_metadata(category);
                CREATE INDEX IF NOT EXISTS idx_memory_last_accessed
                    ON memory_metadata(last_accessed);
                CREATE INDEX IF NOT EXISTS idx_memory_archived
                    ON memory_metadata(archived);
                CREATE INDEX IF NOT EXISTS idx_memory_pinned
                    ON memory_metadata(pinned);

                CREATE TABLE IF NOT EXISTS category_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    summary TEXT,
                    item_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    UNIQUE(user_id, category, scope)
                );

                CREATE TABLE IF NOT EXISTS maintenance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT,
                    details TEXT
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def save_metadata(self, metadata: MemoryMetadata) -> None:
        """Save or update memory metadata."""
        conn = sqlite3.connect(self.db_path)
        try:
            tags_str = ",".join(metadata.tags) if metadata.tags else ""
            expires_str = metadata.expires_at.isoformat() if metadata.expires_at else None
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_metadata
                (memory_id, user_id, scope, category, created_at, updated_at,
                 last_accessed, access_count, source, source_ref, tags, archived, priority,
                 pinned, pin_reason, pin_category, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.memory_id,
                    metadata.user_id,
                    metadata.scope,
                    metadata.category,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    metadata.last_accessed.isoformat(),
                    metadata.access_count,
                    metadata.source,
                    metadata.source_ref,
                    tags_str,
                    int(metadata.archived),
                    metadata.priority,
                    int(metadata.pinned),
                    metadata.pin_reason,
                    metadata.pin_category,
                    expires_str,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_metadata(self, memory_id: str) -> MemoryMetadata | None:
        """Get metadata for a memory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM memory_metadata WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row:
                return self._row_to_metadata(row)
            return None
        finally:
            conn.close()

    def record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE memory_metadata
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE memory_id = ?
                """,
                (memory_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_stale_memories(self, user_id: str, days: int) -> list[str]:
        """Get memory IDs not accessed in N days (excludes pinned memories)."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT memory_id FROM memory_metadata
                WHERE user_id = ?
                AND archived = 0
                AND pinned = 0
                AND last_accessed < datetime('now', ?)
                """,
                (user_id, f"-{days} days"),
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def get_hot_memories(self, user_id: str, threshold: int) -> list[str]:
        """Get frequently accessed memory IDs."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT memory_id FROM memory_metadata
                WHERE user_id = ?
                AND archived = 0
                AND access_count >= ?
                """,
                (user_id, threshold),
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def archive_memory(self, memory_id: str) -> None:
        """Mark a memory as archived."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE memory_metadata SET archived = 1 WHERE memory_id = ?",
                (memory_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def increase_priority(self, memory_id: str, boost: float = 0.1) -> None:
        """Increase memory priority."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE memory_metadata
                SET priority = MIN(priority + ?, 2.0)
                WHERE memory_id = ?
                """,
                (boost, memory_id),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_metadata(self, memory_id: str) -> None:
        """Delete metadata for a memory."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM memory_metadata WHERE memory_id = ?", (memory_id,))
            conn.commit()
        finally:
            conn.close()

    def log_maintenance(
        self, job_type: str, user_id: str, status: str, details: str | None = None
    ) -> int:
        """Log a maintenance job execution."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO maintenance_log (job_type, user_id, status, details)
                VALUES (?, ?, ?, ?)
                """,
                (job_type, user_id, status, details),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def update_maintenance_log(
        self, log_id: int, status: str, details: str | None = None
    ) -> None:
        """Update a maintenance log entry."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE maintenance_log
                SET completed_at = CURRENT_TIMESTAMP, status = ?, details = ?
                WHERE id = ?
                """,
                (status, details, log_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _row_to_metadata(self, row: sqlite3.Row) -> MemoryMetadata:
        """Convert a database row to MemoryMetadata."""
        tags = row["tags"].split(",") if row["tags"] else []
        expires_at = None
        if row["expires_at"]:
            dt = datetime.fromisoformat(row["expires_at"])
            expires_at = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt

        # Convert naive datetimes to UTC-aware
        def to_utc(dt_str: str) -> datetime:
            dt = datetime.fromisoformat(dt_str)
            return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt

        return MemoryMetadata(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            scope=MemoryScope(row["scope"]),
            category=row["category"],
            created_at=to_utc(row["created_at"]),
            updated_at=to_utc(row["updated_at"]),
            last_accessed=to_utc(row["last_accessed"]),
            access_count=row["access_count"],
            source=row["source"],
            source_ref=row["source_ref"],
            tags=tags,
            archived=bool(row["archived"]),
            priority=row["priority"],
            pinned=bool(row["pinned"]),
            pin_reason=row["pin_reason"],
            pin_category=row["pin_category"],
            expires_at=expires_at,
        )

    def pin_memory(
        self, memory_id: str, reason: str, pin_category: str = "guideline"
    ) -> None:
        """Pin a memory to prevent it from being auto-pruned."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE memory_metadata
                SET pinned = 1, pin_reason = ?, pin_category = ?
                WHERE memory_id = ?
                """,
                (reason, pin_category, memory_id),
            )
            conn.commit()
        finally:
            conn.close()

    def unpin_memory(self, memory_id: str) -> None:
        """Unpin a memory."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE memory_metadata
                SET pinned = 0, pin_reason = NULL, pin_category = NULL
                WHERE memory_id = ?
                """,
                (memory_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_pinned_memories(self, user_id: str, pin_category: str | None = None) -> list[str]:
        """Get all pinned memory IDs for a user."""
        conn = sqlite3.connect(self.db_path)
        try:
            if pin_category:
                rows = conn.execute(
                    """
                    SELECT memory_id FROM memory_metadata
                    WHERE user_id = ? AND pinned = 1 AND pin_category = ?
                    """,
                    (user_id, pin_category),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT memory_id FROM memory_metadata
                    WHERE user_id = ? AND pinned = 1
                    """,
                    (user_id,),
                ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def get_memories_by_category(
        self, user_id: str, category: str, scope: MemoryScope | None = None
    ) -> list[str]:
        """Get memory IDs for a category."""
        conn = sqlite3.connect(self.db_path)
        try:
            if scope:
                rows = conn.execute(
                    """
                    SELECT memory_id FROM memory_metadata
                    WHERE user_id = ? AND category = ? AND scope = ? AND archived = 0
                    """,
                    (user_id, category, scope.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT memory_id FROM memory_metadata
                    WHERE user_id = ? AND category = ? AND archived = 0
                    """,
                    (user_id, category),
                ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def get_all_user_memories(self, user_id: str, include_archived: bool = False) -> list[str]:
        """Get all memory IDs for a user."""
        conn = sqlite3.connect(self.db_path)
        try:
            if include_archived:
                rows = conn.execute(
                    "SELECT memory_id FROM memory_metadata WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT memory_id FROM memory_metadata WHERE user_id = ? AND archived = 0",
                    (user_id,),
                ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def get_recent_memories(self, user_id: str, hours: int = 24) -> list[str]:
        """Get memories created in the last N hours."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT memory_id FROM memory_metadata
                WHERE user_id = ?
                AND created_at > datetime('now', ?)
                AND archived = 0
                """,
                (user_id, f"-{hours} hours"),
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def get_old_memories(self, user_id: str, days: int) -> list[str]:
        """Get memories older than N days."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT memory_id FROM memory_metadata
                WHERE user_id = ?
                AND created_at < datetime('now', ?)
                AND archived = 0
                """,
                (user_id, f"-{days} days"),
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def get_all_categories(self, user_id: str, scope: MemoryScope | None = None) -> list[dict]:
        """Get all categories with their memory counts.

        Args:
            user_id: The user ID
            scope: Optional scope filter

        Returns:
            List of dicts with category name and count
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if scope:
                rows = conn.execute(
                    """
                    SELECT category, COUNT(*) as count
                    FROM memory_metadata
                    WHERE user_id = ? AND scope = ? AND archived = 0
                    GROUP BY category
                    ORDER BY count DESC
                    """,
                    (user_id, scope.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT category, scope, COUNT(*) as count
                    FROM memory_metadata
                    WHERE user_id = ? AND archived = 0
                    GROUP BY category, scope
                    ORDER BY count DESC
                    """,
                    (user_id,),
                ).fetchall()

            if scope:
                return [{"category": row[0], "count": row[1]} for row in rows]
            else:
                return [{"category": row[0], "scope": row[1], "count": row[2]} for row in rows]
        finally:
            conn.close()

    def get_recently_accessed(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get recently accessed memories sorted by last access time.

        Args:
            user_id: The user ID
            limit: Maximum number of results

        Returns:
            List of dicts with memory_id, category, access_count, last_accessed
        """
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT memory_id, category, scope, access_count, last_accessed
                FROM memory_metadata
                WHERE user_id = ? AND archived = 0
                ORDER BY last_accessed DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [
                {
                    "memory_id": row[0],
                    "category": row[1],
                    "scope": row[2],
                    "access_count": row[3],
                    "last_accessed": row[4],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def get_category_summary(
        self, user_id: str, category: str, scope: str = "personal"
    ) -> dict | None:
        """Get the summary for a category.

        Args:
            user_id: The user ID
            category: Category name
            scope: "personal" or "shared"

        Returns:
            Summary dict or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT summary, item_count, last_updated, version
                FROM category_summaries
                WHERE user_id = ? AND category = ? AND scope = ?
                """,
                (user_id, category, scope),
            ).fetchone()
            if row:
                return {
                    "summary": row[0],
                    "item_count": row[1],
                    "last_updated": row[2],
                    "version": row[3],
                }
            return None
        finally:
            conn.close()

    def get_all_category_summaries(self, user_id: str, scope: str | None = None) -> list[dict]:
        """Get all category summaries.

        Args:
            user_id: The user ID
            scope: Optional scope filter ("personal" or "shared")

        Returns:
            List of summary dicts
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if scope:
                rows = conn.execute(
                    """
                    SELECT category, scope, summary, item_count, last_updated, version
                    FROM category_summaries
                    WHERE user_id = ? AND scope = ?
                    ORDER BY category
                    """,
                    (user_id, scope),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT category, scope, summary, item_count, last_updated, version
                    FROM category_summaries
                    WHERE user_id = ?
                    ORDER BY category
                    """,
                    (user_id,),
                ).fetchall()
            return [
                {
                    "category": row[0],
                    "scope": row[1],
                    "summary": row[2],
                    "item_count": row[3],
                    "last_updated": row[4],
                    "version": row[5],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def get_archived_memory_ids(self) -> set[str]:
        """Get all archived memory IDs in a single query.
        
        Much faster than checking each memory individually.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT memory_id FROM memory_metadata WHERE archived = 1"
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            conn.close()

    def get_metadata_batch(self, memory_ids: list[str]) -> dict[str, MemoryMetadata]:
        """Get metadata for multiple memories in a single query.
        
        Returns:
            Dict mapping memory_id to MemoryMetadata
        """
        if not memory_ids:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" * len(memory_ids))
            rows = conn.execute(
                f"SELECT * FROM memory_metadata WHERE memory_id IN ({placeholders})",
                memory_ids,
            ).fetchall()
            return {row["memory_id"]: self._row_to_metadata(row) for row in rows}
        finally:
            conn.close()

    def record_access_batch(self, memory_ids: list[str]) -> None:
        """Record access for multiple memories in a single query."""
        if not memory_ids:
            return
        
        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ",".join("?" * len(memory_ids))
            conn.execute(
                f"""
                UPDATE memory_metadata
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE memory_id IN ({placeholders})
                """,
                memory_ids,
            )
            conn.commit()
        finally:
            conn.close()
