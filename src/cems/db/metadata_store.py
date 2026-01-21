"""PostgreSQL-based metadata store for server mode.

This replaces the SQLite MetadataStore when DATABASE_URL is configured.
"""

import logging
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from cems.db.database import get_database
from cems.db.models import CategorySummary as PgCategorySummary
from cems.db.models import MaintenanceLog as PgMaintenanceLog
from cems.db.models import MemoryMetadata as PgMemoryMetadata
from cems.models import CategorySummary, MemoryMetadata, MemoryScope

logger = logging.getLogger(__name__)


class PostgresMetadataStore:
    """PostgreSQL store for extended metadata.

    This implements the same interface as MetadataStore but uses PostgreSQL
    via SQLAlchemy instead of SQLite.
    """

    def __init__(self):
        """Initialize PostgreSQL metadata store."""
        self._db = get_database()

    def save_metadata(self, metadata: MemoryMetadata) -> None:
        """Save or update memory metadata."""
        with self._db.session() as session:
            # Use upsert (INSERT ... ON CONFLICT UPDATE)
            stmt = insert(PgMemoryMetadata).values(
                memory_id=metadata.memory_id,
                scope=metadata.scope,
                category=metadata.category,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
                last_accessed=metadata.last_accessed,
                access_count=metadata.access_count,
                source=metadata.source,
                source_ref=metadata.source_ref,
                tags=metadata.tags,
                archived=metadata.archived,
                priority=metadata.priority,
                pinned=metadata.pinned,
                pin_reason=metadata.pin_reason,
                pin_category=metadata.pin_category,
                expires_at=metadata.expires_at,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["memory_id"],
                set_={
                    "category": stmt.excluded.category,
                    "updated_at": stmt.excluded.updated_at,
                    "last_accessed": stmt.excluded.last_accessed,
                    "access_count": stmt.excluded.access_count,
                    "source": stmt.excluded.source,
                    "source_ref": stmt.excluded.source_ref,
                    "tags": stmt.excluded.tags,
                    "archived": stmt.excluded.archived,
                    "priority": stmt.excluded.priority,
                    "pinned": stmt.excluded.pinned,
                    "pin_reason": stmt.excluded.pin_reason,
                    "pin_category": stmt.excluded.pin_category,
                    "expires_at": stmt.excluded.expires_at,
                },
            )
            session.execute(stmt)

    def get_metadata(self, memory_id: str) -> MemoryMetadata | None:
        """Get metadata for a memory."""
        with self._db.session() as session:
            row = session.execute(
                select(PgMemoryMetadata).where(PgMemoryMetadata.memory_id == memory_id)
            ).scalar_one_or_none()
            if row:
                return self._pg_to_metadata(row)
            return None

    def record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed."""
        with self._db.session() as session:
            session.execute(
                update(PgMemoryMetadata)
                .where(PgMemoryMetadata.memory_id == memory_id)
                .values(
                    last_accessed=datetime.now(UTC),
                    access_count=PgMemoryMetadata.access_count + 1,
                )
            )

    def get_stale_memories(self, user_id: str, days: int) -> list[str]:
        """Get memory IDs not accessed in N days (excludes pinned memories)."""
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(days=days)
        with self._db.session() as session:
            rows = session.execute(
                select(PgMemoryMetadata.memory_id).where(
                    PgMemoryMetadata.archived == False,  # noqa: E712
                    PgMemoryMetadata.pinned == False,  # noqa: E712
                    PgMemoryMetadata.last_accessed < cutoff,
                )
            ).scalars().all()
            return list(rows)

    def get_hot_memories(self, user_id: str, threshold: int) -> list[str]:
        """Get frequently accessed memory IDs."""
        with self._db.session() as session:
            rows = session.execute(
                select(PgMemoryMetadata.memory_id).where(
                    PgMemoryMetadata.archived == False,  # noqa: E712
                    PgMemoryMetadata.access_count >= threshold,
                )
            ).scalars().all()
            return list(rows)

    def archive_memory(self, memory_id: str) -> None:
        """Mark a memory as archived."""
        with self._db.session() as session:
            session.execute(
                update(PgMemoryMetadata)
                .where(PgMemoryMetadata.memory_id == memory_id)
                .values(archived=True)
            )

    def increase_priority(self, memory_id: str, boost: float = 0.1) -> None:
        """Increase memory priority."""
        with self._db.session() as session:
            # Get current priority
            row = session.execute(
                select(PgMemoryMetadata.priority).where(
                    PgMemoryMetadata.memory_id == memory_id
                )
            ).scalar_one_or_none()
            if row is not None:
                new_priority = min(row + boost, 2.0)
                session.execute(
                    update(PgMemoryMetadata)
                    .where(PgMemoryMetadata.memory_id == memory_id)
                    .values(priority=new_priority)
                )

    def delete_metadata(self, memory_id: str) -> None:
        """Delete metadata for a memory."""
        with self._db.session() as session:
            session.execute(
                select(PgMemoryMetadata).where(PgMemoryMetadata.memory_id == memory_id)
            )
            row = session.execute(
                select(PgMemoryMetadata).where(PgMemoryMetadata.memory_id == memory_id)
            ).scalar_one_or_none()
            if row:
                session.delete(row)

    def log_maintenance(
        self, job_type: str, user_id: str, status: str, details: str | None = None
    ) -> int:
        """Log a maintenance job execution."""
        with self._db.session() as session:
            log = PgMaintenanceLog(
                job_type=job_type,
                scope="personal",
                status=status,
                details={"user_id": user_id, "message": details} if details else None,
            )
            session.add(log)
            session.flush()
            return 0  # PostgreSQL uses UUID, not int ID

    def update_maintenance_log(
        self, log_id: int, status: str, details: str | None = None
    ) -> None:
        """Update a maintenance log entry (no-op for PostgreSQL UUID-based logs)."""
        # PostgreSQL version uses UUIDs and different schema
        pass

    def _pg_to_metadata(self, row: PgMemoryMetadata) -> MemoryMetadata:
        """Convert PostgreSQL model to MemoryMetadata dataclass."""
        return MemoryMetadata(
            memory_id=row.memory_id,
            user_id=str(row.user_id) if row.user_id else "unknown",
            scope=MemoryScope(row.scope),
            category=row.category,
            created_at=row.created_at,
            updated_at=row.updated_at,
            last_accessed=row.last_accessed,
            access_count=row.access_count,
            source=row.source,
            source_ref=row.source_ref,
            tags=row.tags or [],
            archived=row.archived,
            priority=row.priority,
            pinned=row.pinned,
            pin_reason=row.pin_reason,
            pin_category=row.pin_category,
            expires_at=row.expires_at,
        )

    def pin_memory(
        self, memory_id: str, reason: str, pin_category: str = "guideline"
    ) -> None:
        """Pin a memory to prevent it from being auto-pruned."""
        with self._db.session() as session:
            session.execute(
                update(PgMemoryMetadata)
                .where(PgMemoryMetadata.memory_id == memory_id)
                .values(pinned=True, pin_reason=reason, pin_category=pin_category)
            )

    def unpin_memory(self, memory_id: str) -> None:
        """Unpin a memory."""
        with self._db.session() as session:
            session.execute(
                update(PgMemoryMetadata)
                .where(PgMemoryMetadata.memory_id == memory_id)
                .values(pinned=False, pin_reason=None, pin_category=None)
            )

    def get_pinned_memories(
        self, user_id: str, pin_category: str | None = None
    ) -> list[str]:
        """Get all pinned memory IDs for a user."""
        with self._db.session() as session:
            query = select(PgMemoryMetadata.memory_id).where(
                PgMemoryMetadata.pinned == True  # noqa: E712
            )
            if pin_category:
                query = query.where(PgMemoryMetadata.pin_category == pin_category)
            rows = session.execute(query).scalars().all()
            return list(rows)

    def get_memories_by_category(
        self, user_id: str, category: str, scope: MemoryScope | None = None
    ) -> list[str]:
        """Get memory IDs for a category."""
        with self._db.session() as session:
            query = select(PgMemoryMetadata.memory_id).where(
                PgMemoryMetadata.category == category,
                PgMemoryMetadata.archived == False,  # noqa: E712
            )
            if scope:
                query = query.where(PgMemoryMetadata.scope == scope.value)
            rows = session.execute(query).scalars().all()
            return list(rows)

    def get_all_user_memories(
        self, user_id: str, include_archived: bool = False
    ) -> list[str]:
        """Get all memory IDs for a user."""
        with self._db.session() as session:
            query = select(PgMemoryMetadata.memory_id)
            if not include_archived:
                query = query.where(PgMemoryMetadata.archived == False)  # noqa: E712
            rows = session.execute(query).scalars().all()
            return list(rows)

    def get_recent_memories(self, user_id: str, hours: int = 24) -> list[str]:
        """Get memories created in the last N hours."""
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        with self._db.session() as session:
            rows = session.execute(
                select(PgMemoryMetadata.memory_id).where(
                    PgMemoryMetadata.created_at > cutoff,
                    PgMemoryMetadata.archived == False,  # noqa: E712
                )
            ).scalars().all()
            return list(rows)

    def get_all_categories(
        self, user_id: str, scope: MemoryScope | None = None
    ) -> list[str]:
        """Get all unique categories for a user."""
        with self._db.session() as session:
            query = select(PgMemoryMetadata.category).distinct()
            if scope:
                query = query.where(PgMemoryMetadata.scope == scope.value)
            rows = session.execute(query).scalars().all()
            return list(rows)

    def get_category_counts(
        self, user_id: str, scope: MemoryScope | None = None
    ) -> dict[str, int]:
        """Get category counts with a single GROUP BY query.
        
        Much faster than iterating over memories and calling get_metadata() for each.
        """
        from sqlalchemy import func
        
        with self._db.session() as session:
            query = (
                select(PgMemoryMetadata.category, func.count(PgMemoryMetadata.memory_id))
                .where(PgMemoryMetadata.archived == False)  # noqa: E712
                .group_by(PgMemoryMetadata.category)
            )
            if scope:
                query = query.where(PgMemoryMetadata.scope == scope.value)
            rows = session.execute(query).all()
            return {cat: count for cat, count in rows if cat}

    def get_recently_accessed(
        self, user_id: str, limit: int = 10, scope: MemoryScope | None = None
    ) -> list[MemoryMetadata]:
        """Get the most recently accessed memories."""
        with self._db.session() as session:
            query = (
                select(PgMemoryMetadata)
                .where(PgMemoryMetadata.archived == False)  # noqa: E712
                .order_by(PgMemoryMetadata.last_accessed.desc())
                .limit(limit)
            )
            if scope:
                query = query.where(PgMemoryMetadata.scope == scope.value)
            rows = session.execute(query).scalars().all()
            return [self._pg_to_metadata(row) for row in rows]

    def get_category_summary(
        self, user_id: str, category: str, scope: MemoryScope
    ) -> CategorySummary | None:
        """Get a category summary."""
        with self._db.session() as session:
            row = session.execute(
                select(PgCategorySummary).where(
                    PgCategorySummary.category == category,
                    PgCategorySummary.scope == scope.value,
                )
            ).scalar_one_or_none()
            if row:
                return CategorySummary(
                    user_id=str(row.scope_id) if row.scope_id else user_id,
                    category=row.category,
                    scope=MemoryScope(row.scope),
                    summary=row.summary,
                    item_count=row.item_count,
                    last_updated=row.last_updated,
                    version=row.version,
                )
            return None

    def save_category_summary(self, summary: CategorySummary) -> None:
        """Save or update a category summary."""
        with self._db.session() as session:
            stmt = insert(PgCategorySummary).values(
                scope=summary.scope.value,
                category=summary.category,
                summary=summary.summary,
                item_count=summary.item_count,
                last_updated=summary.last_updated,
                version=summary.version,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["scope", "scope_id", "category"],
                set_={
                    "summary": stmt.excluded.summary,
                    "item_count": stmt.excluded.item_count,
                    "last_updated": stmt.excluded.last_updated,
                    "version": stmt.excluded.version,
                },
            )
            session.execute(stmt)

    def get_all_category_summaries(
        self, user_id: str, scope: MemoryScope | None = None
    ) -> list[CategorySummary]:
        """Get all category summaries for a user."""
        with self._db.session() as session:
            query = select(PgCategorySummary)
            if scope:
                query = query.where(PgCategorySummary.scope == scope.value)
            rows = session.execute(query).scalars().all()
            return [
                CategorySummary(
                    user_id=str(row.scope_id) if row.scope_id else user_id,
                    category=row.category,
                    scope=MemoryScope(row.scope),
                    summary=row.summary,
                    item_count=row.item_count,
                    last_updated=row.last_updated,
                    version=row.version,
                )
                for row in rows
            ]
