"""PostgreSQL-based metadata store for server mode.

This replaces the SQLite MetadataStore when DATABASE_URL is configured.

NOTE: Many methods were removed because they read from the orphaned
memory_metadata table while all data lives in memory_documents via
DocumentStore. Removed methods: get_stale_memories, get_hot_memories,
get_recent_memories, get_old_memories, get_all_user_memories,
get_memories_by_category, get_metadata_batch, log_maintenance,
update_maintenance_log, save_category_summary.
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from cems.db.database import get_database
from cems.db.models import CategorySummary as PgCategorySummary
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
        # Convert user_id to UUID for the DB column
        user_uuid = None
        if metadata.user_id and metadata.user_id != "unknown":
            try:
                user_uuid = UUID(metadata.user_id)
            except ValueError:
                pass

        with self._db.session() as session:
            # Use upsert (INSERT ... ON CONFLICT UPDATE)
            stmt = insert(PgMemoryMetadata).values(
                memory_id=metadata.memory_id,
                user_id=user_uuid,
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
                    "user_id": stmt.excluded.user_id,
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

    def get_all_categories(
        self, user_id: str, scope: MemoryScope | None = None
    ) -> list[str]:
        """Get all unique categories for a user."""
        with self._db.session() as session:
            query = select(PgMemoryMetadata.category).distinct().where(
                PgMemoryMetadata.user_id == UUID(user_id),
            )
            if scope:
                query = query.where(PgMemoryMetadata.scope == scope.value)
            rows = session.execute(query).scalars().all()
            return list(rows)

    def get_recently_accessed(
        self, user_id: str, limit: int = 10, scope: MemoryScope | None = None
    ) -> list[MemoryMetadata]:
        """Get the most recently accessed memories."""
        with self._db.session() as session:
            query = (
                select(PgMemoryMetadata)
                .where(
                    PgMemoryMetadata.user_id == UUID(user_id),
                    PgMemoryMetadata.archived == False,  # noqa: E712
                )
                .order_by(PgMemoryMetadata.last_accessed.desc())
                .limit(limit)
            )
            if scope:
                query = query.where(PgMemoryMetadata.scope == scope.value)
            rows = session.execute(query).scalars().all()
            return [self._pg_to_metadata(row) for row in rows]

    @staticmethod
    def _row_to_category_summary(row, fallback_user_id: str) -> CategorySummary:
        """Convert a PgCategorySummary row to a CategorySummary model."""
        return CategorySummary(
            user_id=str(row.scope_id) if row.scope_id else fallback_user_id,
            category=row.category,
            scope=MemoryScope(row.scope),
            summary=row.summary,
            item_count=row.item_count,
            last_updated=row.last_updated,
            version=row.version,
        )

    def get_category_summary(
        self, user_id: str, category: str, scope: MemoryScope
    ) -> CategorySummary | None:
        """Get a category summary."""
        with self._db.session() as session:
            row = session.execute(
                select(PgCategorySummary).where(
                    PgCategorySummary.category == category,
                    PgCategorySummary.scope == scope.value,
                    PgCategorySummary.scope_id == UUID(user_id),
                )
            ).scalar_one_or_none()
            if row:
                return self._row_to_category_summary(row, user_id)
            return None

    def get_all_category_summaries(
        self, user_id: str, scope: MemoryScope | None = None
    ) -> list[CategorySummary]:
        """Get all category summaries for a user."""
        with self._db.session() as session:
            query = select(PgCategorySummary).where(
                PgCategorySummary.scope_id == UUID(user_id),
            )
            if scope:
                query = query.where(PgCategorySummary.scope == scope.value)
            rows = session.execute(query).scalars().all()
            return [self._row_to_category_summary(row, user_id) for row in rows]
