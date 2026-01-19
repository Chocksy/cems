"""Database connection and session management for CEMS."""

import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from cems.db.models import Base

logger = logging.getLogger(__name__)

# Global database instance
_database: "Database | None" = None


class Database:
    """Database connection manager supporting both sync and async operations."""

    def __init__(self, database_url: str):
        """Initialize database connection.

        Args:
            database_url: PostgreSQL connection URL.
                Sync: postgresql://user:pass@host:port/db
                Async: postgresql+asyncpg://user:pass@host:port/db
        """
        # Normalize URL: postgres:// -> postgresql:// (SQLAlchemy requires full name)
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        self.database_url = database_url

        # Create sync engine (for CLI operations)
        sync_url = database_url.replace("+asyncpg", "")
        self.sync_engine = create_engine(sync_url, echo=False)
        self.sync_session_factory = sessionmaker(
            bind=self.sync_engine, expire_on_commit=False
        )

        # Create async engine (for server operations)
        if "+asyncpg" not in database_url:
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        else:
            async_url = database_url
        self.async_engine = create_async_engine(async_url, echo=False)
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine, expire_on_commit=False
        )

        logger.info(f"Database initialized: {self._mask_url(database_url)}")

    def _mask_url(self, url: str) -> str:
        """Mask password in URL for logging."""
        import re

        return re.sub(r":([^:@]+)@", r":****@", url)

    def create_tables(self) -> None:
        """Create all tables (sync)."""
        Base.metadata.create_all(self.sync_engine)
        logger.info("Database tables created")

    async def create_tables_async(self) -> None:
        """Create all tables (async)."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a sync database session."""
        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def health_check(self) -> bool:
        """Check if database is accessible."""
        try:
            with self.session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def health_check_async(self) -> bool:
        """Check if database is accessible (async)."""
        try:
            async with self.async_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def close(self) -> None:
        """Close database connections."""
        self.sync_engine.dispose()
        logger.info("Database connections closed")

    async def close_async(self) -> None:
        """Close async database connections."""
        await self.async_engine.dispose()
        logger.info("Async database connections closed")


def init_database(database_url: str) -> Database:
    """Initialize the global database instance.

    Args:
        database_url: PostgreSQL connection URL.

    Returns:
        Database instance.
    """
    global _database
    _database = Database(database_url)
    return _database


def get_database() -> Database:
    """Get the global database instance.

    Returns:
        Database instance.

    Raises:
        RuntimeError: If database not initialized.
    """
    if _database is None:
        raise RuntimeError(
            "Database not initialized. Call init_database() first or set CEMS_DATABASE_URL."
        )
    return _database


def is_database_initialized() -> bool:
    """Check if database is initialized."""
    return _database is not None


def run_migrations() -> None:
    """Run any pending database migrations.

    This is called on server startup to ensure schema is up to date.
    """
    db = get_database()

    migrations = [
        # Fix api_key_prefix column size (10 -> 20 chars)
        (
            "api_key_prefix_size_v1",
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'users'
                    AND column_name = 'api_key_prefix'
                    AND character_maximum_length = 10
                ) THEN
                    ALTER TABLE users ALTER COLUMN api_key_prefix TYPE VARCHAR(20);
                    RAISE NOTICE 'Migrated users.api_key_prefix to VARCHAR(20)';
                END IF;

                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'api_keys'
                    AND column_name = 'key_prefix'
                    AND character_maximum_length = 10
                ) THEN
                    ALTER TABLE api_keys ALTER COLUMN key_prefix TYPE VARCHAR(20);
                    RAISE NOTICE 'Migrated api_keys.key_prefix to VARCHAR(20)';
                END IF;
            END $$;
            """,
        ),
    ]

    with db.session() as session:
        for migration_id, sql in migrations:
            try:
                session.execute(text(sql))
                logger.info(f"Migration applied: {migration_id}")
            except Exception as e:
                logger.warning(f"Migration {migration_id} skipped or failed: {e}")
