"""Database connection and session management for CEMS."""

import hashlib
import logging
import time
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
        # =====================================================================
        # CRITICAL: Create core memory tables if they don't exist.
        #
        # These tables are NOT in SQLAlchemy models (Base.metadata), so
        # create_tables() does NOT create them. Without this migration, any
        # fresh CEMS deployment will 500 on every memory operation because
        # memory_documents and memory_chunks simply don't exist.
        #
        # This was the cause of a production outage on the hubstaff-server
        # CEMS instance (2026-03-27) — the Docker image was up to date but
        # the database had no memory tables.
        # =====================================================================
        (
            "core_memory_tables_v1",
            """
            -- Enable pgvector extension (required for embedding columns)
            CREATE EXTENSION IF NOT EXISTS vector;

            -- Document-level storage (deduplicated by content_hash)
            CREATE TABLE IF NOT EXISTS memory_documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                scope TEXT NOT NULL DEFAULT 'shared',
                category TEXT NOT NULL DEFAULT 'document',
                title TEXT,
                source TEXT,
                source_ref TEXT,
                tags TEXT[] DEFAULT '{}',
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content_bytes INT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT valid_doc_scope CHECK (scope IN ('personal', 'shared'))
            );

            -- Indexes for memory_documents
            CREATE UNIQUE INDEX IF NOT EXISTS memory_documents_hash_user_idx
                ON memory_documents (content_hash, user_id);
            CREATE INDEX IF NOT EXISTS memory_documents_user_id_idx ON memory_documents(user_id);
            CREATE INDEX IF NOT EXISTS memory_documents_scope_idx ON memory_documents(scope);
            CREATE INDEX IF NOT EXISTS memory_documents_category_idx ON memory_documents(category);
            CREATE INDEX IF NOT EXISTS memory_documents_tags_idx ON memory_documents USING gin(tags);
            CREATE INDEX IF NOT EXISTS memory_documents_source_ref_idx ON memory_documents(source_ref);
            CREATE INDEX IF NOT EXISTS memory_documents_created_at_idx ON memory_documents(created_at);

            -- Chunked content with embeddings for vector search
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,
                seq INT NOT NULL,
                pos INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                tokens INT,
                bytes INT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT valid_seq CHECK (seq >= 0),
                CONSTRAINT valid_pos CHECK (pos >= 0)
            );

            CREATE INDEX IF NOT EXISTS memory_chunks_doc_seq_idx
                ON memory_chunks (document_id, seq);
            CREATE INDEX IF NOT EXISTS memory_chunks_embedding_hnsw_idx
                ON memory_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m=16, ef_construction=64);

            -- Graph relations between documents
            CREATE TABLE IF NOT EXISTS memory_relations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_id UUID NOT NULL,
                target_id UUID NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'related',
                weight FLOAT NOT NULL DEFAULT 1.0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT no_self_relation CHECK (source_id != target_id)
            );

            CREATE INDEX IF NOT EXISTS memory_relations_source_idx ON memory_relations(source_id);
            CREATE INDEX IF NOT EXISTS memory_relations_target_idx ON memory_relations(target_id);
            CREATE INDEX IF NOT EXISTS memory_relations_type_idx ON memory_relations(relation_type);
            """,
        ),
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
        # Add soft-delete and feedback tracking to memory_documents
        (
            "soft_delete_feedback_v1",
            """
            DO $$
            BEGIN
                -- Add deleted_at for soft-delete
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memory_documents' AND column_name = 'deleted_at'
                ) THEN
                    ALTER TABLE memory_documents ADD COLUMN deleted_at TIMESTAMP WITH TIME ZONE;
                    RAISE NOTICE 'Added memory_documents.deleted_at';
                END IF;

                -- Add shown_count for feedback tracking
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memory_documents' AND column_name = 'shown_count'
                ) THEN
                    ALTER TABLE memory_documents ADD COLUMN shown_count INT NOT NULL DEFAULT 0;
                    RAISE NOTICE 'Added memory_documents.shown_count';
                END IF;

                -- Add last_shown_at for feedback tracking
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memory_documents' AND column_name = 'last_shown_at'
                ) THEN
                    ALTER TABLE memory_documents ADD COLUMN last_shown_at TIMESTAMP WITH TIME ZONE;
                    RAISE NOTICE 'Added memory_documents.last_shown_at';
                END IF;
            END $$;
            """,
        ),
        # Replace unique content_hash index with partial index (excludes soft-deleted)
        (
            "soft_delete_partial_index_v1",
            """
            DO $$
            BEGIN
                -- Only recreate if deleted_at column exists (migration above ran)
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memory_documents' AND column_name = 'deleted_at'
                ) THEN
                    DROP INDEX IF EXISTS memory_documents_hash_user_idx;
                    CREATE UNIQUE INDEX IF NOT EXISTS memory_documents_hash_user_idx
                        ON memory_documents (content_hash, user_id)
                        WHERE deleted_at IS NULL;
                    RAISE NOTICE 'Recreated memory_documents_hash_user_idx as partial index';
                END IF;
            END $$;
            """,
        ),
        # Fix memory_relations FK to reference memory_documents instead of memories
        (
            "memory_relations_fk_fix_v1",
            """
            DO $$
            BEGIN
                -- Only fix if memory_relations table exists
                IF EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'memory_relations'
                ) THEN
                    -- Drop old FKs (may reference memories.id)
                    ALTER TABLE memory_relations
                        DROP CONSTRAINT IF EXISTS memory_relations_source_id_fkey;
                    ALTER TABLE memory_relations
                        DROP CONSTRAINT IF EXISTS memory_relations_target_id_fkey;

                    -- Add correct FKs referencing memory_documents.id
                    ALTER TABLE memory_relations
                        ADD CONSTRAINT memory_relations_source_id_fkey
                        FOREIGN KEY (source_id) REFERENCES memory_documents(id)
                        ON DELETE CASCADE;
                    ALTER TABLE memory_relations
                        ADD CONSTRAINT memory_relations_target_id_fkey
                        FOREIGN KEY (target_id) REFERENCES memory_documents(id)
                        ON DELETE CASCADE;

                    RAISE NOTICE 'Fixed memory_relations FKs to reference memory_documents';
                END IF;
            END $$;
            """,
        ),
        # Add relevance feedback columns to memory_documents
        (
            "relevance_feedback_v2",
            """
            ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS relevant_count INT NOT NULL DEFAULT 0;
            ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS noise_count INT NOT NULL DEFAULT 0;
            ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS noise_snippet_count INT NOT NULL DEFAULT 0;
            """,
        ),
        # Add content_detailed to memory_documents (used by distillation)
        (
            "content_detailed_v1",
            """
            ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS content_detailed TEXT;
            """,
        ),
        # Add content_tsv to memory_chunks (used by hybrid/full-text search)
        (
            "content_tsv_v1",
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memory_chunks' AND column_name = 'content_tsv'
                ) THEN
                    ALTER TABLE memory_chunks ADD COLUMN content_tsv tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
                    RAISE NOTICE 'Added memory_chunks.content_tsv';
                END IF;
            END $$;
            CREATE INDEX IF NOT EXISTS memory_chunks_tsv_idx
                ON memory_chunks USING gin(content_tsv);
            """,
        ),
        # Add similarity column to memory_relations (used by graph traversal queries)
        (
            "memory_relations_similarity_v1",
            """
            ALTER TABLE memory_relations ADD COLUMN IF NOT EXISTS similarity FLOAT;
            """,
        ),
        # Fix memory_relations FK to point to memory_documents (not legacy memories table)
        # and ensure composite PK for ON CONFLICT upsert in relation builder
        (
            "memory_relations_fk_fix_v2",
            """
            DO $$
            DECLARE
                ref_table TEXT;
                pk_cols TEXT;
            BEGIN
                -- Check if source FK points to wrong table
                SELECT ccu.table_name INTO ref_table
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_name = 'memory_relations'
                  AND tc.constraint_type = 'FOREIGN KEY'
                  AND tc.constraint_name = 'memory_relations_source_id_fkey'
                LIMIT 1;

                IF ref_table IS NOT NULL AND ref_table != 'memory_documents' THEN
                    -- Clear orphaned rows, fix FKs
                    DELETE FROM memory_relations
                    WHERE source_id NOT IN (SELECT id FROM memory_documents)
                       OR target_id NOT IN (SELECT id FROM memory_documents);
                    ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_source_id_fkey;
                    ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_target_id_fkey;
                    ALTER TABLE memory_relations ADD CONSTRAINT memory_relations_source_id_fkey
                        FOREIGN KEY (source_id) REFERENCES memory_documents(id) ON DELETE CASCADE;
                    ALTER TABLE memory_relations ADD CONSTRAINT memory_relations_target_id_fkey
                        FOREIGN KEY (target_id) REFERENCES memory_documents(id) ON DELETE CASCADE;
                    RAISE NOTICE 'Fixed memory_relations FKs: % -> memory_documents', ref_table;
                END IF;

                -- Ensure composite PK (not surrogate id)
                SELECT string_agg(a.attname, ',' ORDER BY array_position(i.indkey, a.attnum))
                INTO pk_cols
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = 'memory_relations'::regclass AND i.indisprimary;

                IF pk_cols IS DISTINCT FROM 'source_id,target_id,relation_type' THEN
                    ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_pkey;
                    ALTER TABLE memory_relations DROP COLUMN IF EXISTS id;
                    ALTER TABLE memory_relations ADD PRIMARY KEY (source_id, target_id, relation_type);
                    RAISE NOTICE 'Fixed memory_relations PK: % -> composite', COALESCE(pk_cols, 'none');
                END IF;
            END $$;
            """,
        ),
        # Memory conflicts table (detected during consolidation, queried by wiki dashboard)
        (
            "memory_conflicts_v1",
            """
            CREATE TABLE IF NOT EXISTS memory_conflicts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                doc_a_id UUID NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,
                doc_b_id UUID NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,
                explanation TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE,
                UNIQUE(doc_a_id, doc_b_id)
            );

            CREATE INDEX IF NOT EXISTS idx_conflicts_user_status
                ON memory_conflicts(user_id, status);
            """,
        ),
        # Remove team_id concept — shared memories are now visible to all users
        (
            "remove_team_id_v1",
            """
            -- Drop team_id from memory_documents (safe if column already absent)
            DROP INDEX IF EXISTS memory_documents_team_id_idx;
            ALTER TABLE memory_documents DROP COLUMN IF EXISTS team_id;

            -- Drop team_id from index tables (guarded for fresh installs where tables may not exist)
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'index_jobs') THEN
                    ALTER TABLE index_jobs DROP COLUMN IF EXISTS team_id;
                END IF;
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'index_patterns') THEN
                    ALTER TABLE index_patterns DROP CONSTRAINT IF EXISTS index_patterns_team_id_name_key;
                    ALTER TABLE index_patterns DROP COLUMN IF EXISTS team_id;
                END IF;
            END $$;

            -- Migrate any legacy scope values before tightening the constraint
            UPDATE memory_documents SET scope = 'shared'
                WHERE scope NOT IN ('personal', 'shared');

            -- Tighten scope constraint (remove unused 'team' and 'company' values)
            ALTER TABLE memory_documents DROP CONSTRAINT IF EXISTS valid_doc_scope;
            ALTER TABLE memory_documents ADD CONSTRAINT valid_doc_scope
                CHECK (scope IN ('personal', 'shared'));

            -- Change default scope to 'shared'
            ALTER TABLE memory_documents ALTER COLUMN scope SET DEFAULT 'shared';

            -- Drop team-related tables (FK constraints cascade)
            DROP TABLE IF EXISTS api_keys CASCADE;
            DROP TABLE IF EXISTS team_members CASCADE;
            DROP TABLE IF EXISTS teams CASCADE;
            """,
        ),
    ]

    # --- Migration runner with tracking, advisory lock, and checksums ---

    # 1. Ensure schema_migrations tracking table exists
    with db.session() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                migration_id VARCHAR(255) PRIMARY KEY,
                checksum VARCHAR(64) NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                execution_ms INTEGER
            );
        """))

    # 2. Acquire advisory lock to prevent concurrent migration runs
    #    (e.g., two containers starting simultaneously)
    MIGRATION_LOCK_ID = 728379  # arbitrary constant
    with db.session() as session:
        session.execute(text(f"SELECT pg_advisory_lock({MIGRATION_LOCK_ID})"))

    try:
        # 3. Get already-applied migrations with checksums
        applied: dict[str, str] = {}
        with db.session() as session:
            rows = session.execute(
                text("SELECT migration_id, checksum FROM schema_migrations")
            ).all()
            applied = {row[0]: row[1] for row in rows}

        pending = len(migrations) - len(applied)
        if applied:
            logger.info(f"Schema migrations: {len(applied)} applied, {pending} pending")

        for migration_id, sql in migrations:
            checksum = hashlib.sha256(sql.encode()).hexdigest()[:16]

            if migration_id in applied:
                # Verify checksum — detect accidental modification of applied migrations
                if applied[migration_id] != checksum:
                    logger.warning(
                        f"Migration {migration_id} checksum mismatch "
                        f"(applied={applied[migration_id]}, current={checksum}). "
                        f"Re-running since all migrations are idempotent."
                    )
                else:
                    continue  # Already applied, checksum matches — skip

            try:
                start = time.monotonic()
                with db.session() as session:
                    session.execute(text(sql))
                    session.execute(
                        text(
                            "INSERT INTO schema_migrations (migration_id, checksum, execution_ms) "
                            "VALUES (:id, :checksum, :ms) "
                            "ON CONFLICT (migration_id) DO UPDATE SET checksum = :checksum, execution_ms = :ms"
                        ),
                        {"id": migration_id, "checksum": checksum, "ms": int((time.monotonic() - start) * 1000)},
                    )
                elapsed = int((time.monotonic() - start) * 1000)
                logger.info(f"Migration applied: {migration_id} ({elapsed}ms)")
            except Exception as e:
                logger.error(f"Migration {migration_id} failed: {e}")
    finally:
        # 4. Release advisory lock
        with db.session() as session:
            session.execute(text(f"SELECT pg_advisory_unlock({MIGRATION_LOCK_ID})"))
