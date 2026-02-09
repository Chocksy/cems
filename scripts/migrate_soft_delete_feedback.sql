-- CEMS Migration: Soft-Delete + Feedback Tracking
-- Adds columns for soft-delete and memory usage feedback.
--
-- Run: psql -d cems -f scripts/migrate_soft_delete_feedback.sql

-- =============================================================================
-- SOFT-DELETE: deleted_at column
-- =============================================================================
-- Soft-deleted documents are hidden from queries but preserved for recovery.
-- Hard delete (DELETE FROM) is still available for permanent removal.

ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;

-- Replace unique content_hash index with partial index (excludes soft-deleted)
-- This allows the same content to be re-added after soft-delete.
DROP INDEX IF EXISTS memory_documents_hash_user_idx;
CREATE UNIQUE INDEX memory_documents_hash_user_idx
    ON memory_documents (content_hash, user_id)
    WHERE deleted_at IS NULL;

-- =============================================================================
-- FEEDBACK TRACKING: shown_count + last_shown_at
-- =============================================================================
-- Track how often a memory is surfaced in search results.
-- Feeds into time decay scoring (last_shown_at replaces created_at for recency).

ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS shown_count INT NOT NULL DEFAULT 0;
ALTER TABLE memory_documents ADD COLUMN IF NOT EXISTS last_shown_at TIMESTAMP WITH TIME ZONE;

-- =============================================================================
-- VERIFICATION
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_documents' AND column_name = 'deleted_at'
    ) THEN
        RAISE EXCEPTION 'deleted_at column was not added';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_documents' AND column_name = 'shown_count'
    ) THEN
        RAISE EXCEPTION 'shown_count column was not added';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_documents' AND column_name = 'last_shown_at'
    ) THEN
        RAISE EXCEPTION 'last_shown_at column was not added';
    END IF;
    RAISE NOTICE 'SUCCESS: soft-delete and feedback columns added to memory_documents';
END $$;
