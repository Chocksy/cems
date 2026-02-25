-- CEMS Migration: Memory Conflicts Table
-- Stores conflicts detected during consolidation for user resolution.
--
-- Run: psql -d cems -f scripts/migrate_conflicts.sql

-- =============================================================================
-- MEMORY CONFLICTS TABLE
-- =============================================================================
-- Detected when two memories contradict each other (e.g., different values
-- for the same attribute). Created by ConsolidationJob, surfaced in
-- SessionStart profile, resolved via API.

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

-- =============================================================================
-- VERIFICATION
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'memory_conflicts'
    ) THEN
        RAISE EXCEPTION 'memory_conflicts table was not created';
    END IF;
    RAISE NOTICE 'SUCCESS: memory_conflicts table created';
END $$;
