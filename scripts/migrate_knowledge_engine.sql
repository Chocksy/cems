-- Knowledge Engine Migration Script
-- Run this on production BEFORE deploying the new version.
-- Safe to re-run (all operations are idempotent).
--
-- Usage (via SSH to Hetzner):
--   docker exec -i <postgres-container> psql -U cems -d cems < migrate_knowledge_engine.sql

-- Step 1: Fix memory_relations FK (currently points to 'memories', should be 'memory_documents')
-- First, clear any orphaned relations that reference old 'memories' table IDs
DELETE FROM memory_relations
WHERE source_id NOT IN (SELECT id FROM memory_documents)
   OR target_id NOT IN (SELECT id FROM memory_documents);

-- Drop old FKs (may fail silently if already correct — that's fine)
ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_source_id_fkey;
ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_target_id_fkey;

-- Add correct FKs pointing to memory_documents
ALTER TABLE memory_relations ADD CONSTRAINT memory_relations_source_id_fkey
    FOREIGN KEY (source_id) REFERENCES memory_documents(id) ON DELETE CASCADE;
ALTER TABLE memory_relations ADD CONSTRAINT memory_relations_target_id_fkey
    FOREIGN KEY (target_id) REFERENCES memory_documents(id) ON DELETE CASCADE;

-- Step 2: Ensure composite PK on (source_id, target_id, relation_type)
-- The legacy schema (database.py) has a surrogate 'id UUID PRIMARY KEY'.
-- We need the composite PK for ON CONFLICT upsert to work.
DO $$
DECLARE
    pk_columns TEXT;
BEGIN
    -- Get current PK column list
    SELECT string_agg(a.attname, ',' ORDER BY array_position(i.indkey, a.attnum))
    INTO pk_columns
    FROM pg_index i
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
    WHERE i.indrelid = 'memory_relations'::regclass AND i.indisprimary;

    IF pk_columns = 'source_id,target_id,relation_type' THEN
        RAISE NOTICE 'Composite PK already correct: %', pk_columns;
    ELSE
        RAISE NOTICE 'Current PK: %. Replacing with composite PK.', COALESCE(pk_columns, 'none');
        -- Drop existing PK (whether surrogate id or wrong composite)
        ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_pkey;
        -- Drop the id column if it exists (legacy surrogate key)
        ALTER TABLE memory_relations DROP COLUMN IF EXISTS id;
        -- Add composite PK
        ALTER TABLE memory_relations ADD PRIMARY KEY (source_id, target_id, relation_type);
        RAISE NOTICE 'Composite PK (source_id, target_id, relation_type) created';
    END IF;
END $$;

-- Step 3: Verify the migration
SELECT 'FK check:' AS check_name,
       tc.constraint_name,
       ccu.table_name AS referenced_table
FROM information_schema.table_constraints tc
JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
WHERE tc.table_name = 'memory_relations' AND tc.constraint_type = 'FOREIGN KEY';
