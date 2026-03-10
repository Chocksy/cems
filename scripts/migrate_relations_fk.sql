-- Migrate memory_relations FK from memories(id) to memory_documents(id)
-- The memory_relations table was created with FKs pointing to the orphaned
-- `memories` table. Fix them to point to `memory_documents`.

-- Drop old FKs (referencing memories.id)
ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_source_id_fkey;
ALTER TABLE memory_relations DROP CONSTRAINT IF EXISTS memory_relations_target_id_fkey;

-- Add new FKs (referencing memory_documents.id)
ALTER TABLE memory_relations
    ADD CONSTRAINT memory_relations_source_id_fkey
    FOREIGN KEY (source_id) REFERENCES memory_documents(id) ON DELETE CASCADE;

ALTER TABLE memory_relations
    ADD CONSTRAINT memory_relations_target_id_fkey
    FOREIGN KEY (target_id) REFERENCES memory_documents(id) ON DELETE CASCADE;
