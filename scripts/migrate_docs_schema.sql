-- CEMS Document + Chunks Schema Migration
-- This migration creates the new document-centric storage model:
-- - memory_documents: Holds the full document metadata (deduplicated by content_hash)
-- - memory_chunks: Holds chunked content with embeddings for search
--
-- Benefits:
-- - No truncation: Long documents are chunked (800 tokens each)
-- - Better recall: Search returns relevant chunks, not entire documents
-- - Deduplication: Same content won't be stored twice
--
-- Run: psql -d cems -f scripts/migrate_docs_schema.sql

-- =============================================================================
-- MEMORY DOCUMENTS TABLE
-- =============================================================================
-- Stores document-level metadata. Each document can have multiple chunks.
-- content_hash ensures deduplication.

CREATE TABLE IF NOT EXISTS memory_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    scope TEXT NOT NULL DEFAULT 'personal',
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
    CONSTRAINT valid_doc_scope CHECK (scope IN ('personal', 'shared', 'team', 'company'))
);

-- Unique constraint on content_hash for deduplication
CREATE UNIQUE INDEX IF NOT EXISTS memory_documents_hash_user_idx
    ON memory_documents (content_hash, user_id);

-- Standard lookup indexes
CREATE INDEX IF NOT EXISTS memory_documents_user_id_idx ON memory_documents(user_id);
CREATE INDEX IF NOT EXISTS memory_documents_team_id_idx ON memory_documents(team_id);
CREATE INDEX IF NOT EXISTS memory_documents_scope_idx ON memory_documents(scope);
CREATE INDEX IF NOT EXISTS memory_documents_category_idx ON memory_documents(category);
CREATE INDEX IF NOT EXISTS memory_documents_tags_idx ON memory_documents USING gin(tags);
CREATE INDEX IF NOT EXISTS memory_documents_source_ref_idx ON memory_documents(source_ref);
CREATE INDEX IF NOT EXISTS memory_documents_created_at_idx ON memory_documents(created_at);

-- =============================================================================
-- MEMORY CHUNKS TABLE
-- =============================================================================
-- Stores chunked content with embeddings. Each chunk belongs to a document.
-- seq: Chunk sequence number within the document (0-indexed)
-- pos: Character position in the original document

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

-- Index for fetching chunks by document in order
CREATE INDEX IF NOT EXISTS memory_chunks_doc_seq_idx
    ON memory_chunks (document_id, seq);

-- HNSW index for fast approximate nearest neighbor search on embeddings
-- Using cosine distance (most common for text embeddings)
-- Parameters: m=16 (max connections), ef_construction=64 (build quality)
CREATE INDEX IF NOT EXISTS memory_chunks_embedding_hnsw_idx
    ON memory_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);

-- GIN index for full-text search on chunk content
ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX IF NOT EXISTS memory_chunks_tsv_idx ON memory_chunks USING gin(content_tsv);

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- View: Chunks with document metadata (for search results)
CREATE OR REPLACE VIEW memory_chunks_with_docs AS
SELECT
    c.id AS chunk_id,
    c.document_id,
    c.seq,
    c.pos,
    c.content AS chunk_content,
    c.embedding,
    c.tokens,
    c.bytes,
    c.created_at AS chunk_created_at,
    d.user_id,
    d.team_id,
    d.scope,
    d.category,
    d.title,
    d.source,
    d.source_ref,
    d.tags,
    d.content AS document_content,
    d.content_hash,
    d.content_bytes,
    d.created_at AS document_created_at
FROM memory_chunks c
JOIN memory_documents d ON c.document_id = d.id;

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to update updated_at timestamp on documents
CREATE OR REPLACE FUNCTION update_document_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at on memory_documents
DROP TRIGGER IF EXISTS memory_documents_updated_at ON memory_documents;
CREATE TRIGGER memory_documents_updated_at
    BEFORE UPDATE ON memory_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_document_updated_at();

-- =============================================================================
-- VERIFICATION
-- =============================================================================

-- Verify tables were created
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'memory_documents') THEN
        RAISE EXCEPTION 'memory_documents table was not created';
    END IF;
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'memory_chunks') THEN
        RAISE EXCEPTION 'memory_chunks table was not created';
    END IF;
    RAISE NOTICE 'SUCCESS: memory_documents and memory_chunks tables created';
END $$;
