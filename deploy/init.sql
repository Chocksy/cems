-- CEMS PostgreSQL Schema for Server Deployment
-- This schema supports multi-user, multi-team memory management
-- Uses pgvector for unified vector + metadata storage (replaces Qdrant)
-- NOTE: SQLAlchemy models auto-create tables, this file is for reference/manual setup

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    api_key_hash VARCHAR(255) NOT NULL,  -- bcrypt hash of API key
    api_key_prefix VARCHAR(20) NOT NULL, -- cems_ak_ + 8 random chars
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    settings JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_users_api_key_prefix ON users(api_key_prefix);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    company_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    settings JSONB DEFAULT '{}'::jsonb
);

-- Team membership
CREATE TABLE IF NOT EXISTS team_members (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'member', -- 'admin', 'member', 'viewer'
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, team_id)
);

-- =============================================================================
-- Unified Memories Table (pgvector - replaces Qdrant + memory_metadata)
-- =============================================================================
-- This is the primary storage for all memories, combining:
-- - Vector embeddings (pgvector VECTOR type)
-- - Full-text search (tsvector for BM25-style hybrid search)
-- - All metadata in one table for ACID transactions
-- =============================================================================

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,  -- embeddinggemma-300M via llama.cpp server
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    scope VARCHAR(50) NOT NULL DEFAULT 'personal',
    category VARCHAR(255) DEFAULT 'general',
    tags TEXT[] DEFAULT '{}',
    source VARCHAR(255),
    source_ref VARCHAR(500),
    priority REAL DEFAULT 1.0,
    pinned BOOLEAN DEFAULT FALSE,
    pin_reason TEXT,
    pin_category VARCHAR(100),
    archived BOOLEAN DEFAULT FALSE,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    -- Generated column for full-text search
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    CONSTRAINT valid_scope CHECK (scope IN ('personal', 'team', 'company', 'shared'))
);

-- HNSW index for fast approximate nearest neighbor search
-- Using cosine distance (most common for text embeddings)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text search (BM25-style ranking)
CREATE INDEX IF NOT EXISTS idx_memories_tsv ON memories USING gin(content_tsv);

-- Standard lookup indexes
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_team_id ON memories(team_id);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at) WHERE expires_at IS NOT NULL;

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_memories_user_scope ON memories(user_id, scope) WHERE archived = FALSE;

-- =============================================================================
-- Memory Relations Table (replaces Kuzu graph for simple relationships)
-- =============================================================================

CREATE TABLE IF NOT EXISTS memory_relations (
    source_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    target_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) DEFAULT 'similar',
    similarity REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_memory_relations_source ON memory_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_target ON memory_relations(target_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_type ON memory_relations(relation_type);

-- =============================================================================
-- Legacy memory_metadata table (kept for migration, will be deprecated)
-- =============================================================================
-- Memory metadata (extended for server mode)
CREATE TABLE IF NOT EXISTS memory_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id VARCHAR(255) UNIQUE NOT NULL, -- Mem0 memory ID
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    scope VARCHAR(50) NOT NULL, -- 'personal', 'team', 'company'
    category VARCHAR(255) DEFAULT 'general',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    source VARCHAR(255), -- 'conversation', 'indexer', 'manual', 'import'
    source_ref VARCHAR(500), -- e.g., 'repo:company/backend:path/to/file.rb'
    tags TEXT[], -- PostgreSQL array for tags
    archived BOOLEAN DEFAULT false,
    priority REAL DEFAULT 1.0,
    -- Pinned/permanent memory support
    pinned BOOLEAN DEFAULT false,
    pin_reason VARCHAR(500), -- Why it's pinned (e.g., 'rspec_convention', 'architecture')
    pin_category VARCHAR(100), -- 'guideline', 'convention', 'architecture', 'standard'
    expires_at TIMESTAMP WITH TIME ZONE, -- NULL = never expires, set for temp memories
    CONSTRAINT valid_scope CHECK (scope IN ('personal', 'team', 'company'))
);

-- Indexes for memory_metadata
CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_team ON memory_metadata(team_id);
CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_metadata(scope);
CREATE INDEX IF NOT EXISTS idx_memory_category ON memory_metadata(category);
CREATE INDEX IF NOT EXISTS idx_memory_pinned ON memory_metadata(pinned);
CREATE INDEX IF NOT EXISTS idx_memory_last_accessed ON memory_metadata(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memory_source ON memory_metadata(source);
CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory_metadata USING GIN(tags);

-- Category summaries
CREATE TABLE IF NOT EXISTS category_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scope VARCHAR(50) NOT NULL,
    scope_id UUID, -- user_id or team_id depending on scope
    category VARCHAR(255) NOT NULL,
    summary TEXT,
    item_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    UNIQUE(scope, scope_id, category)
);

-- Maintenance job logs
CREATE TABLE IF NOT EXISTS maintenance_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(100) NOT NULL,
    scope VARCHAR(50) NOT NULL,
    scope_id UUID,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL, -- 'started', 'completed', 'failed'
    details JSONB,
    memories_processed INTEGER DEFAULT 0,
    memories_affected INTEGER DEFAULT 0
);

-- Index jobs for repository scanning
CREATE TABLE IF NOT EXISTS index_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    repository_url VARCHAR(500) NOT NULL,
    repository_ref VARCHAR(255) DEFAULT 'main', -- branch/tag
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES users(id),
    config JSONB DEFAULT '{}'::jsonb, -- indexing configuration
    result JSONB, -- results summary
    error_message TEXT
);

-- Index patterns configuration
CREATE TABLE IF NOT EXISTS index_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_patterns TEXT[] NOT NULL, -- e.g., ['*.rb', 'spec/**/*_spec.rb']
    extract_type VARCHAR(100) NOT NULL, -- 'rspec_conventions', 'architecture', 'readme', 'comments'
    pin_category VARCHAR(100), -- auto-pin to this category
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, name)
);

-- Insert default index patterns
INSERT INTO index_patterns (id, team_id, name, description, file_patterns, extract_type, pin_category, is_active) VALUES
    (uuid_generate_v4(), NULL, 'rspec_conventions', 'Extract RSpec testing patterns',
     ARRAY['spec/**/*_spec.rb', 'spec/spec_helper.rb', 'spec/rails_helper.rb'],
     'rspec_conventions', 'guideline', true),
    (uuid_generate_v4(), NULL, 'readme_docs', 'Extract README documentation',
     ARRAY['README.md', 'README.rst', 'docs/**/*.md'],
     'readme', 'documentation', true),
    (uuid_generate_v4(), NULL, 'architecture_docs', 'Extract architecture decisions',
     ARRAY['docs/architecture/**/*', 'ADR/**/*.md', 'doc/adr/**/*.md'],
     'architecture', 'architecture', true),
    (uuid_generate_v4(), NULL, 'contributing_guide', 'Extract contribution guidelines',
     ARRAY['CONTRIBUTING.md', '.github/CONTRIBUTING.md', 'docs/contributing.md'],
     'readme', 'guideline', true),
    (uuid_generate_v4(), NULL, 'ci_config', 'Extract CI/CD configuration patterns',
     ARRAY['.github/workflows/*.yml', '.gitlab-ci.yml', 'Jenkinsfile', '.circleci/config.yml'],
     'ci_config', 'convention', true)
ON CONFLICT DO NOTHING;

-- API keys for external integrations
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL, -- bcrypt hashed API key
    key_prefix VARCHAR(20) NOT NULL, -- cems_ak_ + 8 random chars
    permissions JSONB DEFAULT '["read", "write"]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id),
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);

-- Audit log for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address VARCHAR(45), -- Changed from INET for SQLAlchemy compatibility
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);

-- Function to update last_accessed on memory access (for memory_metadata - legacy)
CREATE OR REPLACE FUNCTION update_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_accessed = CURRENT_TIMESTAMP;
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at on memories table
DROP TRIGGER IF EXISTS memories_updated_at ON memories;
CREATE TRIGGER memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- Views for unified memories table
-- =============================================================================

-- View for active (non-archived, non-expired) memories
CREATE OR REPLACE VIEW active_memories_v2 AS
SELECT * FROM memories
WHERE archived = FALSE
AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP);

-- View for pinned memories (never decay)
CREATE OR REPLACE VIEW pinned_memories_v2 AS
SELECT * FROM memories
WHERE pinned = TRUE AND archived = FALSE;

-- =============================================================================
-- Legacy views (for memory_metadata - kept for migration)
-- =============================================================================

-- View for active (non-archived, non-expired) memories
CREATE OR REPLACE VIEW active_memories AS
SELECT * FROM memory_metadata
WHERE archived = false
AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP);

-- View for pinned memories (never decay)
CREATE OR REPLACE VIEW pinned_memories AS
SELECT * FROM memory_metadata
WHERE pinned = true AND archived = false;

-- Grant permissions (for manual setup - Docker uses cems user directly)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cems;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cems;
