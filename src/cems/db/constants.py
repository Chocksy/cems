"""Database constants for CEMS.

Centralizes commonly used SQL fragments to avoid duplication.
"""

# Standard columns for memory SELECT queries (19 columns)
# Used across all memory retrieval operations
MEMORY_COLUMNS = """
    id, content, user_id, team_id, scope, category, tags,
    source, source_ref, priority, pinned, pin_reason,
    pin_category, archived, access_count, created_at,
    updated_at, last_accessed, expires_at
""".strip()

# Memory columns prefixed with table alias (for JOINs)
MEMORY_COLUMNS_PREFIXED = """
    m.id, m.content, m.user_id, m.team_id, m.scope, m.category, m.tags,
    m.source, m.source_ref, m.priority, m.pinned, m.pin_reason,
    m.pin_category, m.archived, m.access_count, m.created_at,
    m.updated_at, m.last_accessed, m.expires_at
""".strip()

# Memory columns with score (for search results)
MEMORY_COLUMNS_WITH_SCORE = f"{MEMORY_COLUMNS}, score"

# Columns for INSERT operations (15 columns, excludes auto-generated)
MEMORY_INSERT_COLUMNS = """
    id, content, embedding, user_id, team_id, scope, category,
    tags, source, source_ref, priority, pinned, pin_reason,
    pin_category, expires_at
""".strip()

# Default embedding dimension for llama.cpp (Embedding Gemma 300M)
# For OpenRouter (text-embedding-3-small), use 1536
DEFAULT_EMBEDDING_DIM = 768
