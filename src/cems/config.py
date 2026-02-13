"""Configuration for CEMS.

Environment Variables for CLI (client mode):
    - CEMS_API_URL: URL of CEMS server (e.g., https://cems.example.com)
    - CEMS_API_KEY: User API key for authentication
    - CEMS_ADMIN_KEY: Admin key for user/team management (optional)

Environment Variables for Server:
    - OPENROUTER_API_KEY: Required for all LLM and embedding operations
    - CEMS_DATABASE_URL: PostgreSQL connection URL for user management

    Optional model configuration:
    - CEMS_EMBEDDING_MODEL: Embedding model (default: openai/text-embedding-3-small)
    - CEMS_LLM_MODEL: Model for maintenance (default: x-ai/grok-4.1-fast)

    OpenRouter provides both LLM and embedding APIs:
    - LLM: https://openrouter.ai/api/v1/chat/completions
    - Embeddings: https://openrouter.ai/api/v1/embeddings
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CEMSConfig(BaseSettings):
    """CEMS configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CEMS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================================
    # Debug Mode - CRITICAL: Let exceptions bubble up in development
    # =========================================================================
    debug_mode: bool = Field(
        default=True,  # ON by default - we want to see errors during development
        description="When True, exceptions bubble up instead of being silently caught. "
                    "Set to False only in production to enable fallbacks.",
    )

    # =========================================================================
    # Client Settings (for CLI and other HTTP clients)
    # =========================================================================
    api_url: str | None = Field(
        default=None,
        description="CEMS server URL (e.g., https://cems.example.com)",
    )
    api_key: str | None = Field(
        default=None,
        description="User API key for authentication",
    )

    # User identification (for server mode)
    user_id: str = Field(default="default", description="Current user ID")
    team_id: str | None = Field(default=None, description="Team ID for shared memory")

    # Storage paths
    storage_dir: Path = Field(
        default=Path.home() / ".cems",
        description="Base directory for CEMS storage",
    )

    # =========================================================================
    # LLM Settings - Single API Key via OpenRouter
    # =========================================================================
    # All LLM and embedding operations use OPENROUTER_API_KEY.
    # OpenRouter provides both chat completions AND embeddings APIs.
    #
    # Required env vars:
    #   - OPENROUTER_API_KEY: For all operations (LLM + embeddings)
    #
    # Model names use OpenRouter format: provider/model
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Embedding model (via OpenRouter)",
    )
    llm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model for maintenance ops (OpenRouter format: provider/model)",
    )

    # =========================================================================
    # Embedding Backend Settings
    # =========================================================================
    # Choose between llama.cpp server (768-dim) or OpenRouter API (1536-dim)
    # Strategy A: Single embedding column at configured dimension
    embedding_backend: Literal["openrouter", "llamacpp_server"] = Field(
        default="openrouter",
        description="Embedding backend: openrouter (1536-dim API) or llamacpp_server (768-dim)",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding dimension (1536 for OpenRouter, 768 for llama.cpp server)",
    )

    # =========================================================================
    # llama.cpp Server Settings
    # =========================================================================
    # These settings are used when embedding_backend="llamacpp_server"
    # Run separate llama.cpp servers for embeddings and reranking
    llamacpp_base_url: str = Field(
        default="http://localhost:8081",
        description="Base URL for llama.cpp embedding server",
    )
    llamacpp_embed_model: str = Field(
        default="embeddinggemma-300M-Q8_0.gguf",
        description="Model name for embeddings (passed to server)",
    )
    llamacpp_embed_path: str = Field(
        default="/v1/embeddings",
        description="Endpoint path for embeddings (OpenAI-compatible)",
    )
    llamacpp_rerank_url: str = Field(
        default="http://localhost:8082",
        description="Base URL for llama.cpp reranker server",
    )
    llamacpp_rerank_model: str = Field(
        default="Qwen3-Reranker-0.6B.Q8_0.gguf",
        description="Model name for reranking",
    )
    llamacpp_rerank_path: str = Field(
        default="/rerank",
        description="Endpoint path for reranking",
    )
    llamacpp_api_key: str | None = Field(
        default=None,
        description="API key for llama.cpp server (if required)",
    )
    llamacpp_timeout_seconds: int = Field(
        default=60,
        description="Timeout for llama.cpp server requests (increased for batch operations)",
    )

    # =========================================================================
    # Vector Store Settings (pgvector - unified with PostgreSQL)
    # =========================================================================
    # pgvector uses the same PostgreSQL database as metadata storage
    # No separate vector store URL needed - uses CEMS_DATABASE_URL
    hybrid_vector_weight: float = Field(
        default=0.4,  # Reduced from 0.7 to favor BM25 for entity matching (temporal queries)
        description="Weight for vector similarity in hybrid search (0-1). Text gets 1-vector_weight.",
    )

    # =========================================================================
    # Graph/Relations Settings
    # =========================================================================
    # Relations are now stored in PostgreSQL memory_relations table
    enable_graph: bool = Field(
        default=True,
        description="Enable memory relations for relationship tracking",
    )

    # =========================================================================
    # Scheduler Settings
    # =========================================================================
    enable_scheduler: bool = Field(
        default=True,
        description="Enable background maintenance jobs",
    )
    nightly_hour: int = Field(default=3, description="Hour for nightly consolidation (0-23)")
    weekly_day: str = Field(default="sun", description="Day for weekly summarization")
    weekly_hour: int = Field(default=4, description="Hour for weekly summarization")
    monthly_day: int = Field(default=1, description="Day of month for monthly reindex")
    monthly_hour: int = Field(default=5, description="Hour for monthly reindex")

    # =========================================================================
    # Retrieval Settings (5-Stage Inference Pipeline)
    # =========================================================================
    enable_query_synthesis: bool = Field(
        default=False,  # EXPERIMENT: Disable LLM query expansion (was slowing down eval)
        description="Enable LLM query expansion (Stage 1 of retrieval pipeline)",
    )
    enable_preference_synthesis: bool = Field(
        default=True,  # Always expand preference queries to bridge semantic gap
        description="Force query synthesis for preference/recommendation queries even when enable_query_synthesis=False",
    )
    relevance_threshold: float = Field(
        default=0.3,  # Raised from 0.005 after Phase 2 data cleanup (584 quality memories)
        description="Minimum similarity score to include in results (Stage 3)",
    )
    default_max_tokens: int = Field(
        default=4000,
        description="Default token budget for retrieval results (Stage 5)",
    )
    max_candidates_per_query: int = Field(
        default=150,  # Increased from 75 for better Recall@All
        description="Max candidates per vector search query",
    )
    rerank_input_limit: int = Field(
        default=40,  # QMD-style cap for reranking quality/latency balance
        description="Max candidates to send to reranker",
    )
    rerank_output_limit: int = Field(
        default=50,  # Increased from 40 - keep more ranked results
        description="Max results to return from LLM reranking",
    )

    # =========================================================================
    # QMD-Style Retrieval Settings (strong-signal skip, lexical stream, RRF)
    # =========================================================================
    # Strong signal detection - skip expansion when BM25 has high confidence
    strong_signal_threshold: float = Field(
        default=0.85,
        description="Skip query expansion if top BM25 score >= this threshold",
    )
    strong_signal_gap: float = Field(
        default=0.15,
        description="AND gap to second result >= this (ensures clear winner)",
    )

    # Lexical stream - add BM25 alongside vector search
    enable_lexical_in_inference: bool = Field(
        default=True,
        description="Add BM25 stream alongside vector in inference pipeline",
    )

    # RRF weights - protect original query from expansion noise
    rrf_original_weight: float = Field(
        default=2.0,
        description="Weight for original query lists in RRF (vector + lexical)",
    )
    rrf_expansion_weight: float = Field(
        default=1.0,
        description="Weight for expansion query lists in RRF",
    )
    rrf_top_rank_bonus_r1: float = Field(
        default=0.05,
        description="Bonus added to rank 1 items in RRF",
    )
    rrf_top_rank_bonus_r23: float = Field(
        default=0.02,
        description="Bonus added to rank 2-3 items in RRF",
    )

    # =========================================================================
    # Reranker Settings
    # =========================================================================
    reranker_backend: Literal["llamacpp_server", "llm", "disabled"] = Field(
        default="disabled",
        description="Reranker backend: llamacpp_server (llama.cpp server), llm (OpenRouter API), disabled",
    )

    # =========================================================================
    # Scoring Penalty Settings
    # =========================================================================
    # These penalties help prioritize memories from the same category/project.
    # They're designed for multi-project, multi-category production usage.
    # Set to False for benchmarks with uniform categories (e.g., LongMemEval).
    enable_cross_category_penalty: bool = Field(
        default=True,
        description="Apply 0.8x penalty when memory category differs from inferred query category",
    )
    enable_project_penalty: bool = Field(
        default=True,
        description="Apply 0.8x penalty for different-project memories (1.3x boost for same-project)",
    )
    cross_category_penalty_factor: float = Field(
        default=0.8,
        description="Multiplier for cross-category memories (lower = more penalty)",
    )
    project_penalty_factor: float = Field(
        default=0.8,
        description="Multiplier for different-project memories",
    )
    project_boost_factor: float = Field(
        default=1.3,
        description="Multiplier for same-project memories (boost)",
    )

    # =========================================================================
    # Decay Settings
    # =========================================================================
    stale_days: int = Field(default=90, description="Days before memory is considered stale")
    archive_days: int = Field(default=180, description="Days before memory is archived")
    hot_access_threshold: int = Field(
        default=5,
        description="Access count to consider memory 'hot'",
    )
    duplicate_similarity_threshold: float = Field(
        default=0.92,
        description="Cosine similarity threshold for duplicate detection",
    )

    # =========================================================================
    # Server Settings
    # =========================================================================
    server_host: str = Field(default="localhost", description="MCP server host")
    server_port: int = Field(default=8765, description="MCP server port")

    # =========================================================================
    # PostgreSQL Settings (required for HTTP mode)
    # =========================================================================
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL. If set, uses PostgreSQL for user/team management.",
    )
    admin_key: str | None = Field(
        default=None,
        description="Admin API key for user management. Required for /admin/* endpoints.",
    )

