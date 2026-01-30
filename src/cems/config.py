"""Configuration for CEMS.

Environment Variables for CLI (client mode):
    - CEMS_API_URL: URL of CEMS server (e.g., https://cems.example.com)
    - CEMS_API_KEY: User API key for authentication
    - CEMS_ADMIN_KEY: Admin key for user/team management (optional)

Environment Variables for Server:
    - OPENROUTER_API_KEY: Required for all LLM and embedding operations
    - CEMS_DATABASE_URL: PostgreSQL connection URL for user management

    Optional model configuration:
    - CEMS_MEM0_MODEL: Model for Mem0 (default: x-ai/grok-4.1-fast)
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

    # Memory engine settings
    memory_backend: Literal["mem0", "local"] = Field(
        default="mem0",
        description="Memory backend to use",
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
    mem0_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model for Mem0 fact extraction (OpenRouter format)",
    )
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Embedding model (via OpenRouter)",
    )
    llm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model for maintenance ops (OpenRouter format: provider/model)",
    )

    # =========================================================================
    # Vector Store Settings (pgvector - unified with PostgreSQL)
    # =========================================================================
    # pgvector uses the same PostgreSQL database as metadata storage
    # No separate vector store URL needed - uses CEMS_DATABASE_URL
    hybrid_vector_weight: float = Field(
        default=0.7,
        description="Weight for vector similarity in hybrid search (0-1). Text gets 1-vector_weight.",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding dimension (1536 for OpenAI text-embedding-3-small)",
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
    relevance_threshold: float = Field(
        default=0.005,  # Lowered further to catch borderline matches in RRF
        description="Minimum similarity score to include in results (Stage 3)",
    )
    default_max_tokens: int = Field(
        default=4000,
        description="Default token budget for retrieval results (Stage 5)",
    )
    max_candidates_per_query: int = Field(
        default=75,  # Increased from 50 for better recall
        description="Max candidates per vector search query",
    )
    rerank_input_limit: int = Field(
        default=60,  # Increased from 40 - LLM sees more candidates
        description="Max candidates to send to LLM for reranking",
    )
    rerank_output_limit: int = Field(
        default=40,  # Increased from 25 - keep more ranked results
        description="Max results to return from LLM reranking",
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

    @property
    def personal_collection(self) -> str:
        """Collection name for personal memories (legacy compatibility)."""
        return f"personal_{self.user_id}"

    @property
    def shared_collection(self) -> str | None:
        """Collection name for shared memories (legacy compatibility)."""
        if self.team_id:
            return f"shared_{self.team_id}"
        return None

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_mem0_model(self) -> str:
        """Get the LLM model for fact extraction (legacy compatibility)."""
        return self.mem0_model
