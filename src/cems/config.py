"""Configuration for CEMS.

Environment Variables:
    Two LLM configurations are required:

    1. Mem0 (fact extraction, ADD/UPDATE/DELETE logic):
       - OPENAI_API_KEY: Required for Mem0's internal operations
       - CEMS_MEM0_MODEL: Model for Mem0 (default: gpt-4o-mini)
       - CEMS_EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small)

    2. Maintenance (summarization, merging via OpenRouter):
       - OPENROUTER_API_KEY: Required for maintenance operations
       - CEMS_LLM_MODEL: Model in OpenRouter format (default: anthropic/claude-3-haiku)
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
        extra="ignore",  # Ignore unknown env vars like CEMS_API_KEY
    )

    # User identification
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
    # LLM Settings - All via OpenRouter
    # =========================================================================
    # Mem0 auto-detects OPENROUTER_API_KEY and uses it for LLM calls.
    # Embeddings still require OPENAI_API_KEY (OpenRouter doesn't do embeddings).
    #
    # Required env vars:
    #   - OPENROUTER_API_KEY: For all LLM calls
    #   - OPENAI_API_KEY: For embeddings only
    mem0_model: str = Field(
        default="gpt-4o-mini",
        description="Model for Mem0 fact extraction (via OpenRouter)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model (uses OpenAI directly)",
    )
    llm_model: str = Field(
        default="anthropic/claude-3-haiku",
        description="Model for maintenance ops (OpenRouter format: provider/model)",
    )

    # =========================================================================
    # Vector Store Settings
    # =========================================================================
    vector_store: Literal["qdrant", "chroma", "lancedb"] = Field(
        default="qdrant",
        description="Vector store backend",
    )
    qdrant_path: Path | None = Field(
        default=None,
        description="Path for Qdrant storage (uses storage_dir/qdrant if not set)",
    )
    qdrant_url: str | None = Field(
        default=None,
        description="URL for Qdrant server (e.g., http://localhost:6333). If set, uses server instead of local storage.",
    )

    # =========================================================================
    # Graph Store Settings (for relationship tracking)
    # =========================================================================
    enable_graph: bool = Field(
        default=True,
        description="Enable knowledge graph for relationship tracking",
    )
    graph_store: Literal["kuzu", "none"] = Field(
        default="kuzu",
        description="Graph database backend (kuzu for embedded, none to disable)",
    )
    kuzu_path: Path | None = Field(
        default=None,
        description="Path for Kuzu database (uses storage_dir/graph if not set)",
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
        default=True,
        description="Enable LLM query expansion (Stage 1 of retrieval pipeline)",
    )
    relevance_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score to include in results (Stage 3)",
    )
    default_max_tokens: int = Field(
        default=2000,
        description="Default token budget for retrieval results (Stage 5)",
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
    api_key: str | None = Field(
        default=None,
        description="API key for HTTP mode authentication. If set, requests must include Bearer token.",
    )

    # =========================================================================
    # PostgreSQL Settings (for multi-user/enterprise deployment)
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
        """Collection name for personal memories."""
        return f"personal_{self.user_id}"

    @property
    def shared_collection(self) -> str | None:
        """Collection name for shared memories."""
        if self.team_id:
            return f"shared_{self.team_id}"
        return None

    @property
    def qdrant_storage_path(self) -> Path:
        """Path for Qdrant storage."""
        return self.qdrant_path or (self.storage_dir / "qdrant")

    @property
    def kuzu_storage_path(self) -> Path:
        """Path for Kuzu graph database."""
        return self.kuzu_path or (self.storage_dir / "graph")

    @property
    def metadata_db_path(self) -> Path:
        """Path for metadata SQLite database."""
        return self.storage_dir / "metadata.db"

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # Only create local Qdrant storage if not using remote URL
        if not self.qdrant_url:
            self.qdrant_storage_path.mkdir(parents=True, exist_ok=True)
        # Note: Don't create kuzu_storage_path - Kuzu creates its own database directory

    def get_mem0_model(self) -> str:
        """Get the LLM model for Mem0."""
        return self.mem0_model
