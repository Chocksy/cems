"""Data models for CEMS."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(UTC)


class MemoryScope(str, Enum):
    """Memory scope - personal or shared."""

    PERSONAL = "personal"
    SHARED = "shared"


class MemoryCategory(str, Enum):
    """Pre-defined memory categories."""

    PREFERENCES = "preferences"
    DECISIONS = "decisions"
    PATTERNS = "patterns"
    CONTEXT = "context"
    LEARNINGS = "learnings"
    GENERAL = "general"
    GATE_RULES = "gate-rules"  # Tool-blocking rules for PreToolUse hooks


class MemoryMetadata(BaseModel):
    """Extended metadata for a memory item."""

    model_config = ConfigDict(use_enum_values=True)

    memory_id: str = Field(description="Unique memory ID (UUID)")
    user_id: str = Field(description="User who owns this memory")
    scope: MemoryScope = Field(description="Personal or shared")
    category: str = Field(default="general", description="Memory category")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    last_accessed: datetime = Field(default_factory=_utcnow)
    source: str | None = Field(default=None, description="Source of the memory")
    source_ref: str | None = Field(default=None, description="Reference (e.g., repo:file:line)")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")


class SearchResult(BaseModel):
    """Search result with metadata."""

    memory_id: str
    content: str
    score: float
    scope: MemoryScope
    metadata: MemoryMetadata | None = None


class CategorySummary(BaseModel):
    """Summary for a memory category."""

    user_id: str | None = None
    category: str
    scope: MemoryScope
    summary: str
    item_count: int
    last_updated: datetime
    version: int = 1
