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


class PinCategory(str, Enum):
    """Categories for pinned memories."""

    GUIDELINE = "guideline"  # Coding guidelines, style guides
    CONVENTION = "convention"  # Team conventions
    ARCHITECTURE = "architecture"  # Architecture decisions
    STANDARD = "standard"  # Industry standards
    DOCUMENTATION = "documentation"  # Important docs


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
    access_count: int = Field(default=0, description="Number of times accessed")
    source: str | None = Field(default=None, description="Source of the memory")
    source_ref: str | None = Field(default=None, description="Reference (e.g., repo:file:line)")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")
    archived: bool = Field(default=False, description="Whether memory is archived")
    priority: float = Field(default=1.0, description="Priority weight for retrieval")
    # Pinned memory support - excluded from decay
    pinned: bool = Field(default=False, description="Pinned memories are never auto-pruned")
    pin_reason: str | None = Field(default=None, description="Why this memory is pinned")
    pin_category: str | None = Field(default=None, description="Pin category (guideline, convention, etc.)")
    expires_at: datetime | None = Field(default=None, description="When memory expires (None = never)")


class SearchResult(BaseModel):
    """Search result with metadata."""

    memory_id: str
    content: str
    score: float
    scope: MemoryScope
    metadata: MemoryMetadata | None = None


class CategorySummary(BaseModel):
    """Summary for a memory category."""

    category: str
    scope: MemoryScope
    summary: str
    item_count: int
    last_updated: datetime
    version: int = 1
