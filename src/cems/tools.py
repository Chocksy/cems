"""MCP tools for CEMS memory operations."""

from typing import Literal

from pydantic import BaseModel, Field


class MemoryAddInput(BaseModel):
    """Input for adding a memory."""

    content: str = Field(description="The content to remember")
    scope: Literal["personal", "shared"] = Field(
        default="personal",
        description="Memory scope: 'personal' for user-only, 'shared' for team",
    )
    category: str = Field(
        default="general",
        description="Category for organization (e.g., 'preferences', 'decisions')",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for organization",
    )


class MemorySearchInput(BaseModel):
    """Input for searching memories."""

    query: str = Field(description="The search query")
    scope: Literal["personal", "shared", "both"] = Field(
        default="both",
        description="Which namespace(s) to search",
    )
    category: str | None = Field(
        default=None,
        description="Optional category filter",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )


class MemoryGetInput(BaseModel):
    """Input for getting a specific memory."""

    memory_id: str = Field(description="The memory ID to retrieve")


class MemoryDeleteInput(BaseModel):
    """Input for deleting a memory."""

    memory_id: str = Field(description="The memory ID to delete")
    hard: bool = Field(
        default=False,
        description="If true, permanently delete. If false, archive.",
    )


class MemoryUpdateInput(BaseModel):
    """Input for updating a memory."""

    memory_id: str = Field(description="The memory ID to update")
    content: str = Field(description="New content for the memory")


class CategoryListInput(BaseModel):
    """Input for listing categories."""

    scope: Literal["personal", "shared", "both"] = Field(
        default="both",
        description="Which namespace(s) to list categories from",
    )


class MaintenanceRunInput(BaseModel):
    """Input for running maintenance."""

    job_type: Literal["consolidation", "summarization", "reindex", "all"] = Field(
        description="Type of maintenance job to run",
    )


class ContextSetInput(BaseModel):
    """Input for setting context."""

    project_id: str | None = Field(
        default=None,
        description="Project ID to set as context",
    )
    team_id: str | None = Field(
        default=None,
        description="Team ID to set as context",
    )


# Response models
class MemorySearchResult(BaseModel):
    """A single search result."""

    memory_id: str
    content: str
    score: float
    scope: str
    category: str | None = None
    access_count: int | None = None


class MemorySearchResponse(BaseModel):
    """Response from memory search."""

    results: list[MemorySearchResult]
    query: str
    total: int


class MemoryAddResponse(BaseModel):
    """Response from adding a memory."""

    success: bool
    message: str
    memory_ids: list[str] = Field(default_factory=list)


class MemoryDeleteResponse(BaseModel):
    """Response from deleting a memory."""

    success: bool
    message: str
    memory_id: str


class CategoryListResponse(BaseModel):
    """Response from listing categories."""

    categories: list[str]
    scope: str


class MaintenanceResponse(BaseModel):
    """Response from maintenance job."""

    success: bool
    job_type: str
    results: dict
