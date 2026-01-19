"""SQLAlchemy models for CEMS PostgreSQL backend.

These models mirror the schema in deploy/init.sql for multi-user deployment.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    ARRAY,
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


class User(Base):
    """User account for CEMS."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    api_key_hash: Mapped[str] = mapped_column(
        String(255), nullable=False
    )  # bcrypt hash
    api_key_prefix: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # First 8 chars for identification
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    team_memberships: Mapped[list["TeamMember"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    memories: Mapped[list["MemoryMetadata"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="created_by_user",
        foreign_keys="ApiKey.created_by",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class Team(Base):
    """Team/organization for shared memories."""

    __tablename__ = "teams"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    company_id: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    members: Mapped[list["TeamMember"]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )
    memories: Mapped[list["MemoryMetadata"]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )
    index_jobs: Mapped[list["IndexJob"]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )
    index_patterns: Mapped[list["IndexPattern"]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Team {self.name}>"


class TeamMember(Base):
    """Team membership with roles."""

    __tablename__ = "team_members"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(
        String(50), default="member"
    )  # 'admin', 'member', 'viewer'
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="team_memberships")
    team: Mapped["Team"] = relationship(back_populates="members")

    def __repr__(self) -> str:
        return f"<TeamMember user={self.user_id} team={self.team_id} role={self.role}>"


class MemoryMetadata(Base):
    """Extended metadata for memories (PostgreSQL version)."""

    __tablename__ = "memory_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    memory_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )  # Mem0 memory ID
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="SET NULL"), nullable=True
    )
    scope: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'personal', 'team', 'company'
    category: Mapped[str] = mapped_column(String(255), default="general")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # 'conversation', 'indexer', 'manual'
    source_ref: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )  # e.g., 'repo:company/backend:path/to/file.rb'
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    priority: Mapped[float] = mapped_column(Float, default=1.0)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    pin_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    pin_category: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # 'guideline', 'convention', etc.
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User | None"] = relationship(back_populates="memories")
    team: Mapped["Team | None"] = relationship(back_populates="memories")

    def __repr__(self) -> str:
        return f"<MemoryMetadata {self.memory_id}>"


class CategorySummary(Base):
    """Category summaries for tiered retrieval."""

    __tablename__ = "category_summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scope: Mapped[str] = mapped_column(String(50), nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )  # user_id or team_id
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    version: Mapped[int] = mapped_column(Integer, default=1)

    __table_args__ = (UniqueConstraint("scope", "scope_id", "category"),)

    def __repr__(self) -> str:
        return f"<CategorySummary {self.scope}:{self.category}>"


class MaintenanceLog(Base):
    """Log of maintenance job executions."""

    __tablename__ = "maintenance_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope: Mapped[str] = mapped_column(String(50), nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'started', 'completed', 'failed'
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    memories_processed: Mapped[int] = mapped_column(Integer, default=0)
    memories_affected: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<MaintenanceLog {self.job_type} {self.status}>"


class IndexJob(Base):
    """Repository indexing job queue."""

    __tablename__ = "index_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    repository_url: Mapped[str] = mapped_column(String(500), nullable=False)
    repository_ref: Mapped[str] = mapped_column(String(255), default="main")
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # 'pending', 'running', 'completed', 'failed'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    team: Mapped["Team"] = relationship(back_populates="index_jobs")

    def __repr__(self) -> str:
        return f"<IndexJob {self.repository_url} {self.status}>"


class IndexPattern(Base):
    """Index patterns for repository scanning."""

    __tablename__ = "index_patterns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_patterns: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    extract_type: Mapped[str] = mapped_column(String(100), nullable=False)
    pin_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    __table_args__ = (UniqueConstraint("team_id", "name"),)

    # Relationships
    team: Mapped["Team | None"] = relationship(back_populates="index_patterns")

    def __repr__(self) -> str:
        return f"<IndexPattern {self.name}>"


class ApiKey(Base):
    """API keys for external integrations."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)
    permissions: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: ["read", "write"]
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    last_used: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    team: Mapped["Team | None"] = relationship(back_populates="api_keys")
    created_by_user: Mapped["User | None"] = relationship(
        back_populates="api_keys", foreign_keys=[created_by]
    )

    def __repr__(self) -> str:
        return f"<ApiKey {self.name} ({self.key_prefix}...)>"


class AuditLog(Base):
    """Audit log for compliance tracking."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} {self.resource_type}>"
