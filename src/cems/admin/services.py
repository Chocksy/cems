"""Admin services for user and team management."""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from cems.admin.auth import generate_api_key, get_key_prefix, verify_api_key
from cems.db.models import AuditLog, Team, TeamMember, User

logger = logging.getLogger(__name__)


def _record_audit(
    session: Session,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict | None = None,
) -> None:
    """Record an audit log entry to the given session."""
    log = AuditLog(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    )
    session.add(log)


@dataclass
class UserCreateResult:
    """Result of user creation."""

    user: User
    api_key: str  # Plain key - show once


@dataclass
class ApiKeyResetResult:
    """Result of API key reset."""

    user: User
    api_key: str  # Plain key - show once


class UserService:
    """Service for user management operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create_user(
        self,
        username: str,
        email: str | None = None,
        is_admin: bool = False,
        settings: dict | None = None,
    ) -> UserCreateResult:
        """Create a new user with API key.

        Args:
            username: Unique username.
            email: Optional email address.
            is_admin: Whether user is an admin.
            settings: Optional user settings dict.

        Returns:
            UserCreateResult with user and plain API key.

        Raises:
            ValueError: If username already exists.
        """
        # Check if username exists
        existing = self.session.execute(
            select(User).where(User.username == username)
        ).scalar_one_or_none()
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        # Check if email exists
        if email:
            existing_email = self.session.execute(
                select(User).where(User.email == email)
            ).scalar_one_or_none()
            if existing_email:
                raise ValueError(f"Email '{email}' already exists")

        # Generate API key
        api_key, key_hash, key_prefix = generate_api_key()

        # Create user
        user = User(
            username=username,
            email=email,
            api_key_hash=key_hash,
            api_key_prefix=key_prefix,
            is_admin=is_admin,
            settings=settings or {},
        )
        self.session.add(user)
        self.session.flush()  # Get the ID

        # Audit log
        self._audit("user_created", "user", str(user.id), {"username": username})

        logger.info(f"Created user: {username} (admin={is_admin})")
        return UserCreateResult(user=user, api_key=api_key)

    def get_user_by_id(self, user_id: uuid.UUID) -> User | None:
        """Get user by ID."""
        return self.session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one_or_none()

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        return self.session.execute(
            select(User).where(User.username == username)
        ).scalar_one_or_none()

    def get_user_by_api_key(self, api_key: str) -> User | None:
        """Get user by API key (validates the key).

        Side effect: Updates user.last_active to now on successful match.

        Args:
            api_key: Plain API key to validate.

        Returns:
            User if key is valid, None otherwise.
        """
        # Get prefix for lookup
        key_prefix = get_key_prefix(api_key)

        # Find users with matching prefix (should be unique but check anyway)
        users = self.session.execute(
            select(User).where(
                User.api_key_prefix == key_prefix, User.is_active == True  # noqa: E712
            )
        ).scalars().all()

        for user in users:
            if verify_api_key(api_key, user.api_key_hash):
                # Update last active
                user.last_active = datetime.now(UTC)
                return user

        return None

    def list_users(
        self, include_inactive: bool = False, limit: int = 100, offset: int = 0
    ) -> list[User]:
        """List all users.

        Args:
            include_inactive: Include inactive users.
            limit: Maximum number of users to return.
            offset: Offset for pagination.

        Returns:
            List of users.
        """
        query = select(User)
        if not include_inactive:
            query = query.where(User.is_active == True)  # noqa: E712
        query = query.order_by(User.created_at.desc()).limit(limit).offset(offset)
        return list(self.session.execute(query).scalars().all())

    def update_user(
        self,
        user_id: uuid.UUID,
        email: str | None = None,
        is_active: bool | None = None,
        is_admin: bool | None = None,
        settings: dict | None = None,
    ) -> User | None:
        """Update user fields.

        Args:
            user_id: User ID to update.
            email: New email (if provided).
            is_active: New active status (if provided).
            is_admin: New admin status (if provided).
            settings: New settings (if provided).

        Returns:
            Updated user or None if not found.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return None

        changed_fields = []
        if email is not None:
            user.email = email
            changed_fields.append("email")
        if is_active is not None:
            user.is_active = is_active
            changed_fields.append("is_active")
        if is_admin is not None:
            user.is_admin = is_admin
            changed_fields.append("is_admin")
        if settings is not None:
            user.settings = settings
            changed_fields.append("settings")

        self._audit("user_updated", "user", str(user_id), {"changed_fields": changed_fields})
        return user

    def delete_user(self, user_id: uuid.UUID) -> bool:
        """Delete a user.

        Args:
            user_id: User ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return False

        username = user.username
        self.session.delete(user)
        self._audit("user_deleted", "user", str(user_id), {"username": username})
        logger.info(f"Deleted user: {username}")
        return True

    def deactivate_user(self, user_id: uuid.UUID) -> User | None:
        """Deactivate a user (soft delete).

        Args:
            user_id: User ID to deactivate.

        Returns:
            Updated user or None if not found.
        """
        return self.update_user(user_id, is_active=False)

    def reset_api_key(self, user_id: uuid.UUID) -> ApiKeyResetResult | None:
        """Generate a new API key for a user.

        Args:
            user_id: User ID.

        Returns:
            ApiKeyResetResult with user and new plain API key, or None if not found.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return None

        # Generate new API key
        api_key, key_hash, key_prefix = generate_api_key()

        user.api_key_hash = key_hash
        user.api_key_prefix = key_prefix

        self._audit("api_key_reset", "user", str(user_id), {"username": user.username})
        logger.info(f"Reset API key for user: {user.username}")
        return ApiKeyResetResult(user=user, api_key=api_key)

    def _audit(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: dict | None = None,
    ) -> None:
        """Record an audit log entry."""
        _record_audit(self.session, action, resource_type, resource_id, details)


class TeamService:
    """Service for team management operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create_team(
        self,
        name: str,
        company_id: str,
        settings: dict | None = None,
    ) -> Team:
        """Create a new team.

        Args:
            name: Unique team name.
            company_id: Company identifier.
            settings: Optional team settings.

        Returns:
            Created team.

        Raises:
            ValueError: If team name already exists.
        """
        # Check if name exists
        existing = self.session.execute(
            select(Team).where(Team.name == name)
        ).scalar_one_or_none()
        if existing:
            raise ValueError(f"Team '{name}' already exists")

        team = Team(
            name=name,
            company_id=company_id,
            settings=settings or {},
        )
        self.session.add(team)
        self.session.flush()

        self._audit("team_created", "team", str(team.id), {"name": name})
        logger.info(f"Created team: {name}")
        return team

    def get_team_by_id(self, team_id: uuid.UUID) -> Team | None:
        """Get team by ID."""
        return self.session.execute(
            select(Team).where(Team.id == team_id)
        ).scalar_one_or_none()

    def get_team_by_name(self, name: str) -> Team | None:
        """Get team by name."""
        return self.session.execute(
            select(Team).where(Team.name == name)
        ).scalar_one_or_none()

    def list_teams(self, limit: int = 100, offset: int = 0) -> list[Team]:
        """List all teams."""
        query = (
            select(Team).order_by(Team.created_at.desc()).limit(limit).offset(offset)
        )
        return list(self.session.execute(query).scalars().all())

    def delete_team(self, team_id: uuid.UUID) -> bool:
        """Delete a team.

        Args:
            team_id: Team ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        team = self.get_team_by_id(team_id)
        if not team:
            return False

        name = team.name
        self.session.delete(team)
        self._audit("team_deleted", "team", str(team_id), {"name": name})
        logger.info(f"Deleted team: {name}")
        return True

    def add_member(
        self,
        team_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str = "member",
    ) -> TeamMember | None:
        """Add a user to a team.

        Args:
            team_id: Team ID.
            user_id: User ID to add.
            role: Role ('admin', 'member', 'viewer').

        Returns:
            TeamMember or None if team/user not found.

        Raises:
            ValueError: If user is already a member.
        """
        if role not in ("admin", "member", "viewer"):
            raise ValueError(f"Invalid role: {role}")

        # Check if already a member
        existing = self.session.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id, TeamMember.user_id == user_id
            )
        ).scalar_one_or_none()
        if existing:
            raise ValueError("User is already a team member")

        member = TeamMember(
            team_id=team_id,
            user_id=user_id,
            role=role,
        )
        self.session.add(member)
        self.session.flush()

        self._audit(
            "member_added",
            "team",
            str(team_id),
            {"user_id": str(user_id), "role": role},
        )
        return member

    def remove_member(self, team_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Remove a user from a team.

        Args:
            team_id: Team ID.
            user_id: User ID to remove.

        Returns:
            True if removed, False if not found.
        """
        member = self.session.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id, TeamMember.user_id == user_id
            )
        ).scalar_one_or_none()
        if not member:
            return False

        self.session.delete(member)
        self._audit(
            "member_removed", "team", str(team_id), {"user_id": str(user_id)}
        )
        return True

    def update_member_role(
        self,
        team_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str,
    ) -> TeamMember | None:
        """Update a member's role.

        Args:
            team_id: Team ID.
            user_id: User ID.
            role: New role.

        Returns:
            Updated TeamMember or None if not found.
        """
        if role not in ("admin", "member", "viewer"):
            raise ValueError(f"Invalid role: {role}")

        member = self.session.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id, TeamMember.user_id == user_id
            )
        ).scalar_one_or_none()
        if not member:
            return None

        member.role = role
        self._audit(
            "member_role_updated",
            "team",
            str(team_id),
            {"user_id": str(user_id), "role": role},
        )
        return member

    def list_members(self, team_id: uuid.UUID) -> list[TeamMember]:
        """List all members of a team (with user info eagerly loaded)."""
        return list(
            self.session.execute(
                select(TeamMember)
                .where(TeamMember.team_id == team_id)
                .options(joinedload(TeamMember.user))
            )
            .scalars()
            .all()
        )

    def get_user_teams(self, user_id: uuid.UUID) -> list[Team]:
        """Get all teams a user belongs to."""
        memberships = self.session.execute(
            select(TeamMember).where(TeamMember.user_id == user_id)
        ).scalars().all()

        team_ids = [m.team_id for m in memberships]
        if not team_ids:
            return []

        return list(
            self.session.execute(select(Team).where(Team.id.in_(team_ids)))
            .scalars()
            .all()
        )

    def _audit(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: dict | None = None,
    ) -> None:
        """Record an audit log entry."""
        _record_audit(self.session, action, resource_type, resource_id, details)
