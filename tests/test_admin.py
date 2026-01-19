"""Tests for admin API routes and services."""

import uuid
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from cems.admin.auth import generate_api_key, get_key_prefix, hash_api_key, verify_api_key


class TestApiKeyAuth:
    """Tests for API key authentication utilities."""

    def test_generate_api_key_format(self):
        """API key should have correct format."""
        api_key, key_hash, key_prefix = generate_api_key()

        assert api_key.startswith("cems_ak_")
        assert len(api_key) > 20
        assert key_prefix.startswith("cems_ak_")
        assert len(key_prefix) < len(api_key)

    def test_generate_api_key_unique(self):
        """Each generated key should be unique."""
        keys = [generate_api_key()[0] for _ in range(10)]
        assert len(set(keys)) == 10

    def test_verify_api_key_correct(self):
        """Correct key should verify."""
        api_key, key_hash, _ = generate_api_key()
        assert verify_api_key(api_key, key_hash)

    def test_verify_api_key_wrong(self):
        """Wrong key should not verify."""
        _, key_hash, _ = generate_api_key()
        assert not verify_api_key("wrong_key", key_hash)

    def test_verify_api_key_invalid_hash(self):
        """Invalid hash should not crash."""
        assert not verify_api_key("any_key", "not_a_valid_hash")

    def test_get_key_prefix(self):
        """Key prefix extraction should work."""
        api_key, _, key_prefix = generate_api_key()
        extracted = get_key_prefix(api_key)
        assert extracted == key_prefix

    def test_get_key_prefix_fallback(self):
        """Fallback for non-standard keys."""
        prefix = get_key_prefix("shortkey")
        assert len(prefix) <= 16

    def test_hash_api_key_different_each_time(self):
        """Same key should produce different hashes (bcrypt salt)."""
        api_key = "test_key_123"
        hash1 = hash_api_key(api_key)
        hash2 = hash_api_key(api_key)
        assert hash1 != hash2  # Different salts
        # But both should verify
        assert verify_api_key(api_key, hash1)
        assert verify_api_key(api_key, hash2)


class TestAdminRoutes:
    """Tests for admin API routes (mocked database)."""

    def test_admin_info_requires_auth(self):
        """Admin info should require authentication."""
        from starlette.applications import Starlette
        from starlette.routing import Route

        from cems.admin.routes import admin_info

        app = Starlette(routes=[Route("/admin", admin_info, methods=["GET"])])
        client = TestClient(app)

        # When admin key is set but no auth header provided
        with patch.dict("os.environ", {"CEMS_ADMIN_KEY": "test_key"}):
            response = client.get("/admin")
            assert response.status_code == 401
            assert "Authorization" in response.json()["error"]

    def test_admin_info_wrong_key(self):
        """Admin info should reject wrong key."""
        from starlette.applications import Starlette
        from starlette.routing import Route

        from cems.admin.routes import admin_info

        app = Starlette(routes=[Route("/admin", admin_info, methods=["GET"])])
        client = TestClient(app)

        with patch.dict("os.environ", {"CEMS_ADMIN_KEY": "correct_key"}):
            response = client.get(
                "/admin", headers={"Authorization": "Bearer wrong_key"}
            )
            assert response.status_code == 403

    def test_admin_info_correct_key(self):
        """Admin info should work with correct key."""
        from starlette.applications import Starlette
        from starlette.routing import Route

        from cems.admin.routes import admin_info

        app = Starlette(routes=[Route("/admin", admin_info, methods=["GET"])])
        client = TestClient(app)

        with patch.dict("os.environ", {"CEMS_ADMIN_KEY": "test_admin_key"}):
            response = client.get(
                "/admin", headers={"Authorization": "Bearer test_admin_key"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["database"] == "not_configured"

    def test_list_users_requires_database(self):
        """List users should require database."""
        from starlette.applications import Starlette
        from starlette.routing import Route

        from cems.admin.routes import list_users

        app = Starlette(routes=[Route("/admin/users", list_users, methods=["GET"])])
        client = TestClient(app)

        with patch.dict("os.environ", {"CEMS_ADMIN_KEY": "test_key"}):
            response = client.get(
                "/admin/users", headers={"Authorization": "Bearer test_key"}
            )
            assert response.status_code == 503
            assert "Database not configured" in response.json()["error"]


class TestUserService:
    """Tests for UserService (mocked session)."""

    def test_create_user_duplicate_username(self):
        """Creating user with duplicate username should fail."""
        from cems.admin.services import UserService
        from cems.db.models import User

        # Mock session
        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = User(
            username="existing"
        )

        service = UserService(mock_session)

        with pytest.raises(ValueError, match="already exists"):
            service.create_user(username="existing")

    def test_get_user_by_api_key_updates_last_active(self):
        """Getting user by API key should update last_active."""
        from datetime import datetime

        from cems.admin.services import UserService
        from cems.db.models import User

        # Generate a real key for testing
        api_key, key_hash, key_prefix = generate_api_key()

        # Create mock user
        mock_user = MagicMock(spec=User)
        mock_user.api_key_hash = key_hash
        mock_user.is_active = True
        mock_user.last_active = datetime(2020, 1, 1, tzinfo=UTC)

        # Mock session
        mock_session = MagicMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            mock_user
        ]

        service = UserService(mock_session)
        result = service.get_user_by_api_key(api_key)

        assert result == mock_user
        # last_active should be updated
        assert mock_user.last_active != datetime(2020, 1, 1, tzinfo=UTC)


class TestTeamService:
    """Tests for TeamService (mocked session)."""

    def test_create_team_duplicate_name(self):
        """Creating team with duplicate name should fail."""
        from cems.admin.services import TeamService
        from cems.db.models import Team

        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = Team(
            name="existing"
        )

        service = TeamService(mock_session)

        with pytest.raises(ValueError, match="already exists"):
            service.create_team(name="existing", company_id="test")

    def test_add_member_invalid_role(self):
        """Adding member with invalid role should fail."""
        from cems.admin.services import TeamService

        mock_session = MagicMock()
        service = TeamService(mock_session)

        with pytest.raises(ValueError, match="Invalid role"):
            service.add_member(
                team_id=uuid.uuid4(), user_id=uuid.uuid4(), role="superuser"
            )

    def test_add_member_already_member(self):
        """Adding existing member should fail."""
        from cems.admin.services import TeamService
        from cems.db.models import TeamMember

        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = TeamMember()

        service = TeamService(mock_session)

        with pytest.raises(ValueError, match="already a team member"):
            service.add_member(
                team_id=uuid.uuid4(), user_id=uuid.uuid4(), role="member"
            )
