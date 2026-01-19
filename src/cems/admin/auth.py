"""Authentication utilities for CEMS admin API."""

import secrets

import bcrypt


def generate_api_key(prefix: str = "cems_ak") -> tuple[str, str, str]:
    """Generate a new API key.

    Args:
        prefix: Prefix for the API key (default: "cems_ak").

    Returns:
        Tuple of (full_key, key_hash, key_prefix).
        The full_key should be shown to the user ONCE.
        The key_hash should be stored in the database.
        The key_prefix is for identification (first 8 chars after prefix).
    """
    # Generate random bytes and encode as hex
    random_part = secrets.token_hex(24)  # 48 characters
    full_key = f"{prefix}_{random_part}"

    # Hash for storage
    key_hash = hash_api_key(full_key)

    # Prefix for identification (first 8 chars of random part)
    key_prefix = f"{prefix}_{random_part[:8]}"

    return full_key, key_hash, key_prefix


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage.

    Args:
        api_key: The plain API key.

    Returns:
        bcrypt hash of the key.
    """
    return bcrypt.hashpw(api_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_api_key(api_key: str, key_hash: str) -> bool:
    """Verify an API key against its hash.

    Args:
        api_key: The plain API key to verify.
        key_hash: The stored bcrypt hash.

    Returns:
        True if the key matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(api_key.encode("utf-8"), key_hash.encode("utf-8"))
    except Exception:
        return False


def get_key_prefix(api_key: str) -> str:
    """Extract the prefix from an API key for lookup.

    Args:
        api_key: Full API key (e.g., "cems_ak_abc123...")

    Returns:
        Key prefix for database lookup (e.g., "cems_ak_abc12345")
    """
    parts = api_key.split("_")
    if len(parts) >= 3:
        # Format: prefix_type_random -> prefix_type_first8
        prefix = "_".join(parts[:2])
        random_part = parts[2] if len(parts) == 3 else "_".join(parts[2:])
        return f"{prefix}_{random_part[:8]}"
    return api_key[:16]  # Fallback
