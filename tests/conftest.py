"""Pytest configuration for CEMS tests."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clean_cems_env(monkeypatch, tmp_path):
    """Clear CEMS environment variables and prevent .env loading for test isolation."""
    # Clear all CEMS_ environment variables
    cems_vars = [k for k in os.environ if k.startswith("CEMS_")]
    for var in cems_vars:
        monkeypatch.delenv(var, raising=False)

    # Also clear API keys that might interfere
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Change to temp directory to avoid loading local .env file
    monkeypatch.chdir(tmp_path)

    yield
