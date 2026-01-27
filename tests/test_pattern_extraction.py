"""Tests for gate rule pattern extraction."""

import re

import pytest

from cems.pattern_extraction import (
    extract_gate_pattern,
    pattern_to_regex,
    extract_severity_from_tags,
    serialize_gate_pattern,
    deserialize_gate_pattern,
)


class TestPatternToRegex:
    """Tests for pattern_to_regex function."""

    def test_simple_pattern(self):
        """Test converting a simple pattern."""
        regex = pattern_to_regex("coolify deploy")
        assert regex.search("coolify deploy app")
        assert regex.search("COOLIFY DEPLOY")  # Case insensitive
        assert regex.search("coolify  deploy")  # Multiple spaces

    def test_pattern_with_spaces(self):
        """Test pattern with multiple words."""
        regex = pattern_to_regex("git push --force")
        assert regex.search("git push --force origin")
        assert regex.search("git  push   --force")  # Flexible whitespace
        assert not regex.search("git push origin")

    def test_glob_wildcard(self):
        """Test glob-style * wildcard."""
        regex = pattern_to_regex("rm -rf *")
        assert regex.search("rm -rf /tmp/test")
        assert regex.search("rm -rf /")

    def test_special_chars_escaped(self):
        """Test that special regex chars are escaped."""
        regex = pattern_to_regex("git push --force")
        # -- should be literal, not regex
        assert regex.search("git push --force")

    def test_partial_match(self):
        """Test that pattern matches anywhere in string."""
        regex = pattern_to_regex("deploy")
        assert regex.search("coolify deploy app")
        assert regex.search("deploy")
        assert regex.search("pre-deploy-hook")


class TestExtractSeverityFromTags:
    """Tests for extract_severity_from_tags function."""

    def test_block_severity(self):
        """Test extracting block severity."""
        assert extract_severity_from_tags(["block", "production"]) == "block"
        assert extract_severity_from_tags(["BLOCK"]) == "block"

    def test_warn_severity(self):
        """Test extracting warn severity."""
        assert extract_severity_from_tags(["warn", "caution"]) == "warn"
        assert extract_severity_from_tags(["safety", "WARN"]) == "warn"

    def test_confirm_severity(self):
        """Test extracting confirm severity."""
        assert extract_severity_from_tags(["confirm"]) == "confirm"

    def test_default_severity(self):
        """Test default severity when no tag matches."""
        assert extract_severity_from_tags(["production", "safety"]) == "warn"
        assert extract_severity_from_tags([]) == "warn"
        assert extract_severity_from_tags(None) == "warn"


class TestExtractGatePattern:
    """Tests for extract_gate_pattern function."""

    def test_basic_pattern(self):
        """Test extracting a basic gate pattern."""
        content = "Bash: coolify deploy — Never use CLI for production"
        result = extract_gate_pattern(content)

        assert result is not None
        assert result["tool"] == "bash"
        assert result["raw_pattern"] == "coolify deploy"
        assert result["reason"] == "Never use CLI for production"
        assert result["severity"] == "warn"
        assert result["project"] is None

    def test_pattern_with_em_dash(self):
        """Test pattern with em dash separator."""
        content = "Bash: rm -rf — Dangerous deletion command"
        result = extract_gate_pattern(content)

        assert result is not None
        assert result["raw_pattern"] == "rm -rf"
        assert result["reason"] == "Dangerous deletion command"

    def test_pattern_with_en_dash(self):
        """Test pattern with en dash separator."""
        content = "Bash: git push --force – Never force push without confirmation"
        result = extract_gate_pattern(content)

        assert result is not None
        assert result["raw_pattern"] == "git push --force"

    def test_pattern_with_hyphen(self):
        """Test pattern with regular hyphen separator (requires spaces around hyphen)."""
        content = "Bash: rm -rf / - Root deletion is forbidden"
        result = extract_gate_pattern(content)

        assert result is not None
        # The hyphen in "rm -rf" is preserved because separator requires " - " with spaces
        assert result["raw_pattern"] == "rm -rf /"
        assert result["reason"] == "Root deletion is forbidden"

    def test_pattern_with_tags(self):
        """Test extracting pattern with severity from tags."""
        content = "Bash: coolify deploy — Never use CLI"
        result = extract_gate_pattern(content, tags=["block", "production"])

        assert result is not None
        assert result["severity"] == "block"

    def test_pattern_with_project_scope(self):
        """Test extracting pattern with project scope."""
        content = "Bash: coolify deploy — Never use CLI for EpicPxls"
        result = extract_gate_pattern(
            content, source_ref="project:EpicCoders/pxls"
        )

        assert result is not None
        assert result["project"] == "EpicCoders/pxls"

    def test_pattern_regex_compiles(self):
        """Test that extracted pattern compiles to valid regex."""
        content = "Bash: coolify deploy — Never use CLI"
        result = extract_gate_pattern(content)

        assert result is not None
        # Test the compiled regex works
        assert result["regex"].search("coolify deploy app")
        assert not result["regex"].search("something else")

    def test_invalid_content_empty(self):
        """Test that empty content returns None."""
        assert extract_gate_pattern("") is None
        assert extract_gate_pattern("   ") is None
        assert extract_gate_pattern(None) is None  # type: ignore

    def test_invalid_content_no_separator(self):
        """Test that content without separator returns None."""
        assert extract_gate_pattern("Bash coolify deploy") is None
        assert extract_gate_pattern("Just some random text") is None

    def test_invalid_content_missing_parts(self):
        """Test that incomplete patterns return None."""
        assert extract_gate_pattern("Bash: — reason only") is None
        assert extract_gate_pattern(": pattern — reason") is None


class TestSerializeDeserialize:
    """Tests for serialization/deserialization."""

    def test_serialize_pattern(self):
        """Test serializing a gate pattern."""
        content = "Bash: coolify deploy — Never use CLI"
        original = extract_gate_pattern(content, tags=["block"])

        serialized = serialize_gate_pattern(original)

        assert "regex" not in serialized  # Regex not serializable
        assert serialized["tool"] == "bash"
        assert serialized["pattern"] == original["pattern"]
        assert serialized["severity"] == "block"

    def test_deserialize_pattern(self):
        """Test deserializing a gate pattern."""
        serialized = {
            "tool": "bash",
            "pattern": r"coolify\s+deploy",
            "raw_pattern": "coolify deploy",
            "reason": "Never use CLI",
            "severity": "block",
            "project": None,
        }

        result = deserialize_gate_pattern(serialized)

        assert result["tool"] == "bash"
        assert result["regex"].search("coolify deploy app")
        assert result["severity"] == "block"

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        content = "Bash: git push --force — Never force push"
        original = extract_gate_pattern(content, tags=["warn"])

        serialized = serialize_gate_pattern(original)
        restored = deserialize_gate_pattern(serialized)

        assert restored["tool"] == original["tool"]
        assert restored["pattern"] == original["pattern"]
        assert restored["raw_pattern"] == original["raw_pattern"]
        assert restored["reason"] == original["reason"]
        assert restored["severity"] == original["severity"]
        # Test regex works the same
        test_cmd = "git push --force origin"
        assert bool(restored["regex"].search(test_cmd)) == bool(
            original["regex"].search(test_cmd)
        )


class TestIntegration:
    """Integration tests for gate rule matching."""

    def test_match_coolify_command(self):
        """Test matching a coolify deploy command."""
        content = "Bash: coolify deploy — Never use CLI for production"
        pattern = extract_gate_pattern(content, tags=["block"])

        # Commands that should match
        assert pattern["regex"].search("coolify deploy app-id")
        assert pattern["regex"].search("coolify deploy rs0cw88...")
        assert pattern["regex"].search("COOLIFY DEPLOY app")

        # Commands that should NOT match
        assert not pattern["regex"].search("coolify status")
        assert not pattern["regex"].search("coolify app list")

    def test_match_force_push(self):
        """Test matching git force push."""
        # Use * (glob wildcard) not .* (regex) - the pattern_to_regex converts * to .*
        content = "Bash: git push*--force — Never force push without confirmation"
        pattern = extract_gate_pattern(content, tags=["warn"])

        # Commands that should match
        assert pattern["regex"].search("git push --force origin main")
        assert pattern["regex"].search("git push origin main --force")

        # Commands that should NOT match
        assert not pattern["regex"].search("git push origin main")
        assert not pattern["regex"].search("git pull --force")  # Wrong command

    def test_match_rm_rf(self):
        """Test matching dangerous rm command."""
        content = "Bash: rm -rf — Destructive deletion requires confirmation"
        pattern = extract_gate_pattern(content, tags=["confirm"])

        # Commands that should match
        assert pattern["regex"].search("rm -rf /tmp/test")
        assert pattern["regex"].search("rm  -rf  /")

        # Commands that should NOT match
        assert not pattern["regex"].search("rm file.txt")
        assert not pattern["regex"].search("rm -r dir/")
