"""Smoke tests for install-server.sh: no Docker needed, uses --dry-run."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install-server.sh"


def run(*args):
    return subprocess.run(["bash", str(SCRIPT), "--dry-run", *args], capture_output=True, text=True)


def test_script_exists_and_is_bash():
    assert SCRIPT.exists()
    assert SCRIPT.read_text().startswith("#!/usr/bin/env bash")


def test_default_mode_needs_openrouter_key():
    r = run("--yes")
    assert r.returncode != 0
    assert "OPENROUTER_API_KEY" in r.stderr + r.stdout


def test_private_cpu_selects_preset_and_profile():
    r = run("--private", "--yes")
    assert r.returncode == 0, r.stderr
    assert ".env.private-cpu.example" in r.stdout
    assert "--profile private" in r.stdout


def test_private_gpu_adds_override_file():
    r = run("--private-gpu", "--yes")
    assert r.returncode == 0, r.stderr
    assert ".env.private-gpu.example" in r.stdout
    assert "docker-compose.gpu.yml" in r.stdout


def test_help_works_when_piped_from_stdin():
    """`curl ... | bash -s -- --help` has no $0 to read, so help must be inline."""
    r = subprocess.run(
        ["bash", "-s", "--", "--help"],
        input=SCRIPT.read_text(),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "--private-gpu" in r.stdout
    assert "--dry-run" in r.stdout
