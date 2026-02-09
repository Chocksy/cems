"""
Hook Test Harness - Test Claude Code hooks without running Claude Code.

Each CEMS hook is a stdin-to-stdout program: it reads JSON from stdin,
makes HTTP calls to CEMS_API_URL, and writes JSON/text to stdout/stderr.
This harness invokes hooks as subprocesses (exactly as Claude Code does)
and captures everything for assertion.

Components:
    RecordingServer  - Starlette app that records all inbound requests and
                       returns configurable canned responses.
    run_hook()       - Invoke a hook script as a subprocess with controlled
                       environment and stdin data.
    HookResult       - Structured result: stdout, stderr, exit_code, and
                       all HTTP requests the hook made to the recording server.

Usage in pytest:
    @pytest.fixture
    def cems_server():
        with RecordingServer() as server:
            yield server

    def test_session_start(cems_server):
        result = run_hook("cems_session_start.py", {...}, server=cems_server)
        assert result.exit_code == 0
        assert "cems-profile" in result.stdout
        assert cems_server.requests[0].path == "/api/memory/profile"
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Hook script location
# ---------------------------------------------------------------------------
HOOKS_DIR = Path(__file__).parent.parent / "hooks"


# ---------------------------------------------------------------------------
# Recorded HTTP request
# ---------------------------------------------------------------------------
@dataclass
class RecordedRequest:
    """A single HTTP request captured by the recording server."""
    method: str
    path: str
    query_string: str
    headers: dict[str, str]
    body: Any  # parsed JSON or raw string
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Canned response configuration
# ---------------------------------------------------------------------------
@dataclass
class CannedResponse:
    """Configure what the recording server returns for a given path."""
    status_code: int = 200
    body: dict | list | str = field(default_factory=lambda: {"success": True})


# ---------------------------------------------------------------------------
# Recording Server
# ---------------------------------------------------------------------------
class RecordingServer:
    """A tiny HTTP server that records every request and returns canned responses.

    Use as a context manager:
        with RecordingServer() as server:
            # server.url is like "http://127.0.0.1:19876"
            # server.requests is a list of RecordedRequest
            ...

    Configure responses before the test:
        server.set_response("/api/memory/search", CannedResponse(
            body={"success": True, "results": [...]}
        ))
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self._requested_port = port
        self.port: int = 0
        self.url: str = ""
        self.requests: list[RecordedRequest] = []
        self._responses: dict[str, CannedResponse] = {}
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def set_response(self, path: str, response: CannedResponse) -> None:
        """Set a canned response for a given path."""
        self._responses[path] = response

    def get_requests(self, path: str | None = None) -> list[RecordedRequest]:
        """Get recorded requests, optionally filtered by path."""
        if path is None:
            return list(self.requests)
        return [r for r in self.requests if r.path == path]

    def clear(self) -> None:
        """Clear all recorded requests."""
        self.requests.clear()

    async def _catch_all(self, request: Request) -> JSONResponse:
        """Handle any request: record it and return canned response."""
        # Read body
        raw_body = await request.body()
        try:
            body = json.loads(raw_body) if raw_body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = raw_body.decode("utf-8", errors="replace") if raw_body else None

        # Record
        recorded = RecordedRequest(
            method=request.method,
            path=request.url.path,
            query_string=str(request.query_params),
            headers=dict(request.headers),
            body=body,
        )
        self.requests.append(recorded)

        # Return canned response (or generic success)
        canned = self._responses.get(request.url.path, CannedResponse())
        resp_body = canned.body
        if isinstance(resp_body, str):
            resp_body = {"message": resp_body}
        return JSONResponse(resp_body, status_code=canned.status_code)

    def _build_app(self) -> Starlette:
        """Build a Starlette app with a catch-all route."""
        return Starlette(
            routes=[
                Route("/{path:path}", self._catch_all, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
            ],
        )

    def __enter__(self) -> RecordingServer:
        import socket

        # Find a free port if none specified
        if self._requested_port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, 0))
                self.port = s.getsockname()[1]
        else:
            self.port = self._requested_port

        self.url = f"http://{self.host}:{self.port}"

        app = self._build_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="error",
            # Prevent uvicorn from installing signal handlers (we're in a thread)
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait for server to be ready (up to 3 seconds)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if self._server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Recording server failed to start within 3 seconds")

        return self

    def __exit__(self, *exc: object) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Hook invocation result
# ---------------------------------------------------------------------------
@dataclass
class HookResult:
    """Result of invoking a hook script."""
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float

    @property
    def stdout_json(self) -> dict | None:
        """Parse stdout as JSON, return None if not valid JSON."""
        try:
            return json.loads(self.stdout)
        except (json.JSONDecodeError, ValueError):
            return None

    @property
    def additional_context(self) -> str | None:
        """Extract additionalContext from hookSpecificOutput, if present."""
        data = self.stdout_json
        if not data:
            return None
        hso = data.get("hookSpecificOutput", {})
        return hso.get("additionalContext")


# ---------------------------------------------------------------------------
# Hook runner
# ---------------------------------------------------------------------------
def run_hook(
    hook_name: str,
    input_data: dict,
    *,
    server: RecordingServer | None = None,
    api_key: str = "test-key-12345",
    extra_env: dict[str, str] | None = None,
    timeout: float = 15.0,
    args: list[str] | None = None,
) -> HookResult:
    """Invoke a hook script as a subprocess, exactly as Claude Code does.

    Args:
        hook_name: Filename of the hook (e.g., "cems_session_start.py")
        input_data: JSON dict to feed on stdin
        server: Recording server to use as CEMS_API_URL (None = no server)
        api_key: API key to set in environment
        extra_env: Additional environment variables
        timeout: Subprocess timeout in seconds
        args: Additional command line arguments

    Returns:
        HookResult with stdout, stderr, exit code, and timing
    """
    hook_path = HOOKS_DIR / hook_name
    if not hook_path.exists():
        raise FileNotFoundError(f"Hook not found: {hook_path}")

    # Build environment: inherit PATH/HOME but control CEMS vars
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "USER": os.environ.get("USER", "test"),
        # Disable .env loading in hooks that use dotenv
        "CEMS_HOOK_TEST": "1",
    }

    if server:
        env["CEMS_API_URL"] = server.url
        env["CEMS_API_KEY"] = api_key
    else:
        # No server = hooks should gracefully skip API calls
        env["CEMS_API_URL"] = ""
        env["CEMS_API_KEY"] = ""

    if extra_env:
        env.update(extra_env)

    # Build command
    cmd = ["uv", "run", str(hook_path)]
    if args:
        cmd.extend(args)

    stdin_data = json.dumps(input_data).encode("utf-8")

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return HookResult(
            exit_code=-1,
            stdout="",
            stderr="TIMEOUT",
            duration_ms=(time.monotonic() - start) * 1000,
        )

    elapsed = (time.monotonic() - start) * 1000

    return HookResult(
        exit_code=proc.returncode,
        stdout=proc.stdout.decode("utf-8", errors="replace").strip(),
        stderr=proc.stderr.decode("utf-8", errors="replace").strip(),
        duration_ms=elapsed,
    )
