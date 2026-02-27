"""Lightweight local debug dashboard server.

Reads hook event logs and serves a single-page debug UI.
No auth needed — reads local files only.
"""

from __future__ import annotations

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cems.debug.indexer import EventIndex

STATIC_DIR = Path(__file__).parent.parent / "static" / "debug"


class DebugHandler(SimpleHTTPRequestHandler):
    """Serves debug dashboard static files and API endpoints."""

    index: EventIndex  # Set by serve()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if path.startswith("/api/"):
            self._handle_api(path, params)
        elif path == "/app.js":
            self._serve_file("app.js", "application/javascript")
        elif path == "/style.css":
            self._serve_file("style.css", "text/css")
        else:
            # Serve index.html for all other paths (SPA with hash routing)
            self._serve_file("index.html", "text/html")

    def _handle_api(self, path: str, params: dict) -> None:
        if path == "/api/sessions":
            limit = _parse_int(params.get("limit", [50])[0], 50)
            data = self.index.get_sessions(limit=limit)
            self._json_response(data)

        elif path.startswith("/api/sessions/"):
            sid = path.split("/api/sessions/")[1]
            # Strip sub-resource suffix
            suffix = ""
            if sid.endswith("/verbose"):
                sid = sid[:-8]
                suffix = "verbose"

            # Path traversal protection
            if "/" in sid or "\\" in sid or ".." in sid:
                self._json_response({"error": "Invalid session ID"}, 400)
                return

            if suffix == "verbose":
                data = self.index.get_session_verbose(sid)
                self._json_response(data)
            else:
                offset = _parse_int(params.get("offset", [0])[0], 0)
                limit = _parse_int(params.get("limit", [200])[0], 200)
                data = self.index.get_session_detail(sid, offset=offset, limit=limit)
                if data is None:
                    self._json_response({"error": "Session not found"}, 404)
                else:
                    self._json_response(data)

        elif path == "/api/retrievals":
            limit = _parse_int(params.get("limit", [50])[0], 50)
            data = self.index.get_retrievals(limit=limit)
            self._json_response(data)

        elif path == "/api/status":
            data = self.index.get_status()
            self._json_response(data)

        elif path == "/api/observer/sessions":
            limit = _parse_int(params.get("limit", [100])[0], 100)
            data = self.index.get_observer_sessions(limit=limit)
            self._json_response(data)

        elif path.startswith("/api/observer/sessions/"):
            sid = path.split("/api/observer/sessions/")[1]
            if "/" in sid or "\\" in sid or ".." in sid:
                self._json_response({"error": "Invalid session ID"}, 400)
                return
            data = self.index.get_observer_session_detail(sid)
            if data is None:
                self._json_response({"error": "Observer session not found"}, 404)
            else:
                self._json_response(data)

        elif path == "/api/observer/stats":
            data = self.index.get_observer_stats()
            self._json_response(data)

        else:
            self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename: str, content_type: str) -> None:
        filepath = STATIC_DIR / filename
        if not filepath.exists():
            self.send_error(404, f"{filename} not found")
            return
        body = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Only log errors, suppress noisy 200s."""
        if args and str(args[0]).startswith(("4", "5")):
            super().log_message(format, *args)


def _parse_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


class ReusableHTTPServer(HTTPServer):
    """HTTPServer with SO_REUSEADDR to avoid 'Address already in use' on restart."""
    allow_reuse_address = True


def serve(port: int = 8767) -> None:
    """Start the debug dashboard server."""
    index = EventIndex()
    index.refresh()

    DebugHandler.index = index

    server = ReusableHTTPServer(("127.0.0.1", port), DebugHandler)
    session_count = len(index.sessions)
    event_count = index._event_count

    print(f"CEMS Debug Dashboard")
    print(f"  http://localhost:{port}")
    print(f"  {event_count:,} events indexed across {session_count} sessions")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
