"""Spotify OAuth PKCE authentication for playback control and metadata."""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_TOKEN_PATH = Path.home() / ".config" / "transm" / "spotify.json"
_REDIRECT_PORT = 8765
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}/callback"
_AUTH_URL = "https://accounts.spotify.com/authorize"
_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"


def _get_client_id() -> str:
    """Get Spotify client ID from environment."""
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    if not client_id:
        msg = (
            "SPOTIFY_CLIENT_ID not set. "
            "Set it in your environment or .env file."
        )
        raise RuntimeError(msg)
    return client_id


def get_access_token() -> str:
    """Return a valid Spotify access token, refreshing or logging in as needed."""
    token_data = _load_token()
    if token_data:
        if _test_token(token_data.get("access_token", "")):
            return token_data["access_token"]
        if "refresh_token" in token_data:
            new_data = _refresh(token_data["refresh_token"])
            if new_data:
                _save_token(new_data)
                return new_data["access_token"]
    msg = "No valid Spotify token. Run 'transm capture --login' first."
    raise RuntimeError(msg)


def login() -> str:
    """Run the full PKCE OAuth flow (opens browser). Returns access token."""
    client_id = _get_client_id()
    verifier = secrets.token_urlsafe(64)
    challenge_bytes = hashlib.sha256(verifier.encode()).digest()
    challenge_b64 = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()

    expected_state = secrets.token_urlsafe(16)
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPES,
        "state": expected_state,
        "code_challenge_method": "S256",
        "code_challenge": challenge_b64,
    }
    auth_url = f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"

    # Start local server to catch the callback.
    # Server loops until a terminal response (valid code or Spotify error) is received,
    # ignoring stray requests to other paths or malformed callbacks.
    result_holder: dict[str, str] = {}
    import html as _html

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)

            # Ignore anything that isn't the callback path — keeps server alive
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            qs = urllib.parse.parse_qs(parsed.query)

            # Check for Spotify error responses (terminal — stop server)
            if "error" in qs:
                error = _html.escape(qs["error"][0])
                result_holder["error"] = error
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    f"<h1>Authentication failed: {error}</h1>".encode()
                )
                return

            # Validate state to prevent CSRF (terminal on mismatch — stop server)
            returned_state = qs.get("state", [None])[0]
            if returned_state != expected_state:
                result_holder["error"] = "state_mismatch"
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<h1>Authentication failed: state mismatch (possible CSRF)</h1>"
                )
                return

            # Extract authorization code (terminal on missing — stop server)
            if "code" not in qs:
                result_holder["error"] = "no_code"
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Authentication failed: no code received</h1>")
                return

            result_holder["code"] = qs["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authenticated! You can close this tab.</h1>")

        def log_message(self, format: str, *args: object) -> None:
            pass  # Suppress server logs

    server = http.server.HTTPServer(("localhost", _REDIRECT_PORT), CallbackHandler)
    server.timeout = 120  # Overall timeout for the server

    def _serve_until_result() -> None:
        """Handle requests until we get a terminal result (code or error)."""
        while "code" not in result_holder and "error" not in result_holder:
            server.handle_request()

    server_thread = threading.Thread(target=_serve_until_result, daemon=True)
    server_thread.start()

    logger.info("Opening browser for Spotify login...")
    webbrowser.open(auth_url)
    server_thread.join(timeout=120)
    server.server_close()

    if "error" in result_holder:
        msg = f"Spotify authentication failed: {result_holder['error']}"
        raise RuntimeError(msg)
    if "code" not in result_holder:
        msg = "Did not receive auth code from Spotify (timed out?)."
        raise RuntimeError(msg)

    # Exchange code for tokens
    resp = requests.post(
        _TOKEN_URL,
        data={
            "client_id": client_id,
            "grant_type": "authorization_code",
            "code": result_holder["code"],
            "redirect_uri": _REDIRECT_URI,
            "code_verifier": verifier,
        },
        timeout=15,
    )
    resp.raise_for_status()
    token_data = resp.json()
    _save_token(token_data)
    logger.info("Spotify authentication successful.")
    return token_data["access_token"]


def _refresh(refresh_token: str) -> dict | None:
    """Refresh an access token using the refresh token."""
    try:
        client_id = _get_client_id()
        resp = requests.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if "refresh_token" not in data:
            data["refresh_token"] = refresh_token
        return data
    except Exception:
        logger.warning("Failed to refresh Spotify token.")
        return None


def _test_token(token: str) -> bool:
    """Quick check if a token is still valid."""
    if not token:
        return False
    try:
        resp = requests.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _load_token() -> dict | None:
    """Load cached token data from disk."""
    if _TOKEN_PATH.exists():
        try:
            return json.loads(_TOKEN_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_token(data: dict) -> None:
    """Save token data to disk."""
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_PATH.write_text(json.dumps(data, indent=2))
    _TOKEN_PATH.chmod(0o600)
