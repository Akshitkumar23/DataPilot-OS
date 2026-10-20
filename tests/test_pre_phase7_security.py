"""
Pre-Phase 7 Security & CORS Hardening Test Suite.
Validates:
1. CORS configuration restricts access to legitimate local renderer origins (5173, 8011, null)
2. Unauthorized origins (e.g. attacker.com) are rejected by CORS headers
3. Quarantined arbitrary code execution remains blocked
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app, _allowed_origins


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_origins_contain_desktop_renderers():
    """Verify that allowed origins cover local dev and packaged Electron environments."""
    assert "http://localhost:5173" in _allowed_origins
    assert "http://127.0.0.1:5173" in _allowed_origins
    assert "http://localhost:8011" in _allowed_origins
    assert "http://127.0.0.1:8011" in _allowed_origins
    assert "null" in _allowed_origins  # Packaged Electron file:// origin


def test_cors_allows_legitimate_dev_origin(client):
    """CORS response for Vite dev server origin http://localhost:5173 must allow access."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET"
    }
    response = client.options("/api/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_allows_electron_file_origin(client):
    """CORS response for packaged Electron 'null' origin must allow access."""
    headers = {
        "Origin": "null",
        "Access-Control-Request-Method": "GET"
    }
    response = client.options("/api/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "null"


def test_cors_rejects_unauthorized_external_origin(client):
    """CORS response for unauthorized external origin must NOT return allow-origin header."""
    headers = {
        "Origin": "https://evil-attacker.example.com",
        "Access-Control-Request-Method": "GET"
    }
    response = client.options("/api/health", headers=headers)
    # Origin should not be reflected in access-control-allow-origin
    allow_origin = response.headers.get("access-control-allow-origin")
    assert allow_origin != "https://evil-attacker.example.com"
    assert allow_origin != "*"


def test_cors_rejects_arbitrary_http_origin(client):
    """CORS response for untrusted HTTP domain must NOT return allow-origin header."""
    headers = {
        "Origin": "http://malicious-site.org",
        "Access-Control-Request-Method": "POST"
    }
    response = client.options("/api/operations/apply", headers=headers)
    allow_origin = response.headers.get("access-control-allow-origin")
    assert allow_origin != "http://malicious-site.org"
    assert allow_origin != "*"


def test_arbitrary_terminal_execution_blocked(client):
    """Verify that arbitrary terminal/shell execution is not exposed as an active endpoint."""
    res = client.post("/api/terminal/execute", json={"command": "import os; os.system('echo 1')"})
    assert res.status_code in (404, 405)
