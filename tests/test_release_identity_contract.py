"""Release-identity contract for the packaged desktop/backend handoff."""

from backend.main import liveness_check


def test_liveness_exposes_electron_supplied_release_identity(monkeypatch):
    """A desktop shell can reject a stale warm backend without querying Ollama."""
    monkeypatch.setenv("DATAPILOT_RELEASE_ID", "test-package-fingerprint")

    status = liveness_check()

    assert status["backend"] == "DataPilot OS"
    assert status["status"] == "live"
    assert status["release_id"] == "test-package-fingerprint"
