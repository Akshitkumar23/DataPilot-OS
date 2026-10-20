from fastapi.testclient import TestClient

from backend.main import app


def _client(monkeypatch):
    monkeypatch.delenv("DATAPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("DATAPILOT_ENABLE_LEGACY_CODEGEN", raising=False)
    monkeypatch.delenv("DATAPILOT_ENABLE_LEGACY_PRODUCTION_EXECUTION", raising=False)
    return TestClient(app)


def test_arbitrary_execution_router_is_replaced_by_authenticated_tombstone(monkeypatch):
    response = _client(monkeypatch).post("/api/execution/run", json={"code": "print('unsafe')"})
    assert response.status_code == 410
    assert response.json()["code"] == "QUARANTINED_ENDPOINT"


def test_duplicate_intent_router_is_replaced_by_tombstone(monkeypatch):
    response = _client(monkeypatch).post(
        "/api/intent/analyze",
        json={"query": "average sales", "dataset_id": "missing"},
    )
    assert response.status_code == 410
    assert response.json()["code"] == "QUARANTINED_ENDPOINT"


def test_free_form_codegen_is_default_deny(monkeypatch):
    response = _client(monkeypatch).post(
        "/api/code/generate",
        json={
            "workflow_id": "wf",
            "node_id": "node",
            "capability": "data_analysis",
            "intent": {},
            "dataset_id": "dataset",
        },
    )
    assert response.status_code == 403
    assert response.json()["code"] == "LEGACY_CODEGEN_DISABLED"


def test_generated_code_production_execution_is_default_deny(monkeypatch):
    response = _client(monkeypatch).post(
        "/api/production/execute",
        json={
            "workflow_id": "wf",
            "production_dataset_id": "dataset",
            "user_approved": True,
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "LEGACY_PRODUCTION_EXECUTION_DISABLED"


def test_canonical_chat_route_remains_mounted():
    paths = set()
    for route in app.routes:
        original_router = getattr(route, "original_router", None)
        nested_routes = getattr(original_router, "routes", None) or getattr(route, "routes", None)
        if nested_routes:
            paths.update(getattr(nested, "path", None) for nested in nested_routes)
        else:
            paths.add(getattr(route, "path", None))
    assert "/api/chat" in paths
