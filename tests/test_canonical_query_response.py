"""
Tests for DataPilot OS Canonical Query Response Contract (Phase 1).
Verifies canonical response structure, decision derivation, analysis_spec parity,
and FastAPI endpoint status code mappings.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from modules.models.execution_result import CanonicalQueryResponse, QueryErrorDetail


client = TestClient(app)


def test_canonical_query_response_structure_succeeded():
    resp = CanonicalQueryResponse(
        request_id="req-001",
        status="SUCCEEDED",
        plan={"objective": "describe", "task": "summary"},
        result={
            "response": "Data summary computed.",
            "manifest": {"operator": "eda_stats", "rows": 10},
            "validation": {"is_valid": True, "issues": []}
        }
    )

    data = resp.model_dump()
    assert data["request_id"] == "req-001"
    assert data["status"] == "SUCCEEDED"
    assert data["decision"] == "execute"
    assert data["analysis_spec"] == {"objective": "describe", "task": "summary"}
    assert data["manifest"] == {"operator": "eda_stats", "rows": 10}
    assert data["validation"] == {"is_valid": True, "issues": []}
    assert data["error"] is None


def test_canonical_query_response_structure_clarification():
    resp = CanonicalQueryResponse(
        request_id="req-002",
        status="NEEDS_CLARIFICATION",
        clarification="Multiple date columns found. Which one should be used?",
        error=QueryErrorDetail(
            code="AMBIGUOUS_COLUMN",
            message="Multiple date columns found.",
            candidate_columns=["order_date", "ship_date"],
            required_action="Specify the date column."
        )
    )

    data = resp.model_dump()
    assert data["request_id"] == "req-002"
    assert data["status"] == "NEEDS_CLARIFICATION"
    assert data["decision"] == "clarify"
    assert data["error"]["code"] == "AMBIGUOUS_COLUMN"
    assert data["error"]["candidate_columns"] == ["order_date", "ship_date"]


def test_canonical_query_response_structure_unsupported():
    resp = CanonicalQueryResponse(
        request_id="req-003",
        status="UNSUPPORTED",
        error=QueryErrorDetail(
            code="UNSUPPORTED_OPERATION",
            message="Deep reinforcement learning is not supported.",
            required_action="Choose a supported statistical or ML operation."
        )
    )

    data = resp.model_dump()
    assert data["request_id"] == "req-003"
    assert data["status"] == "UNSUPPORTED"
    assert data["decision"] == "unsupported"
    assert data["error"]["code"] == "UNSUPPORTED_OPERATION"


def test_canonical_query_response_structure_unsafe():
    resp = CanonicalQueryResponse(
        request_id="req-004",
        status="UNSAFE",
        error=QueryErrorDetail(
            code="TARGET_LEAKAGE_DETECTED",
            message="Feature 'outcome_flag' has 100% deterministic target leakage.",
            required_action="Remove leaked feature from predictor list."
        )
    )

    data = resp.model_dump()
    assert data["request_id"] == "req-004"
    assert data["status"] == "UNSAFE"
    assert data["decision"] == "unsafe"
    assert data["error"]["code"] == "TARGET_LEAKAGE_DETECTED"


def test_api_chat_empty_query_returns_400():
    response = client.post("/api/chat", json={"query": ""})
    assert response.status_code == 400
    detail = response.json().get("detail", response.json())
    assert detail["status"] == "FAILED"
    assert detail["error"]["code"] == "INVALID_QUERY"
    assert "cannot be empty" in detail["error"]["message"]


def test_api_quarantined_routes_return_410():
    res_exec = client.post("/api/execution/run", json={"code": "print(1)"})
    assert res_exec.status_code == 410
    d_exec = res_exec.json()
    assert d_exec["error"] is True
    assert d_exec["code"] == "QUARANTINED_ENDPOINT"

    res_intent = client.post("/api/intent/analyze", json={"query": "test"})
    assert res_intent.status_code == 410
    d_intent = res_intent.json()
    assert d_intent["error"] is True
    assert d_intent["code"] == "QUARANTINED_ENDPOINT"
