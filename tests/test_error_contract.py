"""
Tests for DataPilot OS Canonical Error Contract (Phase 1).
Verifies error envelopes, HTTP status mappings, and raw traceback sanitization.
"""

import pytest
from backend.core.error_contract import (
    CanonicalStatus,
    CanonicalDecision,
    ErrorEnvelope,
    create_error_envelope,
    map_status_to_http_code,
    sanitize_traceback
)
from backend.core.error_states import DataPilotError, ErrorState


def test_error_envelope_structure():
    env = create_error_envelope(
        code="TEST_ERROR",
        message="Test user message",
        required_action="Fix the input",
        stage="planning",
        recoverable=True,
        candidate_columns=["col_a", "col_b"],
        diagnostics={"internal_key": 123},
        request_id="req-123"
    )

    d = env.to_dict()
    assert d["error"] is True
    assert d["code"] == "TEST_ERROR"
    assert d["error_code"] == "TEST_ERROR"
    assert d["message"] == "Test user message"
    assert d["required_action"] == "Fix the input"
    assert d["stage"] == "planning"
    assert d["recoverable"] is True
    assert d["candidate_columns"] == ["col_a", "col_b"]
    assert d["diagnostics"] == {"internal_key": 123}
    assert d["request_id"] == "req-123"


def test_sanitize_traceback_removes_internal_details():
    raw_tb = """Traceback (most recent call last):
  File "C:\\Users\\Mr\\Desktop\\DataPilot OS\\backend\\main.py", line 120, in some_function
    raise ValueError("Invalid column calculation")
ValueError: Invalid column calculation"""

    sanitized = sanitize_traceback(raw_tb)
    assert "Traceback" not in sanitized
    assert "File \"" not in sanitized
    assert "line 120" not in sanitized
    assert "C:\\Users" not in sanitized
    assert len(sanitized) > 0


def test_sanitize_traceback_handles_empty():
    assert sanitize_traceback("") == "An internal system error occurred."
    assert sanitize_traceback(None) == "An internal system error occurred."


def test_http_status_mapping_succeeded():
    assert map_status_to_http_code("SUCCEEDED") == 200
    assert map_status_to_http_code("succeeded") == 200


def test_http_status_mapping_clarification_unsupported_unsafe():
    assert map_status_to_http_code("NEEDS_CLARIFICATION") == 422
    assert map_status_to_http_code("UNSUPPORTED") == 400
    assert map_status_to_http_code("UNSAFE") == 400


def test_http_status_mapping_failed_codes():
    # 400 Bad Request
    assert map_status_to_http_code("FAILED", "INVALID_QUERY") == 400
    assert map_status_to_http_code("FAILED", "EMPTY_QUERY") == 400

    # 503 Service Unavailable
    assert map_status_to_http_code("FAILED", "OLLAMA_UNAVAILABLE") == 503
    assert map_status_to_http_code("FAILED", "INTENT_PLANNER_UNAVAILABLE") == 503
    assert map_status_to_http_code("FAILED", "OLLAMA_TIMEOUT") == 503

    # 422 Unprocessable Entity
    assert map_status_to_http_code("FAILED", "INTENT_PLAN_INVALID") == 422
    assert map_status_to_http_code("FAILED", "COLUMN_NOT_FOUND") == 422
    assert map_status_to_http_code("FAILED", "AMBIGUOUS_INTENT") == 422
    assert map_status_to_http_code("FAILED", "SECURITY_VIOLATION") == 422

    # 500 Unexpected runtime error
    assert map_status_to_http_code("FAILED", "UNKNOWN_RUNTIME_BUG") == 500
    assert map_status_to_http_code("FAILED", None) == 500


def test_datapilot_error_to_response_and_envelope():
    err = DataPilotError(
        code=ErrorState.DATASET_NOT_CONNECTED,
        message="No dataset connected",
        required_action="Connect a dataset first"
    )

    resp = err.to_response()
    assert resp["error"] is True
    assert resp["code"] == "DATASET_NOT_CONNECTED"
    assert resp["error_code"] == "DATASET_NOT_CONNECTED"
    assert resp["message"] == "No dataset connected"
    assert resp["required_action"] == "Connect a dataset first"

    env = err.to_envelope()
    assert isinstance(env, ErrorEnvelope)
    assert env.code == "DATASET_NOT_CONNECTED"
    assert env.required_action == "Connect a dataset first"
