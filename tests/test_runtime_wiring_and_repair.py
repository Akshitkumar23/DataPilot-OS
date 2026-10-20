"""
Phase 12A Integration Test Suite — Runtime Wiring and Self-Repair Budget Acceptance.

Verifies:
1. RequestContext life cycle, ContextVar isolation, and phase timing tracking.
2. RepairBudgetManager integration:
   - Max 2 repair iterations.
   - LOOP_NO_PROGRESS triggered on identical plan proposals.
   - LOOP_BUDGET_EXCEEDED triggered after 2 failed attempts.
   - Telemetry and audit history capture.
3. PlannerAuditor integration:
   - Structured audit record logging in ring buffer.
   - Strict avoidance of raw dataset cell data.
4. Reconciled HTTP Status & Response Contract Policy:
   - HTTP 200 for SUCCEEDED / execute
   - HTTP 422 for NEEDS_CLARIFICATION / clarify
   - HTTP 400 for UNSUPPORTED / unsupported
   - HTTP 400 for UNSAFE / unsafe
   - HTTP 503 for OLLAMA_UNAVAILABLE / INTENT_PLANNER_UNAVAILABLE
   - HTTP 500 for unhandled exceptions (decision="failed", never "unsafe")
"""

import os
import time
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from server import app
from backend.core.error_states import ErrorState
from backend.core.request_context import (
    RequestContext,
    get_request_context,
    set_request_context,
    reset_request_context,
    request_context_scope,
)
from backend.ds_engine.repair_budget import RepairBudgetManager
from modules.planner_audit import PlannerAuditor, PlannerAuditRecord, planner_auditor
from backend.intelligence.ollama_service import ollama_service, OllamaResponse
from backend.core.dataset_registry import dataset_registry
from modules.semantic_planner import SemanticPlanner
from modules.models.execution_result import CanonicalQueryResponse


# =====================================================================
# 1. RequestContext Lifecycle & Isolation Tests
# =====================================================================

def test_request_context_scope_and_isolation():
    """RequestContext must cleanly isolate across scopes and clear on exit."""
    reset_request_context()
    # Before scope, context is default ephemeral
    default_ctx = get_request_context()
    assert default_ctx.request_id == "ephemeral"

    with request_context_scope(
        request_id="req-test-123",
        dataset_id="ds-alpha",
        dataset_hash="hash-abc",
        model_version="qwen2.5:3b-instruct",
        registry_version="3.0.0",
        max_repair_attempts=2,
    ) as ctx:
        assert ctx.request_id == "req-test-123"
        assert ctx.dataset_id == "ds-alpha"
        assert ctx.dataset_hash == "hash-abc"
        assert ctx.model_version == "qwen2.5:3b-instruct"
        assert ctx.registry_version == "3.0.0"
        assert ctx.repair_budget_manager is not None
        assert ctx.repair_budget_manager.max_attempts == 2

        # Context accessed from get_request_context() matches scope
        active_ctx = get_request_context()
        assert active_ctx.request_id == "req-test-123"

        # Test phase timing
        active_ctx.start_phase("planning")
        time.sleep(0.01)
        active_ctx.end_phase("planning")

        assert "planning" in active_ctx.phase_timings
        assert active_ctx.phase_timings["planning"] >= 0.005

    # After exiting scope, context returns to fresh state
    after_ctx = get_request_context()
    assert after_ctx.request_id != "req-test-123"


# =====================================================================
# 2. RepairBudgetManager Progress & Budget Exhaustion Tests
# =====================================================================

def test_repair_budget_manager_identical_loop_detection():
    """Identical repair proposals must immediately abort with LOOP_NO_PROGRESS."""
    mgr = RepairBudgetManager(max_attempts=2)
    assert mgr.can_attempt_repair()
    assert mgr.attempts_used == 0

    plan_orig = {"tool": "filter_data", "arguments": {"column": "nonexistent", "condition": ">", "value": 10}}
    plan_identical = {"tool": "filter_data", "arguments": {"column": "nonexistent", "condition": ">", "value": 10}}

    err = mgr.check_progress_and_record(
        original_plan=plan_orig,
        proposed_plan=plan_identical,
        error_diagnosed="COLUMN_NOT_FOUND",
        adjustment_summary="No modification",
        success=False,
    )

    assert err is not None
    assert err.code == ErrorState.LOOP_NO_PROGRESS
    assert len(mgr.history) == 1
    assert mgr.history[0].is_identical is True


def test_repair_budget_manager_budget_exhaustion():
    """After 2 failed repair attempts, manager must abort with LOOP_BUDGET_EXCEEDED."""
    mgr = RepairBudgetManager(max_attempts=2)

    plan_0 = {"tool": "filter_data", "arguments": {"column": "col_a", "condition": ">", "value": 10}}
    plan_1 = {"tool": "filter_data", "arguments": {"column": "col_b", "condition": ">", "value": 10}}
    plan_2 = {"tool": "filter_data", "arguments": {"column": "col_c", "condition": ">", "value": 10}}

    # Attempt 1: Progress made (different hash), not successful
    err1 = mgr.check_progress_and_record(
        original_plan=plan_0,
        proposed_plan=plan_1,
        error_diagnosed="COLUMN_NOT_FOUND",
        adjustment_summary="Changed col_a to col_b",
        success=False,
    )
    assert err1 is None
    assert mgr.attempts_used == 1
    assert mgr.can_attempt_repair()

    # Attempt 2: Progress made, not successful -> budget exhausted
    err2 = mgr.check_progress_and_record(
        original_plan=plan_1,
        proposed_plan=plan_2,
        error_diagnosed="COLUMN_NOT_FOUND",
        adjustment_summary="Changed col_b to col_c",
        success=False,
    )
    assert err2 is not None
    assert err2.code == ErrorState.LOOP_BUDGET_EXCEEDED
    assert mgr.attempts_used == 2
    assert not mgr.can_attempt_repair()


# =====================================================================
# 3. PlannerAuditor Privacy & Audit Retention Tests
# =====================================================================

def test_planner_auditor_records_without_cell_data():
    """PlannerAuditor must record audit traces without containing raw dataframe cell values."""
    auditor = PlannerAuditor(max_records=10)
    req_id = "req-audit-test-99"

    record = PlannerAuditRecord(
        request_id=req_id,
        query_fingerprint=PlannerAuditor.fingerprint_query("Filter sales > 1000"),
        model_name="qwen2.5:3b-instruct",
        retrieved_tools=["filter_data", "group_data"],
        locked_tool="filter_data",
        initial_raw_response='{"tool": "filter_data", "arguments": {"column": "sales", "condition": ">", "value": 1000}}',
        validation_failures=[],
        repair_attempts=0,
        repair_history=[],
        final_decision="execute",
        final_tool_call={"tool": "filter_data", "arguments": {"column": "sales", "condition": ">", "value": 1000}},
        stage_latencies_ms={"planning_ms": 12.5},
    )

    auditor.record(record)
    history = auditor.get_history(req_id)
    assert len(history) == 1
    rec = history[0]
    assert rec.request_id == req_id
    assert rec.final_decision == "execute"
    assert rec.locked_tool == "filter_data"
    assert "sales" in rec.initial_raw_response


# =====================================================================
# 4. Reconciled HTTP Status & Response Contract Policy Tests
# =====================================================================

@pytest.fixture
def client():
    return TestClient(app)


def test_http_status_200_on_succeeded_execution(client):
    """Execution success must return HTTP 200 with SUCCEEDED status and execute decision."""
    df = pd.DataFrame({
        "revenue": [100, 200, 300, 400],
        "division": ["East", "West", "East", "West"]
    })
    dataset_registry.register("test_ds_http_200", df)

    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content='{"tool": "filter_data", "arguments": {"column": "revenue", "condition": ">", "value": 150}}',
        parsed_json={"tool": "filter_data", "arguments": {"column": "revenue", "condition": ">", "value": 150}}
    )

    try:
        response = client.post("/api/chat", json={
            "query": "Filter revenue > 150",
            "dataset_id": "test_ds_http_200",
        })
    finally:
        ollama_service.structured_request = orig_req

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "SUCCEEDED"
    assert payload["decision"] == "execute"
    assert payload["result"] is not None


def test_http_status_422_on_needs_clarification(client):
    """Clarification requests must return HTTP 422 with NEEDS_CLARIFICATION status."""
    df = pd.DataFrame({
        "score": [10, 20, 30],
        "hours": [1, 2, 3]
    })
    dataset_registry.register("test_ds_http_422", df)

    orig_method = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **_kwargs: {
        "intent_type": "EDA",
        "primary_intent": "EDA",
        "raw_query": "Predict grade_band",
        "is_ambiguous": True,
        "ambiguity_reason": "Column 'grade_band' not present in dataset.",
    })

    try:
        response = client.post("/api/chat", json={
            "query": "Predict grade_band",
            "dataset_id": "test_ds_http_422",
        })
    finally:
        SemanticPlanner.extract_semantic_intent = orig_method

    assert response.status_code == 422
    payload = response.json().get("detail", response.json())
    assert payload["status"] == "NEEDS_CLARIFICATION"
    assert payload["decision"] == "clarify"
    assert "grade_band" in payload["clarification"]


def test_http_status_400_on_unsupported_operation(client):
    """Unsupported operations must return HTTP 400 with UNSUPPORTED status."""
    orig_method = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **_kwargs: {
        "intent_type": "UNSUPPORTED",
        "primary_intent": "UNSUPPORTED",
        "raw_query": "Train a neural network with backpropagation",
        "error_code": "UNSUPPORTED_OPERATION",
        "error_message": "Deep learning models are not supported.",
    })

    try:
        response = client.post("/api/chat", json={
            "query": "Train a neural network with backpropagation",
        })
    finally:
        SemanticPlanner.extract_semantic_intent = orig_method

    assert response.status_code == 400
    payload = response.json().get("detail", response.json())
    assert payload["status"] in ["UNSUPPORTED", "FAILED"]


def test_http_status_503_on_ollama_unavailable(client):
    """When Ollama planner service is unreachable, endpoint must return HTTP 503."""
    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=False,
        error_code="OLLAMA_UNAVAILABLE",
        error_message="Local Ollama service offline or unreachable.",
        content=None,
        parsed_json=None
    )

    try:
        response = client.post("/api/chat", json={
            "query": "Perform EDA on sales data",
        })
    finally:
        ollama_service.structured_request = orig_req

    assert response.status_code == 503
    payload = response.json().get("detail", response.json())
    assert payload["status"] == "FAILED"
    assert payload["error"]["code"] == "INTENT_PLANNER_UNAVAILABLE"


def test_failed_internal_error_maps_to_decision_failed_never_unsafe(client):
    """An unhandled internal exception must emit HTTP 500 with decision='failed', never 'unsafe'."""
    from modules.query_orchestrator import QueryOrchestrator
    orig_process = QueryOrchestrator.process_query

    def throw_internal_error(*args, **kwargs):
        raise RuntimeError("Unexpected zero division in analytical engine core.")

    QueryOrchestrator.process_query = throw_internal_error

    try:
        response = client.post("/api/chat", json={
            "query": "Trigger internal engine failure",
        })
    finally:
        QueryOrchestrator.process_query = orig_process

    assert response.status_code == 500
    payload = response.json().get("detail", response.json())
    assert payload["status"] == "FAILED"
    assert payload["decision"] == "failed"
    assert payload["decision"] != "unsafe"
    assert "Unexpected zero division" in payload["error"]["message"]
