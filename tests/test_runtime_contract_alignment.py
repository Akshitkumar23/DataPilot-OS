"""Regression tests for the live desktop-to-engine API contract."""

from types import SimpleNamespace

import pandas as pd

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.semantic_feasibility import SemanticFeasibilityValidator
from modules.semantic_planner import SemanticPlanner
from modules.query_orchestrator import QueryOrchestrator


def test_planner_rejects_tool_outside_retrieved_contract():
    allowed = {"event_window_analysis", "clarification_needed"}

    assert not SemanticPlanner._uses_only_selected_tools(
        {"tool": "eda_data", "arguments": {}}, allowed
    )
    assert SemanticPlanner._uses_only_selected_tools(
        {"tool": "event_window_analysis", "arguments": {}}, allowed
    )


def test_event_window_requires_explicit_lookback_before_execution():
    df = pd.DataFrame(
        {
            "sequence": [1, 2, 3, 4],
            "failure": [0, 0, 1, 0],
            "vibration": [0.1, 0.3, 1.2, 0.2],
        }
    )
    profile = DatasetProfiler.profile(df, dataset_name="event-window-contract")

    result = SemanticFeasibilityValidator.validate_plan(
        "event_window_analysis",
        {
            "columns": ["vibration"],
            "event_column": "failure",
            "event_predicate": {"column": "failure", "operator": "==", "value": 1},
            "sequence_order": "sequence",
        },
        profile,
    )

    assert not result.is_feasible
    assert result.error_code == "CLARIFICATION_REQUIRED"
    assert "lookback window" in result.clarification_prompt.lower()


def test_planner_contract_failure_is_reported_without_invented_clarification(monkeypatch):
    df = pd.DataFrame(
        {
            "sequence": [1, 2, 3, 4],
            "failure": [0, 0, 1, 0],
            "vibration": [0.1, 0.3, 1.2, 0.2],
        }
    )
    monkeypatch.setenv("OLLAMA_TOOL_RETRIEVER", "lexical")
    monkeypatch.setattr(
        "modules.query_orchestrator.ollama_service.verify_ready",
        lambda: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(
        "modules.query_orchestrator.SemanticPlanner.extract_semantic_intent",
        lambda **_: None,
    )

    response = QueryOrchestrator.process_query(
        "Compare variance before a failure event",
        df=df,
    )

    assert response.status == "FAILED"
    assert response.error.code == "INTENT_PLAN_INVALID"
    assert response.clarification is None
