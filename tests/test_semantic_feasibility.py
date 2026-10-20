"""Contract tests for registry-driven, zero-keyword feasibility validation."""

import pandas as pd
import pytest

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.semantic_feasibility import (
    SemanticFeasibilityValidator,
    _find_sequence_candidates,
)
from backend.intelligence.ollama_service import OllamaReadinessStatus, ollama_service
from modules.query_orchestrator import QueryOrchestrator
from modules.semantic_planner import SemanticPlanner


@pytest.fixture
def ordered_event_df():
    return pd.DataFrame({
        "opaque_counter": list(range(1, 101)),
        "signal": [float(i) for i in range(100)],
        "incident_flag": [0] * 90 + [1] * 10,
        "category": ["a"] * 50 + ["b"] * 50,
    })


@pytest.fixture
def temporal_df():
    return pd.DataFrame({
        "observed_at": pd.date_range("2026-01-01", periods=32, freq="h"),
        "signal": [float(i) for i in range(32)],
        "incident_flag": [0] * 28 + [1] * 4,
    })


def test_sequence_candidates_use_structure_not_column_names(ordered_event_df):
    profile = DatasetProfiler.profile(ordered_event_df, "ordered")
    # Both columns meet the structural ordering contract.  The validator must
    # expose that ambiguity instead of choosing one from its name or position.
    assert _find_sequence_candidates(profile) == ["opaque_counter", "signal"]


def test_event_tool_without_ordering_requests_explicit_choice(ordered_event_df):
    profile = DatasetProfiler.profile(ordered_event_df, "ordered")
    result = SemanticFeasibilityValidator.validate_plan(
        "event_window_analysis",
        {
            "columns": ["signal"],
            "event_column": "incident_flag",
            "event_predicate": {"column": "incident_flag", "operator": "==", "value": 1},
            "pre_event_window": 5,
        },
        profile,
        query_raw="This text must not affect feasibility routing.",
    )
    assert result.error_code == "CLARIFICATION_REQUIRED"
    assert result.candidate_columns == ["opaque_counter", "signal"]


def test_event_tool_accepts_explicit_structural_sequence(ordered_event_df):
    profile = DatasetProfiler.profile(ordered_event_df, "ordered")
    args = {
        "columns": ["signal"],
        "sequence_order": "opaque_counter",
        "event_column": "incident_flag",
        "event_predicate": {"column": "incident_flag", "operator": "==", "value": 1},
        "pre_event_window": 5,
    }
    result = SemanticFeasibilityValidator.validate_plan("event_window_analysis", args, profile)
    assert result.is_feasible
    assert args["sequence_column"] == "opaque_counter"


def test_event_tool_never_guesses_event_column(ordered_event_df):
    profile = DatasetProfiler.profile(ordered_event_df, "ordered")
    result = SemanticFeasibilityValidator.validate_plan(
        "event_window_analysis",
        {
            "sequence_order": "opaque_counter",
            "event_predicate": {"operator": "==", "value": 1},
        },
        profile,
    )
    assert result.error_code == "CLARIFICATION_REQUIRED"


def test_event_predicate_requires_operator(ordered_event_df):
    profile = DatasetProfiler.profile(ordered_event_df, "ordered")
    result = SemanticFeasibilityValidator.validate_plan(
        "event_window_analysis",
        {
            "sequence_order": "opaque_counter",
            "event_column": "incident_flag",
            "event_predicate": {"column": "incident_flag", "value": 1},
        },
        profile,
    )
    assert result.error_code == "EVENT_PREDICATE_REQUIRED"


def test_temporal_tool_requires_explicit_temporal_column(temporal_df):
    profile = DatasetProfiler.profile(temporal_df, "temporal")
    result = SemanticFeasibilityValidator.validate_plan(
        "forecast_data", {"value_column": "signal", "horizon": 3}, profile
    )
    assert result.error_code == "CLARIFICATION_REQUIRED"
    assert result.candidate_columns == ["observed_at"]


def test_temporal_tool_rejects_sequence_as_timestamp(ordered_event_df):
    profile = DatasetProfiler.profile(ordered_event_df, "ordered")
    result = SemanticFeasibilityValidator.validate_plan(
        "forecast_data", {"value_column": "signal", "time_column": "opaque_counter", "horizon": 3}, profile
    )
    assert result.error_code == "INVALID_TEMPORAL_COLUMN"


def test_ambiguous_planner_response_does_not_invent_an_empty_analytical_call(monkeypatch):
    """Retrieval is not evidence that the user's selected arguments are absent."""
    df = pd.DataFrame({
        "UDI": [1, 2, 3, 4],
        "vibration_x_g": [0.1, 0.2, 0.7, 1.1],
        "vibration_y_g": [0.1, 0.3, 0.6, 1.2],
        "vibration_z_g": [0.2, 0.2, 0.8, 1.0],
        "Machine_failure": [0, 0, 0, 1],
    })
    monkeypatch.setattr(
        ollama_service,
        "verify_ready",
        lambda: OllamaReadinessStatus(ready=True, model="test-model", model_available=True),
    )
    monkeypatch.setattr(
        SemanticPlanner,
        "extract_semantic_intent",
        classmethod(lambda cls, **_kwargs: {
            "commands": [{
                "tool": "clarification_needed",
                "arguments": {"reason": "Please provide more details."},
            }],
        }),
    )

    response = QueryOrchestrator.process_query(
        "Before machine failure, compare variance in vibration_x_g, vibration_y_g, vibration_z_g.",
        df=df,
    )

    assert response.status == "NEEDS_CLARIFICATION"
    assert response.error.code == "CLARIFICATION_REQUIRED"
    assert response.error.message == "Please provide more details."
    assert response.error.candidate_columns == []
