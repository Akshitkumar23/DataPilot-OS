"""Protocol-contract coverage for local-model function-call envelopes."""

from modules.semantic_planner import SemanticPlanner
from backend.intelligence.ollama_service import OllamaResponse
import pandas as pd


def test_model_name_envelope_is_canonicalized_before_tool_gate():
    payload = {
        "commands": [
            {
                "name": "correlate_data",
                "arguments": {"columns": ["attendance_rate_pct", "lms_hours_weekly"]},
            }
        ]
    }

    normalized = SemanticPlanner._normalise_tool_protocol(payload)

    assert normalized["commands"][0]["tool"] == "correlate_data"
    assert SemanticPlanner._uses_only_selected_tools(normalized, {"correlate_data"})


def test_conflicting_provider_and_canonical_tool_names_fail_closed():
    payload = {
        "name": "correlate_data",
        "tool": "forecast_data",
        "arguments": {},
    }

    normalized = SemanticPlanner._normalise_tool_protocol(payload)

    assert not SemanticPlanner._uses_only_selected_tools(normalized, {"correlate_data"})


def test_name_envelope_reaches_strict_registry_resolution(monkeypatch):
    response = OllamaResponse(
        success=True,
        content='{"name": "correlate_data"}',
        parsed_json={
            "name": "correlate_data",
            "arguments": {
                "columns": ["attendance_rate_pct", "lms_hours_weekly"],
                "analysis_mode": "matrix",
                "relationship_form": "linear",
                "correlation_methods": ["pearson"],
                "chart_type": "heatmap",
                "smoother": "none",
                "expected_direction": "unspecified",
            },
        },
    )
    monkeypatch.setattr(
        "modules.semantic_planner.ollama_service.structured_request",
        lambda _request: response,
    )

    result = SemanticPlanner.extract_semantic_intent(
        "Build a correlation matrix for attendance_rate_pct and lms_hours_weekly.",
        df=pd.DataFrame({
            "attendance_rate_pct": [78.0, 84.0, 91.0],
            "lms_hours_weekly": [2.3, 4.1, 6.4],
        }),
    )

    assert result is not None
    assert result["tool_name"] == "correlate_data"
    assert result["execution_plan"][0]["arguments"]["chart_type"] == "heatmap"
