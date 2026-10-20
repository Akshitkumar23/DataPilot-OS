"""End-to-end contracts for generic conditional cohort-rate analysis."""

import pandas as pd

from backend.ds_engine.command_resolver import CommandResolver
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.result_validator import SemanticResultValidator
from backend.ds_engine.semantic_feasibility import SemanticFeasibilityValidator
from backend.ds_engine.typed_operators import TypedOperatorRegistry
from modules.capability_registry import capability_registry
from modules.query_orchestrator import _map_intent_to_v2


def _execute(command, frame):
    resolution = CommandResolver.resolve_command(command, df=frame, raw_query="generic conditional-rate request")
    assert resolution.is_valid, resolution.error_message
    v2_intent = _map_intent_to_v2(resolution.semantic_intent, "generic conditional-rate request", frame)
    output = TypedOperatorRegistry.execute_intent(frame, v2_intent)
    return resolution, v2_intent, output


def test_conditional_rate_is_live_registry_capability_and_reports_both_counts():
    frame = pd.DataFrame({
        "team": ["north", "north", "north", "north", "south", "south", "south", "tiny"],
        "decision": ["approved", "declined", "approved", "declined", "approved", "declined", "declined", "approved"],
    })
    command = {
        "tool": "conditional_rate",
        "arguments": {
            "group_by": ["team"],
            "numerator_condition": {"column": "decision", "operator": "eq", "value": "approved"},
            "min_group_size": 2,
            "sort_by": "rate_percent",
            "direction": "desc",
            "chart_type": "bar",
        },
    }

    assert capability_registry.get_tool_spec("conditional_rate") is not None
    assert capability_registry.get_tool_intent_map()["conditional_rate"].value == "GROUP_AGGREGATE"

    _, intent, output = _execute(command, frame)
    assert intent.visualization.type == "bar"
    assert output.status == "SUCCEEDED"
    assert output.manifest.operator == "conditional_rate"
    assert output.manifest.columns_used == ["team", "decision"]
    assert output.manifest.statistics_computed == ["matching_count", "total_count", "rate", "rate_percent"]

    rows = output.data["records"]
    assert [row["team"] for row in rows] == ["north", "south"]
    assert rows[0]["matching_count"] == 2
    assert rows[0]["total_count"] == 4
    assert rows[0]["rate_percent"] == 50.0
    assert rows[1]["matching_count"] == 1
    assert rows[1]["total_count"] == 3
    assert round(rows[1]["rate_percent"], 6) == round(100 / 3, 6)
    assert output.plot_image_base64 is not None

    validation = SemanticResultValidator.validate_manifest(intent, output.manifest, output.data)
    assert validation.is_valid, validation.issues


def test_conditional_rate_supports_non_binary_numeric_predicates_without_dataset_assumptions():
    frame = pd.DataFrame({
        "region": ["east", "east", "east", "west", "west", "west"],
        "latency_ms": [42, 130, 165, 99, 101, 220],
    })
    command = {
        "tool": "conditional_rate",
        "arguments": {
            "group_by": ["region"],
            "numerator_condition": {"column": "latency_ms", "operator": "gt", "value": 100},
            "sort_by": "matching_count",
            "direction": "desc",
            "chart_type": "none",
        },
    }

    _, _, output = _execute(command, frame)
    assert output.status == "SUCCEEDED"
    rows = {row["region"]: row for row in output.data["records"]}
    assert rows["east"]["matching_count"] == 2
    assert rows["east"]["total_count"] == 3
    assert rows["west"]["matching_count"] == 2
    assert rows["west"]["total_count"] == 3
    assert output.manifest.chart_type == "none"


def test_conditional_rate_rejects_missing_predicate_column_instead_of_guessing():
    frame = pd.DataFrame({"segment": ["a", "b"], "outcome": [1, 0]})
    command = {
        "tool": "conditional_rate",
        "arguments": {
            "group_by": ["segment"],
            "numerator_condition": {"column": "not_a_column", "operator": "eq", "value": 1},
            "chart_type": "none",
        },
    }

    _, _, output = _execute(command, frame)
    assert output.status == "FAILED"
    assert output.error_code == "COLUMN_NOT_FOUND"
    assert "not_a_column" in output.error_message


def test_conditional_rate_feasibility_is_declared_by_toolspec_metadata():
    frame = pd.DataFrame({"segment": ["a", "b"], "outcome": ["yes", "no"]})
    profile = DatasetProfiler.profile(frame)
    arguments = {
        "group_by": ["segment"],
        "numerator_condition": {"column": "outcome", "condition": "==", "value": "yes"},
        "chart_type": "bar",
    }

    result = SemanticFeasibilityValidator.validate_plan("conditional_rate", arguments, profile)
    assert result.is_feasible, result.error_message
    assert arguments["group_by"] == ["segment"]
    assert arguments["numerator_condition"] == {"column": "outcome", "operator": "eq", "value": "yes"}

    arguments["numerator_condition"] = {"column": "invented", "operator": "eq", "value": "yes"}
    result = SemanticFeasibilityValidator.validate_plan("conditional_rate", arguments, profile)
    assert not result.is_feasible
    assert result.error_code == "COLUMN_NOT_FOUND"
