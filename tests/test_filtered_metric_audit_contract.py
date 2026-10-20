"""Regression contracts for generic filtered metric audits.

The cases deliberately use neutral column names.  They protect the registry
contract itself, rather than encoding a particular demo dataset or prompt.
"""

import pandas as pd

from backend.ds_engine.command_resolver import CommandResolver
from backend.ds_engine.result_validator import SemanticResultValidator
from backend.ds_engine.typed_operators import TypedOperatorRegistry
from modules.query_orchestrator import QueryOrchestrator, _map_intent_to_v2
from modules.semantic_planner import SemanticPlanner


def _resolve_to_v2(command, dataframe):
    resolution = CommandResolver.resolve_command(command, dataframe, raw_query="generic audit test")
    assert resolution.is_valid, resolution.error_message
    return _map_intent_to_v2(resolution.semantic_intent, "generic audit test")


def test_overall_filtered_count_does_not_require_grouping_dimension():
    dataframe = pd.DataFrame(
        {
            "operating_status": ["off", "off", "on", "off"],
            "volume": [0, 5, 7, 0],
        }
    )
    intent = _resolve_to_v2(
        {
            "tool": "aggregate_data",
            "arguments": {
                "column": "operating_status",
                "aggregation": "count",
                "filters": [{"column": "operating_status", "condition": "==", "value": "off"}],
            },
        },
        dataframe,
    )

    assert intent.requires_grouping is False
    output = TypedOperatorRegistry.execute_intent(dataframe, intent)
    assert output.status == "SUCCEEDED", output.error_message
    assert output.data["records"] == [{"operating_status": 3}]
    assert SemanticResultValidator.validate_manifest(intent, output.manifest, output.data).is_valid


def test_measure_expression_is_rejected_before_execution_not_treated_as_column():
    """Planner display syntax is never a substitute for an exact column name."""
    dataframe = pd.DataFrame({"segment": ["a", "b"], "amount": [10.0, 20.0]})
    resolution = CommandResolver.resolve_command(
        {
            "tool": "group_data",
            "arguments": {
                "group_by": ["segment"],
                "metric": "sum(amount)",
                "aggregation": "sum",
                "datetime_parses": [],
                "chart_type": "bar",
            },
        },
        dataframe,
        raw_query="Show total amount by segment as a bar chart.",
    )

    assert resolution.is_valid is False
    assert resolution.error_code == "INVALID_NUMERIC_MEASURE"
    assert "exact column names only" in resolution.error_message


def test_filtered_metric_distribution_reports_observed_zero_evidence():
    dataframe = pd.DataFrame(
        {
            "operating_status": ["off", "off", "on", "off"],
            "volume": [0, 5, 7, 0],
        }
    )
    intent = _resolve_to_v2(
        {
            "tool": "audit_metric_distribution",
            "arguments": {
                "metric": "volume",
                "filters": [{"column": "operating_status", "condition": "==", "value": "off"}],
                "chart_type": "box",
            },
        },
        dataframe,
    )

    output = TypedOperatorRegistry.execute_intent(dataframe, intent)
    assert output.status == "SUCCEEDED", output.error_message
    record = output.data["records"][0]
    assert record["count"] == 3
    assert record["zero_count"] == 2
    assert record["zero_rate_percent"] == 66.6667
    assert output.manifest.chart_type == "box"
    assert SemanticResultValidator.validate_manifest(intent, output.manifest, output.data).is_valid


def test_orchestrator_preserves_filtered_audit_contract_end_to_end(monkeypatch):
    """The V2 route must not lose metric, filter, zero evidence, or chart."""
    dataframe = pd.DataFrame(
        {
            "operating_status": ["off", "off", "on", "off"],
            "volume": [0, 5, 7, 0],
        }
    )
    planner_command = {
        "tool": "audit_metric_distribution",
        "arguments": {
            "metric": "volume",
            "filters": [{"column": "operating_status", "condition": "==", "value": "off"}],
            "chart_type": "box",
        },
    }
    monkeypatch.setattr(
        SemanticPlanner,
        "extract_semantic_intent",
        classmethod(lambda cls, **_kwargs: planner_command),
    )

    response = QueryOrchestrator.process_query("generic filtered metric audit", dataframe)

    assert response.status == "SUCCEEDED"
    result = response.result
    assert result["manifest"]["operator"] == "distribution_comparison"
    assert result["manifest"]["chart_type"] == "box"
    assert result["manifest"]["filters_applied"] == [
        {"column": "operating_status", "operator": "eq", "value": "off"}
    ]
    assert result["data"]["records"][0]["zero_count"] == 2
