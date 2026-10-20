import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from backend.ds_engine.command_resolver import command_resolver
from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily, TimeContext, VisualizationSpec
from backend.ds_engine.typed_operators import TimeSeriesForecastOperator, TypedOperatorRegistry
from modules.capability_registry import ForecastArgs, capability_registry
from modules.query_orchestrator import _map_intent_to_v2


def test_deprecated_pandas_hour_alias_is_canonicalized_and_preserves_time():
    """Pandas 3 rejects H, but a valid hourly user/planner interval stays hourly."""
    arguments = ForecastArgs.model_validate({
        "value_column": "demand",
        "time_column": "observed_at",
        "horizon": 2,
        "frequency": "H",
        "method": "naive_last_value",
        "chart_type": "none",
    })
    assert arguments.frequency == "h"

    df = pd.DataFrame({
        "observed_at": pd.date_range("2025-01-01", periods=6, freq="h"),
        "demand": [1., 2., 3., 4., 5., 6.],
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.TIME_SERIES_FORECAST,
        target_column="demand",
        subjects=["demand"],
        time_context=TimeContext(time_column="observed_at", horizon=2),
        visualization=VisualizationSpec(type="none"),
        arguments=arguments.model_dump(),
    )
    result = TimeSeriesForecastOperator().execute(df, intent)
    assert result.status == "SUCCEEDED"
    assert result.manifest.assumptions[0] == "Declared frequency=h"
    assert result.data["records"][0]["ds"].endswith("06:00:00")
    assert result.data["records"][1]["ds"].endswith("07:00:00")


def test_forecast_horizon_must_be_grounded_in_the_user_request():
    """A profile cannot be turned into a forecast by inventing a horizon."""
    base_args = {
        "value_column": "Rented Bike Count",
        "time_column": "Date",
        "horizon": 24,
        "frequency": "h",
        "method": "naive_last_value",
        "chart_type": "line",
    }
    profile_query = (
        "Operations needs an hourly demand profile. Parse Date as DD/MM/YYYY, then show "
        "average Rented Bike Count by Hour and Seasons with a line chart."
    )
    rejected = command_resolver.resolve_command(
        {"tool": "forecast_data", "arguments": {**base_args, "horizon_evidence": "hourly demand profile"}},
        raw_query=profile_query,
    )
    assert not rejected.is_valid
    assert rejected.error_code == "UNGROUNDED_REQUEST_ARGUMENT"

    accepted = command_resolver.resolve_command(
        {"tool": "forecast_data", "arguments": {**base_args, "horizon": 7, "horizon_evidence": "next 7 hours"}},
        raw_query="Forecast Rented Bike Count for the next 7 hours.",
    )
    assert accepted.is_valid, accepted.error_message


def test_grouped_numeric_aggregate_rejects_a_date_bound_as_the_measure():
    """A datetime parse source cannot silently become a mean/average metric."""
    df = pd.DataFrame({
        "Date": ["01/12/2017", "02/12/2017"],
        "Hour": [0, 1],
        "Rented Bike Count": [100, 200],
    })
    query = "Show average Rented Bike Count by Hour after parsing Date as DD/MM/YYYY."
    rejected = command_resolver.resolve_command(
        {
            "tool": "group_data",
            "arguments": {
                "group_by": ["Hour"],
                "metric": "Date",
                "aggregation": "mean",
                "chart_type": "line",
                "datetime_parses": [{"column": "Date", "format": "%d/%m/%Y"}],
            },
        },
        df=df,
        raw_query=query,
    )
    assert not rejected.is_valid
    assert rejected.error_code == "INVALID_NUMERIC_MEASURE"

    accepted = command_resolver.resolve_command(
        {
            "tool": "group_data",
            "arguments": {
                "group_by": ["Hour"],
                "metric": "Rented Bike Count",
                "aggregation": "mean",
                "chart_type": "line",
                "datetime_parses": [{"column": "Date", "format": "%d/%m/%Y"}],
            },
        },
        df=df,
        raw_query=query,
    )
    assert accepted.is_valid, accepted.error_message


def test_historical_hourly_profile_is_retrieved_and_executes_as_grouped_mean_line_chart():
    """A historical grouped profile must remain a grouping, never a forecast."""
    query = (
        "Operations needs an hourly demand profile. Parse Date as DD/MM/YYYY, then show "
        "average Rented Bike Count by Hour and Seasons with a line chart."
    )
    selected = capability_registry.select_planner_tools(query, has_dataset=True)
    assert "group_data" in [spec.name for spec in selected]

    df = pd.DataFrame({
        "Date": ["01/12/2017"] * 8,
        "Hour": [0, 1, 0, 1, 0, 1, 0, 1],
        "Seasons": ["Winter", "Winter", "Winter", "Winter", "Summer", "Summer", "Summer", "Summer"],
        "Rented Bike Count": [100, 200, 120, 220, 300, 400, 320, 420],
    })
    resolution = command_resolver.resolve_command({
        "tool": "group_data",
        "arguments": {
            "group_by": ["Hour", "Seasons"],
            "metric": "Rented Bike Count",
            "aggregation": "mean",
            "chart_type": "line",
            "datetime_parses": [{"column": "Date", "format": "%d/%m/%Y"}],
        },
    }, df=df, raw_query=query)
    assert resolution.is_valid, resolution.error_message

    intent = _map_intent_to_v2(resolution.semantic_intent, query, df)
    result = TypedOperatorRegistry.execute_intent(df, intent)
    assert result.status == "SUCCEEDED"
    assert result.manifest.operator == "group_aggregation"
    assert result.manifest.statistics_computed == ["mean"]
    assert result.manifest.chart_type == "line"
    assert result.data["row_count"] == 4
    assert result.plot_image_base64 is not None
    assert "Date" in result.manifest.columns_used
    assert result.manifest.assumptions == ["Parsed Date using declared format=%d/%m/%Y"]


def test_table_only_eda_cannot_promise_a_bar_and_registry_expands_chart_repair_choices():
    """A chart request must fail at plan validation, never after fake execution."""
    rejected = command_resolver.resolve_command(
        {"tool": "eda_data", "arguments": {"chart_type": "bar"}},
        raw_query="Show a bar chart.",
    )
    assert not rejected.is_valid
    assert rejected.error_code == "UNSUPPORTED_VISUALIZATION"

    selected = [capability_registry.get_tool_spec("eda_data")]
    repaired_choices = capability_registry.expand_tools_for_requested_visualizations(
        selected,
        {"tool": "eda_data", "arguments": {"chart_type": "bar"}},
    )
    assert "group_data" in [spec.name for spec in repaired_choices]
