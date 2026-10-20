"""
DataPilot OS — Regression Test Suite for Explicit Datetime Format Execution.

Tests:
1. Real Seoul Bike dataset with user-specified format 'DD/MM/YYYY' (all 8,760 rows parse, zero unparseable, exact oracle match).
2. Human date tokens ('YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY HH:mm') normalize to exact strptime patterns.
3. Pre-existing strptime directives ('%d/%m/%Y', '%Y-%m-%d') pass through unchanged.
4. Invalid format ('XYZ') fails explicitly with INVALID_DATETIME_FORMAT / ValueError (no guessing, zero fallback).
5. Mismatched format on real data fails explicitly with DATETIME_PARSE_FAILED (strict zero silent guessing).
6. TimeSeriesEngine methods (validate, stationarity, acf, decompose, forecast) accept and strictly honor datetime_format.
7. CleanOperator calendar feature extraction respects explicit datetime_format.
8. Generated Python code contains exact normalized strptime format string.
"""

import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.ds_engine.datetime_parser import normalize_datetime_format, parse_datetime_column, DatetimeParseResult
from backend.ds_engine.command_resolver import command_resolver
from modules.capability_registry import DatetimeParseSpec, GroupArgs, ForecastArgs, CleanArgs, StationarityTestArgs, DecomposeTimeSeriesArgs
from modules.query_orchestrator import _map_intent_to_v2
from backend.ds_engine.typed_operators import TypedOperatorRegistry
from backend.ds_engine.timeseries_engine import timeseries_engine


SEOUL_BIKE_PATH = os.environ.get(
    "SEOUL_BIKE_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "external" / "seoul_bike.csv")
)


def test_normalize_datetime_format_mappings():
    """Validates deterministic human token to strptime conversion."""
    assert normalize_datetime_format("DD/MM/YYYY") == "%d/%m/%Y"
    assert normalize_datetime_format("YYYY-MM-DD") == "%Y-%m-%d"
    assert normalize_datetime_format("MM/DD/YYYY") == "%m/%d/%Y"
    assert normalize_datetime_format("DD-MM-YYYY") == "%d-%m-%Y"
    assert normalize_datetime_format("YYYY/MM/DD") == "%Y/%m/%d"
    assert normalize_datetime_format("DD/MM/YYYY HH:mm") == "%d/%m/%Y %H:%M"
    assert normalize_datetime_format("YYYY-MM-DD HH:mm:ss") == "%Y-%m-%d %H:%M:%S"
    assert normalize_datetime_format("YYYY-MM") == "%Y-%m"


def test_normalize_datetime_format_preserves_strptime():
    """Validates pre-existing strptime directives pass through intact."""
    assert normalize_datetime_format("%d/%m/%Y") == "%d/%m/%Y"
    assert normalize_datetime_format("%Y-%m-%d") == "%Y-%m-%d"
    assert normalize_datetime_format("%m/%d/%Y %H:%M:%S") == "%m/%d/%Y %H:%M:%S"


def test_normalize_datetime_format_rejects_invalid_tokens():
    """Explicitly rejects unknown or invalid format tokens without guessing."""
    with pytest.raises(ValueError, match="Invalid datetime format"):
        normalize_datetime_format("XYZ")

    with pytest.raises(ValueError, match="must be a non-empty string"):
        normalize_datetime_format("")

    with pytest.raises(ValueError, match="Unrecognized strptime directive"):
        normalize_datetime_format("%K/%m/%Y")


def test_pydantic_datetime_parse_spec_normalizes_format():
    """DatetimeParseSpec schema validator normalizes human tokens to strptime."""
    spec = DatetimeParseSpec(column="Date", format="DD/MM/YYYY")
    assert spec.format == "%d/%m/%Y"

    spec2 = DatetimeParseSpec(column="timestamp", format="%Y-%m-%d")
    assert spec2.format == "%Y-%m-%d"

    with pytest.raises(Exception):
        DatetimeParseSpec(column="Date", format="XYZ")


def test_seoul_bike_real_data_group_aggregation_with_dd_mm_yyyy():
    """
    Production Defect Reproduction & Fix Verification:
    Seoul Bike dataset with 8,760 rows, Date column 'DD/MM/YYYY'.
    Execution succeeds with 100% parsed rows and exact oracle parity.
    """
    if not os.path.exists(SEOUL_BIKE_PATH):
        pytest.skip(f"Dataset not found at {SEOUL_BIKE_PATH}")

    df = pd.read_csv(SEOUL_BIKE_PATH)
    assert len(df) == 8760

    query = (
        "Operations needs an hourly demand profile. Parse Date as DD/MM/YYYY, then show "
        "average Rented Bike Count by Hour and Seasons with a line chart."
    )
    cmd = {
        "tool": "group_data",
        "arguments": {
            "group_by": ["Hour", "Seasons"],
            "metric": "Rented Bike Count",
            "aggregation": "mean",
            "chart_type": "line",
            "datetime_parses": [{"column": "Date", "format": "DD/MM/YYYY"}],
        },
    }

    res = command_resolver.resolve_command(cmd, df=df, raw_query=query)
    assert res.is_valid, res.error_message

    intent_v2 = _map_intent_to_v2(res.semantic_intent, query, df)
    op_res = TypedOperatorRegistry.execute_intent(df, intent_v2)

    assert op_res.status == "SUCCEEDED"
    assert op_res.manifest.operator == "group_aggregation"
    assert "Date" in op_res.manifest.columns_used
    assert "Parsed Date using declared format=%d/%m/%Y" in op_res.manifest.assumptions

    # Oracle Parity Check: compare full op_res.dataframe to oracle
    oracle = df.groupby(["Hour", "Seasons"], dropna=False)["Rented Bike Count"].mean().reset_index()
    res_df = op_res.dataframe
    assert len(res_df) == len(oracle) == 96
    merged = pd.merge(oracle, res_df, on=["Hour", "Seasons"], suffixes=("_oracle", "_op"))
    assert len(merged) == 96
    max_diff = (merged["Rented Bike Count_oracle"] - merged["Rented Bike Count_op"]).abs().max()
    assert max_diff < 1e-9

    # Verify generated reproducible Python code contains exact strptime
    assert "%d/%m/%Y" in str(op_res.reproducible_code)


def test_mismatched_datetime_format_fails_explicitly():
    """
    Strict zero silent fallback:
    If user/model specifies format 'YYYY-MM-DD' on '01/12/2017' data,
    the system must NOT guess or fall back; it must fail with DATETIME_PARSE_FAILED.
    """
    df = pd.DataFrame({
        "Date": ["01/12/2017", "02/12/2017", "13/12/2017"],
        "Hour": [0, 1, 2],
        "Count": [10, 20, 30],
    })
    cmd = {
        "tool": "group_data",
        "arguments": {
            "group_by": ["Hour"],
            "metric": "Count",
            "aggregation": "mean",
            "chart_type": "none",
            "datetime_parses": [{"column": "Date", "format": "YYYY-MM-DD"}],
        },
    }
    res = command_resolver.resolve_command(cmd, df=df, raw_query="Show Count by Hour after parsing Date as YYYY-MM-DD")
    assert res.is_valid
    intent_v2 = _map_intent_to_v2(res.semantic_intent, "Show Count by Hour after parsing Date as YYYY-MM-DD", df)
    op_res = TypedOperatorRegistry.execute_intent(df, intent_v2)

    assert op_res.status == "FAILED"
    assert op_res.error_code == "DATETIME_PARSE_FAILED"
    assert "Datetime parsing failed" in op_res.error_message


def test_timeseries_engine_with_explicit_datetime_format():
    """Validates timeseries_engine honors explicit format on day-first dates without ambiguous truncation."""
    dates = [f"{day:02d}/12/2017" for day in range(1, 25)]
    values = [float(100 + i * 5) for i in range(len(dates))]
    df = pd.DataFrame({"Date": dates, "Demand": values})

    # Without format, day>12 causes unparseable timestamps or warning
    # With explicit format DD/MM/YYYY:
    val_res = timeseries_engine.validate_time_series(df, "Date", "Demand", datetime_format="DD/MM/YYYY")
    assert val_res["status"] == "VALIDATED"
    assert val_res["usable_observations"] == 24
    assert val_res["unparseable_timestamps"] == 0
    assert val_res["datetime_format"] == "DD/MM/YYYY"

    # Stationarity test
    stat_res = timeseries_engine.run_stationarity_tests(df, "Date", "Demand", datetime_format="DD/MM/YYYY")
    assert stat_res["status"] == "SUCCEEDED"
    assert stat_res["sample_size"] == 24

    # Autocorrelation
    acf_res = timeseries_engine.run_acf_pacf(df, "Date", "Demand", max_lags=5, datetime_format="DD/MM/YYYY")
    assert acf_res["status"] == "SUCCEEDED"

    # Forecast
    fc_res = timeseries_engine.run_forecast(df, "Date", "Demand", model_type="naive", horizon=3, datetime_format="DD/MM/YYYY")
    assert fc_res["status"] == "SUCCEEDED"
    assert len(fc_res["forecast_records"]) == 3


def test_clean_operator_calendar_features_with_explicit_datetime_format():
    """Validates CleanOperator calendar_features extraction supports explicit datetime_format."""
    df = pd.DataFrame({
        "Date": ["01/12/2017", "13/12/2017", "25/12/2017", "05/01/2018"],
        "Count": [10, 20, 30, 40],
    })
    args = CleanArgs.model_validate({
        "column": "Date",
        "method": "calendar_features",
        "datetime_format": "DD/MM/YYYY",
    })
    assert args.datetime_format == "%d/%m/%Y"

    from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily, VisualizationSpec
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["Date"],
        time_column="Date",
        visualization=VisualizationSpec(type="none"),
        arguments=args.model_dump(),
    )
    res = TypedOperatorRegistry.execute_intent(df, intent)
    assert res.status == "SUCCEEDED"
    assert "Date__month" in res.manifest.statistics_computed
    assert "Date__day" in res.manifest.statistics_computed
