"""
Tests for time-series grain clarification, duplicate timestamp detection,
and irregular temporal grid gating (Phase 6).
"""

import pandas as pd
import pytest

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.time_series_grain import (
    TimeSeriesGrainDiagnostics,
    analyze_time_series_grain,
)
from backend.ds_engine.semantic_feasibility import SemanticFeasibilityValidator
from backend.ds_engine.feasibility_rules import FeasibilityRulesEngine


@pytest.fixture
def regular_ts_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="D"),
        "sales": [float(i * 2) for i in range(50)],
    })


@pytest.fixture
def duplicate_ts_df():
    # 50 records with duplicate timestamps
    dates = pd.date_range("2026-01-01", periods=25, freq="D").tolist() * 2
    return pd.DataFrame({
        "timestamp": sorted(dates),
        "sales": [float(i * 2) for i in range(50)],
    })


@pytest.fixture
def irregular_grid_df():
    # Gaps between timestamps (e.g. day 1, 2, 5, 12, 19, 35)
    dates = [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-01-19"),
        pd.Timestamp("2026-02-05"),
    ]
    return pd.DataFrame({
        "timestamp": dates,
        "sales": [10.0, 15.0, 12.0, 18.0, 20.0, 25.0],
    })


def test_regular_time_series_grain_passes(regular_ts_df):
    diag = analyze_time_series_grain(regular_ts_df, "timestamp")
    assert not diag.has_duplicates
    assert diag.is_regular_grid
    assert not diag.requires_clarification

    profile = DatasetProfiler.profile(regular_ts_df, "regular_ts")
    result = SemanticFeasibilityValidator.validate_plan(
        "forecast_data",
        {"time_column": "timestamp", "value_column": "sales", "horizon": 7},
        profile,
        df=regular_ts_df,
    )
    assert result.is_feasible


def test_duplicate_timestamps_trigger_grain_clarification(duplicate_ts_df):
    diag = analyze_time_series_grain(duplicate_ts_df, "timestamp")
    assert diag.has_duplicates
    assert diag.duplicate_count == 25
    assert diag.requires_clarification
    assert "duplicate timestamp" in diag.clarification_reason.lower()

    profile = DatasetProfiler.profile(duplicate_ts_df, "duplicate_ts")
    result = SemanticFeasibilityValidator.validate_plan(
        "forecast_data",
        {"time_column": "timestamp", "value_column": "sales", "horizon": 7},
        profile,
        df=duplicate_ts_df,
    )
    assert not result.is_feasible
    assert result.error_code == "CLARIFICATION_REQUIRED"
    assert "duplicate timestamp" in result.error_message.lower()
    assert "suggested_grains" in result.suggested_adjustments


def test_irregular_temporal_grid_requests_explicit_grain(irregular_grid_df):
    diag = analyze_time_series_grain(irregular_grid_df, "timestamp")
    assert not diag.has_duplicates
    assert not diag.is_regular_grid
    assert diag.requires_clarification
    assert "irregular time grid" in diag.clarification_reason.lower()

    profile = DatasetProfiler.profile(irregular_grid_df, "irregular_ts")
    result = SemanticFeasibilityValidator.validate_plan(
        "autocorrelation",
        {"time_column": "timestamp", "value_column": "sales"},
        profile,
        df=irregular_grid_df,
    )
    assert not result.is_feasible
    assert result.error_code == "CLARIFICATION_REQUIRED"
    assert "irregular time grid" in result.error_message.lower()
    assert len(result.suggested_adjustments.get("suggested_grains", [])) > 0


def test_feasibility_rules_engine_direct_call(duplicate_ts_df):
    res = FeasibilityRulesEngine.check_time_series_grain(
        "decompose_time_series",
        {"time_column": "timestamp", "value_column": "sales"},
        duplicate_ts_df,
    )
    assert res is not None
    assert not res.is_feasible
    assert res.error_code == "TIME_SERIES_GRAIN_CLARIFICATION"
    assert "duplicate timestamp" in res.error_message.lower()
