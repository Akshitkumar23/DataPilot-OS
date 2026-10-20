"""
DataPilot OS — Phase 6: Real Time-Series & Forecasting Studio Full Test Suite.
Verifies time-series validation, stationarity diagnostics (ADF, KPSS), ACF/PACF,
seasonal decomposition, and forecasting (Naive, AutoReg, ARIMA, SARIMAX) with
strict chronological validation, zero active dataset mutation, and JSON safety.
"""

import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_registry import dataset_registry
from backend.ds_engine.timeseries_engine import timeseries_engine, sanitize_obj
from backend.persistence.provenance_store import provenance_store


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure dataset registry is clear before each test."""
    dataset_registry.clear()
    yield
    dataset_registry.clear()


@pytest.fixture
def synthetic_stationary_ts():
    """Synthetic stationary AR(1) process with daily timestamps."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    # AR(1) with phi=0.3 -> stationary
    vals = [10.0]
    for _ in range(1, n):
        vals.append(0.3 * vals[-1] + np.random.normal(0, 1.0))
    return pd.DataFrame({
        "recorded_at": dates,
        "sensor_reading": vals
    })


@pytest.fixture
def synthetic_trending_seasonal_ts():
    """Synthetic non-stationary time series with trend and weekly seasonality."""
    np.random.seed(99)
    n = 70
    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    t = np.arange(n)
    trend = 0.5 * t
    seasonal = 5.0 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 0.5, size=n)
    vals = 50.0 + trend + seasonal + noise
    return pd.DataFrame({
        "timestamp_col": dates,
        "metric_value": vals
    })


def test_time_series_validation_clean_and_diagnostics(synthetic_stationary_ts):
    """Verifies temporal index validation, monotonic check, and frequency detection."""
    df = synthetic_stationary_ts
    res = timeseries_engine.validate_time_series(df, "recorded_at", "sensor_reading")

    assert res["status"] == "VALIDATED"
    assert res["time_column"] == "recorded_at"
    assert res["target_column"] == "sensor_reading"
    assert res["usable_observations"] == 60
    assert res["null_timestamps"] == 0
    assert res["has_duplicate_timestamps"] is False
    assert res["is_chronologically_sorted"] is True
    assert res["detected_frequency"] in {"D", "1D"}
    assert res["target_summary"]["mean"] is not None


def test_time_series_validation_rejections(synthetic_stationary_ts):
    """Verifies rejection of invalid columns, non-numeric targets, and insufficient data."""
    df = synthetic_stationary_ts

    # Non-existent time column
    with pytest.raises(ValueError, match="Temporal column 'invalid_col' not found"):
        timeseries_engine.validate_time_series(df, "invalid_col", "sensor_reading")

    # Non-existent target column
    with pytest.raises(ValueError, match="Target column 'missing_target' not found"):
        timeseries_engine.validate_time_series(df, "recorded_at", "missing_target")

    # Non-numeric target
    df_str = df.copy()
    df_str["text_col"] = ["value"] * len(df)
    with pytest.raises(ValueError, match="must be numeric"):
        timeseries_engine.validate_time_series(df_str, "recorded_at", "text_col")

    # Insufficient rows
    df_short = df.head(3)
    with pytest.raises(ValueError, match="Minimum 5 required"):
        timeseries_engine.validate_time_series(df_short, "recorded_at", "sensor_reading")


def test_stationarity_diagnostics_adf_and_kpss(synthetic_stationary_ts, synthetic_trending_seasonal_ts):
    """Verifies ADF and KPSS hypothesis testing and synthesized stationarity conclusions."""
    # Stationary dataset
    stat_res = timeseries_engine.run_stationarity_tests(synthetic_stationary_ts, "recorded_at", "sensor_reading")
    assert stat_res["status"] == "SUCCEEDED"
    assert "adf_test" in stat_res
    assert "kpss_test" in stat_res
    assert stat_res["adf_test"]["p_value"] < 0.05  # Rejects H0 (stationary)
    assert stat_res["adf_test"]["is_stationary"] is True
    assert stat_res["conclusion"] in {"STATIONARY", "DIFFERENCE_STATIONARY"}

    # Trending non-stationary dataset
    nonstat_res = timeseries_engine.run_stationarity_tests(synthetic_trending_seasonal_ts, "timestamp_col", "metric_value")
    assert nonstat_res["status"] == "SUCCEEDED"
    assert nonstat_res["adf_test"]["p_value"] > 0.05  # Cannot reject H0 (has unit root / trend)
    assert nonstat_res["adf_test"]["is_stationary"] is False


def test_acf_and_pacf_computation(synthetic_stationary_ts):
    """Verifies ACF and PACF computation, 95% confidence bounds, and visualization generation."""
    res = timeseries_engine.run_acf_pacf(synthetic_stationary_ts, "recorded_at", "sensor_reading", max_lags=15)
    assert res["status"] == "SUCCEEDED"
    assert res["max_lags"] == 15
    assert len(res["lag_records"]) == 16  # lags 0 through 15
    assert res["significance_threshold"] > 0
    assert res["plot_base64"] is not None
    assert len(res["plot_base64"]) > 500

    # Lags exceeding length limit rejection
    with pytest.raises(ValueError, match="cannot exceed half the series length"):
        timeseries_engine.run_acf_pacf(synthetic_stationary_ts, "recorded_at", "sensor_reading", max_lags=50)


def test_seasonal_decomposition_additive_and_multiplicative(synthetic_trending_seasonal_ts):
    """Verifies seasonal decomposition into trend, seasonal, and residual components."""
    df = synthetic_trending_seasonal_ts
    res = timeseries_engine.run_decomposition(df, "timestamp_col", "metric_value", period=7, model="additive")
    assert res["status"] == "SUCCEEDED"
    assert res["period"] == 7
    assert res["model"] == "additive"
    assert res["plot_base64"] is not None

    # Multiplicative model on positive series
    res_mul = timeseries_engine.run_decomposition(df, "timestamp_col", "metric_value", period=7, model="multiplicative")
    assert res_mul["status"] == "SUCCEEDED"
    assert res_mul["model"] == "multiplicative"

    # Invalid period rejection
    with pytest.raises(ValueError, match="Seasonal period must be an integer >= 2"):
        timeseries_engine.run_decomposition(df, "timestamp_col", "metric_value", period=1)


def test_forecasting_models_chronological_evaluation_and_metrics(synthetic_trending_seasonal_ts):
    """Verifies forecasting with Naive, AutoReg, ARIMA, and SARIMAX including chronological evaluation."""
    df = synthetic_trending_seasonal_ts
    models = ["naive", "autoreg", "arima", "sarimax"]

    for m in models:
        params = {"p": 1, "d": 1, "q": 1, "P": 1, "D": 0, "Q": 1, "s": 7, "lags": 2}
        res = timeseries_engine.run_forecast(
            df=df,
            time_column="timestamp_col",
            target_column="metric_value",
            model_type=m,
            horizon=7,
            parameters=params,
            val_size=7,
            dataset_id="ds_test",
            dataset_name="TestTS"
        )
        assert res["status"] == "SUCCEEDED"
        assert res["model"] == m
        assert res["horizon"] == 7
        assert len(res["forecast_records"]) == 7
        assert "metrics" in res
        assert "rmse" in res["metrics"]
        assert res["metrics"]["rmse"] >= 0.0
        assert "mae" in res["metrics"]
        assert res["metrics"]["mae"] >= 0.0
        assert "baseline_metrics" in res
        assert res["baseline_metrics"]["rmse"] >= 0.0
        assert res["plot_base64"] is not None
        assert "reproducible_code" in res
        assert res["provenance_event_id"] is not None

        # Verify 95% confidence intervals are ordered
        for rec in res["forecast_records"]:
            assert rec["lower_ci_95"] <= rec["upper_ci_95"]


def test_zero_dataset_mutation_guarantee(synthetic_trending_seasonal_ts):
    """Guarantees that forecasting and diagnostics strictly NEVER mutate active DataFrame."""
    df = synthetic_trending_seasonal_ts
    original_shape = df.shape
    original_columns = list(df.columns)
    original_fingerprint = str(pd.util.hash_pandas_object(df).sum())

    # Run full spectrum of time series operations
    timeseries_engine.validate_time_series(df, "timestamp_col", "metric_value")
    timeseries_engine.run_stationarity_tests(df, "timestamp_col", "metric_value")
    timeseries_engine.run_acf_pacf(df, "timestamp_col", "metric_value")
    timeseries_engine.run_decomposition(df, "timestamp_col", "metric_value", period=7)
    timeseries_engine.run_forecast(df, "timestamp_col", "metric_value", model_type="arima", horizon=7)

    # Assert completely untouched
    assert df.shape == original_shape
    assert list(df.columns) == original_columns
    assert str(pd.util.hash_pandas_object(df).sum()) == original_fingerprint


def test_cross_dataset_state_isolation(synthetic_stationary_ts, synthetic_trending_seasonal_ts):
    """Verifies that time series results on Dataset 1 are completely isolated from Dataset 2."""
    ds1_res = timeseries_engine.run_forecast(
        df=synthetic_stationary_ts,
        time_column="recorded_at",
        target_column="sensor_reading",
        model_type="naive",
        horizon=5,
        dataset_id="ds_1"
    )

    ds2_res = timeseries_engine.run_forecast(
        df=synthetic_trending_seasonal_ts,
        time_column="timestamp_col",
        target_column="metric_value",
        model_type="arima",
        horizon=7,
        dataset_id="ds_2"
    )

    assert ds1_res["time_column"] == "recorded_at"
    assert ds1_res["target_column"] == "sensor_reading"
    assert ds1_res["horizon"] == 5

    assert ds2_res["time_column"] == "timestamp_col"
    assert ds2_res["target_column"] == "metric_value"
    assert ds2_res["horizon"] == 7

    assert ds1_res["provenance_event_id"] != ds2_res["provenance_event_id"]


def test_api_timeseries_endpoints_and_provenance(synthetic_trending_seasonal_ts):
    """Verifies all FastAPI /api/timeseries/* endpoints and provenance audit logging."""
    from backend.core.dataset_registry import DatasetEntry
    client = TestClient(app)

    ds_id = "test_ts_ds_api"
    entry = DatasetEntry(
        dataset_id=ds_id,
        name="test_ts.csv",
        file_path="test_ts.csv",
        format="csv",
        row_count=len(synthetic_trending_seasonal_ts),
        column_count=len(synthetic_trending_seasonal_ts.columns),
        knowledge_model={"columns": list(synthetic_trending_seasonal_ts.columns)},
        loaded_at="2026-10-02T12:00:00",
        file_hash="hash123",
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = synthetic_trending_seasonal_ts.copy(deep=True)
    dataset_registry.set_active(ds_id)

    # 1. /validate
    val_resp = client.post("/api/timeseries/validate", json={
        "dataset_id": ds_id,
        "time_column": "timestamp_col",
        "target_column": "metric_value"
    })
    assert val_resp.status_code == 200
    assert val_resp.json()["status"] == "VALIDATED"

    # 2. /stationarity
    stat_resp = client.post("/api/timeseries/stationarity", json={
        "dataset_id": ds_id,
        "time_column": "timestamp_col",
        "target_column": "metric_value"
    })
    assert stat_resp.status_code == 200
    assert stat_resp.json()["status"] == "SUCCEEDED"

    # 3. /acf-pacf
    acf_resp = client.post("/api/timeseries/acf-pacf", json={
        "dataset_id": ds_id,
        "time_column": "timestamp_col",
        "target_column": "metric_value",
        "max_lags": 10
    })
    assert acf_resp.status_code == 200
    assert acf_resp.json()["status"] == "SUCCEEDED"

    # 4. /decompose
    dec_resp = client.post("/api/timeseries/decompose", json={
        "dataset_id": ds_id,
        "time_column": "timestamp_col",
        "target_column": "metric_value",
        "period": 7,
        "model": "additive"
    })
    assert dec_resp.status_code == 200
    assert dec_resp.json()["status"] == "SUCCEEDED"

    # 5. /forecast
    fc_resp = client.post("/api/timeseries/forecast", json={
        "dataset_id": ds_id,
        "time_column": "timestamp_col",
        "target_column": "metric_value",
        "model_type": "arima",
        "horizon": 7,
        "parameters": {"p": 1, "d": 1, "q": 1}
    })
    assert fc_resp.status_code == 200
    fc_data = fc_resp.json()
    assert fc_data["status"] == "SUCCEEDED"
    assert fc_data["provenance_event_id"] is not None

    # Verify provenance store
    events = provenance_store.get_dataset_lineage(ds_id)
    ts_events = [e for e in events if e.operation == "TIME_SERIES_FORECAST"]
    assert len(ts_events) >= 1
    assert ts_events[0].parameters["model"] == "arima"
    assert ts_events[0].parameters["horizon"] == 7


def test_json_serialization_safety(synthetic_trending_seasonal_ts):
    """Verifies that all time series analytical payloads are 100% JSON-compliant without NaNs/Infs."""
    df = synthetic_trending_seasonal_ts
    diag = timeseries_engine.run_stationarity_tests(df, "timestamp_col", "metric_value")
    acf_res = timeseries_engine.run_acf_pacf(df, "timestamp_col", "metric_value")
    fc_res = timeseries_engine.run_forecast(df, "timestamp_col", "metric_value", model_type="arima", horizon=7)

    # Exclude binary base64 plot payloads from substring checks
    diag_data = {k: v for k, v in diag.items() if k != "plot_base64"}
    acf_data = {k: v for k, v in acf_res.items() if k != "plot_base64"}
    fc_data = {k: v for k, v in fc_res.items() if k != "plot_base64"}

    json_str1 = json.dumps(diag_data)
    json_str2 = json.dumps(acf_data)
    json_str3 = json.dumps(fc_data)

    assert "NaN" not in json_str1
    assert "Infinity" not in json_str1
    assert "NaN" not in json_str2
    assert "NaN" not in json_str3
