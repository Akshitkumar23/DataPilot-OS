"""
DataPilot OS — FW-03 Automated Tabular Feature Engineering Test Suite.
Verifies leakage-safe temporal feature engineering across:
1. create_lag (single-series, grouped, boundary NaNs, k >= 1 enforcement)
2. create_rolling (trailing only, mean/std/min/max/sum, target leakage guard)
3. create_difference (first and higher order, grouped differencing)
4. create_calendar_features (datetime components, cardinality checks)
5. Phase 4 TEMPORAL_LEAKAGE_GUARD in ml_engine.train_model
6. Real Datasets:
   - Parkinson's longitudinal dataset (42 subject entities, motor_UPDRS)
   - Bike Sharing hourly dataset (24-step lag & rolling)
   - AI4I Predictive Maintenance
   - CO2 time series
"""
import pytest
import numpy as np
import pandas as pd

from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily
from backend.ds_engine.typed_operators import FeatureTransformationOperator
from backend.dataset_engine.ml_engine import ml_engine
from backend.api.operations import GovernedOperationRequest, map_operation_to_tool_call


@pytest.fixture
def op():
    return FeatureTransformationOperator()


# ── 1. CREATE_LAG TESTS ───────────────────────────────────────────────────────

def test_create_lag_single_series(op):
    df = pd.DataFrame({
        "time": [1, 2, 3, 4, 5],
        "val": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={
            "method": "create_lag",
            "column": "val",
            "time_column": "time",
            "lag_order": 1
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    assert "val__lag_1" in out.columns
    # Check shift values and boundary NaN
    assert pd.isna(out["val__lag_1"].iloc[0])
    assert out["val__lag_1"].iloc[1] == 10.0
    assert out["val__lag_1"].iloc[4] == 40.0
    assert res.data["nan_count_introduced"] == 1


def test_create_lag_grouped_entity_boundary_isolation(op):
    # Customer A and Customer B: customer A's lag must NEVER contaminate B's lag!
    df = pd.DataFrame({
        "entity": ["A", "A", "A", "B", "B", "B"],
        "time": [1, 2, 3, 1, 2, 3],
        "sales": [100.0, 110.0, 120.0, 500.0, 550.0, 600.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["sales"],
        arguments={
            "method": "create_lag",
            "column": "sales",
            "time_column": "time",
            "group_column": "entity",
            "lag_order": 1
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    col = "sales__lag_1"
    # Row 0 (A, t=1): NaN
    assert pd.isna(out[col].iloc[0])
    # Row 1 (A, t=2): 100
    assert out[col].iloc[1] == 100.0
    # Row 2 (A, t=3): 110
    assert out[col].iloc[2] == 110.0
    # Row 3 (B, t=1): NaN (Crucial: Must NOT be 120.0 from Customer A!)
    assert pd.isna(out[col].iloc[3])
    # Row 4 (B, t=2): 500
    assert out[col].iloc[4] == 500.0
    assert res.data["nan_count_introduced"] == 2
    assert res.data["affected_entities"] == 2


def test_create_lag_k_greater_than_one(op):
    df = pd.DataFrame({
        "time": [1, 2, 3, 4],
        "val": [10.0, 20.0, 30.0, 40.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={
            "method": "create_lag",
            "column": "val",
            "time_column": "time",
            "lag_order": 2
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    assert "val__lag_2" in out.columns
    assert pd.isna(out["val__lag_2"].iloc[0])
    assert pd.isna(out["val__lag_2"].iloc[1])
    assert out["val__lag_2"].iloc[2] == 10.0
    assert out["val__lag_2"].iloc[3] == 20.0


def test_create_lag_guards_and_failures(op):
    df = pd.DataFrame({"time": [1, 2], "val": [10, 20]})

    # Missing time_column -> TEMPORAL_ORDER_REQUIRED
    intent_no_time = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={"method": "create_lag", "column": "val", "lag_order": 1}
    )
    res_no_time = op.execute(df, intent_no_time)
    assert res_no_time.status == "FAILED"
    assert res_no_time.error_code == "TEMPORAL_ORDER_REQUIRED"

    # Missing lag_order -> CLARIFICATION_REQUIRED (No default k=1 guessing)
    intent_no_lag = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={"method": "create_lag", "column": "val", "time_column": "time"}
    )
    res_no_lag = op.execute(df, intent_no_lag)
    assert res_no_lag.status == "FAILED"
    assert res_no_lag.error_code == "CLARIFICATION_REQUIRED"

    # Invalid lag_order (k < 1) -> INVALID_LAG_ORDER
    intent_zero_lag = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={"method": "create_lag", "column": "val", "time_column": "time", "lag_order": 0}
    )
    res_zero = op.execute(df, intent_zero_lag)
    assert res_zero.status == "FAILED"
    assert res_zero.error_code == "INVALID_LAG_ORDER"


# ── 2. CREATE_ROLLING TESTS ──────────────────────────────────────────────────

def test_create_rolling_mean_trailing(op):
    df = pd.DataFrame({
        "time": [1, 2, 3, 4],
        "metric": [10.0, 20.0, 30.0, 40.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["metric"],
        arguments={
            "method": "create_rolling",
            "column": "metric",
            "time_column": "time",
            "window_size": 2,
            "aggregation": "mean"
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    col = "metric__roll_mean_2"
    assert col in out.columns
    assert pd.isna(out[col].iloc[0])
    assert out[col].iloc[1] == 15.0  # mean(10, 20)
    assert out[col].iloc[2] == 25.0  # mean(20, 30)
    assert out[col].iloc[3] == 35.0  # mean(30, 40)


def test_create_rolling_all_aggregations(op):
    df = pd.DataFrame({
        "time": [1, 2, 3],
        "metric": [10.0, 20.0, 30.0]
    })
    for agg, expected_last in [
        ("mean", 25.0),
        ("min", 20.0),
        ("max", 30.0),
        ("sum", 50.0),
        ("std", np.std([20.0, 30.0], ddof=1))
    ]:
        intent = AnalysisIntentV2(
            task_family=TaskFamily.FEATURE_ENGINEERING,
            subjects=["metric"],
            arguments={
                "method": "create_rolling",
                "column": "metric",
                "time_column": "time",
                "window_size": 2,
                "aggregation": agg
            }
        )
        res = op.execute(df, intent)
        assert res.status == "SUCCEEDED"
        out = res.dataframe
        assert np.isclose(out[f"metric__roll_{agg}_2"].iloc[-1], expected_last)


def test_create_rolling_rejects_centered_window(op):
    df = pd.DataFrame({"time": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={
            "method": "create_rolling",
            "column": "val",
            "time_column": "time",
            "window_size": 2,
            "aggregation": "mean",
            "center": True
        }
    )
    res = op.execute(df, intent)
    assert res.status == "FAILED"
    assert res.error_code == "CENTERED_ROLLING_PROHIBITED"


def test_create_rolling_target_leakage_protection(op):
    # When rolling is applied to the target column, shift(1) is mandatory
    df = pd.DataFrame({"time": [1, 2, 3, 4], "y": [10.0, 20.0, 30.0, 40.0]})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["y"],
        target_column="y",
        arguments={
            "method": "create_rolling",
            "column": "y",
            "time_column": "time",
            "window_size": 2,
            "aggregation": "mean",
            "is_target": True
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    col = "y__roll_mean_2"
    # Target shift(1) means row 1 has only 1 observation (from row 0: 10), row 2 has mean(10, 20)=15.0
    assert pd.isna(out[col].iloc[0])
    assert pd.isna(out[col].iloc[1])
    assert out[col].iloc[2] == 15.0
    assert out[col].iloc[3] == 25.0


# ── 3. CREATE_DIFFERENCE TESTS ───────────────────────────────────────────────

def test_create_difference_first_order(op):
    df = pd.DataFrame({
        "time": [1, 2, 3, 4],
        "val": [100.0, 105.0, 102.0, 110.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={
            "method": "create_difference",
            "column": "val",
            "time_column": "time",
            "lag_order": 1
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    col = "val__diff_1"
    assert col in out.columns
    assert pd.isna(out[col].iloc[0])
    assert out[col].iloc[1] == 5.0
    assert out[col].iloc[2] == -3.0
    assert out[col].iloc[3] == 8.0


def test_create_difference_grouped(op):
    df = pd.DataFrame({
        "entity": ["A", "A", "B", "B"],
        "time": [1, 2, 1, 2],
        "val": [10.0, 15.0, 100.0, 120.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={
            "method": "create_difference",
            "column": "val",
            "time_column": "time",
            "group_column": "entity",
            "lag_order": 1
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    col = "val__diff_1"
    assert pd.isna(out[col].iloc[0])
    assert out[col].iloc[1] == 5.0
    assert pd.isna(out[col].iloc[2])  # Entity B initial NaN
    assert out[col].iloc[3] == 20.0


# ── 4. CREATE_CALENDAR_FEATURES TESTS ────────────────────────────────────────

def test_create_calendar_features(op):
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"timestamp": dates, "val": range(10)})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["timestamp"],
        arguments={
            "method": "create_calendar_features",
            "time_column": "timestamp"
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    assert "timestamp__day" in out.columns
    assert "timestamp__dayofweek" in out.columns
    assert "timestamp__is_weekend" in out.columns


# ── 5. GOVERNED OPERATION API ROUTER MAPPING TESTS ───────────────────────────

def test_map_operation_to_tool_call_fw03():
    req_lag = GovernedOperationRequest(
        operation="create_lag",
        column="speed",
        time_column="epoch",
        group_column="device_id",
        lag_order=3,
        new_name="custom_lag_feature"
    )
    mapped_lag = map_operation_to_tool_call(req_lag)
    assert mapped_lag["tool"] == "transform_data"
    assert mapped_lag["arguments"]["method"] == "lag"
    assert mapped_lag["arguments"]["time_column"] == "epoch"
    assert mapped_lag["arguments"]["group_column"] == "device_id"
    assert mapped_lag["arguments"]["lag_order"] == 3
    assert mapped_lag["arguments"]["new_name"] == "custom_lag_feature"

    req_roll = GovernedOperationRequest(
        operation="create_rolling",
        column="temp",
        time_column="epoch",
        window_size=5,
        aggregation="std"
    )
    mapped_roll = map_operation_to_tool_call(req_roll)
    assert mapped_roll["arguments"]["method"] == "rolling"
    assert mapped_roll["arguments"]["window_size"] == 5
    assert mapped_roll["arguments"]["aggregation"] == "std"


# ── 6. PHASE 4 TEMPORAL LEAKAGE GUARD TESTS ──────────────────────────────────

def test_ml_engine_temporal_leakage_guard():
    # Build dataset with a lag feature
    df = pd.DataFrame({
        "time": range(20),
        "x": np.random.randn(20),
        "x__lag_1": np.random.randn(20),
        "target": np.random.randn(20)
    })

    # Case A: Random split should be blocked with TEMPORAL_LEAKAGE_GUARD
    with pytest.raises(ValueError) as excinfo:
        ml_engine.train_model(
            df=df,
            target="target",
            features=["x", "x__lag_1"],
            task="regression",
            validation_strategy="split"
        )
    assert "TEMPORAL_LEAKAGE_GUARD" in str(excinfo.value)

    # Case B: Random CV should also be blocked
    with pytest.raises(ValueError) as excinfo_cv:
        ml_engine.train_model(
            df=df,
            target="target",
            features=["x", "x__lag_1"],
            task="regression",
            validation_strategy="cv"
        )
    assert "TEMPORAL_LEAKAGE_GUARD" in str(excinfo_cv.value)

    # Case C: Explicit TimeSeriesSplit with time_column succeeds
    res = ml_engine.train_model(
        df=df,
        target="target",
        features=["x", "x__lag_1"],
        task="regression",
        validation_strategy="time_series_split",
        time_column="time"
    )
    assert res["model_id"] is not None
    assert res["time_column"] == "time"
    assert "metrics" in res


# ── 7. REAL REPOSITORY DATASETS BENCHMARK TESTS ──────────────────────────────

def test_real_dataset_parkinsons_longitudinal(op):
    path = "data/real_world_evaluation/dataset_i_multi_entity_longitudinal.csv"
    df = pd.read_csv(path)
    # Parkinson's: 42 subjects, longitudinal time series per subject
    assert "subject#" in df.columns
    assert "test_time" in df.columns
    assert "motor_UPDRS" in df.columns
    n_subjects = df["subject#"].nunique()
    assert n_subjects == 42

    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["motor_UPDRS"],
        arguments={
            "method": "create_lag",
            "column": "motor_UPDRS",
            "time_column": "test_time",
            "group_column": "subject#",
            "lag_order": 1
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    col = "motor_UPDRS__lag_1"
    assert col in out.columns
    # Exactly 42 NaNs introduced — 1 per subject boundary!
    assert out[col].isna().sum() == 42
    assert res.data["nan_count_introduced"] == 42
    assert res.data["affected_entities"] == 42


def test_real_dataset_bike_sharing_hourly(op):
    path = "data/real_benchmarks/bike_sharing_hourly.csv"
    df = pd.read_csv(path)
    assert "instant" in df.columns
    assert "cnt" in df.columns

    # 24-hour lag (daily seasonality)
    intent_lag = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["cnt"],
        arguments={
            "method": "create_lag",
            "column": "cnt",
            "time_column": "instant",
            "lag_order": 24
        }
    )
    res_lag = op.execute(df, intent_lag)
    assert res_lag.status == "SUCCEEDED"
    out_lag = res_lag.dataframe
    assert "cnt__lag_24" in out_lag.columns
    assert out_lag["cnt__lag_24"].isna().sum() == 24

    # 24-hour trailing rolling mean
    intent_roll = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["cnt"],
        arguments={
            "method": "create_rolling",
            "column": "cnt",
            "time_column": "instant",
            "window_size": 24,
            "aggregation": "mean"
        }
    )
    res_roll = op.execute(df, intent_roll)
    assert res_roll.status == "SUCCEEDED"
    out_roll = res_roll.dataframe
    assert "cnt__roll_mean_24" in out_roll.columns
    assert out_roll["cnt__roll_mean_24"].isna().sum() == 23


def test_real_dataset_co2_differencing(op):
    path = "data/real_world_evaluation/dataset_g_co2_timeseries.csv"
    df = pd.read_csv(path)
    assert "date" in df.columns
    assert "co2_ppm" in df.columns

    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["co2_ppm"],
        arguments={
            "method": "create_difference",
            "column": "co2_ppm",
            "time_column": "date",
            "lag_order": 1
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    assert "co2_ppm__diff_1" in out.columns
    assert out["co2_ppm__diff_1"].isna().sum() == 1
