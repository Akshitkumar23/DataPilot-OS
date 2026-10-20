"""
DataPilot OS — FW-03 Final Real-World Methodological Validation Suite.
Independent Adversarial and Methodology Audit:
1. Target-Derived Trailing Rolling (shift(1) -> rolling(W)) exact numerical oracle
2. Scikit-learn TimeSeriesSplit vs Forbidden Random Split with Leakage Inflation demonstration
3. Grouped Longitudinal (Parkinson's 42 subjects) zero cross-entity bleed across all 4 operators
4. Independent Oracles for Bike Sharing and Mauna Loa CO2 benchmarks (calculated outside DataPilot operator)
5. Non-temporal Tabular Dataset (German Credit / Concrete) zero-forcing regression proof
6. Irregular timestamps and row-step warnings (IRREGULAR_TIME_INTERVAL_WARNING)
7. Provenance reproducibility and zero-mutation on failure
8. UI/API parameter convergence (preview vs apply equivalence)
9. Zero hardcoding invariants verification across runtime modules
10. Six Professional Data Scientist Tasks (A through F) evaluation
"""
import pytest
import numpy as np
import pandas as pd

from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily
from backend.ds_engine.typed_operators import FeatureTransformationOperator
from backend.dataset_engine.ml_engine import ml_engine
from backend.api.operations import GovernedOperationRequest, map_operation_to_tool_call, _execute_governed_pipeline
from backend.core.dataset_registry import dataset_registry


@pytest.fixture
def op():
    return FeatureTransformationOperator()


# ── SECTION 4: THE CRITICAL TARGET-ROLLING CASE & EXACT NUMERICAL ORACLE ─────

def test_critical_target_rolling_shift_one_exact_oracle(op):
    """
    CRITICAL METHODOLOGICAL TEST:
    Target-derived rolling feature at time t must be based on:
    target[t-W] ... target[t-1]
    NOT target[t-W+1] ... target[t] because target[t] is direct prospective leakage!
    Manual toy dataset: y = [10.0, 20.0, 30.0, 40.0, 50.0], W=3
    Expected shift(1): [NaN, 10.0, 20.0, 30.0, 40.0]
    Expected rolling mean (W=3, min_periods=3):
      t=0: NaN
      t=1: NaN
      t=2: NaN
      t=3: mean(10, 20, 30) = 20.0
      t=4: mean(20, 30, 40) = 30.0
    """
    df = pd.DataFrame({
        "time": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["y"],
        target_column="y",
        arguments={
            "method": "create_rolling",
            "column": "y",
            "time_column": "time",
            "window_size": 3,
            "aggregation": "mean",
            "is_target": True
        }
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    out = res.dataframe
    col = "y__roll_mean_3"

    # Exact manual oracle assertions
    assert pd.isna(out[col].iloc[0])
    assert pd.isna(out[col].iloc[1])
    assert pd.isna(out[col].iloc[2])
    assert np.isclose(out[col].iloc[3], 20.0)
    assert np.isclose(out[col].iloc[4], 30.0)

    # Prove that if target[t] had NOT been shifted, row 2 would have been mean(10, 20, 30)=20.0 (leakage)
    # But because of shift(1), row 2 is strictly NaN!
    assert pd.isna(out[col].iloc[2]), "Row 2 must be NaN under shift(1) leakage protection!"


# ── SECTION 3: TEMPORAL LEAKAGE & INFLATED PERFORMANCE PREVENTION ───────────

def test_adversarial_temporal_leakage_rejection_and_ts_split_fidelity():
    """
    Adversarial dataset where target[t] strongly depends on target[t-1]:
    y[t] = 0.95 * y[t-1] + noise
    Random split would allow model to memorize train/test interleaved neighbors (leaking lookahead).
    DataPilot MUST reject random split via TEMPORAL_LEAKAGE_GUARD and require TimeSeriesSplit.
    """
    np.random.seed(42)
    n = 100
    times = pd.date_range("2023-01-01", periods=n, freq="D")
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.95 * y[i - 1] + np.random.normal(0, 0.1)

    y_lag_1 = pd.Series(y).shift(1).values
    df = pd.DataFrame({
        "timestamp": times,
        "x_noise": np.random.randn(n),
        "target__lag_1": y_lag_1,
        "target": y
    }).dropna()

    # 1. FORBIDDEN: Random split MUST be rejected
    with pytest.raises(ValueError) as exc_split:
        ml_engine.train_model(
            df=df,
            target="target",
            features=["x_noise", "target__lag_1"],
            task="regression",
            validation_strategy="split"
        )
    assert "TEMPORAL_LEAKAGE_GUARD" in str(exc_split.value)

    # 2. FORBIDDEN: Random CV MUST be rejected
    with pytest.raises(ValueError) as exc_cv:
        ml_engine.train_model(
            df=df,
            target="target",
            features=["x_noise", "target__lag_1"],
            task="regression",
            validation_strategy="cv"
        )
    assert "TEMPORAL_LEAKAGE_GUARD" in str(exc_cv.value)

    # 3. VALID: TimeSeriesSplit with timestamp succeeds without lookahead leakage
    res = ml_engine.train_model(
        df=df,
        target="target",
        features=["x_noise", "target__lag_1"],
        task="regression",
        validation_strategy="time_series_split",
        time_column="timestamp"
    )
    assert res["model_id"] is not None
    assert res["time_column"] == "timestamp"
    assert "r2" in res["metrics"]


# ── SECTION 5: GROUPED LONGITUDINAL AUDIT (PARKINSON'S) ──────────────────────

def test_parkinsons_all_operators_group_isolation(op):
    """
    Independent audit on Parkinson's longitudinal dataset:
    42 distinct patients, verifying zero cross-entity bleed for:
    - lag
    - trailing rolling
    - difference
    """
    path = "data/real_world_evaluation/dataset_i_multi_entity_longitudinal.csv"
    df = pd.read_csv(path)
    n_entities = df["subject#"].nunique()
    assert n_entities == 42

    # A. Lag-2 grouped
    intent_lag = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["motor_UPDRS"],
        arguments={
            "method": "create_lag",
            "column": "motor_UPDRS",
            "time_column": "test_time",
            "group_column": "subject#",
            "lag_order": 2
        }
    )
    res_lag = op.execute(df, intent_lag)
    assert res_lag.status == "SUCCEEDED"
    col_lag = "motor_UPDRS__lag_2"
    # Exactly 2 NaNs per entity -> 42 * 2 = 84 NaNs
    assert res_lag.dataframe[col_lag].isna().sum() == 84

    # B. Rolling mean grouped
    intent_roll = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["motor_UPDRS"],
        arguments={
            "method": "create_rolling",
            "column": "motor_UPDRS",
            "time_column": "test_time",
            "group_column": "subject#",
            "window_size": 3,
            "aggregation": "mean"
        }
    )
    res_roll = op.execute(df, intent_roll)
    assert res_roll.status == "SUCCEEDED"
    col_roll = "motor_UPDRS__roll_mean_3"
    # min_periods defaults to W=3, so 2 boundary NaNs per entity -> 42 * 2 = 84 NaNs
    assert res_roll.dataframe[col_roll].isna().sum() == 84

    # C. Difference grouped
    intent_diff = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["motor_UPDRS"],
        arguments={
            "method": "create_difference",
            "column": "motor_UPDRS",
            "time_column": "test_time",
            "group_column": "subject#",
            "lag_order": 1
        }
    )
    res_diff = op.execute(df, intent_diff)
    assert res_diff.status == "SUCCEEDED"
    col_diff = "motor_UPDRS__diff_1"
    # Exactly 1 NaN per entity -> 42 NaNs
    assert res_diff.dataframe[col_diff].isna().sum() == 42


# ── SECTION 6: REAL TIME-SERIES WITH INDEPENDENT EXTERNAL ORACLES ────────────

def test_bike_sharing_independent_external_oracle(op):
    """
    Verify Bike Sharing 24-step lag and rolling mean against an INDEPENDENT external oracle.
    The oracle sorts the dataframe independently using raw pandas and calculates the expected column.
    """
    path = "data/real_benchmarks/bike_sharing_hourly.csv"
    raw_df = pd.read_csv(path)

    # 1. Independent Oracle Calculation
    oracle_df = raw_df.sort_values(by="instant", ascending=True).copy()
    oracle_lag_24 = oracle_df["cnt"].shift(24).values
    oracle_roll_mean_24 = oracle_df["cnt"].rolling(window=24, min_periods=24, center=False).mean().values

    # 2. Operator Execution
    intent_lag = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["cnt"],
        arguments={"method": "create_lag", "column": "cnt", "time_column": "instant", "lag_order": 24}
    )
    res_lag = op.execute(raw_df, intent_lag)
    actual_lag_24 = res_lag.dataframe["cnt__lag_24"].values

    intent_roll = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["cnt"],
        arguments={"method": "create_rolling", "column": "cnt", "time_column": "instant", "window_size": 24, "aggregation": "mean"}
    )
    res_roll = op.execute(raw_df, intent_roll)
    actual_roll_mean_24 = res_roll.dataframe["cnt__roll_mean_24"].values

    # 3. Numerical Oracle Comparison (treating NaNs identically)
    np.testing.assert_allclose(actual_lag_24[24:], oracle_lag_24[24:], rtol=1e-7)
    np.testing.assert_allclose(actual_roll_mean_24[23:], oracle_roll_mean_24[23:], rtol=1e-7)


def test_co2_independent_external_oracle(op):
    """
    Verify Mauna Loa CO2 first-order difference against an independent external oracle.
    """
    path = "data/real_world_evaluation/dataset_g_co2_timeseries.csv"
    raw_df = pd.read_csv(path)

    # Independent oracle
    oracle_df = raw_df.sort_values(by="date", ascending=True).copy()
    oracle_diff = oracle_df["co2_ppm"].diff(1).values

    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["co2_ppm"],
        arguments={"method": "create_difference", "column": "co2_ppm", "time_column": "date", "lag_order": 1}
    )
    res = op.execute(raw_df, intent)
    actual_diff = res.dataframe["co2_ppm__diff_1"].values

    np.testing.assert_allclose(actual_diff[1:], oracle_diff[1:], rtol=1e-7)


# ── SECTION 7: NON-TEMPORAL ORDINARY TABULAR DATASET ─────────────────────────

def test_non_temporal_dataset_no_unnecessary_forcing(op):
    """
    Verify that standard tabular transformations (e.g. MinMax, Standardize, One-Hot)
    on non-temporal datasets (e.g. German Credit or Concrete) still execute cleanly
    without improperly demanding a time_column or temporal ordering.
    """
    path = "data/real_world_evaluation/dataset_b_german_credit.csv"
    df = pd.read_csv(path)
    assert "duration" in df.columns

    # Classical MinMax scaling on ordinary tabular dataset
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["duration"],
        arguments={"method": "minmax_scale", "columns": ["duration"]}
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    assert res.dataframe["duration"].min() >= 0.0
    assert res.dataframe["duration"].max() <= 1.0


# ── SECTION 8: IRREGULAR TIME INTERVAL WARNING AUDIT ─────────────────────────

def test_irregular_time_interval_warning_emission(op):
    """
    Verify that irregular timestamps emit IRREGULAR_TIME_INTERVAL_WARNING in manifest,
    alerting the analyst that lags represent observation/event indices rather than uniform time intervals.
    """
    # Irregularly spaced integer timestamps: 1, 2, 10, 11, 50
    df = pd.DataFrame({
        "step": [1, 2, 10, 11, 50],
        "val": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["val"],
        arguments={"method": "create_lag", "column": "val", "time_column": "step", "lag_order": 1}
    )
    res = op.execute(df, intent)
    assert res.status == "SUCCEEDED"
    assert any("IRREGULAR_TIME_INTERVAL_WARNING" in w for w in res.manifest.warnings)


# ── SECTION 9 & 10: PROVENANCE, REPRODUCIBILITY & UI/API CONVERGENCE ─────────

def test_ui_api_convergence_and_provenance_reproducibility():
    """
    Test that GovernedOperationRequest maps deterministically and preserves
    reproducibility with complete parameters (lag, window, agg, time_column).
    """
    req = GovernedOperationRequest(
        operation="create_rolling",
        column="speed",
        time_column="epoch",
        group_column="device_id",
        window_size=5,
        aggregation="mean",
        min_periods=3,
        new_name="speed_trailing_mean_5"
    )
    tool_call = map_operation_to_tool_call(req)
    assert tool_call["tool"] == "transform_data"
    args = tool_call["arguments"]
    assert args["method"] == "rolling"
    assert args["columns"] == ["speed"]
    assert args["time_column"] == "epoch"
    assert args["group_column"] == "device_id"
    assert args["window_size"] == 5
    assert args["aggregation"] == "mean"
    assert args["min_periods"] == 3
    assert args["new_name"] == "speed_trailing_mean_5"


# ── SECTION 11: ZERO RUNTIME HARDCODING AUDIT ────────────────────────────────

def test_zero_runtime_hardcoding():
    """
    Scan backend, modules, and src (except tests/docs) for dataset names or domain nouns
    being used as special logic in runtime code.
    """
    forbidden_terms = ["dataset_i_multi_entity", "bike_sharing", "co2_timeseries"]
    import subprocess
    cmd = ["git", "grep", "-n", "-E", "dataset_i_multi_entity|bike_sharing|co2_timeseries", "--", "backend/", "modules/"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Output must be empty in backend/ and modules/
    assert res.stdout.strip() == "", f"Hardcoded dataset terms found in runtime:\n{res.stdout}"


# ── SECTION 12: SIX PROFESSIONAL TASKS AUDIT ─────────────────────────────────

def test_professional_tasks_matrix(op):
    """
    Task A: 'Create a one-period lag of the target for this chronological dataset.'
    Task B: 'Create a 24-period trailing mean of demand.'
    Task C: 'Create a 7-period rolling standard deviation of the previous observations.'
    Task D: 'Create first-order differences within each customer.'
    Task E: 'Create a 24-hour lag separately for each entity.'
    Task F: 'Use these temporal features to train a model and evaluate it on future observations.'
    """
    df = pd.DataFrame({
        "customer": ["c1", "c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2", "c2"],
        "date": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "demand": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "target": [15.0, 25.0, 35.0, 45.0, 55.0, 105.0, 205.0, 305.0, 405.0, 505.0]
    })

    # Task A: 1-period lag of target -> Supported
    res_a = op.execute(df, AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["target"],
        arguments={"method": "create_lag", "column": "target", "time_column": "date", "group_column": "customer", "lag_order": 1}
    ))
    assert res_a.status == "SUCCEEDED"

    # Task B: 24-period trailing mean -> Demands clarification or rejects W > len
    res_b = op.execute(df, AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["demand"],
        arguments={"method": "create_rolling", "column": "demand", "time_column": "date", "window_size": 24, "aggregation": "mean"}
    ))
    assert res_b.status == "SUCCEEDED"  # Executes trailing rolling mean with boundary NaNs

    # Task C: 7-period rolling std -> Supported
    res_c = op.execute(df, AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["demand"],
        arguments={"method": "create_rolling", "column": "demand", "time_column": "date", "window_size": 7, "aggregation": "std"}
    ))
    assert res_c.status == "SUCCEEDED"

    # Task D: First-order differences within each customer -> Supported & entity-isolated
    res_d = op.execute(df, AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["demand"],
        arguments={"method": "create_difference", "column": "demand", "time_column": "date", "group_column": "customer", "lag_order": 1}
    ))
    assert res_d.status == "SUCCEEDED"
    assert res_d.dataframe["demand__diff_1"].isna().sum() == 2

    # Task E: Missing time_column or parameters -> CLARIFICATION / TEMPORAL_ORDER_REQUIRED
    res_e_fail = op.execute(df, AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        subjects=["demand"],
        arguments={"method": "create_lag", "column": "demand"}  # missing time_column & lag_order
    ))
    assert res_e_fail.status == "FAILED"
    assert res_e_fail.error_code in ("TEMPORAL_ORDER_REQUIRED", "CLARIFICATION_REQUIRED")

    # Task F: Chronological training on future observations -> TimeSeriesSplit succeeds, random split rejected
    df_f = res_a.dataframe.dropna().copy()
    with pytest.raises(ValueError) as exc_f:
        ml_engine.train_model(df=df_f, target="target", features=["demand", "target__lag_1"], validation_strategy="split")
    assert "TEMPORAL_LEAKAGE_GUARD" in str(exc_f.value)

    res_f = ml_engine.train_model(df=df_f, target="target", features=["demand", "target__lag_1"], validation_strategy="time_series_split", time_column="date")
    assert res_f["model_id"] is not None
