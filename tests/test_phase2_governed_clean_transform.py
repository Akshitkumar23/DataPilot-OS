"""
Phase 2 Test Suite: Governed Data Cleaning & Transformation.
Tests A through I covering:
- Governed clean operations (drop nulls, mean/median/mode/constant imputation, duplicate removal, IQR outlier capping).
- Governed transform operations (minmax scale, standardize/z-score, one-hot encode, yeo-johnson, log transform).
- Strict schema grounding: non-existent columns halt with NEEDS_CLARIFICATION and candidate columns; zero silent guessing.
- Type safety: non-numeric columns passed to numeric operations fail with NUMERIC_COLUMNS_REQUIRED.
- Preview vs Apply lifecycle: preview does not mutate dataset; apply updates dataset_registry and re-profiles.
- Profile consistency: Phase 1 real profile recalculates row count, missingness, duplicates, and columns from updated DataFrame.
"""
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.intent_schema import TaskFamily, AnalysisIntentV2
from backend.ds_engine.typed_operators import (
    TypedOperatorRegistry,
    DataCleaningOperator,
    FeatureTransformationOperator,
)
from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import LifecycleState


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean dataset registry before and after every test."""
    dataset_registry.clear()
    yield
    dataset_registry.clear()


@pytest.fixture
def sample_cleaning_df():
    """Deterministic DataFrame with known missing cells, duplicates, and outliers."""
    return pd.DataFrame({
        "age": [20.0, np.nan, 20.0, 40.0, 1000.0],  # 1 null, 1 duplicate pair (20.0), 1 outlier (1000.0)
        "score": [80.0, 90.0, 80.0, 70.0, 60.0],
        "category": ["A", "B", "A", "C", "A"],
        "city": ["New York", "London", "New York", "Tokyo", "Paris"]
    })


@pytest.fixture
def registered_dataset(sample_cleaning_df):
    """Register sample_cleaning_df in dataset_registry as active dataset."""
    ds_id = "test_ds_phase2"
    prof = DatasetProfiler.profile(sample_cleaning_df, dataset_name="sample_test.csv")
    entry = DatasetEntry(
        dataset_id=ds_id,
        name="sample_test.csv",
        file_path="/tmp/sample_test.csv",
        format="csv",
        row_count=len(sample_cleaning_df),
        column_count=len(sample_cleaning_df.columns),
        knowledge_model=prof.model_dump(),
        loaded_at="2026-09-28T00:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = sample_cleaning_df.copy(deep=True)
    dataset_registry.set_active(ds_id)
    return ds_id


client = TestClient(app)


# ── TEST A: Governed Clean — Drop Nulls ───────────────────────────────────────
def test_governed_clean_drop_nulls(sample_cleaning_df):
    """Test A: drop_nulls removes rows containing nulls in the declared column."""
    intent = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "drop_nulls", "columns": ["age"]},
    )
    op = DataCleaningOperator()
    out = op.execute(sample_cleaning_df, intent)
    assert out.status == "SUCCEEDED"
    assert out.dataframe is not None
    assert len(out.dataframe) == 4
    assert out.dataframe["age"].isna().sum() == 0


# ── TEST B: Governed Clean — Imputation ───────────────────────────────────────
def test_governed_clean_imputation(sample_cleaning_df):
    """Test B: mean, median, mode, and constant imputation."""
    op = DataCleaningOperator()

    # Mean imputation
    intent_mean = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "impute_mean", "columns": ["age"]},
    )
    out_mean = op.execute(sample_cleaning_df, intent_mean)
    assert out_mean.status == "SUCCEEDED"
    expected_mean = sample_cleaning_df["age"].mean()
    assert out_mean.dataframe["age"].isna().sum() == 0
    assert out_mean.dataframe.loc[1, "age"] == pytest.approx(expected_mean)

    # Median imputation
    intent_median = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "impute_median", "columns": ["age"]},
    )
    out_median = op.execute(sample_cleaning_df, intent_median)
    assert out_median.status == "SUCCEEDED"
    expected_median = sample_cleaning_df["age"].median()
    assert out_median.dataframe.loc[1, "age"] == pytest.approx(expected_median)

    # Constant imputation
    intent_const = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "impute_constant", "columns": ["age"], "fill_value": 99.0},
    )
    out_const = op.execute(sample_cleaning_df, intent_const)
    assert out_const.status == "SUCCEEDED"
    assert out_const.dataframe.loc[1, "age"] == 99.0

    # Mode imputation on categorical
    df_cat = pd.DataFrame({"cat": ["A", "B", np.nan, "A", "C"]})
    intent_mode = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "impute_mode", "columns": ["cat"]},
    )
    out_mode = op.execute(df_cat, intent_mode)
    assert out_mode.status == "SUCCEEDED"
    assert out_mode.dataframe.loc[2, "cat"] == "A"


# ── TEST C: Governed Clean — Duplicate Removal ────────────────────────────────
def test_governed_clean_duplicate_removal():
    """Test C: duplicate removal drops identical rows."""
    df_dup = pd.DataFrame({
        "a": [1, 2, 1, 3],
        "b": ["x", "y", "x", "z"]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "drop_duplicates"},
    )
    op = DataCleaningOperator()
    out = op.execute(df_dup, intent)
    assert out.status == "SUCCEEDED"
    assert len(out.dataframe) == 3


# ── TEST D: Governed Clean — IQR Outlier Capping ──────────────────────────────
def test_governed_clean_iqr_clipping():
    """Test D: IQR outlier capping caps values at Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    df_outliers = pd.DataFrame({
        "val": [10.0, 11.0, 12.0, 13.0, 14.0, 100.0]  # 100.0 is an extreme outlier
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "clip_iqr", "columns": ["val"]},
    )
    op = DataCleaningOperator()
    out = op.execute(df_outliers, intent)
    assert out.status == "SUCCEEDED"
    assert out.dataframe["val"].max() < 100.0

    # Non-numeric column should fail with NUMERIC_COLUMNS_REQUIRED
    df_str = pd.DataFrame({"text": ["a", "b", "c"]})
    intent_str = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        arguments={"method": "clip_iqr", "columns": ["text"]},
    )
    out_str = op.execute(df_str, intent_str)
    assert out_str.status == "FAILED"
    assert out_str.error_code == "NUMERIC_COLUMNS_REQUIRED"


# ── TEST E: Governed Transform — Min-Max Scaling & Standardization ────────────
def test_governed_transform_scaling_and_standardization():
    """Test E: Min-Max scaling to [0, 1] and Z-score standardization to μ≈0, σ≈1."""
    df_num = pd.DataFrame({
        "feature": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    op = FeatureTransformationOperator()

    # Min-Max
    intent_minmax = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        arguments={"method": "minmax_scale", "columns": ["feature"]},
    )
    out_minmax = op.execute(df_num, intent_minmax)
    assert out_minmax.status == "SUCCEEDED"
    assert out_minmax.dataframe["feature"].min() == pytest.approx(0.0)
    assert out_minmax.dataframe["feature"].max() == pytest.approx(1.0)

    # Standardization
    intent_std = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        arguments={"method": "standardize", "columns": ["feature"]},
    )
    out_std = op.execute(df_num, intent_std)
    assert out_std.status == "SUCCEEDED"
    assert out_std.dataframe["feature"].mean() == pytest.approx(0.0, abs=1e-6)
    assert out_std.dataframe["feature"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    # Non-numeric validation
    df_cat = pd.DataFrame({"dept": ["Sales", "HR", "Engineering"]})
    intent_invalid = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        arguments={"method": "minmax_scale", "columns": ["dept"]},
    )
    out_invalid = op.execute(df_cat, intent_invalid)
    assert out_invalid.status == "FAILED"
    assert out_invalid.error_code == "NUMERIC_COLUMNS_REQUIRED"


# ── TEST F: Governed Transform — One-Hot Encoding ─────────────────────────────
def test_governed_transform_one_hot_encode():
    """Test F: One-Hot Encoding expands categorical column into dummy columns."""
    df_cat = pd.DataFrame({
        "color": ["red", "blue", "green"],
        "price": [10, 20, 30]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        arguments={"method": "one_hot_encode", "columns": ["color"]},
    )
    op = FeatureTransformationOperator()
    out = op.execute(df_cat, intent)
    assert out.status == "SUCCEEDED"
    assert "color" not in out.dataframe.columns
    assert "color_red" in out.dataframe.columns
    assert "color_blue" in out.dataframe.columns
    assert "color_green" in out.dataframe.columns
    assert out.dataframe["price"].tolist() == [10, 20, 30]


# ── TEST G: Governed Transform — Yeo-Johnson & Log Transform ──────────────────
def test_governed_transform_power_and_log():
    """Test G: Yeo-Johnson power transform and log transform."""
    df_skew = pd.DataFrame({
        "income": [10.0, 100.0, 1000.0, 10000.0, 100000.0]
    })
    op = FeatureTransformationOperator()

    # Log transform
    intent_log = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        arguments={"method": "log_transform", "columns": ["income"]},
    )
    out_log = op.execute(df_skew, intent_log)
    assert out_log.status == "SUCCEEDED"
    assert out_log.dataframe["income"].iloc[0] == pytest.approx(np.log1p(10.0))

    # Yeo-Johnson
    intent_yj = AnalysisIntentV2(
        task_family=TaskFamily.FEATURE_ENGINEERING,
        arguments={"method": "yeo_johnson", "columns": ["income"]},
    )
    out_yj = op.execute(df_skew, intent_yj)
    assert out_yj.status == "SUCCEEDED"
    assert not out_yj.dataframe["income"].isna().any()


# ── TEST H: Strict Schema Grounding & Clarification ───────────────────────────
def test_strict_schema_grounding_nonexistent_column(sample_cleaning_df):
    """Test H: non-existent columns halt with NEEDS_CLARIFICATION and candidate columns; zero silent guessing."""
    tool_call = {
        "tool": "clean_data",
        "arguments": {
            "method": "drop_nulls",
            "columns": ["ThisColumnDefinitelyDoesNotExist"]
        }
    }
    prof = DatasetProfiler.profile(sample_cleaning_df)
    res = QueryOrchestrator._process_intent(
        query_raw="drop nulls in ThisColumnDefinitelyDoesNotExist",
        df=sample_cleaning_df,
        intent_dict=tool_call,
        profile=prof,
        conversation_history=None,
        context={},
        request_id="test_grounding",
        t_start=0.0
    )
    assert res.status == LifecycleState.NEEDS_CLARIFICATION.value
    assert "ThisColumnDefinitelyDoesNotExist" in res.clarification
    assert len(res.candidate_columns) > 0


# ── TEST I: Preview vs Apply Lifecycle & Profile Consistency ──────────────────
def test_preview_vs_apply_lifecycle(registered_dataset):
    """Test I: Preview does NOT mutate dataset state; Apply mutates dataset state and updates profile."""
    # Step 1: Check initial profile
    res_prof_initial = client.get("/api/datasets/profile")
    assert res_prof_initial.status_code == 200
    init_data = res_prof_initial.json()
    assert init_data["row_count"] == 5
    assert init_data["missing_cells"] == 1
    assert init_data["duplicate_rows"] == 1

    # Step 2: POST /api/operations/preview for drop_duplicates
    res_preview = client.post("/api/operations/preview", json={
        "dataset_id": registered_dataset,
        "operation": "drop_duplicates"
    })
    assert res_preview.status_code == 200
    prev_data = res_preview.json()
    assert prev_data["status"] == "SUCCEEDED"
    assert prev_data["row_count_before"] == 5
    assert prev_data["row_count_after"] == 4

    # CRITICAL: Verify that preview did NOT mutate dataset state in registry!
    res_prof_after_prev = client.get("/api/datasets/profile")
    assert res_prof_after_prev.status_code == 200
    after_prev_data = res_prof_after_prev.json()
    assert after_prev_data["row_count"] == 5  # STILL 5!

    # Step 3: POST /api/operations/apply for drop_duplicates
    res_apply = client.post("/api/operations/apply", json={
        "dataset_id": registered_dataset,
        "operation": "drop_duplicates"
    })
    assert res_apply.status_code == 200
    apply_data = res_apply.json()
    assert apply_data["status"] == "SUCCEEDED"
    assert apply_data["row_count"] == 4

    # CRITICAL: Verify that apply DID mutate dataset state and re-profiled!
    res_prof_after_apply = client.get("/api/datasets/profile")
    assert res_prof_after_apply.status_code == 200
    after_apply_data = res_prof_after_apply.json()
    assert after_apply_data["row_count"] == 4  # Now 4!
    assert after_apply_data["duplicate_rows"] == 0  # 0 duplicates remaining!

    # Step 4: Verify operation failure never mutates dataset state
    res_fail = client.post("/api/operations/apply", json={
        "dataset_id": registered_dataset,
        "operation": "minmax_scale",
        "column": "category"  # Non-numeric!
    })
    assert res_fail.status_code == 422

    # Verify state remains intact at 4 rows
    res_prof_after_fail = client.get("/api/datasets/profile")
    assert res_prof_after_fail.status_code == 200
    assert res_prof_after_fail.json()["row_count"] == 4
