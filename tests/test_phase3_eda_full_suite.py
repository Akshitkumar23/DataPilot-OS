"""
DataPilot OS — Phase 3 Real EDA & Statistics Comprehensive Test Suite.
100% Deterministic, Dataset-Agnostic, Zero-Hardcoding Mathematical Verification.

Covers all 10 core verification areas:
1. Descriptive Statistics: exact match with Pandas mathematical ground truth (mean, median, std, min, max, quartiles, IQR, variance).
2. Missingness Analysis: factual column-wise and overall missingness metrics.
3. Distribution Analysis: accurate histogram binning, bounds, counts, and empirical outliers.
4. Correlation Analysis: Pearson & Spearman correlation matrix calculation and ranked association pairs.
5. Skewness & Normality: skewness, kurtosis, classification, and transformation recommendations.
6. Cross-Dataset Isolation: seamless analytical execution across distinct schemas (HR vs. eCommerce).
7. Schema Validation & Error Handling: controlled 422 errors for COLUMN_NOT_FOUND, NON_NUMERIC_COLUMN, INSUFFICIENT_NUMERIC_COLUMNS.
8. Zero-Mutation Invariant: pure read-only execution leaving DataFrame and dataset fingerprint completely unchanged.
9. JSON Safety: robust sanitization of NaN, +Inf, -Inf to None/null.
10. Provenance Integration: audit logging with capability="EDA", non-mutating version tracking, and audit querying.
"""
import uuid
import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.persistence.provenance_store import provenance_store
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.dataset_engine.eda_engine import eda_engine, sanitize_float, sanitize_obj


def register_test_dataset(df: pd.DataFrame, name: str = "test_dataset.csv") -> DatasetEntry:
    """Helper to profile and register a DataFrame into the active dataset registry."""
    ds_id = f"ds_{uuid.uuid4().hex[:8]}"
    prof = DatasetProfiler.profile(df, dataset_name=name)
    entry = DatasetEntry(
        dataset_id=ds_id,
        name=name,
        file_path=f"/tmp/{name}",
        format="csv",
        row_count=len(df),
        column_count=len(df.columns),
        knowledge_model=prof.model_dump(),
        loaded_at="2026-10-01T00:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = df.copy(deep=True)
    dataset_registry.set_active(ds_id)
    return entry


@pytest.fixture(autouse=True)
def clean_test_env():
    """Ensure a clean registry and provenance store for every test."""
    provenance_store.clear()
    dataset_registry._datasets.clear()
    dataset_registry._dataframes_cache.clear()
    yield
    provenance_store.clear()
    dataset_registry._datasets.clear()
    dataset_registry._dataframes_cache.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ── TEST 1: Descriptive Statistics Calculation ──────────────────────────────────
def test_1_descriptive_statistics_calculation(client):
    """
    Verify descriptive statistics against independent Pandas calculations:
    count, mean, median, std, min, max, q25, q50, q75, variance, iqr.
    """
    df = pd.DataFrame({
        "metric_a": [10.0, 20.0, 30.0, 40.0, 50.0],
        "metric_b": [5.0, 15.0, None, 35.0, 45.0],
        "category": ["X", "Y", "Z", "W", "V"]
    })
    entry = register_test_dataset(df, name="metrics.csv")

    resp = client.post("/api/eda/descriptive", json={
        "dataset_id": entry.dataset_id
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "DESCRIPTIVE_STATS"

    stats = {row["column"]: row for row in data["result"]["statistics"]}
    assert "metric_a" in stats
    assert "metric_b" in stats
    assert "category" not in stats  # Non-numeric excluded by default

    # Verify metric_a ground truth
    ma = stats["metric_a"]
    assert ma["count"] == 5
    assert ma["null_count"] == 0
    assert ma["null_percentage"] == 0.0
    assert pytest.approx(ma["mean"], 0.001) == 30.0
    assert pytest.approx(ma["median"], 0.001) == 30.0
    assert pytest.approx(ma["min"], 0.001) == 10.0
    assert pytest.approx(ma["max"], 0.001) == 50.0
    assert pytest.approx(ma["q25"], 0.001) == 20.0
    assert pytest.approx(ma["q50"], 0.001) == 30.0
    assert pytest.approx(ma["q75"], 0.001) == 40.0
    assert pytest.approx(ma["iqr"], 0.001) == 20.0
    assert pytest.approx(ma["std"], 0.001) == float(df["metric_a"].std())
    assert pytest.approx(ma["variance"], 0.001) == float(df["metric_a"].var())

    # Verify metric_b ground truth (with nulls)
    mb = stats["metric_b"]
    assert mb["count"] == 4
    assert mb["null_count"] == 1
    assert mb["null_percentage"] == 20.0
    valid_b = df["metric_b"].dropna()
    assert pytest.approx(mb["mean"], 0.001) == float(valid_b.mean())
    assert pytest.approx(mb["median"], 0.001) == float(valid_b.median())
    assert pytest.approx(mb["min"], 0.001) == float(valid_b.min())
    assert pytest.approx(mb["max"], 0.001) == float(valid_b.max())


# ── TEST 2: Missingness Analysis ───────────────────────────────────────────────
def test_2_missingness_analysis(client):
    """
    Verify exact column-wise missing counts, percentages, non-missing counts,
    and overall dataset missingness metrics.
    """
    df = pd.DataFrame({
        "col1": [1.0, 2.0, None, 4.0, 5.0],      # 1 missing (20%)
        "col2": [None, None, None, None, "ok"],   # 4 missing (80%)
        "col3": [10, 20, 30, 40, 50],             # 0 missing (0%)
    })
    entry = register_test_dataset(df, name="missingness_test.csv")

    resp = client.post("/api/eda/missingness", json={
        "dataset_id": entry.dataset_id
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "MISSINGNESS_ANALYSIS"

    summary = data["result"]["summary"]
    assert summary["total_rows"] == 5
    assert summary["total_columns"] == 3
    assert summary["total_cells"] == 15
    assert summary["total_missing_cells"] == 5
    assert pytest.approx(summary["overall_missing_percentage"], 0.01) == round(5 / 15 * 100, 2)
    assert summary["complete_rows"] == 1  # Only row 4 has all 3 values
    assert summary["rows_with_missing"] == 4

    col_map = {c["column"]: c for c in data["result"]["columns"]}
    assert col_map["col1"]["missing_count"] == 1
    assert col_map["col1"]["non_missing_count"] == 4
    assert col_map["col1"]["missing_percentage"] == 20.0
    assert col_map["col1"]["status"] == "Moderate Missingness"

    assert col_map["col2"]["missing_count"] == 4
    assert col_map["col2"]["non_missing_count"] == 1
    assert col_map["col2"]["missing_percentage"] == 80.0
    assert col_map["col2"]["status"] == "High Missingness"

    assert col_map["col3"]["missing_count"] == 0
    assert col_map["col3"]["non_missing_count"] == 5
    assert col_map["col3"]["missing_percentage"] == 0.0
    assert col_map["col3"]["status"] == "Complete"


# ── TEST 3: Distribution Analysis ──────────────────────────────────────────────
def test_3_distribution_analysis(client):
    """
    Verify histogram binning, empirical ranges, bin percentages, and sample bounds.
    """
    np.random.seed(42)
    vals = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 35.0, 100.0]  # 100 is an outlier
    df = pd.DataFrame({"score": vals})
    entry = register_test_dataset(df, name="dist_test.csv")

    resp = client.post("/api/eda/distribution", json={
        "dataset_id": entry.dataset_id,
        "column": "score",
        "n_bins": 5
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "DISTRIBUTION_ANALYSIS"

    res = data["result"]
    assert res["column"] == "score"
    assert res["sample_size"] == 10
    assert res["n_bins"] == 5
    assert len(res["bins"]) == 5

    # Check that sum of bin counts equals sample size
    total_binned = sum(b["count"] for b in res["bins"])
    assert total_binned == 10

    # Check percentages sum to ~100%
    total_pct = sum(b["percentage"] for b in res["bins"])
    assert pytest.approx(total_pct, 0.5) == 100.0

    # Summary bounds
    assert res["summary"]["min"] == 10.0
    assert res["summary"]["max"] == 100.0
    assert res["summary"]["outlier_count"] >= 1  # 100.0 is > Q3 + 1.5*IQR


# ── TEST 4: Distribution Analysis with Constant Series ─────────────────────────
def test_4_distribution_constant_series(client):
    """
    Verify distribution safely handles a column where all values are identical.
    """
    df = pd.DataFrame({"constant_col": [42.0, 42.0, 42.0, 42.0]})
    entry = register_test_dataset(df, name="constant_test.csv")

    resp = client.post("/api/eda/distribution", json={
        "dataset_id": entry.dataset_id,
        "column": "constant_col",
        "n_bins": 5
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    res = data["result"]
    assert res["sample_size"] == 4
    assert len(res["bins"]) == 1
    assert res["bins"][0]["count"] == 4
    assert res["bins"][0]["bin_start"] == 42.0
    assert res["bins"][0]["bin_end"] == 42.0


# ── TEST 5: Correlation Matrix and Ranked Pairs ─────────────────────────────────
def test_5_correlation_matrix_and_ranked_pairs(client):
    """
    Verify Pearson and Spearman correlation computation, symmetric matrix,
    and ranked pairwise associations.
    """
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfectly correlated with x (r = 1.0)
        "z": [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfectly negatively correlated with x (r = -1.0)
        "w": [1.0, 0.0, 1.0, 0.0, 1.0],  # Weakly correlated
    })
    entry = register_test_dataset(df, name="corr_test.csv")

    # 1. Pearson correlation
    resp = client.post("/api/eda/correlation", json={
        "dataset_id": entry.dataset_id,
        "method": "pearson"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "CORRELATION_ANALYSIS"

    res = data["result"]
    cols = res["columns"]
    assert cols == ["x", "y", "z", "w"]
    matrix = res["matrix"]
    assert len(matrix) == 4
    assert len(matrix[0]) == 4

    # Diagonal is 1.0
    for i in range(4):
        assert pytest.approx(matrix[i][i], 0.001) == 1.0

    # Check x and y correlation
    idx_x = cols.index("x")
    idx_y = cols.index("y")
    idx_z = cols.index("z")
    assert pytest.approx(matrix[idx_x][idx_y], 0.001) == 1.0
    assert pytest.approx(matrix[idx_x][idx_z], 0.001) == -1.0

    # Check ranked pairs
    pairs = res["pairs"]
    assert len(pairs) == 6  # 4 choose 2 = 6
    # Top pair should have abs_correlation == 1.0
    assert pytest.approx(pairs[0]["abs_correlation"], 0.001) == 1.0
    assert pairs[0]["strength"] == "Very Strong"

    # 2. Spearman correlation
    resp_spearman = client.post("/api/eda/correlation", json={
        "dataset_id": entry.dataset_id,
        "method": "spearman"
    })
    assert resp_spearman.status_code == 200
    assert resp_spearman.json()["result"]["method"] == "spearman"


# ── TEST 6: Skewness & Normality Analysis ───────────────────────────────────────
def test_6_skewness_and_normality(client):
    """
    Verify skewness, kurtosis, classification, and transformation recommendations.
    """
    # Create right-skewed dataset
    df = pd.DataFrame({
        "symmetric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "right_skewed": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 100.0],
    })
    entry = register_test_dataset(df, name="skew_test.csv")

    resp = client.post("/api/eda/skewness", json={
        "dataset_id": entry.dataset_id
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "SKEWNESS_ANALYSIS"

    res_map = {r["column"]: r for r in data["result"]["results"]}

    # Symmetric column
    sym = res_map["symmetric"]
    assert pytest.approx(sym["skewness"], 0.01) == 0.0
    assert sym["classification"] == "Symmetric / Normal"
    assert sym["direction"] == "Symmetric"

    # Right-skewed column
    skewed = res_map["right_skewed"]
    assert skewed["skewness"] > 1.0
    assert skewed["classification"] == "Highly Skewed"
    assert skewed["direction"] == "Positive (Right-tailed)"
    assert "Log transform" in skewed["recommendation"]


# ── TEST 7: Cross-Dataset Isolation (Zero Hardcoding) ───────────────────────────
def test_7_cross_dataset_isolation(client):
    """
    Verify that Dataset A and completely different Dataset B execute cleanly
    through the identical unspecialized EDA pipeline without state leakage.
    """
    # Dataset A: HR Domain
    df_a = pd.DataFrame({
        "emp_age": [25.0, 30.0, 35.0, 40.0],
        "emp_salary": [50000.0, 60000.0, 70000.0, 80000.0]
    })
    entry_a = register_test_dataset(df_a, name="dataset_hr.csv")

    # Dataset B: Supply Chain Domain
    df_b = pd.DataFrame({
        "shipment_weight_kg": [10.5, 20.0, 15.2, 45.0, 120.0],
        "transit_days": [2.0, 3.0, 1.0, 7.0, 14.0],
        "freight_cost_usd": [150.0, 220.0, 180.0, 490.0, 1100.0]
    })
    entry_b = register_test_dataset(df_b, name="dataset_logistics.csv")

    # Run descriptive stats on Dataset A
    resp_a = client.post("/api/eda/descriptive", json={"dataset_id": entry_a.dataset_id})
    assert resp_a.status_code == 200
    cols_a = [c["column"] for c in resp_a.json()["result"]["statistics"]]
    assert set(cols_a) == {"emp_age", "emp_salary"}

    # Run descriptive stats on Dataset B
    resp_b = client.post("/api/eda/descriptive", json={"dataset_id": entry_b.dataset_id})
    assert resp_b.status_code == 200
    cols_b = [c["column"] for c in resp_b.json()["result"]["statistics"]]
    assert set(cols_b) == {"shipment_weight_kg", "transit_days", "freight_cost_usd"}

    # Ensure zero overlap or leakage
    assert not set(cols_a).intersection(set(cols_b))

    # Provenance events isolation
    events_a = provenance_store.get_events(dataset_id=entry_a.dataset_id)
    events_b = provenance_store.get_events(dataset_id=entry_b.dataset_id)
    assert len(events_a) == 1
    assert len(events_b) == 1
    assert events_a[0].dataset_name == "dataset_hr.csv"
    assert events_b[0].dataset_name == "dataset_logistics.csv"


# ── TEST 8: Strict Schema & Precondition Validation ─────────────────────────────
def test_8_schema_validation_and_controlled_errors(client):
    """
    Verify controlled 422 HTTP exceptions and failed provenance logging for:
    - COLUMN_NOT_FOUND
    - NON_NUMERIC_COLUMN
    - INSUFFICIENT_NUMERIC_COLUMNS
    """
    df = pd.DataFrame({
        "category": ["A", "B", "C", "D"],
        "single_num": [1.0, 2.0, 3.0, 4.0]
    })
    entry = register_test_dataset(df, name="validation_test.csv")

    # 1. Non-existent column
    resp1 = client.post("/api/eda/distribution", json={
        "dataset_id": entry.dataset_id,
        "column": "non_existent_column"
    })
    assert resp1.status_code == 422
    err1 = resp1.json()["detail"]
    assert err1["error_code"] == "COLUMN_NOT_FOUND"

    # 2. Non-numeric column for distribution
    resp2 = client.post("/api/eda/distribution", json={
        "dataset_id": entry.dataset_id,
        "column": "category"
    })
    assert resp2.status_code == 422
    err2 = resp2.json()["detail"]
    assert err2["error_code"] == "NON_NUMERIC_COLUMN"

    # 3. Insufficient numeric columns for correlation (only 1 numeric column exists)
    resp3 = client.post("/api/eda/correlation", json={
        "dataset_id": entry.dataset_id
    })
    assert resp3.status_code == 422
    err3 = resp3.json()["detail"]
    assert err3["error_code"] == "INSUFFICIENT_NUMERIC_COLUMNS"

    # Verify all 3 recorded FAILED provenance events
    failed_events = [e for e in provenance_store.get_events(dataset_id=entry.dataset_id) if e.status == "FAILED"]
    assert len(failed_events) == 3
    assert {e.error["error_code"] for e in failed_events} == {
        "COLUMN_NOT_FOUND", "NON_NUMERIC_COLUMN", "INSUFFICIENT_NUMERIC_COLUMNS"
    }


# ── TEST 9: Zero-Mutation & JSON Safety Invariant ───────────────────────────────
def test_9_zero_mutation_and_json_safety(client):
    """
    Verify:
    1. Read-only EDA operations NEVER mutate the underlying DataFrame or fingerprint.
    2. DataFrames containing NaN, +Inf, -Inf produce valid, JSON-safe responses without crashing.
    """
    df = pd.DataFrame({
        "val_with_nan": [1.0, np.nan, 3.0, 4.0],
        "val_with_inf": [10.0, np.inf, -np.inf, 40.0],
    })
    entry = register_test_dataset(df, name="extreme_values.csv")

    df_before = dataset_registry.get_dataframe(entry.dataset_id).copy(deep=True)
    hash_before = DatasetProfiler.compute_hash(df_before)

    # Execute descriptive statistics
    resp = client.post("/api/eda/descriptive", json={"dataset_id": entry.dataset_id})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCEEDED"

    # Verify no NaN or Inf in JSON response
    import json
    json_str = json.dumps(data)
    assert "NaN" not in json_str
    assert "Infinity" not in json_str

    # Verify DataFrame and hash are completely identical
    df_after = dataset_registry.get_dataframe(entry.dataset_id)
    hash_after = DatasetProfiler.compute_hash(df_after)
    assert hash_before == hash_after
    pd.testing.assert_frame_equal(df_before, df_after)

    # Check provenance event has identical input and output fingerprints
    events = provenance_store.get_events(dataset_id=entry.dataset_id)
    assert len(events) == 1
    evt = events[0]
    assert evt.before_state.fingerprint == evt.after_state.fingerprint
    assert evt.input_dataset_version == evt.output_dataset_version


# ── TEST 10: Unified /api/eda/execute Interface ────────────────────────────────
def test_10_unified_execute_interface(client):
    """
    Verify the unified /api/eda/execute endpoint correctly routes all operations:
    descriptive, missingness, distribution, correlation, skewness.
    """
    df = pd.DataFrame({
        "p": [10.0, 20.0, 30.0],
        "q": [100.0, 200.0, 300.0]
    })
    entry = register_test_dataset(df, name="unified.csv")

    ops = ["descriptive", "missingness", "correlation", "skewness"]
    for op in ops:
        resp = client.post("/api/eda/execute", json={
            "dataset_id": entry.dataset_id,
            "operation": op
        })
        assert resp.status_code == 200, f"Failed for {op}: {resp.text}"
        data = resp.json()
        assert data["status"] == "SUCCEEDED"

    # Distribution requires target column
    resp_dist = client.post("/api/eda/execute", json={
        "dataset_id": entry.dataset_id,
        "operation": "distribution",
        "column": "p"
    })
    assert resp_dist.status_code == 200
    assert resp_dist.json()["status"] == "SUCCEEDED"

    # Check total provenance events recorded
    events = provenance_store.get_events(dataset_id=entry.dataset_id)
    assert len(events) == 5
    capabilities = {e.capability for e in events}
    assert capabilities == {"EDA"}
