"""
Phase 1 Audit & Verification Tests — Trustworthy Data Profile + Data Grid.
Verifies mathematical accuracy of data profiling, zero hardcoded values,
API endpoints, and dataset switching without stale state.
"""
import os
import tempfile
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.datasets import compute_profile_summary, router
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_dataset_a_clean_ground_truth():
    """Dataset A: Clean dataset with zero nulls and zero duplicates."""
    df_clean = pd.DataFrame({
        "id": list(range(1, 11)),
        "category": [f"cat_{i % 3}" for i in range(10)],
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "is_active": [True, False] * 5
    })

    prof = compute_profile_summary(df_clean, "ds_clean", "clean.csv")

    assert prof["row_count"] == 10
    assert prof["column_count"] == 4
    assert prof["total_cells"] == 40
    assert prof["missing_cells"] == 0
    assert prof["missing_percentage"] == 0.0
    assert prof["duplicate_rows"] == 0
    assert prof["duplicate_percentage"] == 0.0
    assert prof["schema_status"]["is_valid"] is True
    assert prof["schema_status"]["status"] == "Valid"

    col_map = {c["name"]: c for c in prof["columns"]}
    assert col_map["id"]["normalized_type"] == "NUM"
    assert col_map["id"]["null_count"] == 0
    assert col_map["id"]["unique_count"] == 10

    assert col_map["category"]["normalized_type"] == "STR"
    assert col_map["category"]["null_count"] == 0
    assert col_map["category"]["unique_count"] == 3

    assert col_map["timestamp"]["normalized_type"] == "DATE"
    assert col_map["timestamp"]["null_count"] == 0
    assert col_map["timestamp"]["unique_count"] == 10

    assert col_map["is_active"]["normalized_type"] == "BOOL"
    assert col_map["is_active"]["null_count"] == 0
    assert col_map["is_active"]["unique_count"] == 2


def test_dataset_b_known_missingness_ground_truth():
    """Dataset B: Dataset with known missingness (exact null count and percentage)."""
    df_missing = pd.DataFrame({
        "col_a": [1.0, None, 3.0, None, 5.0, 6.0, 7.0, None, 9.0, None],  # 4 nulls out of 10 = 40%
        "col_b": ["a", "b", None, "d", "e", "f", "g", "h", None, "j"],      # 2 nulls out of 10 = 20%
        "col_c": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],                 # 0 nulls = 0%
        "col_d": [True] * 10                                                  # 0 nulls = 0%
    })

    # Total cells = 10 * 4 = 40. Total nulls = 4 + 2 = 6. Overall missing % = 6/40 * 100 = 15.0%
    prof = compute_profile_summary(df_missing, "ds_missing", "missing.csv")

    assert prof["row_count"] == 10
    assert prof["column_count"] == 4
    assert prof["total_cells"] == 40
    assert prof["missing_cells"] == 6
    assert prof["missing_percentage"] == 15.0

    col_map = {c["name"]: c for c in prof["columns"]}
    assert col_map["col_a"]["null_count"] == 4
    assert col_map["col_a"]["null_percentage"] == 40.0
    assert col_map["col_a"]["unique_count"] == 6

    assert col_map["col_b"]["null_count"] == 2
    assert col_map["col_b"]["null_percentage"] == 20.0
    assert col_map["col_b"]["unique_count"] == 8

    assert col_map["col_c"]["null_count"] == 0
    assert col_map["col_c"]["null_percentage"] == 0.0

    assert col_map["col_d"]["null_count"] == 0
    assert col_map["col_d"]["null_percentage"] == 0.0


def test_dataset_c_known_duplicates_ground_truth():
    """Dataset C: Dataset with known duplicates."""
    row_unique_1 = {"id": 1, "val": "alpha"}
    row_unique_2 = {"id": 2, "val": "beta"}
    row_unique_3 = {"id": 3, "val": "gamma"}

    # 8 rows total: row_unique_1 appears 3 times (2 duplicates), row_unique_2 appears 2 times (1 duplicate)
    # Total duplicates = 2 + 1 = 3 duplicates out of 8 rows = 37.5%
    data = [
        row_unique_1,
        row_unique_2,
        row_unique_3,
        row_unique_1,  # duplicate #1
        row_unique_1,  # duplicate #2
        row_unique_2,  # duplicate #3
        {"id": 4, "val": "delta"},
        {"id": 5, "val": "epsilon"}
    ]
    df_dups = pd.DataFrame(data)

    prof = compute_profile_summary(df_dups, "ds_dups", "dups.csv")

    assert prof["row_count"] == 8
    assert prof["duplicate_rows"] == 3
    assert prof["duplicate_percentage"] == 37.5
    assert prof["missing_cells"] == 0


def test_dataset_d_mixed_types_and_date_strings():
    """Dataset D: Object column containing parseable dates, and another column with mixed types."""
    df_mixed = pd.DataFrame({
        "date_strings": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "mixed_col": [1, "two", 3, "four", 5]  # mixed int and str
    })

    prof = compute_profile_summary(df_mixed, "ds_mixed", "mixed.csv")

    assert prof["schema_status"]["is_valid"] is False
    assert prof["schema_status"]["status"] == "Warning"
    assert "mixed_col" in prof["schema_status"]["mixed_columns"]

    col_map = {c["name"]: c for c in prof["columns"]}
    assert col_map["date_strings"]["normalized_type"] == "DATE"


def test_api_endpoints_and_dataset_switching():
    """Verify GET /data, GET /profile, and active dataset switching without stale state."""
    # Register Dataset 1
    df1 = pd.DataFrame({
        "metric": [10.0, 20.0, 30.0],
        "tag": ["a", "b", "c"]
    })
    entry1 = DatasetEntry(
        dataset_id="ds_test_1",
        name="test_1.csv",
        file_path="/tmp/test_1.csv",
        format="csv",
        row_count=3,
        column_count=2,
        knowledge_model={},
        loaded_at="2026-09-27T00:00:00",
        file_hash="hash1"
    )
    dataset_registry.register("ds_test_1", entry1)
    dataset_registry._dataframes_cache["ds_test_1"] = df1
    dataset_registry.set_active("ds_test_1")

    # Register Dataset 2 (with known nulls & duplicates)
    df2 = pd.DataFrame({
        "score": [100, None, 100],  # 1 null, 2 identical rows across both cols
        "label": ["win", "loss", "win"]
    })
    entry2 = DatasetEntry(
        dataset_id="ds_test_2",
        name="test_2.csv",
        file_path="/tmp/test_2.csv",
        format="csv",
        row_count=3,
        column_count=2,
        knowledge_model={},
        loaded_at="2026-09-27T00:00:00",
        file_hash="hash2"
    )
    dataset_registry.register("ds_test_2", entry2)
    dataset_registry._dataframes_cache["ds_test_2"] = df2

    # 1. Test GET /api/datasets/profile for ds_test_1
    res1 = client.get("/api/datasets/profile?dataset_id=ds_test_1")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["dataset_id"] == "ds_test_1"
    assert data1["missing_cells"] == 0
    assert data1["duplicate_rows"] == 0
    assert data1["row_count"] == 3

    # 2. Test GET /api/datasets/data returns both rows and profile for ds_test_1
    data_res1 = client.get("/api/datasets/data?dataset_id=ds_test_1")
    assert data_res1.status_code == 200
    data_body1 = data_res1.json()
    assert "profile" in data_body1
    assert data_body1["profile"]["missing_cells"] == 0
    assert len(data_body1["rows"]) == 3

    # 3. Switch active dataset to ds_test_2
    dataset_registry.set_active("ds_test_2")

    # 4. Test GET /api/datasets/profile for active dataset (now ds_test_2)
    res2 = client.get("/api/datasets/profile")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["dataset_id"] == "ds_test_2"
    assert data2["missing_cells"] == 1
    assert data2["duplicate_rows"] == 1
    assert round(data2["missing_percentage"], 1) == 16.7  # 1 null out of 6 cells = 16.67%
    assert round(data2["duplicate_percentage"], 1) == 33.3  # 1 duplicate out of 3 rows = 33.33%

    # 5. Test GET /api/datasets/{dataset_id}/profile endpoint
    res_by_id = client.get("/api/datasets/ds_test_2/profile")
    assert res_by_id.status_code == 200
    assert res_by_id.json()["dataset_id"] == "ds_test_2"
    assert res_by_id.json()["duplicate_rows"] == 1


def test_zero_fake_domain_or_quality_in_code():
    """Verify that DataPreview.jsx contains no hardcoded fake Domain or Quality badges."""
    preview_jsx_path = os.path.join(os.path.dirname(__file__), "..", "src", "components", "DataPreview.jsx")
    with open(preview_jsx_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Domain: Retail Sales" not in content
    assert "Quality: 92/100" not in content
    assert "0.0% Zero Dups" not in content
    assert "100% Inferred Safe" not in content
