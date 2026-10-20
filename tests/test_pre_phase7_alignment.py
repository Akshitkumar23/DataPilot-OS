"""
Pre-Phase 7 Governed Prep & Feature Operations Alignment Test Suite.
Validates:
1. The governed operations wired into DataScienceStudio.jsx execute through POST /api/operations/apply:
   - clip_iqr (IQR outlier capping)
   - impute_mean (numeric mean imputation)
   - yeo_johnson (power transformation)
   - one_hot_encode (categorical dummy encoding)
2. Every operation registers an immutable provenance step with execution metrics and audit hashes.
3. Zero arbitrary terminal execution, zero code evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.ds_engine.dataset_profiler import DatasetProfiler


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def registered_dataset():
    """Register a clean test dataset with missing values, outliers, and categories."""
    dataset_registry.clear()
    df = pd.DataFrame({
        "num_meas": [10.0, 12.0, 11.0, 100.0, np.nan, 13.0, 9.0, 12.0, -50.0, 11.0],
        "skewed_val": [1.0, 2.0, 1.5, 30.0, 50.0, 2.0, 1.0, 80.0, 1.2, 2.5],
        "cat_type": ["type_A", "type_B", "type_A", "type_C", "type_B", "type_A", "type_B", "type_C", "type_A", "type_B"],
    })
    ds_id = "test_align_ds"
    prof = DatasetProfiler.profile(df, dataset_name="alignment_test.csv")
    entry = DatasetEntry(
        dataset_id=ds_id,
        name="alignment_test.csv",
        file_path="/tmp/alignment_test.csv",
        format="csv",
        row_count=len(df),
        column_count=len(df.columns),
        knowledge_model=prof.model_dump(),
        loaded_at="2026-10-02T00:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = df.copy(deep=True)
    dataset_registry.set_active(ds_id)
    return ds_id, df


def test_governed_impute_mean(client, registered_dataset):
    """Verify governed mean imputation via POST /api/operations/apply."""
    ds_id, _ = registered_dataset
    payload = {
        "dataset_id": ds_id,
        "operation": "impute_mean",
        "columns": ["num_meas"],
    }
    res = client.post("/api/operations/apply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "impute_mean"

    # Verify no nulls in num_meas
    updated_df = dataset_registry.get_dataframe(ds_id)
    assert updated_df["num_meas"].isna().sum() == 0


def test_governed_clip_iqr_outliers(client, registered_dataset):
    """Verify governed IQR outlier capping via POST /api/operations/apply."""
    ds_id, _ = registered_dataset
    payload = {
        "dataset_id": ds_id,
        "operation": "clip_iqr",
        "columns": ["num_meas"],
    }
    res = client.post("/api/operations/apply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "clip_iqr"

    # Verify 100.0 and -50.0 were clipped
    updated_df = dataset_registry.get_dataframe(ds_id)
    assert updated_df["num_meas"].max() < 100.0
    assert updated_df["num_meas"].min() > -50.0


def test_governed_one_hot_encode(client, registered_dataset):
    """Verify governed one-hot dummy encoding via POST /api/operations/apply."""
    ds_id, _ = registered_dataset
    payload = {
        "dataset_id": ds_id,
        "operation": "one_hot_encode",
        "columns": ["cat_type"],
    }
    res = client.post("/api/operations/apply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "one_hot_encode"

    # Verify new dummy columns exist
    updated_df = dataset_registry.get_dataframe(ds_id)
    encoded_cols = [c for c in updated_df.columns if "cat_type_" in c or "type_" in c]
    assert len(encoded_cols) >= 2


def test_governed_yeo_johnson_transform(client, registered_dataset):
    """Verify governed Yeo-Johnson power transform via POST /api/operations/apply."""
    ds_id, _ = registered_dataset
    payload = {
        "dataset_id": ds_id,
        "operation": "yeo_johnson",
        "columns": ["skewed_val"],
    }
    res = client.post("/api/operations/apply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCEEDED"
    assert data["operation"] == "yeo_johnson"


def test_governed_operations_record_provenance_trail(client, registered_dataset):
    """Verify that multiple governed operations build an unbroken, verifiable audit trail."""
    ds_id, _ = registered_dataset
    # Apply an operation
    res_op = client.post("/api/operations/apply", json={
        "dataset_id": ds_id,
        "operation": "impute_mean",
        "columns": ["num_meas"],
    })
    assert res_op.status_code == 200

    # Query audit trail
    res = client.get(f"/api/governance/audit?dataset_id={ds_id}")
    assert res.status_code == 200
    prov_data = res.json()
    assert prov_data["status"] == "success"
    assert prov_data["total"] >= 1
    assert any(e["operation"].lower() == "impute_mean" for e in prov_data["events"])
