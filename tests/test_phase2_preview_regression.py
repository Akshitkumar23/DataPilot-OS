"""
Phase 2 Regression Tests: Preview JSON Serialization & State Isolation.
Verifies that:
1. POST /api/operations/preview on columns containing Pandas NaN returns HTTP 200.
2. Missing values are represented in preview payload as JSON null (Python None).
3. The underlying DataFrame continues to contain Pandas NaN with unchanged numeric dtype.
4. No dataset mutation occurs during Preview (row count, values, missingness, duplicates intact).
5. Normal numeric columns without NaN produce unchanged expected preview values.
6. Columns with infinite values (+inf, -inf) serialize safely to JSON null.
"""
import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.ds_engine.dataset_profiler import DatasetProfiler

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean dataset registry before and after every test."""
    dataset_registry.clear()
    yield
    dataset_registry.clear()


@pytest.fixture
def user_specified_df():
    """Exact test dataset specified in the specification."""
    csv_data = """age,income,department
20,30000,Sales
25,40000,HR
,50000,Sales
30,60000,Engineering
30,60000,Engineering
"""
    return pd.read_csv(io.StringIO(csv_data))


@pytest.fixture
def registered_user_dataset(user_specified_df):
    """Register user_specified_df in dataset_registry as active dataset."""
    ds_id = "test_ds_preview_regress"
    prof = DatasetProfiler.profile(user_specified_df, dataset_name="user_dataset.csv")
    entry = DatasetEntry(
        dataset_id=ds_id,
        name="user_dataset.csv",
        file_path="/tmp/user_dataset.csv",
        format="csv",
        row_count=len(user_specified_df),
        column_count=len(user_specified_df.columns),
        knowledge_model=prof.model_dump(),
        loaded_at="2026-09-28T00:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = user_specified_df.copy(deep=True)
    dataset_registry.set_active(ds_id)
    return ds_id


def test_preview_nan_serialization_and_isolation(registered_user_dataset):
    """
    Regression Test: drop_nulls on 'age' containing NaN.
    Must return HTTP 200, serialize NaN as null, preserve underlying DataFrame.
    """
    df_before = dataset_registry.get_dataframe(registered_user_dataset)
    assert len(df_before) == 5
    assert df_before["age"].isna().sum() == 1
    assert pd.api.types.is_float_dtype(df_before["age"])

    # 1. POST /api/operations/preview on 'age' (which contains NaN)
    res = client.post("/api/operations/preview", json={
        "dataset_id": registered_user_dataset,
        "operation": "drop_nulls",
        "column": "age"
    })

    # Expected: HTTP 200 with valid JSON response
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["status"] == "SUCCEEDED"
    assert "preview" in data
    preview = data["preview"]

    # 2. Before values must represent missing age as JSON null (None in Python)
    before_vals = preview["before"]
    assert len(before_vals) == 5
    assert before_vals[0] == 20.0
    assert before_vals[1] == 25.0
    assert before_vals[2] is None, f"Expected None/null at index 2, got {before_vals[2]}"
    assert before_vals[3] == 30.0
    assert before_vals[4] == 30.0

    # 3. After values must contain expected post-operation values (4 non-null rows)
    after_vals = preview["after"]
    assert len(after_vals) == 4
    assert after_vals == [20.0, 25.0, 30.0, 30.0]

    # 4. Preview stats
    assert preview["before_stats"]["nulls"] == 1
    assert preview["before_stats"]["rows"] == 5
    assert preview["after_stats"]["nulls"] == 0
    assert preview["after_stats"]["rows"] == 4

    # 5. Dataset in registry MUST be strictly unmutated
    df_after = dataset_registry.get_dataframe(registered_user_dataset)
    assert len(df_after) == 5, "Dataset row count changed during preview!"
    assert df_after["age"].isna().sum() == 1, "Dataset NaN was mutated during preview!"
    assert pd.isna(df_after.loc[2, "age"]), "Row 2 age was altered!"
    assert df_after.loc[0, "age"] == 20.0
    assert df_after.loc[1, "age"] == 25.0
    assert df_after.loc[3, "age"] == 30.0
    assert df_after.loc[4, "age"] == 30.0

    # 6. DataFrame column dtype MUST remain normal numeric float64 (NOT object)
    assert pd.api.types.is_float_dtype(df_after["age"]), f"Column dtype corrupted: {df_after['age'].dtype}"
    assert df_after["age"].dtype == np.float64 or df_after["age"].dtype == "float64"


def test_preview_normal_numeric_column_unmodified(registered_user_dataset):
    """
    Verification: normal numeric column without NaN ('income') preview behaves normally.
    """
    df_before = dataset_registry.get_dataframe(registered_user_dataset)
    assert df_before["income"].isna().sum() == 0

    res = client.post("/api/operations/preview", json={
        "dataset_id": registered_user_dataset,
        "operation": "normalize_minmax",
        "column": "income"
    })

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCEEDED"
    preview = data["preview"]

    assert preview["before"] == [30000, 40000, 50000, 60000, 60000]
    expected_after = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0, 1.0]
    assert np.allclose(preview["after"], expected_after)

    # Dataset remains completely unmutated
    df_after = dataset_registry.get_dataframe(registered_user_dataset)
    assert len(df_after) == 5
    assert df_after["income"].tolist() == [30000, 40000, 50000, 60000, 60000]


def test_preview_infinite_float_serialization():
    """
    Verification: DataFrame with +inf, -inf, and nan serializes to JSON null without error.
    """
    df_inf = pd.DataFrame({
        "val": [1.0, float("inf"), float("-inf"), np.nan, 5.0]
    })
    ds_id = "test_ds_inf"
    prof = DatasetProfiler.profile(df_inf, dataset_name="inf_test.csv")
    entry = DatasetEntry(
        dataset_id=ds_id,
        name="inf_test.csv",
        file_path="/tmp/inf_test.csv",
        format="csv",
        row_count=len(df_inf),
        column_count=1,
        knowledge_model=prof.model_dump(),
        loaded_at="2026-09-28T00:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = df_inf.copy(deep=True)
    dataset_registry.set_active(ds_id)

    res = client.post("/api/operations/preview", json={
        "dataset_id": ds_id,
        "operation": "drop_nulls",
        "column": "val"
    })

    assert res.status_code == 200
    preview = res.json()["preview"]
    # 1.0 stays 1.0, inf/-inf/nan become null (None), 5.0 stays 5.0
    assert preview["before"][0] == 1.0
    assert preview["before"][1] is None
    assert preview["before"][2] is None
    assert preview["before"][3] is None
    assert preview["before"][4] == 5.0

    # Underlying DataFrame preserved with its infinite/nan values
    df_after = dataset_registry.get_dataframe(ds_id)
    assert np.isinf(df_after.loc[1, "val"])
    assert np.isneginf(df_after.loc[2, "val"])
    assert np.isnan(df_after.loc[3, "val"])
