"""
DataPilot OS — Phase 5 Unsupervised Learning Full Test Suite.
Verifies K-Means Clustering, PCA Dimensionality Reduction, Elbow Curve Calculation,
Feature Validation, Zero Dataset Mutation, Cross-Dataset State Isolation,
Provenance Integration, and API Endpoints via TestClient.
"""

import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.unsupervised_engine import UnsupervisedEngine, sanitize_obj
from backend.persistence.provenance_store import provenance_store


@pytest.fixture
def unsupervised_engine():
    """UnsupervisedEngine instance."""
    return UnsupervisedEngine()


@pytest.fixture
def sample_numeric_df():
    """Synthetic dataset with 3 distinct clusters in 3-dimensional space."""
    np.random.seed(42)
    n_per_cluster = 30
    
    c1 = np.random.normal(loc=[10.0, 10.0, 5.0], scale=1.0, size=(n_per_cluster, 3))
    c2 = np.random.normal(loc=[30.0, 5.0, 15.0], scale=1.2, size=(n_per_cluster, 3))
    c3 = np.random.normal(loc=[5.0, 40.0, 25.0], scale=1.1, size=(n_per_cluster, 3))
    
    data = np.vstack([c1, c2, c3])
    df = pd.DataFrame(data, columns=["dim_alpha", "dim_beta", "dim_gamma"])
    df["categorical_meta"] = ["group_a"] * 30 + ["group_b"] * 30 + ["group_c"] * 30
    return df


@pytest.fixture
def registered_dataset(sample_numeric_df):
    """Registers sample_numeric_df into dataset_registry and cleans up after test."""
    dataset_id = "test_unsup_ds_1"
    prof = DatasetProfiler.profile(sample_numeric_df, dataset_name="Test Unsup Dataset")

    entry = DatasetEntry(
        dataset_id=dataset_id,
        name="Test Unsup Dataset",
        file_path="/tmp/test_unsup.csv",
        format="csv",
        row_count=len(sample_numeric_df),
        column_count=len(sample_numeric_df.columns),
        knowledge_model=prof.model_dump(),
        loaded_at="2026-10-02T12:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(dataset_id, entry)
    dataset_registry._dataframes_cache[dataset_id] = sample_numeric_df.copy(deep=True)
    dataset_registry.set_active(dataset_id)
    yield dataset_id, sample_numeric_df
    dataset_registry._datasets.pop(dataset_id, None)
    dataset_registry._dataframes_cache.pop(dataset_id, None)


# ── TEST 1: K-Means Clustering Workflow with Standard Scaling ─────────────────
def test_kmeans_clustering_standard_scaling(unsupervised_engine, sample_numeric_df):
    result = unsupervised_engine.run_kmeans(
        df=sample_numeric_df,
        features=["dim_alpha", "dim_beta", "dim_gamma"],
        n_clusters=3,
        scaling="standard",
        random_state=42
    )

    assert result["algorithm"] == "kmeans"
    assert result["n_clusters"] == 3
    assert result["scaling"] == "standard"
    assert result["features"] == ["dim_alpha", "dim_beta", "dim_gamma"]
    
    # Mathematical metrics validation
    assert "inertia" in result
    assert result["inertia"] > 0.0
    
    assert "silhouette_score" in result
    assert -1.0 <= result["silhouette_score"] <= 1.0
    # Three well-separated Gaussian clusters should yield high silhouette
    assert result["silhouette_score"] > 0.5

    # Cluster sizes validation
    cluster_sizes = result["cluster_sizes"]
    assert len(cluster_sizes) == 3
    assert sum(cluster_sizes.values()) == len(sample_numeric_df)
    for c_id, size in cluster_sizes.items():
        assert size > 0

    # Centroids validation
    centroids = result["centroids"]
    assert len(centroids) == 3
    for c in centroids:
        c_label = c.get("cluster_label", c.get("cluster"))
        assert c_label in [0, 1, 2]
        for f in ["dim_alpha", "dim_beta", "dim_gamma"]:
            assert f in c
            assert isinstance(c[f], (int, float))

    # Cluster sample and visualizations
    assert len(result["cluster_sample"]) > 0
    assert "cluster_label" in result["cluster_sample"][0]
    assert result["chart"] is not None and len(result["chart"]) > 50
    assert result["code"] is not None and "KMeans" in result["code"]


# ── TEST 2: K-Means Clustering with No Scaling ────────────────────────────────
def test_kmeans_clustering_no_scaling(unsupervised_engine, sample_numeric_df):
    result = unsupervised_engine.run_kmeans(
        df=sample_numeric_df,
        features=["dim_alpha", "dim_beta"],
        n_clusters=2,
        scaling="none",
        random_state=42
    )

    assert result["n_clusters"] == 2
    assert result["scaling"] == "none"
    assert len(result["cluster_sizes"]) == 2
    assert sum(result["cluster_sizes"].values()) == len(sample_numeric_df)
    assert result["inertia"] > 0.0
    assert -1.0 <= result["silhouette_score"] <= 1.0


# ── TEST 3: K-Means Validation & Rejections ───────────────────────────────────
def test_kmeans_validation_rejections(unsupervised_engine, sample_numeric_df):
    # k < 2
    with pytest.raises(ValueError, match="Number of clusters.*must be at least 2"):
        unsupervised_engine.run_kmeans(
            df=sample_numeric_df,
            features=["dim_alpha", "dim_beta"],
            n_clusters=1
        )

    # k >= len(df)
    with pytest.raises(ValueError, match="less than number of samples"):
        unsupervised_engine.run_kmeans(
            df=sample_numeric_df,
            features=["dim_alpha", "dim_beta"],
            n_clusters=len(sample_numeric_df)
        )

    # Non-existent feature column
    with pytest.raises(ValueError, match="Feature column.*not found"):
        unsupervised_engine.run_kmeans(
            df=sample_numeric_df,
            features=["dim_alpha", "nonexistent_feature"],
            n_clusters=3
        )

    # Non-numeric feature column
    with pytest.raises(ValueError, match="must be numeric"):
        unsupervised_engine.run_kmeans(
            df=sample_numeric_df,
            features=["dim_alpha", "categorical_meta"],
            n_clusters=3
        )

    # Fewer than 2 features
    with pytest.raises(ValueError, match="At least 2 feature columns are required"):
        unsupervised_engine.run_kmeans(
            df=sample_numeric_df,
            features=["dim_alpha"],
            n_clusters=3
        )


# ── TEST 4: Elbow Curve Computation ───────────────────────────────────────────
def test_elbow_curve_computation(unsupervised_engine, sample_numeric_df):
    elbow_result = unsupervised_engine.compute_elbow(
        df=sample_numeric_df,
        features=["dim_alpha", "dim_beta", "dim_gamma"],
        min_k=2,
        max_k=5,
        scaling="standard"
    )

    assert elbow_result["features"] == ["dim_alpha", "dim_beta", "dim_gamma"]
    curve = elbow_result["curve"]
    assert len(curve) == 4  # k=2, 3, 4, 5

    ks = [pt["k"] for pt in curve]
    assert ks == [2, 3, 4, 5]

    for pt in curve:
        assert pt["inertia"] > 0.0
        assert -1.0 <= pt["silhouette_score"] <= 1.0

    # Inertia must be monotonically non-increasing as k increases
    inertias = [pt["inertia"] for pt in curve]
    for i in range(len(inertias) - 1):
        assert inertias[i] >= inertias[i + 1]

    # Test invalid range
    with pytest.raises(ValueError, match="min_k must be at least 2"):
        unsupervised_engine.compute_elbow(sample_numeric_df, ["dim_alpha", "dim_beta"], min_k=1, max_k=4)

    with pytest.raises(ValueError, match="min_k must be strictly less than max_k"):
        unsupervised_engine.compute_elbow(sample_numeric_df, ["dim_alpha", "dim_beta"], min_k=4, max_k=4)


# ── TEST 5: PCA Dimensionality Reduction and Loadings ─────────────────────────
def test_pca_execution_and_loadings(unsupervised_engine, sample_numeric_df):
    result = unsupervised_engine.run_pca(
        df=sample_numeric_df,
        features=["dim_alpha", "dim_beta", "dim_gamma"],
        n_components=2,
        scaling="standard"
    )

    assert result["algorithm"] == "pca"
    assert result["n_components"] == 2
    assert result["scaling"] == "standard"
    assert result["features"] == ["dim_alpha", "dim_beta", "dim_gamma"]

    # Variance ratios
    evr = result["explained_variance_ratio"]
    assert len(evr) == 2
    assert all(0.0 <= v <= 1.0 for v in evr)
    assert sum(evr) <= 1.0001

    cvr = result["cumulative_variance_ratio"]
    assert len(cvr) == 2
    assert abs(cvr[0] - evr[0]) < 1e-3
    assert abs(cvr[1] - sum(evr)) < 1e-3

    # Loadings matrix: length matches n_components, each containing all feature weights
    loadings = result["loadings"]
    assert len(loadings) == 2  # 2 components
    for row in loadings:
        assert "component" in row
        for f in ["dim_alpha", "dim_beta", "dim_gamma"]:
            assert f in row
            assert isinstance(row[f], float)

    # Transformed sample and visualizations
    assert len(result["transformed_sample"]) > 0
    assert "PC1" in result["transformed_sample"][0]
    assert "PC2" in result["transformed_sample"][0]
    assert result["chart"] is not None and len(result["chart"]) > 50
    assert result["code"] is not None and "PCA" in result["code"]


# ── TEST 6: PCA Validation & Rejections ───────────────────────────────────────
def test_pca_validation_rejections(unsupervised_engine, sample_numeric_df):
    # n_components > features
    with pytest.raises(ValueError, match="n_components.*cannot exceed"):
        unsupervised_engine.run_pca(
            df=sample_numeric_df,
            features=["dim_alpha", "dim_beta"],
            n_components=3
        )

    # n_components < 1
    with pytest.raises(ValueError, match="n_components must be at least 1"):
        unsupervised_engine.run_pca(
            df=sample_numeric_df,
            features=["dim_alpha", "dim_beta"],
            n_components=0
        )

    # Non-numeric feature
    with pytest.raises(ValueError, match="must be numeric"):
        unsupervised_engine.run_pca(
            df=sample_numeric_df,
            features=["dim_alpha", "categorical_meta"],
            n_components=2
        )

    # Non-existent feature
    with pytest.raises(ValueError, match="Feature column.*not found"):
        unsupervised_engine.run_pca(
            df=sample_numeric_df,
            features=["dim_alpha", "missing_col"],
            n_components=2
        )


# ── TEST 7: Zero Dataset Mutation Guarantee ───────────────────────────────────
def test_zero_dataset_mutation_guarantee(unsupervised_engine, registered_dataset):
    dataset_id, original_df = registered_dataset
    
    # Store exact snapshot
    original_shape = original_df.shape
    original_columns = list(original_df.columns)
    original_dtypes = original_df.dtypes.to_dict()
    original_values = original_df.values.copy()

    # 1. Run K-Means
    unsupervised_engine.run_kmeans(
        df=original_df,
        features=["dim_alpha", "dim_beta"],
        n_clusters=3,
        scaling="standard"
    )

    # Verify original dataframe in registry cache is unmodified
    current_df = dataset_registry.get_dataframe(dataset_id)
    assert current_df.shape == original_shape
    assert list(current_df.columns) == original_columns
    assert current_df.dtypes.to_dict() == original_dtypes
    np.testing.assert_array_equal(current_df.values, original_values)

    # 2. Run Elbow
    unsupervised_engine.compute_elbow(
        df=original_df,
        features=["dim_alpha", "dim_beta"],
        min_k=2,
        max_k=4
    )

    # Verify unmodified
    current_df = dataset_registry.get_dataframe(dataset_id)
    assert current_df.shape == original_shape
    assert list(current_df.columns) == original_columns
    np.testing.assert_array_equal(current_df.values, original_values)

    # 3. Run PCA
    unsupervised_engine.run_pca(
        df=original_df,
        features=["dim_alpha", "dim_beta", "dim_gamma"],
        n_components=2
    )

    # Verify unmodified
    current_df = dataset_registry.get_dataframe(dataset_id)
    assert current_df.shape == original_shape
    assert list(current_df.columns) == original_columns
    np.testing.assert_array_equal(current_df.values, original_values)


# ── TEST 8: Cross-Dataset State Isolation ─────────────────────────────────────
def test_cross_dataset_state_isolation(unsupervised_engine):
    # Dataset A
    df_a = pd.DataFrame({
        "var_a1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "var_a2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    })
    # Dataset B with completely distinct schema
    df_b = pd.DataFrame({
        "sensor_b1": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        "sensor_b2": [5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        "sensor_b3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    })

    # Execute on A
    res_a = unsupervised_engine.run_kmeans(df_a, ["var_a1", "var_a2"], n_clusters=2)
    assert res_a["features"] == ["var_a1", "var_a2"]

    # Attempting to run with Dataset A's features on Dataset B must fail
    with pytest.raises(ValueError, match="Feature column 'var_a1' not found"):
        unsupervised_engine.run_kmeans(df_b, ["var_a1", "var_a2"], n_clusters=2)

    # Execute PCA on B
    res_b = unsupervised_engine.run_pca(df_b, ["sensor_b1", "sensor_b2", "sensor_b3"], n_components=2)
    assert res_b["features"] == ["sensor_b1", "sensor_b2", "sensor_b3"]
    assert len(res_b["loadings"]) == 2
    assert all(f in res_b["loadings"][0] for f in ["sensor_b1", "sensor_b2", "sensor_b3"])


# ── TEST 9: API Endpoints E2E via TestClient & Provenance Logging ──────────────
def test_api_endpoints_e2e_and_provenance(registered_dataset):
    dataset_id, _ = registered_dataset
    client = TestClient(app)

    # 1. K-Means Endpoint
    payload_kmeans = {
        "dataset_id": dataset_id,
        "features": ["dim_alpha", "dim_beta"],
        "n_clusters": 3,
        "scaling": "standard",
    }
    resp_k = client.post("/api/unsupervised/kmeans", json=payload_kmeans)
    assert resp_k.status_code == 200
    data_k = resp_k.json()
    assert data_k["success"] is True
    assert data_k["algorithm"] == "kmeans"
    assert data_k["n_clusters"] == 3
    assert "audit_event_id" in data_k
    assert data_k["audit_event_id"].startswith(("evt_", "prov_"))

    # Verify provenance event logged
    events = provenance_store.get_dataset_lineage(dataset_id)
    kmeans_events = [e for e in events if e.operation == "KMEANS_CLUSTERING"]
    assert len(kmeans_events) > 0
    assert kmeans_events[-1].parameters["n_clusters"] == 3
    assert kmeans_events[-1].parameters["features"] == ["dim_alpha", "dim_beta"]

    # 2. Elbow Endpoint
    payload_elbow = {
        "dataset_id": dataset_id,
        "features": ["dim_alpha", "dim_beta"],
        "max_k": 4,
        "scaling": "standard"
    }
    resp_e = client.post("/api/unsupervised/elbow", json=payload_elbow)
    assert resp_e.status_code == 200
    data_e = resp_e.json()
    assert data_e["success"] is True
    assert len(data_e["curve"]) == 3

    # 3. PCA Endpoint
    payload_pca = {
        "dataset_id": dataset_id,
        "features": ["dim_alpha", "dim_beta", "dim_gamma"],
        "n_components": 2,
        "scaling": "standard"
    }
    resp_p = client.post("/api/unsupervised/pca", json=payload_pca)
    assert resp_p.status_code == 200
    data_p = resp_p.json()
    assert data_p["success"] is True
    assert data_p["algorithm"] == "pca"
    assert data_p["n_components"] == 2
    assert "audit_event_id" in data_p
    assert data_p["audit_event_id"].startswith(("evt_", "prov_"))

    # Verify provenance event logged for PCA
    events = provenance_store.get_dataset_lineage(dataset_id)
    pca_events = [e for e in events if e.operation == "PCA_DIMENSIONALITY_REDUCTION"]
    assert len(pca_events) > 0
    assert pca_events[-1].parameters["n_components"] == 2
    assert pca_events[-1].parameters["features"] == ["dim_alpha", "dim_beta", "dim_gamma"]

    # 4. Error handling: Non-existent dataset
    resp_404 = client.post("/api/unsupervised/kmeans", json={
        "dataset_id": "nonexistent_id",
        "features": ["dim_alpha", "dim_beta"],
        "n_clusters": 2
    })
    assert resp_404.status_code == 404

    # 5. Error handling: Invalid parameters (e.g. k=1)
    resp_422 = client.post("/api/unsupervised/kmeans", json={
        "dataset_id": dataset_id,
        "features": ["dim_alpha", "dim_beta"],
        "n_clusters": 1
    })
    assert resp_422.status_code in [400, 422]


# ── TEST 10: JSON Serialization Safety ────────────────────────────────────────
def test_json_serialization_safety():
    raw_data = {
        "int64_val": np.int64(42),
        "float64_val": np.float64(3.14159),
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "arr_1d": np.array([1.0, 2.0, 3.0]),
        "arr_2d": np.array([[10, 20], [30, 40]]),
        "nested_dict": {
            "inner_arr": np.array([0.5, 0.25]),
            "inner_int": np.int32(100),
            "inner_nan": np.nan
        }
    }

    sanitized = sanitize_obj(raw_data)
    json_str = json.dumps(sanitized)
    deserialized = json.loads(json_str)

    assert deserialized["int64_val"] == 42
    assert abs(deserialized["float64_val"] - 3.14159) < 1e-4
    assert deserialized["nan_val"] is None
    assert deserialized["inf_val"] is None
    assert deserialized["arr_1d"] == [1.0, 2.0, 3.0]
    assert deserialized["arr_2d"] == [[10, 20], [30, 40]]
    assert deserialized["nested_dict"]["inner_arr"] == [0.5, 0.25]
    assert deserialized["nested_dict"]["inner_nan"] is None
