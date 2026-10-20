"""
DataPilot OS — Phase 4 Machine Learning Full Test Suite.
Verifies Classification, Regression, AutoML Tournament, Target/Feature Validation,
Data Leakage Prevention, Cross-Validation, Model Persistence, Schema Mismatch Guards,
and Provenance Integration across multiple schema-isolated datasets.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.dataset_engine.ml_engine import MLEngine, sanitize_obj
from backend.main import app
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.persistence.provenance_store import provenance_store


@pytest.fixture
def temp_ml_engine():
    """Isolated MLEngine instance with temporary storage."""
    temp_dir = tempfile.mkdtemp()
    engine = MLEngine(models_dir=temp_dir)
    yield engine
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_classification_df():
    """Synthetic dataset for classification with numeric and categorical features."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feat_num1": np.random.normal(10, 2, n),
        "feat_num2": np.random.uniform(0, 100, n),
        "feat_cat": np.random.choice(["cat_a", "cat_b", "cat_c"], n),
        "label": np.random.choice(["class_0", "class_1"], n),
    })


@pytest.fixture
def sample_regression_df():
    """Synthetic dataset for continuous regression."""
    np.random.seed(123)
    n = 120
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.uniform(1, 20, n)
    cat = np.random.choice(["alpha", "beta"], n)
    # y = 2.5*x1 - 1.2*x2 + noise
    noise = np.random.normal(0, 2, n)
    y = 2.5 * x1 - 1.2 * x2 + noise
    return pd.DataFrame({
        "feature_x1": x1,
        "feature_x2": x2,
        "category_dim": cat,
        "continuous_target": y,
    })


# ── TEST 1: Supervised Classification Full Workflow ────────────────────────────
def test_classification_training_and_metrics(temp_ml_engine, sample_classification_df):
    result = temp_ml_engine.train_model(
        df=sample_classification_df,
        target="label",
        task="classification",
        algorithm="random_forest",
        validation_strategy="split",
        test_size=0.2,
    )

    assert result["task"] == "classification"
    assert result["algorithm"] == "random_forest"
    assert result["target"] == "label"
    assert set(result["features"]) == {"feat_num1", "feat_num2", "feat_cat"}
    assert result["sample_counts"]["total"] == 100
    assert result["sample_counts"]["train"] == 80
    assert result["sample_counts"]["test"] == 20

    metrics = result["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_weighted"] <= 1.0
    assert 0.0 <= metrics["precision_macro"] <= 1.0
    assert 0.0 <= metrics["recall_macro"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert len(cm["classes"]) == 2
    assert len(cm["matrix"]) == 2
    assert sum(sum(row) for row in cm["matrix"]) == 20  # Matches test sample count

    assert len(result["feature_importances"]) > 0
    assert result["model_id"].startswith("model_")


# ── TEST 2: Supervised Regression Full Workflow ────────────────────────────────
def test_regression_training_and_metrics(temp_ml_engine, sample_regression_df):
    result = temp_ml_engine.train_model(
        df=sample_regression_df,
        target="continuous_target",
        task="regression",
        algorithm="linear_regression",
        validation_strategy="split",
        test_size=0.2,
    )

    assert result["task"] == "regression"
    assert result["algorithm"] == "linear_regression"
    assert result["target"] == "continuous_target"

    metrics = result["metrics"]
    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    # On synthetic linear data with small noise, R2 should be high (> 0.8)
    assert metrics["r2"] > 0.8

    residuals = metrics["residuals"]
    assert "min" in residuals
    assert "max" in residuals
    assert "mean" in residuals
    assert "std" in residuals


# ── TEST 3: AutoML Tournament Model Comparison ─────────────────────────────────
def test_automl_tournament_ranking(temp_ml_engine, sample_classification_df):
    result = temp_ml_engine.train_model(
        df=sample_classification_df,
        target="label",
        algorithm="auto",  # Triggers tournament across all supported classification models
    )

    tournament = result["tournament"]
    assert len(tournament) == 3  # random_forest, decision_tree, logistic_regression

    alg_ids = [c["algorithm_id"] for c in tournament]
    assert "random_forest" in alg_ids
    assert "decision_tree" in alg_ids
    assert "logistic_regression" in alg_ids

    # Ranked in descending order of primary score
    scores = [c["primary_score"] for c in tournament]
    assert scores == sorted(scores, reverse=True)
    assert tournament[0]["rank"] == 1
    assert result["algorithm"] == tournament[0]["algorithm_id"]


# ── TEST 4: Cross-Validation Support ───────────────────────────────────────────
def test_k_fold_cross_validation(temp_ml_engine, sample_regression_df):
    result = temp_ml_engine.train_model(
        df=sample_regression_df,
        target="continuous_target",
        task="regression",
        algorithm="decision_tree",
        validation_strategy="cv",
        cv_folds=5,
    )

    cv_data = result["cross_validation"]
    assert cv_data["performed"] is True
    assert len(cv_data["scores"]) == 5
    assert cv_data["mean"] is not None
    assert cv_data["std"] is not None


# ── TEST 5: Target Validation & Precondition Guards ────────────────────────────
def test_target_validation_errors(temp_ml_engine, sample_classification_df):
    # 1. Nonexistent target
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(sample_classification_df, target="ghost_col")
    assert "COLUMN_NOT_FOUND" in str(exc_info.value)

    # 2. Single-class target for classification
    single_class_df = sample_classification_df.copy()
    single_class_df["label"] = "only_one_class"
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(single_class_df, target="label", task="classification")
    assert "SINGLE_CLASS_TARGET" in str(exc_info.value)

    # 3. Non-numeric target for regression
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(sample_classification_df, target="label", task="regression")
    assert "NON_NUMERIC_REGRESSION_TARGET" in str(exc_info.value)

    # 4. Insufficient samples (< 5)
    tiny_df = sample_classification_df.head(3)
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(tiny_df, target="label")
    assert "INSUFFICIENT_SAMPLES" in str(exc_info.value)


# ── TEST 6: Feature Validation & Data Leakage Prevention ───────────────────────
def test_feature_validation_and_target_leakage(temp_ml_engine, sample_classification_df):
    # 1. Target leakage: target explicitly passed in feature list
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(
            sample_classification_df,
            target="label",
            features=["feat_num1", "label", "feat_num2"],
        )
    assert "TARGET_LEAKAGE" in str(exc_info.value)

    # 2. Nonexistent feature in list
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(
            sample_classification_df,
            target="label",
            features=["feat_num1", "phantom_feature"],
        )
    assert "COLUMN_NOT_FOUND" in str(exc_info.value)

    # 3. Duplicate features in selection (must deduplicate cleanly)
    result = temp_ml_engine.train_model(
        sample_classification_df,
        target="label",
        features=["feat_num1", "feat_num2", "feat_num1"],
    )
    assert result["features"] == ["feat_num1", "feat_num2"]


# ── TEST 7: Unsupported Algorithm / Task Rejection ─────────────────────────────
def test_unsupported_algorithm_and_task(temp_ml_engine, sample_classification_df):
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(
            sample_classification_df,
            target="label",
            task="classification",
            algorithm="quantum_super_neural_net",
        )
    assert "UNSUPPORTED_ALGORITHM" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.train_model(
            sample_classification_df,
            target="label",
            task="unsupported_quantum_task",
        )
    assert "UNSUPPORTED_TASK" in str(exc_info.value)


# ── TEST 8: Model Artifact Persistence & Live Inference ────────────────────────
def test_model_persistence_and_prediction(temp_ml_engine, sample_classification_df):
    train_res = temp_ml_engine.train_model(
        df=sample_classification_df,
        target="label",
        task="classification",
        algorithm="decision_tree",
    )
    model_id = train_res["model_id"]

    # Verify model artifact exists in models directory
    model_file = os.path.join(temp_ml_engine.models_dir, f"{model_id}.joblib")
    assert os.path.exists(model_file)

    # Test inference on single sample dict
    sample_input = [{
        "feat_num1": 11.2,
        "feat_num2": 45.0,
        "feat_cat": "cat_a",
    }]
    pred_res = temp_ml_engine.predict_with_model(model_id, sample_input)
    assert pred_res["model_id"] == model_id
    assert pred_res["prediction_count"] == 1
    assert pred_res["predictions"][0] in ["class_0", "class_1"]

    # Test schema mismatch: missing required feature
    invalid_input = [{
        "feat_num1": 11.2,
        # missing feat_num2 and feat_cat
    }]
    with pytest.raises(ValueError) as exc_info:
        temp_ml_engine.predict_with_model(model_id, invalid_input)
    assert "MODEL_SCHEMA_MISMATCH" in str(exc_info.value)


# ── TEST 9: Non-Mutation Invariant & JSON Safety ────────────────────────────────
def test_zero_mutation_and_json_safety(temp_ml_engine, sample_regression_df):
    df_copy = sample_regression_df.copy(deep=True)

    # Insert infinite floats in a non-target column to verify JSON sanitization
    sample_regression_df.loc[0, "feature_x1"] = np.nan
    sample_regression_df.loc[1, "feature_x2"] = np.inf

    result = temp_ml_engine.train_model(
        df=sample_regression_df,
        target="continuous_target",
        task="regression",
        algorithm="linear_regression",
    )

    # Ensure result is completely serializable to standard JSON
    import json
    json_str = json.dumps(result)
    assert json_str is not None
    assert "NaN" not in json_str
    assert "Infinity" not in json_str

    # Ensure original DataFrame structure was not corrupted
    assert len(sample_regression_df) == len(df_copy)
    assert list(sample_regression_df.columns) == list(df_copy.columns)


# ── TEST 10: API Router Endpoints & Provenance Integration ─────────────────────
def test_ml_api_router_and_provenance(sample_classification_df):
    client = TestClient(app)

    # Register active dataset in dataset_registry
    ds_id = "ds_test_ml_p4"
    name = "test_ml_dataset.csv"
    prof = DatasetProfiler.profile(sample_classification_df, dataset_name=name)
    entry = DatasetEntry(
        dataset_id=ds_id,
        name=name,
        file_path=f"/tmp/{name}",
        format="csv",
        row_count=len(sample_classification_df),
        column_count=len(sample_classification_df.columns),
        knowledge_model=prof.model_dump(),
        loaded_at="2026-10-02T00:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = sample_classification_df.copy(deep=True)
    dataset_registry.set_active(ds_id)

    # 1. Test /api/ml/train (Success)
    resp = client.post("/api/ml/train", json={
        "dataset_id": "ds_test_ml_p4",
        "target": "label",
        "task": "classification",
        "algorithm": "decision_tree",
        "validation_strategy": "split",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCEEDED"
    assert "result" in data
    assert "provenance_event" in data
    prov = data["provenance_event"]
    assert prov["capability"] == "ML"
    assert prov["operation"] == "ML_TRAIN"
    model_id = data["result"]["model_id"]

    # 2. Test /api/ml/models (Listing)
    list_resp = client.get("/api/ml/models")
    assert list_resp.status_code == 200
    assert any(m["model_id"] == model_id for m in list_resp.json()["models"])

    # 3. Test /api/ml/models/{id}/predict
    pred_resp = client.post(f"/api/ml/models/{model_id}/predict", json={
        "rows": [{
            "feat_num1": 10.5,
            "feat_num2": 50.0,
            "feat_cat": "cat_b",
        }]
    })
    assert pred_resp.status_code == 200
    assert pred_resp.json()["status"] == "SUCCEEDED"
    assert len(pred_resp.json()["result"]["predictions"]) == 1

    # 4. Test /api/ml/train (Controlled Error 422 for missing target)
    err_resp = client.post("/api/ml/train", json={
        "dataset_id": "ds_test_ml_p4",
        "target": "nonexistent_target_column",
    })
    assert err_resp.status_code == 422
    err_data = err_resp.json()
    detail = err_data.get("detail", {})
    assert detail.get("error_code") == "COLUMN_NOT_FOUND"
