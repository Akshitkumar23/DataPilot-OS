"""
DataPilot OS — FW-02 Production Implementation Test Suite.
Verifies Dynamic Asymmetric Cost-Matrix Optimization, Generalized Bayes Thresholds,
Model-Aware Calibration Routing, Zero Test Leakage, Deterministic Tie-Breaking,
Real-World Dataset Savings, API Endpoints, and Provenance Store.
"""

import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.dataset_engine.ml_engine import ml_engine
from backend.persistence.provenance_store import provenance_store
from backend.main import app

client = TestClient(app)

GERMAN_CREDIT_PATH = "data/real_world_evaluation/dataset_b_german_credit.csv"
HEART_DISEASE_PATH = "data/real_world_evaluation/dataset_e_heart_disease.csv"
MUSHROOM_PATH = "data/real_world_evaluation/dataset_h_mushroom_high_cardinality.csv"


# =========================================================================
# 1. MATHEMATICAL FORMULATION & VALIDATION
# =========================================================================

def test_generalized_bayes_threshold_calculation():
    """Verify that generalized 4-cell Bayes threshold reduces to 2-cell when TP=TN=0."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)
    # Scenario B: FP=1, FN=5, TP=0, TN=0 -> tau* = 1 / (1 + 5) = 1/6 ~= 0.1667
    res = ml_engine.optimize_classification_threshold(
        df=df,
        target="class",
        positive_class="bad",
        false_positive_cost=1.0,
        false_negative_cost=5.0,
        true_positive_cost=0.0,
        true_negative_cost=0.0,
        algorithm="logistic_regression",
    )
    assert res["status"] == "OPTIMIZATION_SUCCESSFUL"
    assert np.isclose(res["cost_policy"]["theoretical_bayes_threshold"], 1.0 / 6.0, atol=1e-4)
    assert np.isclose(res["cost_policy"]["loss_ratio"], 5.0)

    # 4-cell with non-zero benefit/cost: FP=2, FN=8, TP=-1, TN=-0.5
    # denom = (2 - (-0.5)) + (8 - (-1)) = 2.5 + 9 = 11.5
    # num = 2 - (-0.5) = 2.5
    # tau* = 2.5 / 11.5 ~= 0.21739
    res_4cell = ml_engine.optimize_classification_threshold(
        df=df,
        target="class",
        positive_class="bad",
        false_positive_cost=2.0,
        false_negative_cost=8.0,
        true_positive_cost=-1.0,
        true_negative_cost=-0.5,
        algorithm="logistic_regression",
    )
    expected_bayes = 2.5 / 11.5
    assert np.isclose(res_4cell["cost_policy"]["theoretical_bayes_threshold"], expected_bayes, atol=1e-4)


def test_invalid_cost_parameters_rejected():
    """Verify strict validation of cost parameters: positive costs and rationality."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)

    # Negative / zero FP cost
    with pytest.raises(ValueError, match="INVALID_COST_VALUES"):
        ml_engine.optimize_classification_threshold(
            df=df, target="class", positive_class="bad",
            false_positive_cost=-1.0, false_negative_cost=5.0
        )

    with pytest.raises(ValueError, match="INVALID_COST_VALUES"):
        ml_engine.optimize_classification_threshold(
            df=df, target="class", positive_class="bad",
            false_positive_cost=0.0, false_negative_cost=5.0
        )

    # Irrational cost where errors cost less than correct classifications:
    # (FP - TN) + (FN - TP) <= 0
    # Let FP=1, TN=5 -> FP - TN = -4; FN=1, TP=5 -> FN - TP = -4; sum = -8 <= 0
    with pytest.raises(ValueError, match="INVALID_COST_RATIONALITY"):
        ml_engine.optimize_classification_threshold(
            df=df, target="class", positive_class="bad",
            false_positive_cost=1.0, false_negative_cost=1.0,
            true_positive_cost=5.0, true_negative_cost=5.0
        )


def test_missing_or_invalid_classes_rejected():
    """Verify missing positive class and multiclass targets are cleanly rejected."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)

    # Non-existent positive class
    with pytest.raises(ValueError, match="MISSING_POSITIVE_CLASS"):
        ml_engine.optimize_classification_threshold(
            df=df, target="class", positive_class="definitely_not_a_class",
            false_positive_cost=1.0, false_negative_cost=5.0
        )

    # Multiclass dataset: create 3 classes
    df_multi = df.copy()
    df_multi.loc[:50, "class"] = "third_class"
    with pytest.raises(ValueError, match="MULTICLASS_NOT_SUPPORTED"):
        ml_engine.optimize_classification_threshold(
            df=df_multi, target="class", positive_class="bad",
            false_positive_cost=1.0, false_negative_cost=5.0
        )


def test_symmetric_costs_emit_warning():
    """Verify that symmetric costs issue an EQUAL_COSTS_INFO warning."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)
    res = ml_engine.optimize_classification_threshold(
        df=df,
        target="class",
        positive_class="bad",
        false_positive_cost=1.0,
        false_negative_cost=1.0,
        algorithm="logistic_regression",
    )
    warning_codes = [w["code"] for w in res["warnings"]]
    assert "EQUAL_COSTS_INFO" in warning_codes


# =========================================================================
# 2. CALIBRATION ROUTING & SAMPLE GUARDS
# =========================================================================

def test_model_aware_calibration_routing():
    """Verify that RF and DT receive sigmoid calibration by default, while LR remains uncalibrated."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)

    # Random Forest -> Auto should select Sigmoid
    res_rf = ml_engine.optimize_classification_threshold(
        df=df, target="class", positive_class="bad",
        false_positive_cost=1.0, false_negative_cost=5.0,
        algorithm="random_forest", calibration_policy="auto"
    )
    assert res_rf["calibration"]["applied"] is True
    assert res_rf["calibration"]["method"] == "sigmoid"

    # Decision Tree -> Auto should select Sigmoid
    res_dt = ml_engine.optimize_classification_threshold(
        df=df, target="class", positive_class="bad",
        false_positive_cost=1.0, false_negative_cost=5.0,
        algorithm="decision_tree", calibration_policy="auto"
    )
    assert res_dt["calibration"]["applied"] is True
    assert res_dt["calibration"]["method"] == "sigmoid"

    # Logistic Regression -> Auto should select None (well-calibrated via log-odds)
    res_lr = ml_engine.optimize_classification_threshold(
        df=df, target="class", positive_class="bad",
        false_positive_cost=1.0, false_negative_cost=5.0,
        algorithm="logistic_regression", calibration_policy="auto"
    )
    assert res_lr["calibration"]["applied"] is False
    assert res_lr["calibration"]["method"] == "none"


def test_isotonic_sample_size_guard():
    """Verify that Isotonic calibration is rejected when N_train < 2000."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)  # 1000 rows, train is ~800
    assert len(df) < 2000

    with pytest.raises(ValueError, match="ISOTONIC_SAMPLE_SIZE_VIOLATION"):
        ml_engine.optimize_classification_threshold(
            df=df, target="class", positive_class="bad",
            false_positive_cost=1.0, false_negative_cost=5.0,
            algorithm="random_forest", calibration_policy="isotonic"
        )


def test_discrete_decision_tree_warning():
    """Verify warning when optimizing uncalibrated Decision Tree."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)
    res = ml_engine.optimize_classification_threshold(
        df=df, target="class", positive_class="bad",
        false_positive_cost=1.0, false_negative_cost=5.0,
        algorithm="decision_tree", calibration_policy="none"
    )
    warning_codes = [w["code"] for w in res["warnings"]]
    assert "DISCRETE_STEP_PROBABILITY_WARNING" in warning_codes


# =========================================================================
# 3. ZERO DATA LEAKAGE & OUT-OF-SAMPLE INTEGRITY
# =========================================================================

def test_zero_test_leakage_and_savings():
    """Verify that test evaluation is performed on pristine 20% held-out test split."""
    df = pd.read_csv(GERMAN_CREDIT_PATH)
    res = ml_engine.optimize_classification_threshold(
        df=df,
        target="class",
        positive_class="bad",
        false_positive_cost=1.0,
        false_negative_cost=5.0,
        algorithm="random_forest",
        test_size=0.2,
    )

    # 1000 total rows -> 800 train, 200 test
    assert res["sample_counts"]["train"] == 800
    assert res["sample_counts"]["test"] == 200

    # Test set confusion matrix sum equals test sample count
    b_cm = res["business_decision_quality"]["baseline_confusion_matrix"]
    o_cm = res["business_decision_quality"]["optimized_confusion_matrix"]
    assert b_cm["tn"] + b_cm["fp"] + b_cm["fn"] + b_cm["tp"] == 200
    assert o_cm["tn"] + o_cm["fp"] + o_cm["fn"] + o_cm["tp"] == 200

    # Business decision savings
    base_cost = res["business_decision_quality"]["baseline_cost"]
    opt_cost = res["business_decision_quality"]["optimized_cost"]
    abs_red = res["business_decision_quality"]["absolute_cost_reduction"]
    pct_red = res["business_decision_quality"]["percentage_cost_reduction"]

    assert np.isclose(base_cost - opt_cost, abs_red, atol=1e-3)
    assert opt_cost < base_cost, "Optimization must strictly reduce business cost on German Credit test set"
    assert pct_red > 15.0, f"Expected >15% cost reduction on German Credit, got {pct_red}%"

    # Threshold stability across folds
    stability = res["threshold_selection"]["stability"]
    assert len(stability["folds"]) == 5
    assert stability["min"] <= stability["median"] <= stability["max"]
    assert stability["std"] >= 0.0

    # 50-point cost curve present
    assert len(res["cost_curve"]) == 50
    assert all("threshold" in pt and "cost" in pt for pt in res["cost_curve"])


# =========================================================================
# 4. REAL-WORLD DATASETS BENCHMARK
# =========================================================================

def test_heart_disease_threshold_optimization():
    """Verify cost reduction on real-world Heart Disease dataset."""
    df = pd.read_csv(HEART_DISEASE_PATH)
    # Target > 0 indicates disease presence
    df["has_disease"] = (df["target"] > 0).astype(int)

    # Medical diagnosis scenario: Missing disease (FN) is 5x more costly than false alarm (FP)
    res = ml_engine.optimize_classification_threshold(
        df=df,
        target="has_disease",
        positive_class=1,
        false_positive_cost=1.0,
        false_negative_cost=5.0,
        algorithm="random_forest",
        test_size=0.2,
    )
    assert res["status"] == "OPTIMIZATION_SUCCESSFUL"
    # Optimal threshold shifts lower than 0.50 to catch more true positives
    assert res["threshold_selection"]["optimized_threshold"] < 0.50
    assert res["business_decision_quality"]["optimized_cost"] <= res["business_decision_quality"]["baseline_cost"]
    assert res["business_decision_quality"]["percentage_cost_reduction"] >= 0.0


def test_mushroom_threshold_optimization():
    """Verify cost reduction on real-world Mushroom dataset."""
    df = pd.read_csv(MUSHROOM_PATH)
    # Poisonous ('p') vs Edible ('e'). Eating poisonous mushroom (FN) is 10x worse than discarding edible (FP)
    res = ml_engine.optimize_classification_threshold(
        df=df,
        target="class",
        positive_class="p",
        false_positive_cost=1.0,
        false_negative_cost=10.0,
        algorithm="random_forest",
        test_size=0.2,
    )
    assert res["status"] == "OPTIMIZATION_SUCCESSFUL"
    assert res["business_decision_quality"]["optimized_cost"] <= res["business_decision_quality"]["baseline_cost"]


# =========================================================================
# 5. API ENDPOINT & PROVENANCE INTEGRATION
# =========================================================================

def test_api_optimize_threshold_success(monkeypatch):
    """Verify POST /api/ml/optimize-threshold produces successful response and records provenance."""
    # Ensure active dataset is registered
    from backend.core.dataset_registry import dataset_registry, DatasetEntry
    df = pd.read_csv(GERMAN_CREDIT_PATH)
    ds_id = "german_credit_test"
    entry = DatasetEntry(
        dataset_id=ds_id,
        name="German Credit Test",
        file_path=GERMAN_CREDIT_PATH,
        format="csv",
        columns=list(df.columns),
        row_count=len(df),
        column_count=len(df.columns),
        knowledge_model={"columns": list(df.columns)},
        loaded_at="2026-10-04T00:00:00Z",
        file_hash="mock_hash_german",
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = df.copy()
    dataset_registry.set_active(ds_id)

    payload = {
        "dataset_id": ds_id,
        "target": "class",
        "positive_class": "bad",
        "false_positive_cost": 1.0,
        "false_negative_cost": 5.0,
        "algorithm": "random_forest",
        "calibration_policy": "auto",
        "grid_resolution": 100,
        "test_size": 0.2,
        "cv_folds": 5
    }

    response = client.post("/api/ml/optimize-threshold", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCEEDED"
    assert data["dataset_id"] == ds_id
    assert "result" in data
    assert "provenance_event" in data

    prov = data["provenance_event"]
    assert prov["operation"] == "ML_THRESHOLD_OPTIMIZATION"
    assert prov["capability"] == "ML"
    assert prov["summary"]["algorithm"] == "random_forest"
    assert prov["summary"]["percentage_cost_reduction"] > 0.0

    # Verify event in provenance store
    events = provenance_store.get_events(dataset_id=ds_id)
    opt_events = [e for e in events if e.operation == "ML_THRESHOLD_OPTIMIZATION"]
    assert len(opt_events) >= 1
    latest_event = opt_events[-1]
    assert latest_event.status == "SUCCESS"
    assert latest_event.parameters["false_negative_cost"] == 5.0


def test_api_optimize_threshold_validation_error():
    """Verify POST /api/ml/optimize-threshold returns 422 on invalid cost policy and records failure."""
    from backend.core.dataset_registry import dataset_registry, DatasetEntry
    df = pd.read_csv(GERMAN_CREDIT_PATH)
    ds_id = "german_credit_invalid_cost"
    entry = DatasetEntry(
        dataset_id=ds_id,
        name="German Credit Invalid Cost",
        file_path=GERMAN_CREDIT_PATH,
        format="csv",
        columns=list(df.columns),
        row_count=len(df),
        column_count=len(df.columns),
        knowledge_model={"columns": list(df.columns)},
        loaded_at="2026-10-04T00:00:00Z",
        file_hash="mock_hash_german_inv",
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = df.copy()
    dataset_registry.set_active(ds_id)

    # Invalid negative cost
    payload = {
        "dataset_id": ds_id,
        "target": "class",
        "positive_class": "bad",
        "false_positive_cost": -10.0,
        "false_negative_cost": 5.0,
    }

    response = client.post("/api/ml/optimize-threshold", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error_code"] == "INVALID_COST_VALUES"

    # Verify failed event logged in provenance store
    events = provenance_store.get_events(dataset_id=ds_id)
    fail_events = [e for e in events if e.operation == "ML_THRESHOLD_OPTIMIZATION" and e.status == "FAILED"]
    assert len(fail_events) >= 1
    assert fail_events[-1].error["error_code"] == "INVALID_COST_VALUES"
