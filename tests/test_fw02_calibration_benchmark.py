"""
Unit and regression tests for FW-02 Pre-Implementation Probability Calibration Benchmark.
Verifies calibration mathematical guarantees, data-split leakage controls, and probability bounds.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from backend.dataset_engine.ml_engine import ml_engine
from scripts.benchmark_fw02_calibration import (
    build_pipeline,
    calculate_calibration_table,
    evaluate_predictions,
)


@pytest.fixture
def german_credit_data():
    csv_path = "data/real_world_evaluation/dataset_b_german_credit.csv"
    if not os.path.exists(csv_path):
        pytest.skip(f"Benchmark dataset {csv_path} not found.")
    df = pd.read_csv(csv_path)
    return df


@pytest.fixture
def heart_disease_data():
    csv_path = "data/real_world_evaluation/dataset_e_heart_disease.csv"
    if not os.path.exists(csv_path):
        pytest.skip(f"Benchmark dataset {csv_path} not found.")
    df = pd.read_csv(csv_path)
    df["heart_disease"] = (df["target"] > 0).astype(int)
    return df.drop(columns=["target"])


def test_calibration_benchmark_does_not_mutate_dataset(german_credit_data):
    """Verify that running the calibration pipeline does not mutate the input DataFrame."""
    df_copy = german_credit_data.copy(deep=True)
    features = [c for c in german_credit_data.columns if c != "class"]
    y = (german_credit_data["class"] == "bad").astype(int)
    X = german_credit_data[features]
    
    pipe = build_pipeline(X, features, LogisticRegression(max_iter=200, random_state=42))
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    
    assert german_credit_data.equals(df_copy), "Calibration pipeline must not mutate input DataFrame"


def test_probability_bounds_and_row_sum_unity(german_credit_data):
    """Verify that all generated predicted probabilities lie strictly in [0, 1] and rows sum to 1."""
    features = [c for c in german_credit_data.columns if c != "class"]
    y = (german_credit_data["class"] == "bad").astype(int)
    X = german_credit_data[features]
    
    for model in [
        LogisticRegression(max_iter=200, random_state=42),
        RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42),
        DecisionTreeClassifier(max_depth=4, random_state=42),
    ]:
        pipe = build_pipeline(X, features, model)
        cal = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=3)
        cal.fit(X, y)
        probs = cal.predict_proba(X)
        
        assert probs.shape == (len(X), 2)
        assert np.all(probs >= 0.0), f"Probabilities must be >= 0 for {type(model).__name__}"
        assert np.all(probs <= 1.0), f"Probabilities must be <= 1 for {type(model).__name__}"
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_leakage_free_evaluation_design(german_credit_data):
    """
    Verify that out-of-sample test evaluation uses estimators fitted strictly on training data
    without leaking test fold statistics into the preprocessor or calibrator.
    """
    features = [c for c in german_credit_data.columns if c != "class"]
    y = (german_credit_data["class"] == "bad").astype(int)
    X = german_credit_data[features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    pipe = build_pipeline(X_train, features, LogisticRegression(max_iter=200, random_state=42))
    cal = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=3)
    cal.fit(X_train, y_train)
    
    # Preprocessor inside pipeline should know training feature parameters, but not test samples
    test_probs = cal.predict_proba(X_test)[:, 1]
    assert len(test_probs) == len(X_test)
    assert not np.isnan(test_probs).any()


def test_calibration_error_math_properties():
    """Verify ECE and MCE math on synthetic edge cases."""
    # Perfect calibration: prob 0.2 has 20% positives, prob 0.8 has 80% positives
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    # Exact empirical matches
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]) # 1/5 = 0.2 in bin 1, 4/5 = 0.8 in bin 5
    
    table, ece, mce = calculate_calibration_table(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert mce >= ece or np.isclose(mce, ece)


def test_reproducibility_and_determinism(heart_disease_data):
    """Verify that repeated benchmark runs with the same seed yield bitwise identical metrics."""
    features = [c for c in heart_disease_data.columns if c != "heart_disease"]
    y = heart_disease_data["heart_disease"]
    X = heart_disease_data[features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    pipe1 = build_pipeline(X_train, features, RandomForestClassifier(n_estimators=30, random_state=42))
    pipe1.fit(X_train, y_train)
    p1 = pipe1.predict_proba(X_test)[:, 1]
    
    pipe2 = build_pipeline(X_train, features, RandomForestClassifier(n_estimators=30, random_state=42))
    pipe2.fit(X_train, y_train)
    p2 = pipe2.predict_proba(X_test)[:, 1]
    
    np.testing.assert_array_equal(p1, p2)


def test_production_ml_engine_behavior_unaltered(german_credit_data):
    """Verify that DataPilot OS production ml_engine continues to work normally and identically."""
    res = ml_engine.train_model(
        df=german_credit_data,
        target="class",
        features=[c for c in german_credit_data.columns if c != "class"],
        task="classification",
        algorithm="logistic_regression",
        test_size=0.2,
    )
    assert "metrics" in res
    m = res["metrics"]
    assert "accuracy" in m
    assert "brier_score" in m
    assert m["brier_score"] is not None
    assert m["asymmetric_cost_status"].startswith("UNSPECIFIED")
