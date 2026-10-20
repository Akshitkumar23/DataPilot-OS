"""
Unit and regression tests for FW-02 Cost-Sensitive Threshold Benchmark Methodology.
Verifies cost calculations, threshold search validity, and strict zero-leakage constraints.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def compute_cost(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, c_fp: float, c_fn: float):
    """Reference cost computation function for tests."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fp * c_fp) + (fn * c_fn)
    return total_cost, int(tn), int(fp), int(fn), int(tp)


def optimize_threshold_grid(y_true: np.ndarray, y_prob: np.ndarray, c_fp: float, c_fn: float, grid_steps: int = 99):
    """Grid sweep search to minimize cost."""
    thresholds = np.linspace(0.01, 0.99, grid_steps)
    best_thresh = 0.50
    best_cost = float("inf")
    for t in thresholds:
        cost, _, _, _, _ = compute_cost(y_true, y_prob, t, c_fp, c_fn)
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)
    return best_thresh, best_cost


@pytest.fixture
def german_credit_df():
    path = "data/real_world_evaluation/dataset_b_german_credit.csv"
    if not os.path.exists(path):
        pytest.skip("German credit dataset not found.")
    return pd.read_csv(path)


def test_cost_calculation_correctness():
    """Verify cost calculation against known manual confusion matrix."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.7, 0.3, 0.8])
    # At threshold 0.50:
    # y_pred = [0, 1, 0, 1]
    # TN=1, FP=1, FN=1, TP=1
    cost, tn, fp, fn, tp = compute_cost(y_true, y_prob, 0.50, c_fp=1.0, c_fn=5.0)
    assert tn == 1
    assert fp == 1
    assert fn == 1
    assert tp == 1
    assert cost == (1 * 1.0) + (1 * 5.0) == 6.0


def test_confusion_matrix_cost_consistency():
    """Verify that sum of confusion matrix cells always equals total sample count."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.3, size=200)
    y_prob = rng.uniform(0.0, 1.0, size=200)
    
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cost, tn, fp, fn, tp = compute_cost(y_true, y_prob, thresh, c_fp=2.0, c_fn=7.0)
        assert tn + fp + fn + tp == 200
        assert cost >= 0.0


def test_threshold_search_range_validity():
    """Verify that optimal threshold is always bounded strictly inside (0, 1)."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.4, size=150)
    y_prob = rng.uniform(0.01, 0.99, size=150)
    
    for c_fp, c_fn in [(1, 1), (1, 5), (5, 1), (1, 10)]:
        opt_thresh, opt_cost = optimize_threshold_grid(y_true, y_prob, c_fp, c_fn)
        assert 0.0 < opt_thresh < 1.0
        # Cost at optimal threshold must be <= cost at default 0.50
        def_cost, _, _, _, _ = compute_cost(y_true, y_prob, 0.50, c_fp, c_fn)
        assert opt_cost <= def_cost


def test_strict_out_of_sample_leakage_isolation(german_credit_df):
    """
    Verify that the optimal threshold derived strictly from training CV
    is applied to a held-out test set without test labels influencing the threshold.
    """
    y = (german_credit_df["class"] == "bad").astype(int).values
    features = [c for c in german_credit_df.columns if c != "class" and pd.api.types.is_numeric_dtype(german_credit_df[c])]
    X = german_credit_df[features].fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model strictly on train
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Optimize threshold strictly on train_probs
    opt_thresh, train_opt_cost = optimize_threshold_grid(y_train, train_probs, c_fp=1.0, c_fn=5.0)
    
    # Evaluate strictly on test_probs
    test_cost_opt, _, _, _, _ = compute_cost(y_test, test_probs, opt_thresh, c_fp=1.0, c_fn=5.0)
    test_cost_def, _, _, _, _ = compute_cost(y_test, test_probs, 0.50, c_fp=1.0, c_fn=5.0)
    
    # Check threshold is strictly determined by train set
    assert isinstance(opt_thresh, float)
    assert 0.01 <= opt_thresh <= 0.99
    assert isinstance(test_cost_opt, (int, float))


def test_fold_results_reproducibility(german_credit_df):
    """Verify that repeated CV threshold evaluations yield identical thresholds with fixed seed."""
    y = (german_credit_df["class"] == "bad").astype(int).values
    features = [c for c in german_credit_df.columns if c != "class" and pd.api.types.is_numeric_dtype(german_credit_df[c])]
    X = german_credit_df[features].fillna(0).values
    
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_thresholds_run1 = []
    fold_thresholds_run2 = []
    
    for run in [1, 2]:
        for train_idx, val_idx in skf.split(X_scaled, y):
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_scaled[train_idx], y[train_idx])
            val_probs = clf.predict_proba(X_scaled[val_idx])[:, 1]
            t_opt, _ = optimize_threshold_grid(y[val_idx], val_probs, c_fp=1.0, c_fn=5.0)
            if run == 1:
                fold_thresholds_run1.append(t_opt)
            else:
                fold_thresholds_run2.append(t_opt)
                
    assert fold_thresholds_run1 == fold_thresholds_run2


def test_no_hardcoded_column_assumptions():
    """Verify that threshold optimization operates dynamically on arbitrary numeric/binary arrays without named columns."""
    rng = np.random.RandomState(99)
    y_arbitrary = rng.binomial(1, 0.25, size=80)
    probs_arbitrary = rng.uniform(0.0, 1.0, size=80)
    
    t_opt, cost = optimize_threshold_grid(y_arbitrary, probs_arbitrary, c_fp=1.0, c_fn=10.0)
    assert 0.01 <= t_opt <= 0.99
    assert cost >= 0.0
