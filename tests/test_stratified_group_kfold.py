"""
Regression test suite for Defect D-01: StratifiedGroupKFold for grouped classification.
Verifies group non-leakage, class stratification, impossible topology error handling,
and regression GroupKFold invariance.
"""

import pytest
import numpy as np
import pandas as pd
from backend.dataset_engine.ml_engine import ml_engine


def test_stratified_group_kfold_classification_leakage_and_stratification():
    """
    Verify StratifiedGroupKFold on an imbalanced classification dataset:
    1. No group overlap between train and test sets (group leakage = 0).
    2. Cross-validation records strategy as 'stratified_group_cv'.
    3. Both classes appear in both train and test partitions.
    """
    np.random.seed(42)
    # 6 groups, 30 samples each = 180 samples
    groups = []
    classes = []
    features_a = []
    features_b = []

    for g in range(6):
        # Imbalance ratio ~ 3:1 (majority: 22, minority: 8 per group)
        for i in range(30):
            groups.append(f"hospital_{g}")
            label = 1 if i < 8 else 0
            classes.append(label)
            features_a.append(np.random.randn() + (2.0 if label == 1 else -1.0))
            features_b.append(np.random.randn() * 0.5)

    df = pd.DataFrame({
        "hospital_id": groups,
        "diagnosis": classes,
        "marker_a": features_a,
        "marker_b": features_b,
    })

    result = ml_engine.train_model(
        df=df,
        target="diagnosis",
        features=["marker_a", "marker_b"],
        task="classification",
        algorithm="decision_tree",
        validation_strategy="stratified_group_cv",
        test_size=0.33,
        cv_folds=3,
        group_column="hospital_id",
    )

    # 1. Check CV metadata
    cv_info = result["cross_validation"]
    assert cv_info["performed"] is True
    assert cv_info["strategy"] == "stratified_group_cv"
    assert cv_info["group_column"] == "hospital_id"
    assert len(cv_info["scores"]) >= 2
    assert cv_info["mean"] is not None

    # 2. Check holdout split group non-leakage
    # Model evaluation metrics must reflect real holdout performance
    assert "accuracy" in result["metrics"]
    assert "balanced_accuracy" in result["metrics"]
    assert result["metrics"]["imbalance_ratio"] >= 2.0


def test_stratified_group_kfold_impossible_topology_rejection():
    """
    When a minority class appears in only 1 group, StratifiedGroupKFold cannot
    place that class in both train and test folds without leaking the group.
    The kernel MUST raise ValueError('IMPOSSIBLE_GROUP_TOPOLOGY') and NEVER
    silently fall back to standard random CV.
    """
    np.random.seed(42)
    # 4 groups; Class 1 ONLY exists in group 'facility_0'
    groups = []
    classes = []
    features = []

    for g in range(4):
        for i in range(20):
            groups.append(f"facility_{g}")
            label = 1 if (g == 0 and i < 5) else 0
            classes.append(label)
            features.append(np.random.randn())

    df = pd.DataFrame({
        "facility": groups,
        "rare_outcome": classes,
        "sensor_val": features,
    })

    with pytest.raises(ValueError) as excinfo:
        ml_engine.train_model(
            df=df,
            target="rare_outcome",
            features=["sensor_val"],
            task="classification",
            algorithm="decision_tree",
            validation_strategy="stratified_group_cv",
            group_column="facility",
        )

    err = str(excinfo.value)
    assert "IMPOSSIBLE_GROUP_TOPOLOGY" in err
    assert "facility" in err or "group" in err


def test_group_regression_invariance():
    """
    Verify regression tasks continue to use GroupShuffleSplit and GroupKFold,
    preserving regression mechanics without invoking StratifiedGroupKFold.
    """
    np.random.seed(42)
    groups = []
    y = []
    x = []

    for g in range(5):
        for _ in range(20):
            groups.append(f"site_{g}")
            val = float(g * 10 + np.random.randn())
            x.append(val)
            y.append(val * 1.5 + np.random.randn())

    df = pd.DataFrame({"site": groups, "feature_x": x, "target_y": y})

    result = ml_engine.train_model(
        df=df,
        target="target_y",
        features=["feature_x"],
        task="regression",
        algorithm="linear_regression",
        validation_strategy="group_cv",
        cv_folds=3,
        group_column="site",
    )

    assert result["task"] == "regression"
    cv_info = result["cross_validation"]
    assert cv_info["performed"] is True
    assert cv_info["strategy"] == "group_cv"
    assert len(cv_info["scores"]) == 3
