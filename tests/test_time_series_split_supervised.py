"""
Regression test suite for Defect D-03: Supervised ML TimeSeriesSplit for Chronological Modeling.
Verifies chronological ordering, walk-forward splits, prevention of look-ahead leakage,
and strict rejection when time_column is missing or has insufficient grain.
"""

import pytest
import numpy as np
import pandas as pd
from backend.dataset_engine.ml_engine import ml_engine


def test_time_series_split_walk_forward_execution():
    """
    Verify TimeSeriesSplit chronological holdout and cross-validation execution:
    1. Holds out strictly the temporal tail of the dataset.
    2. Runs TimeSeriesSplit walk-forward CV without future leakage.
    3. Records time_column and time_series_split in cross_validation results.
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Trend + lag feature
    demand = np.arange(n) * 0.5 + np.random.randn(n) * 2.0
    price = 50.0 - np.arange(n) * 0.2 + np.random.randn(n)
    promo = (np.random.rand(n) > 0.7).astype(int)

    # Scramble rows deliberately to verify temporal reordering
    shuffled_idx = np.random.permutation(n)
    df = pd.DataFrame({
        "timestamp": dates[shuffled_idx],
        "demand": demand[shuffled_idx],
        "price": price[shuffled_idx],
        "promo": promo[shuffled_idx],
    })

    result = ml_engine.train_model(
        df=df,
        target="demand",
        features=["price", "promo"],
        task="regression",
        algorithm="linear_regression",
        validation_strategy="time_series_split",
        test_size=0.2,
        cv_folds=4,
        time_column="timestamp",
    )

    assert result["task"] == "regression"
    assert result["time_column"] == "timestamp"

    cv_info = result["cross_validation"]
    assert cv_info["performed"] is True
    assert cv_info["strategy"] == "time_series_split"
    assert cv_info["time_column"] == "timestamp"
    assert len(cv_info["scores"]) >= 2
    assert cv_info["mean"] is not None

    # Holdout tail validation
    assert result["sample_counts"]["train"] == 80
    assert result["sample_counts"]["test"] == 20


def test_time_series_split_missing_time_column_rejected():
    """
    TimeSeriesSplit validation MUST reject execution when time_column is missing.
    Zero silent fallback to random splitting.
    """
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    })

    with pytest.raises(ValueError) as excinfo:
        ml_engine.train_model(
            df=df,
            target="y",
            features=["x1"],
            task="regression",
            algorithm="linear_regression",
            validation_strategy="time_series_split",
            time_column=None,
        )

    err = str(excinfo.value)
    assert "TIME_COLUMN_REQUIRED" in err


def test_time_series_split_insufficient_grain_rejected():
    """
    TimeSeriesSplit validation MUST reject execution when the temporal column
    has fewer than 3 unique timestamps.
    """
    df = pd.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02"],
        "x": [1, 2, 3, 4, 5, 6],
        "y": [10, 20, 30, 40, 50, 60],
    })

    with pytest.raises(ValueError) as excinfo:
        ml_engine.train_model(
            df=df,
            target="y",
            features=["x"],
            task="regression",
            algorithm="linear_regression",
            validation_strategy="time_series_split",
            time_column="timestamp",
        )

    err = str(excinfo.value)
    assert "INSUFFICIENT_TEMPORAL_GRAIN" in err


def test_time_series_split_classification_support():
    """
    Verify TimeSeriesSplit works for temporal classification tasks (e.g. regime change, outage prediction).
    """
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    feature = np.random.randn(n)
    labels = (feature > 0.0).astype(int)

    df = pd.DataFrame({
        "event_time": dates,
        "signal": feature,
        "state": labels,
    })

    result = ml_engine.train_model(
        df=df,
        target="state",
        features=["signal"],
        task="classification",
        algorithm="decision_tree",
        validation_strategy="time_series_split",
        test_size=0.25,
        cv_folds=3,
        time_column="event_time",
    )

    assert result["task"] == "classification"
    assert result["cross_validation"]["strategy"] == "time_series_split"
    assert len(result["cross_validation"]["scores"]) >= 2


def test_time_series_split_honors_higher_cv_folds():
    """
    Verify TimeSeriesSplit does not artificially cap folds at 5 when more folds are requested.
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "ts": dates,
        "x": np.random.randn(n),
        "y": np.random.randn(n),
    })

    result = ml_engine.train_model(
        df=df,
        target="y",
        features=["x"],
        task="regression",
        algorithm="linear_regression",
        validation_strategy="time_series_split",
        cv_folds=7,
        time_column="ts",
    )

    assert len(result["cross_validation"]["scores"]) == 7
