"""
Tests for statistical assumption gating, sample size thresholds, zero-variance checks,
and supervised ML feasibility contracts (Phase 6).
"""

import pandas as pd
import pytest

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.feasibility_rules import FeasibilityRulesEngine
from backend.ds_engine.semantic_feasibility import SemanticFeasibilityValidator


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "group_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "group_b": [2.5, 3.5, 4.5, 5.5, 6.5],
        "constant_col": [5.0, 5.0, 5.0, 5.0, 5.0],
        "tiny_col": [1.0, 2.0, None, None, None],
        "cat_binary": ["yes", "no", "yes", "no", "yes"],
        "cat_uniform": ["A", "A", "A", "A", "A"],
        "target": [0, 1, 0, 1, 0],
    })


def test_statistical_test_requires_test_type():
    res = FeasibilityRulesEngine.check_statistical_assumptions(
        "statistical_test",
        {"columns": ["group_a", "group_b"]},
        None,
    )
    assert res is not None
    assert not res.is_feasible
    assert res.error_code == "TEST_TYPE_REQUIRED"


def test_statistical_test_requires_at_least_two_columns():
    res = FeasibilityRulesEngine.check_statistical_assumptions(
        "statistical_test",
        {"test_type": "independent_ttest", "columns": ["group_a"]},
        None,
    )
    assert res is not None
    assert not res.is_feasible
    assert res.error_code == "INSUFFICIENT_COLUMNS"


def test_statistical_test_rejects_missing_column(sample_df):
    res = FeasibilityRulesEngine.check_statistical_assumptions(
        "statistical_test",
        {"test_type": "independent_ttest", "columns": ["group_a", "nonexistent"]},
        sample_df,
    )
    assert res is not None
    assert not res.is_feasible
    assert res.error_code == "COLUMN_NOT_FOUND"


def test_independent_ttest_rejects_insufficient_sample_size(sample_df):
    res = FeasibilityRulesEngine.check_statistical_assumptions(
        "statistical_test",
        {"test_type": "independent_ttest", "columns": ["group_a", "tiny_col"]},
        sample_df,
    )
    assert res is not None
    assert not res.is_feasible
    assert res.error_code == "INSUFFICIENT_SAMPLE_SIZE"


def test_independent_ttest_rejects_zero_variance(sample_df):
    res = FeasibilityRulesEngine.check_statistical_assumptions(
        "statistical_test",
        {"test_type": "independent_ttest", "columns": ["group_a", "constant_col"]},
        sample_df,
    )
    assert res is not None
    assert not res.is_feasible
    assert res.error_code == "ZERO_VARIANCE"


def test_chi_square_rejects_degenerate_contingency_table(sample_df):
    res = FeasibilityRulesEngine.check_statistical_assumptions(
        "statistical_test",
        {"test_type": "chi_square_independence", "columns": ["cat_binary", "cat_uniform"]},
        sample_df,
    )
    assert res is not None
    assert not res.is_feasible
    assert res.error_code == "DEGENERATE_CONTINGENCY_TABLE"


def test_valid_ttest_passes(sample_df):
    res = FeasibilityRulesEngine.check_statistical_assumptions(
        "statistical_test",
        {"test_type": "independent_ttest", "columns": ["group_a", "group_b"]},
        sample_df,
    )
    assert res is None


def test_train_model_requires_target_and_features():
    res = FeasibilityRulesEngine.check_supervised_ml_contract(
        "train_model",
        {"target_column": "", "feature_columns": []},
        None,
    )
    assert res is not None
    assert res.error_code == "TARGET_COLUMN_REQUIRED"

    res = FeasibilityRulesEngine.check_supervised_ml_contract(
        "train_model",
        {"target_column": "target", "feature_columns": []},
        None,
    )
    assert res is not None
    assert res.error_code == "FEATURE_COLUMNS_REQUIRED"


def test_train_model_detects_target_in_features():
    res = FeasibilityRulesEngine.check_supervised_ml_contract(
        "train_model",
        {
            "target_column": "target",
            "feature_columns": ["group_a", "target"],
            "prediction_context": "prospective_prediction",
        },
        None,
    )
    assert res is not None
    assert res.error_code == "TARGET_LEAKAGE_DETECTED"


def test_train_model_requires_prediction_context():
    res = FeasibilityRulesEngine.check_supervised_ml_contract(
        "train_model",
        {
            "target_column": "target",
            "feature_columns": ["group_a", "group_b"],
        },
        None,
    )
    assert res is not None
    assert res.error_code == "PREDICTION_CONTEXT_REQUIRED"
