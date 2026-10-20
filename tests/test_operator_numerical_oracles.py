"""
DataPilot OS — Numerical Oracle Tests (Phase 7).
Validates operator calculations against independent mathematical ground truth
(scipy.stats, numpy, sklearn) and ensures strict enforcement of numerical invariants.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from backend.ds_engine.operator_contracts import (
    NumericalInvariantValidator,
    NumericalInvariantViolation,
)
from backend.ds_engine.typed_operators import (
    CorrelationOperator,
    GroupAggregationOperator,
    StatisticalTestOperator,
    PCADimensionalityReductionOperator,
    ClusteringOperator,
)
from backend.ds_engine.intent_schema import (
    AnalysisIntentV2,
    CorrelationSpec,
    TaskFamily,
)


@pytest.fixture
def numerical_oracle_df():
    np.random.seed(42)
    n = 100
    x = np.random.normal(50, 15, n)
    y = 2.5 * x + np.random.normal(0, 10, n)
    z = np.random.exponential(5, n)
    category = np.random.choice(["Alpha", "Beta", "Gamma"], n)
    return pd.DataFrame({
        "var_x": x,
        "var_y": y,
        "var_z": z,
        "category": category,
    })


def test_correlation_oracle_matches_scipy(numerical_oracle_df):
    """Verify Pearson & Spearman correlations match scipy.stats ground truth."""
    df = numerical_oracle_df
    # Ground truth Pearson
    r_xy_scipy, _ = stats.pearsonr(df["var_x"], df["var_y"])
    r_xz_scipy, _ = stats.pearsonr(df["var_x"], df["var_z"])

    op = CorrelationOperator()
    intent = AnalysisIntentV2(
        task_family=TaskFamily.CORRELATION_ANALYSIS,
        subjects=["var_x", "var_y", "var_z"],
        correlation_spec=CorrelationSpec(
            analysis_mode="matrix",
            relationship_form="linear",
            correlation_methods=["pearson"],
            chart_type="none",
            smoother="none",
            expected_direction="unspecified",
        ),
    )
    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    corr_df = res.dataframe
    assert corr_df is not None
    matrix = corr_df.set_index("feature") if "feature" in corr_df.columns else corr_df

    # Check against scipy oracle
    assert np.isclose(matrix.loc["var_x", "var_y"], r_xy_scipy, atol=1e-4)
    assert np.isclose(matrix.loc["var_x", "var_z"], r_xz_scipy, atol=1e-4)

    # Invariant checks
    NumericalInvariantValidator.validate_correlation_matrix(matrix)


def test_group_aggregation_oracle_matches_pandas(numerical_oracle_df):
    """Verify group aggregation operations (mean, sum, count) match pandas oracle."""
    df = numerical_oracle_df
    oracle_grouped = df.groupby("category")["var_x"].agg(["mean", "sum", "count"])

    op = GroupAggregationOperator()
    for agg in ("mean", "sum", "count"):
        intent = AnalysisIntentV2(
            task_family=TaskFamily.GROUP_AGGREGATION,
            grouping=["category"],
            subjects=["var_x"],
            statistical_requirements=[agg],
        )
        res = op.execute(df, intent)
        assert res.status == "SUCCEEDED"
        res_df = res.dataframe
        assert res_df is not None
        for cat in ["Alpha", "Beta", "Gamma"]:
            expected_val = oracle_grouped.loc[cat, agg]
            actual_val = res_df.loc[res_df["category"] == cat, f"var_x_{agg}"].values[0] if f"var_x_{agg}" in res_df.columns else res_df.loc[res_df["category"] == cat, "var_x"].values[0]
            assert np.isclose(actual_val, expected_val, atol=1e-4)


def test_statistical_test_ttest_oracle_matches_scipy():
    """Verify independent t-test t-statistic and p-value against scipy.stats."""
    np.random.seed(123)
    sample_1 = np.random.normal(10.0, 2.0, 40)
    sample_2 = np.random.normal(12.5, 2.5, 40)
    df = pd.DataFrame({"s1": sample_1, "s2": sample_2})

    scipy_res = stats.ttest_ind(sample_1, sample_2, equal_var=False)

    op = StatisticalTestOperator()
    intent = AnalysisIntentV2(
        task_family=TaskFamily.STATISTICAL_TEST,
        subjects=["s1", "s2"],
        arguments={"test_type": "independent_ttest", "columns": ["s1", "s2"], "alpha": 0.05},
    )
    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    t_stat = res.dataframe["statistic"].iloc[0]
    p_val = res.dataframe["p_value"].iloc[0]

    assert np.isclose(t_stat, scipy_res.statistic, atol=1e-4)
    assert np.isclose(p_val, scipy_res.pvalue, atol=1e-4)

    NumericalInvariantValidator.validate_statistical_test_result(t_stat, p_val)


def test_pca_oracle_matches_sklearn(numerical_oracle_df):
    """Verify PCA explained variance ratios match sklearn.decomposition.PCA."""
    from sklearn.preprocessing import StandardScaler
    df = numerical_oracle_df[["var_x", "var_y", "var_z"]]
    scaled = StandardScaler().fit_transform(df)
    sklearn_pca = PCA(n_components=2)
    sklearn_pca.fit(scaled)
    expected_ratios = sklearn_pca.explained_variance_ratio_.tolist()

    op = PCADimensionalityReductionOperator()
    intent = AnalysisIntentV2(
        task_family=TaskFamily.DIMENSIONALITY_REDUCTION,
        subjects=["var_x", "var_y", "var_z"],
        arguments={"n_components": 2},
    )
    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    actual_ratios = [rec["explained_variance_ratio"] for rec in res.data.get("records", [])]
    assert len(actual_ratios) == 2
    for actual, expected in zip(actual_ratios, expected_ratios):
        assert np.isclose(actual, expected, atol=1e-4)

    NumericalInvariantValidator.validate_pca_variance_ratios(actual_ratios)


def test_invariant_validator_rejects_asymmetric_correlation():
    asymmetric_df = pd.DataFrame(
        [[1.0, 0.8], [0.5, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    with pytest.raises(NumericalInvariantViolation, match="asymmetric"):
        NumericalInvariantValidator.validate_correlation_matrix(asymmetric_df)


def test_invariant_validator_rejects_out_of_bound_correlation():
    bad_bounds_df = pd.DataFrame(
        [[1.0, 1.5], [1.5, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    with pytest.raises(NumericalInvariantViolation, match="out of bounds"):
        NumericalInvariantValidator.validate_correlation_matrix(bad_bounds_df)


def test_invariant_validator_rejects_invalid_p_value():
    with pytest.raises(NumericalInvariantViolation, match="p-value"):
        NumericalInvariantValidator.validate_statistical_test_result(2.5, 1.25)

    with pytest.raises(NumericalInvariantViolation, match="p-value"):
        NumericalInvariantValidator.validate_statistical_test_result(2.5, -0.01)


def test_invariant_validator_rejects_negative_confusion_matrix():
    bad_cm = np.array([[10, -1], [2, 15]])
    with pytest.raises(NumericalInvariantViolation, match="cannot be negative"):
        NumericalInvariantValidator.validate_confusion_matrix(bad_cm)


def test_invariant_validator_rejects_ascending_pca_variance():
    bad_pca_ratios = [0.2, 0.8]  # Ascending order is invalid for PCA
    with pytest.raises(NumericalInvariantViolation, match="descending variance"):
        NumericalInvariantValidator.validate_pca_variance_ratios(bad_pca_ratios)
