"""Regression tests for explicit scientific-method contracts."""

import numpy as np
import pandas as pd

from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily
from backend.ds_engine.typed_operators import (
    ClusteringOperator,
    PCADimensionalityReductionOperator,
    StatisticalTestOperator,
    TimeSeriesForecastOperator,
)
from modules.capability_registry import capability_registry


def test_statistical_test_toolspec_normalizes_alias_but_keeps_explicit_test_choice():
    args = capability_registry.validate_tool_call(
        "statistical_test",
        {"metrics": ["sample_a", "sample_b"], "test_type": "t_test"},
    )
    assert args.columns == ["sample_a", "sample_b"]
    assert args.test_type == "independent_ttest"
    assert args.alpha == 0.05


def test_statistical_operator_never_guesses_a_test_from_column_types():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    output = StatisticalTestOperator().execute(
        df,
        AnalysisIntentV2(task_family=TaskFamily.STATISTICAL_TEST, subjects=["a", "b"]),
    )
    assert output.status == "FAILED"
    assert output.error_code == "TEST_TYPE_REQUIRED"


def test_one_way_anova_uses_declared_column_roles_and_reports_effect_size():
    df = pd.DataFrame({
        "score": [1.0, 1.5, 2.0, 4.0, 4.5, 5.0],
        "group": ["control", "control", "control", "treatment", "treatment", "treatment"],
    })
    output = StatisticalTestOperator().execute(
        df,
        AnalysisIntentV2(
            task_family=TaskFamily.STATISTICAL_TEST,
            arguments={"test_type": "one_way_anova", "columns": ["score", "group"], "alpha": 0.01},
        ),
    )
    assert output.status == "SUCCEEDED"
    row = output.data["records"][0]
    assert row["test_type"] == "one_way_anova"
    assert row["effect_size_name"] == "eta_squared"
    assert row["alpha"] == 0.01
    assert "Test selected explicitly: one_way_anova" in output.manifest.assumptions


def test_chi_square_warns_when_expected_cell_counts_are_small():
    df = pd.DataFrame({"source": ["A", "A", "B", "B"], "outcome": ["yes", "no", "yes", "no"]})
    output = StatisticalTestOperator().execute(
        df,
        AnalysisIntentV2(
            task_family=TaskFamily.STATISTICAL_TEST,
            arguments={"test_type": "chi_square_independence", "columns": ["source", "outcome"], "alpha": 0.05},
        ),
    )
    assert output.status == "SUCCEEDED"
    assert output.data["records"][0]["effect_size_name"] == "cramers_v"
    assert output.manifest.warnings


def test_reproducible_code_uses_the_actual_cluster_and_pca_parameters():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"x": rng.normal(size=24), "y": rng.normal(size=24), "z": rng.normal(size=24)})

    cluster_out = ClusteringOperator().execute(
        df,
        AnalysisIntentV2(task_family=TaskFamily.CLUSTERING, subjects=["x", "y"], arguments={"n_clusters": 2, "scaling": "standard"}),
    )
    assert cluster_out.status == "SUCCEEDED"
    assert "n_clusters = 2" in cluster_out.reproducible_code
    assert "n_clusters=3" not in cluster_out.reproducible_code

    pca_out = PCADimensionalityReductionOperator().execute(
        df,
        AnalysisIntentV2(task_family=TaskFamily.DIMENSIONALITY_REDUCTION, subjects=["x", "y", "z"], arguments={"n_components": 2}),
    )
    assert pca_out.status == "SUCCEEDED"
    assert "n_components = 2" in pca_out.reproducible_code
    assert "min(len(subjects), 5)" not in pca_out.reproducible_code


def test_forecast_requires_a_method_and_frequency_and_honours_table_only_output():
    df = pd.DataFrame({
        "observed_at": pd.date_range("2025-01-01", periods=8, freq="D"),
        "demand": [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 15.0],
    })
    operator = TimeSeriesForecastOperator()
    missing_contract = operator.execute(
        df,
        AnalysisIntentV2(
            task_family=TaskFamily.TIME_SERIES_FORECAST,
            target_column="demand",
            time_context={"time_column": "observed_at", "horizon": 2},
            arguments={},
        ),
    )
    assert missing_contract.status == "FAILED"
    assert missing_contract.error_code == "FORECAST_METHOD_REQUIRED"

    table_only = operator.execute(
        df,
        AnalysisIntentV2(
            task_family=TaskFamily.TIME_SERIES_FORECAST,
            target_column="demand",
            time_context={"time_column": "observed_at", "horizon": 2},
            arguments={"method": "naive_last_value", "frequency": "D", "chart_type": "none"},
        ),
    )
    assert table_only.status == "SUCCEEDED"
    assert table_only.plot_image_base64 is None
    assert table_only.plotly_spec is None
    assert table_only.manifest.assumptions[-1] == "No calibrated prediction interval was computed."


def test_cluster_toolspec_declares_scaling_and_operator_rejects_missing_persisted_contract():
    validated = capability_registry.validate_tool_call(
        "cluster_data", {"columns": ["x", "y"], "n_clusters": 2, "scaling": "standard"}
    )
    assert validated.scaling == "standard"

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    output = ClusteringOperator().execute(
        df,
        AnalysisIntentV2(task_family=TaskFamily.CLUSTERING, subjects=["x", "y"], arguments={"n_clusters": 2}),
    )
    assert output.status == "FAILED"
    assert output.error_code == "SCALING_REQUIRED"
