"""Regression coverage for typed, evidence-preserving correlation requests."""

import warnings

import numpy as np
import pandas as pd

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.intent_schema import AnalysisIntentV2, CorrelationSpec, TaskFamily
from backend.ds_engine.typed_operators import CorrelationOperator
from backend.ds_engine.command_resolver import command_resolver
from modules.query_orchestrator import _map_intent_to_v2


NONLINEAR_PAIRWISE_ARGS = {
    "columns": ["x_signal", "y_signal"],
    "analysis_mode": "pairwise",
    "relationship_form": "nonlinear",
    "correlation_methods": ["pearson", "spearman"],
    "chart_type": "scatter",
    "smoother": "lowess",
    "expected_direction": "negative",
}


def test_correlation_rejects_columns_only_contract_instead_of_silent_heatmap():
    result = command_resolver.resolve_command(
        {"tool": "correlate_data", "arguments": {"columns": ["x_signal", "y_signal"]}},
        df=pd.DataFrame({"x_signal": [1, 2], "y_signal": [2, 1]}),
    )

    assert not result.is_valid
    assert result.error_code == "INVALID_ARGUMENTS"


def test_nonlinear_pairwise_request_preserves_stats_and_lowess_plot():
    x = np.linspace(0.1, 5.0, 180)
    df = pd.DataFrame({"x_signal": x, "y_signal": -np.square(x)})
    resolution = command_resolver.resolve_command(
        {"tool": "correlate_data", "arguments": NONLINEAR_PAIRWISE_ARGS},
        df=df,
        raw_query="A generic nonlinear inverse relationship request",
    )

    assert resolution.is_valid
    analysis_intent = _map_intent_to_v2(resolution.semantic_intent, raw_query="generic")
    output = CorrelationOperator().execute(df, analysis_intent)

    assert output.status == "SUCCEEDED"
    assert output.manifest.chart_type == "scatter"
    assert {"pearson_correlation", "spearman_rank_correlation", "lowess_descriptive_smoother"}.issubset(
        output.manifest.statistics_computed
    )
    assert output.plot_image_base64
    result = output.dataframe.iloc[0]
    assert result["pearson_r"] < 0
    assert result["spearman_rho"] < 0
    assert bool(result["direction_matches_request"])


def test_matrix_contract_executes_only_the_explicit_method_and_heatmap():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 4, 9]})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.CORRELATION_ANALYSIS,
        subjects=["a", "b", "c"],
        correlation_spec=CorrelationSpec(
            analysis_mode="matrix",
            relationship_form="monotonic",
            correlation_methods=["spearman"],
            chart_type="heatmap",
            smoother="none",
            expected_direction="unspecified",
        ),
    )

    output = CorrelationOperator().execute(df, intent)

    assert output.status == "SUCCEEDED"
    assert output.manifest.statistics_computed == ["spearman_correlation"]
    assert output.manifest.chart_type == "heatmap"
    assert "Spearman" in output.written_finding


def test_profiler_does_not_emit_format_inference_warning_for_text_columns():
    df = pd.DataFrame({"label": ["alpha", "beta", "gamma"], "value": [1, 2, 3]})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        DatasetProfiler.profile(df, dataset_name="generic-text")

    assert not any("Could not infer format" in str(w.message) for w in captured)
