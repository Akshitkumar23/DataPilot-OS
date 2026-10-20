"""
Pre-Phase 7 Natural Language Capability Parity Test Suite.
Validates end-to-end parity from ToolSpec -> CommandResolver -> TypedOperatorRegistry -> OperatorOutput for:
1. Stationarity (ADF & KPSS unit root testing)
2. Elbow Analysis (K-Means optimal cluster count via inertia & silhouette)
3. Principal Component Analysis (PCA)
4. Autocorrelation (ACF & PACF with temporal sorting)
5. Time Series Decomposition (additive/multiplicative trend, season, residual)

All tests verify zero domain hardcoding, zero column heuristics, and zero silent fallbacks.
"""

import pytest
import pandas as pd
import numpy as np

from modules.capability_registry import capability_registry
from backend.ds_engine.command_resolver import command_resolver
from backend.ds_engine.typed_operators import TypedOperatorRegistry
from modules.query_orchestrator import _map_intent_to_v2


@pytest.fixture
def timeseries_data():
    """Synthetic time-series dataset with arbitrary column names."""
    rng = np.random.default_rng(42)
    n = 40
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    trend = np.linspace(10, 50, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.normal(0, 1, n)
    vals = trend + seasonal + noise

    return pd.DataFrame({
        "time_tau": dates,
        "signal_alpha": vals,
        "signal_beta": rng.normal(0, 1, n)  # Stationary white noise
    })


@pytest.fixture
def clustering_data():
    """Synthetic multi-feature dataset with arbitrary column names."""
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({
        "dim_alpha": np.concatenate([rng.normal(2, 0.5, 15), rng.normal(8, 0.5, 15)]),
        "dim_beta": np.concatenate([rng.normal(3, 0.5, 15), rng.normal(9, 0.5, 15)]),
        "dim_gamma": np.concatenate([rng.normal(1, 0.5, 15), rng.normal(7, 0.5, 15)]),
    })


def test_stationarity_toolspec_registered():
    """Verify stationarity_test ToolSpec is correctly registered in CapabilityRegistry."""
    spec = capability_registry.get_tool_spec("stationarity_test")
    assert spec is not None
    assert spec.name == "stationarity_test"
    assert spec.category == "statistical_test"
    assert "stationarity" in spec.synonyms
    assert "adf_test" in spec.synonyms


def test_stationarity_natural_language_pipeline(timeseries_data):
    """Verify stationarity testing executes end-to-end through command_resolver and operator."""
    res_cmd = command_resolver.resolve_command(
        command_data={
            "tool": "stationarity_test",
            "arguments": {
                "time_column": "time_tau",
                "target_column": "signal_alpha"
            }
        },
        df=timeseries_data,
        raw_query="is signal_alpha over time_tau stationary"
    )
    assert res_cmd.is_valid, f"Command resolution failed: {res_cmd.error_message}"

    v2_intent = _map_intent_to_v2(res_cmd.semantic_intent, raw_query="is signal_alpha over time_tau stationary", df=timeseries_data)
    op_out = TypedOperatorRegistry.execute_intent(timeseries_data, v2_intent)

    assert op_out.status == "SUCCEEDED"
    assert "conclusion" in op_out.data
    assert "adf_test" in op_out.data
    assert "kpss_test" in op_out.data
    assert op_out.data["adf_test"]["test_name"] == "Augmented Dickey-Fuller (ADF)"
    assert op_out.data["kpss_test"]["test_name"] == "Kwiatkowski-Phillips-Schmidt-Shin (KPSS)"
    assert len(op_out.data["records"]) == 2  # ADF and KPSS rows
    assert "signal_alpha" in op_out.manifest.columns_used
    assert "time_tau" in op_out.manifest.columns_used
    assert op_out.manifest.operator == "stationarity_test"


def test_stationarity_via_statistical_test_tool(timeseries_data):
    """Verify statistical_test tool with test_type='stationarity' also routes cleanly."""
    res_cmd = command_resolver.resolve_command(
        command_data={
            "tool": "statistical_test",
            "arguments": {
                "test_type": "stationarity",
                "columns": ["time_tau", "signal_alpha"]
            }
        },
        df=timeseries_data,
        raw_query="run stationarity test on time_tau and signal_alpha"
    )
    assert res_cmd.is_valid, f"Command resolution failed: {res_cmd.error_message}"

    v2_intent = _map_intent_to_v2(res_cmd.semantic_intent, raw_query="run stationarity test", df=timeseries_data)
    op_out = TypedOperatorRegistry.execute_intent(timeseries_data, v2_intent)

    assert op_out.status == "SUCCEEDED"
    assert "conclusion" in op_out.data
    assert op_out.manifest.operator == "stationarity_test"


def test_elbow_analysis_toolspec_registered():
    """Verify elbow_analysis ToolSpec is correctly registered in CapabilityRegistry."""
    spec = capability_registry.get_tool_spec("elbow_analysis")
    assert spec is not None
    assert spec.name == "elbow_analysis"
    assert spec.category == "clustering"
    assert "elbow_method" in spec.synonyms
    assert "optimal_clusters" in spec.synonyms


def test_elbow_analysis_natural_language_pipeline(clustering_data):
    """Verify elbow curve analysis executes end-to-end through command_resolver and operator."""
    res_cmd = command_resolver.resolve_command(
        command_data={
            "tool": "elbow_analysis",
            "arguments": {
                "columns": ["dim_alpha", "dim_beta", "dim_gamma"],
                "min_k": 2,
                "max_k": 5
            }
        },
        df=clustering_data,
        raw_query="find optimal clusters for dim_alpha, dim_beta, dim_gamma"
    )
    assert res_cmd.is_valid, f"Command resolution failed: {res_cmd.error_message}"

    v2_intent = _map_intent_to_v2(res_cmd.semantic_intent, raw_query="find optimal clusters", df=clustering_data)
    op_out = TypedOperatorRegistry.execute_intent(clustering_data, v2_intent)

    assert op_out.status == "SUCCEEDED"
    assert "curve" in op_out.data
    assert len(op_out.data["curve"]) == 4  # k=2, 3, 4, 5
    assert all("inertia" in pt and "k" in pt for pt in op_out.data["curve"])
    assert op_out.plot_image_base64 is not None
    assert op_out.manifest.operator == "elbow_analysis"
    assert "dim_alpha" in op_out.manifest.columns_used


def test_pca_analysis_natural_language_pipeline(clustering_data):
    """Verify PCA dimensionality reduction executes end-to-end through command_resolver and operator."""
    res_cmd = command_resolver.resolve_command(
        command_data={
            "tool": "pca_analysis",
            "arguments": {
                "columns": ["dim_alpha", "dim_beta", "dim_gamma"],
                "n_components": 2,
                "scaling": "standard"
            }
        },
        df=clustering_data,
        raw_query="run pca on dim_alpha, dim_beta, dim_gamma"
    )
    assert res_cmd.is_valid, f"Command resolution failed: {res_cmd.error_message}"

    v2_intent = _map_intent_to_v2(res_cmd.semantic_intent, raw_query="run pca", df=clustering_data)
    op_out = TypedOperatorRegistry.execute_intent(clustering_data, v2_intent)

    assert op_out.status == "SUCCEEDED"
    assert "explained_variance_ratio" in op_out.data
    assert len(op_out.data["explained_variance_ratio"]) == 2
    assert "feature_loadings" in op_out.data
    assert op_out.manifest.operator == "pca_dimensionality_reduction"


def test_autocorrelation_natural_language_pipeline(timeseries_data):
    """Verify Autocorrelation (ACF/PACF) executes end-to-end with time column ordering."""
    res_cmd = command_resolver.resolve_command(
        command_data={
            "tool": "autocorrelation",
            "arguments": {
                "time_column": "time_tau",
                "value_column": "signal_alpha",
                "max_lags": 12
            }
        },
        df=timeseries_data,
        raw_query="autocorrelation of signal_alpha over time_tau"
    )
    assert res_cmd.is_valid, f"Command resolution failed: {res_cmd.error_message}"

    v2_intent = _map_intent_to_v2(res_cmd.semantic_intent, raw_query="autocorrelation", df=timeseries_data)
    op_out = TypedOperatorRegistry.execute_intent(timeseries_data, v2_intent)

    assert op_out.status == "SUCCEEDED"
    assert len(op_out.data["records"]) == 13  # lag 0 to 12
    assert "acf" in op_out.data["records"][0]
    assert "pacf" in op_out.data["records"][0]
    assert op_out.manifest.sorted_by == "time_tau"


def test_time_series_decomposition_natural_language_pipeline(timeseries_data):
    """Verify seasonal decomposition executes end-to-end through command_resolver and operator."""
    res_cmd = command_resolver.resolve_command(
        command_data={
            "tool": "decompose_time_series",
            "arguments": {
                "time_column": "time_tau",
                "target_column": "signal_alpha",
                "period": 7,
                "model": "additive"
            }
        },
        df=timeseries_data,
        raw_query="decompose signal_alpha over time_tau"
    )
    assert res_cmd.is_valid, f"Command resolution failed: {res_cmd.error_message}"

    v2_intent = _map_intent_to_v2(res_cmd.semantic_intent, raw_query="decompose", df=timeseries_data)
    op_out = TypedOperatorRegistry.execute_intent(timeseries_data, v2_intent)

    assert op_out.status == "SUCCEEDED"
    assert len(op_out.data["records"]) == len(timeseries_data)
    assert "trend" in op_out.data["records"][0]
    assert "seasonal" in op_out.data["records"][0]
    assert "residual" in op_out.data["records"][0]
    assert op_out.manifest.operator == "time_series_decomposition"
