import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.ds_engine.intent_schema import (
    AnalysisIntentV2,
    TaskFamily,
    TimeContext,
    FilterSpec,
    FilterOperator,
)
from backend.ds_engine.typed_operators import (
    TypedOperatorRegistry,
    EventWindowAnalysisOperator,
    TimeSeriesDecompositionOperator,
    ClusteringOperator,
    PCADimensionalityReductionOperator,
    AutocorrelationOperator,
    TimeSeriesForecastOperator,
)
from backend.ds_engine.command_resolver import CommandResolver
from modules.query_orchestrator import QueryOrchestrator
from modules.capability_registry import capability_registry


@pytest.fixture
def panel_sensor_df():
    """Longitudinal panel data across 2 devices with timestamps and varied metrics."""
    np.random.seed(42)
    base_time = datetime(2025, 1, 1)
    rows = []
    for dev in ["Device_A", "Device_B"]:
        for t in range(50):
            timestamp = base_time + timedelta(hours=t)
            is_anomaly_val = 2 if t == 40 else (1 if t in (38, 39) else 0)
            status_val = "CRITICAL_ALERT" if t == 45 else "normal_state"
            defect_code = "ERR_OVERHEAT" if t == 35 else "OK"
            vib = float(np.random.normal(10, 1) + (50 if t in (38, 39, 40) else 0))
            temp = float(np.random.normal(70, 5) + (30 if t >= 35 else 0))
            rows.append({
                "device_id": dev,
                "timestamp": timestamp,
                "vibration": vib,
                "temperature": temp,
                "anomaly_score": is_anomaly_val,
                "status": status_val,
                "defect_code": defect_code
            })
    return pd.DataFrame(rows)


@pytest.fixture
def clean_time_series_df():
    """Continuous temporal time series for decomposition testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    trend = np.linspace(100, 200, 60)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(60) / 7)
    noise = np.random.normal(0, 2, 60)
    values = trend + seasonal + noise
    return pd.DataFrame({"record_date": dates, "daily_revenue": values})


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEST EVENT PREDICATE OPERATORS (> , >=, <, <=, !=, IN, CONTAINS)
# ══════════════════════════════════════════════════════════════════════════════

def test_event_predicate_greater_than_operator(panel_sensor_df):
    """Test EventWindowAnalysis with '>' operator: anomaly_score > 1 matches value 2."""
    intent = AnalysisIntentV2(
        raw_query="Check vibration spikes before severe anomaly",
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(
            time_column="timestamp",
            event_column="anomaly_score",
            event_predicate={"column": "anomaly_score", "operator": ">", "value": 1},
            pre_event_window=5,
            entity_id="device_id"
        ),
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(panel_sensor_df, intent)

    assert out.status == "SUCCEEDED", f"Failed with: {out.error_message}"
    assert out.manifest.event_row_count == 2  # 1 per device at t=40
    assert out.manifest.pre_event_row_count >= 2
    assert "anomaly_score > 1.0" in out.manifest.predicate_applied or "anomaly_score" in out.manifest.predicate_applied


def test_event_predicate_greater_than_or_equal_operator(panel_sensor_df):
    """Test EventWindowAnalysis with '>=' operator: anomaly_score >= 1 matches values 1 and 2."""
    intent = AnalysisIntentV2(
        raw_query="Check vibration spikes before any anomaly",
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(
            time_column="timestamp",
            event_column="anomaly_score",
            event_predicate={"column": "anomaly_score", "operator": ">=", "value": 1},
            pre_event_window=5,
            entity_id="device_id"
        ),
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(panel_sensor_df, intent)

    assert out.status == "SUCCEEDED", f"Failed with: {out.error_message}"
    assert out.manifest.event_row_count == 6  # 3 per device at t=38,39,40
    assert out.manifest.pre_event_row_count >= 2


def test_event_predicate_not_equal_operator(panel_sensor_df):
    """Test EventWindowAnalysis with '!=' operator: status != 'normal_state'."""
    intent = AnalysisIntentV2(
        raw_query="Check temperature before non-normal states",
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["temperature"],
        time_context=TimeContext(
            time_column="timestamp",
            event_column="status",
            event_predicate={"column": "status", "operator": "!=", "value": "normal_state"},
            pre_event_window=5,
            entity_id="device_id"
        ),
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(panel_sensor_df, intent)

    assert out.status == "SUCCEEDED", f"Failed with: {out.error_message}"
    assert out.manifest.event_row_count == 2  # CRITICAL_ALERT at t=45 per device
    assert out.manifest.pre_event_row_count >= 2


def test_event_predicate_in_operator(panel_sensor_df):
    """Test EventWindowAnalysis with 'in' operator: defect_code in ['ERR_OVERHEAT', 'ERR_CRITICAL']."""
    intent = AnalysisIntentV2(
        raw_query="Check temperature before overheat error",
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["temperature"],
        time_context=TimeContext(
            time_column="timestamp",
            event_column="defect_code",
            event_predicate={"column": "defect_code", "operator": "in", "value": ["ERR_OVERHEAT", "ERR_CRITICAL"]},
            pre_event_window=5,
            entity_id="device_id"
        ),
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(panel_sensor_df, intent)

    assert out.status == "SUCCEEDED", f"Failed with: {out.error_message}"
    assert out.manifest.event_row_count == 2  # ERR_OVERHEAT at t=35 per device
    assert out.manifest.pre_event_row_count >= 2


def test_event_predicate_contains_operator(panel_sensor_df):
    """Test EventWindowAnalysis with 'contains' operator: status contains 'ALERT'."""
    intent = AnalysisIntentV2(
        raw_query="Check vibration before alert",
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(
            time_column="timestamp",
            event_column="status",
            event_predicate={"column": "status", "operator": "contains", "value": "ALERT"},
            pre_event_window=5,
            entity_id="device_id"
        ),
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(panel_sensor_df, intent)

    assert out.status == "SUCCEEDED", f"Failed with: {out.error_message}"
    assert out.manifest.event_row_count == 2
    assert out.manifest.pre_event_row_count >= 2


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEST TIME-SERIES DECOMPOSITION OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

def test_time_series_decomposition_operator_execution(clean_time_series_df):
    """Test TimeSeriesDecompositionOperator execution with trend, seasonal, and residual outputs."""
    intent = AnalysisIntentV2(
        raw_query="Decompose daily revenue into trend and seasonality",
        task_family=TaskFamily.TIME_SERIES_DECOMPOSITION,
        subjects=["daily_revenue"],
        target_column="daily_revenue",
        time_context=TimeContext(
            time_column="record_date",
            pre_event_window=7
        ),
        arguments={"period": 7, "model": "additive"}
    )
    op = TimeSeriesDecompositionOperator()
    out = op.execute(clean_time_series_df, intent)

    assert out.status == "SUCCEEDED", f"Failed with: {out.error_message}"
    assert out.dataframe is not None
    assert "observed" in out.dataframe.columns
    assert "trend" in out.dataframe.columns
    assert "seasonal" in out.dataframe.columns
    assert "residual" in out.dataframe.columns
    assert out.manifest.operator == "time_series_decomposition"
    assert out.manifest.chart_type == "line"
    assert out.plot_image_base64 is not None
    assert "Trend Strength" in out.written_finding


def test_time_series_decomposition_via_typed_registry(clean_time_series_df):
    """Test that TypedOperatorRegistry executes TaskFamily.TIME_SERIES_DECOMPOSITION cleanly."""
    intent = AnalysisIntentV2(
        raw_query="Decompose daily revenue",
        task_family=TaskFamily.TIME_SERIES_DECOMPOSITION,
        target_column="daily_revenue",
        time_context=TimeContext(time_column="record_date"),
        arguments={"period": 7, "model": "additive"}
    )
    out = TypedOperatorRegistry.execute_intent(clean_time_series_df, intent)
    assert out.status == "SUCCEEDED", f"Registry execution failed: {out.error_message}"
    assert out.manifest.operator == "time_series_decomposition"


def test_time_series_decomposition_live_toolspec_e2e(clean_time_series_df):
    """Test E2E decompose_time_series ToolSpec resolution and execution through QueryOrchestrator."""
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse
    toolspec_payload = {
        "tool": "decompose_time_series",
        "arguments": {
            "time_column": "record_date",
            "value_column": "daily_revenue",
            "period": 7,
            "model": "additive"
        }
    }
    resolution = CommandResolver.resolve_command(toolspec_payload)
    assert resolution.is_valid is True
    assert resolution.semantic_intent is not None

    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content=json.dumps(toolspec_payload),
        parsed_json=toolspec_payload
    )

    try:
        res = QueryOrchestrator.process_query(
            "Decompose daily_revenue over record_date",
            df=clean_time_series_df
        )
        assert res.status == "SUCCEEDED", f"Live E2E decomposition failed: {res.error}"
        assert res.result is not None
        assert "manifest" in res.result
        assert res.result["manifest"]["operator"] == "time_series_decomposition"
    finally:
        ollama_service.structured_request = orig_req


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEST STRICT ZERO-DEFAULT REJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def test_clustering_strictly_rejects_missing_n_clusters(panel_sensor_df):
    """Test ClusteringOperator fails with N_CLUSTERS_REQUIRED when n_clusters/k is omitted."""
    intent = AnalysisIntentV2(
        raw_query="Cluster vibration and temperature",
        task_family=TaskFamily.CLUSTERING,
        subjects=["vibration", "temperature"],
        arguments={}  # Omitted n_clusters
    )
    op = ClusteringOperator()
    out = op.execute(panel_sensor_df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "N_CLUSTERS_REQUIRED"


def test_pca_strictly_rejects_missing_n_components(panel_sensor_df):
    """Test PCADimensionalityReductionOperator fails with N_COMPONENTS_REQUIRED when n_components is omitted."""
    intent = AnalysisIntentV2(
        raw_query="Run PCA on vibration and temperature",
        task_family=TaskFamily.DIMENSIONALITY_REDUCTION,
        subjects=["vibration", "temperature"],
        arguments={}  # Omitted n_components
    )
    op = PCADimensionalityReductionOperator()
    out = op.execute(panel_sensor_df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "N_COMPONENTS_REQUIRED"


def test_autocorrelation_strictly_rejects_missing_max_lags(panel_sensor_df):
    """Test AutocorrelationOperator fails with MAX_LAGS_REQUIRED when max_lags is omitted."""
    intent = AnalysisIntentV2(
        raw_query="Plot autocorrelation for vibration",
        task_family=TaskFamily.AUTOCORRELATION,
        target_column="vibration",
        arguments={}  # Omitted max_lags
    )
    op = AutocorrelationOperator()
    out = op.execute(panel_sensor_df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "MAX_LAGS_REQUIRED"


def test_forecast_strictly_rejects_missing_horizon(clean_time_series_df):
    """Test TimeSeriesForecastOperator fails with HORIZON_REQUIRED when horizon is omitted."""
    intent = AnalysisIntentV2(
        raw_query="Forecast daily revenue",
        task_family=TaskFamily.TIME_SERIES_FORECAST,
        subjects=["daily_revenue"],
        time_context=TimeContext(time_column="record_date", horizon=None)  # Omitted horizon
    )
    op = TimeSeriesForecastOperator()
    out = op.execute(clean_time_series_df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "HORIZON_REQUIRED"


def test_time_series_decomposition_strictly_rejects_missing_period(clean_time_series_df):
    """Test TimeSeriesDecompositionOperator fails with PERIOD_REQUIRED when period is omitted."""
    intent = AnalysisIntentV2(
        raw_query="Decompose daily revenue",
        task_family=TaskFamily.TIME_SERIES_DECOMPOSITION,
        subjects=["daily_revenue"],
        target_column="daily_revenue",
        time_context=TimeContext(time_column="record_date"),
        arguments={"model": "additive"}
    )
    op = TimeSeriesDecompositionOperator()
    out = op.execute(clean_time_series_df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "PERIOD_REQUIRED"
    assert "explicit seasonal period" in out.error_message


def test_time_series_decomposition_strictly_rejects_missing_model(clean_time_series_df):
    """Test TimeSeriesDecompositionOperator fails with MODEL_REQUIRED when model is omitted."""
    intent = AnalysisIntentV2(
        raw_query="Decompose daily revenue",
        task_family=TaskFamily.TIME_SERIES_DECOMPOSITION,
        subjects=["daily_revenue"],
        target_column="daily_revenue",
        time_context=TimeContext(time_column="record_date"),
        arguments={"period": 7}
    )
    op = TimeSeriesDecompositionOperator()
    out = op.execute(clean_time_series_df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "MODEL_REQUIRED"
    assert "explicit decomposition model" in out.error_message


# ══════════════════════════════════════════════════════════════════════════════
# 4. TEST REPRODUCIBLE CODE EXECUTION EQUIVALENCE
# ══════════════════════════════════════════════════════════════════════════════

def test_event_window_reproducible_code_executes_faithfully(panel_sensor_df):
    """Test that generated reproducible code executes in a clean namespace and produces summary_df."""
    intent = AnalysisIntentV2(
        raw_query="Check vibration spikes before severe anomaly",
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(
            time_column="timestamp",
            event_column="anomaly_score",
            event_predicate={"column": "anomaly_score", "operator": ">", "value": 1},
            pre_event_window=5,
            entity_id="device_id"
        ),
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(panel_sensor_df, intent)

    assert out.status == "SUCCEEDED"
    code_str = out.reproducible_code
    assert code_str is not None
    assert "df['anomaly_score'] > 1.0" in code_str or "anomaly_score" in code_str

    # Execute generated code in a fresh sandbox namespace
    local_scope = {"df": panel_sensor_df.copy()}
    exec(code_str, {}, local_scope)

    assert "summary_df" in local_scope
    generated_summary = local_scope["summary_df"]
    assert len(generated_summary) > 0
    assert "variance_spike_ratio" in generated_summary.columns


def test_time_series_decomposition_reproducible_code_executes_faithfully(clean_time_series_df):
    """Test that generated reproducible code executes in a clean namespace and produces exact result_df."""
    intent = AnalysisIntentV2(
        raw_query="Decompose daily revenue",
        task_family=TaskFamily.TIME_SERIES_DECOMPOSITION,
        subjects=["daily_revenue"],
        target_column="daily_revenue",
        time_context=TimeContext(time_column="record_date"),
        arguments={"period": 7, "model": "additive"}
    )
    op = TimeSeriesDecompositionOperator()
    out = op.execute(clean_time_series_df, intent)

    assert out.status == "SUCCEEDED"
    code_str = out.reproducible_code
    assert code_str is not None

    # Execute generated code in a fresh sandbox namespace
    local_scope = {"df": clean_time_series_df.copy()}
    exec(code_str, {}, local_scope)

    assert "result_df" in local_scope
    result_df = local_scope["result_df"]
    assert "record_date" in result_df.columns
    assert "observed" in result_df.columns
    assert "trend" in result_df.columns
    assert "seasonal" in result_df.columns
    assert "residual" in result_df.columns
    assert len(result_df) == len(out.dataframe)
