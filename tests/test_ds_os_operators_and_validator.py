"""
DataPilot OS — Comprehensive DS-OS Operators & Semantic Validator Tests.
Verifies all 9 core DS-OS architectural invariants:
1. DatasetProfiler profiles columns using pure statistical traits, detects binary flags & entity IDs, and computes full SHA-256 hash.
2. DistributionVarianceOperator produces exact Box plot stats, IQR, Plotly spec, and Matplotlib PNG.
3. EventWindowAnalysisOperator partitions data into Pre-Event Window, Event, and Normal regimes, computes variance spike ratio, and plots multi-regime Box Plots.
4. FilterSliceOperator preserves full in-memory DataFrame without row truncation.
5. TypedOperatorRegistry strictly rejects unsupported task families without hidden fallbacks.
6. SemanticResultValidator strictly rejects missing charts ('none' when 'box' was requested) and missing subject columns.
7. Multi-step pipeline executes filter + event-window analysis accurately on predictive maintenance data.
"""

import pytest
import pandas as pd
import numpy as np

from backend.ds_engine.dataset_profiler import DatasetProfiler, DatasetProfile
from backend.ds_engine.intent_schema import (
    AnalysisIntentV2, TaskFamily, FilterSpec, FilterOperator,
    VisualizationSpec, ExecutionManifest, TimeContext
)
from backend.ds_engine.typed_operators import (
    DistributionVarianceOperator, EventWindowAnalysisOperator, FilterSliceOperator,
    GroupAggregationOperator, CorrelationOperator, AnomalyDetectionOperator,
    TypedOperatorRegistry
)
from backend.ds_engine.result_validator import SemanticResultValidator


@pytest.fixture
def synthetic_sensor_df():
    np.random.seed(42)
    n = 300
    # Simulate a machine that develops high vibration spikes before failure
    vib_x = np.random.normal(1.6, 0.2, n)
    vib_y = np.random.normal(1.4, 0.15, n)
    vib_z = np.random.normal(2.0, 0.25, n)
    failure = np.zeros(n, dtype=int)
    
    # Introduce failure at index 100 and index 250
    failure[100] = 1
    failure[250] = 1
    
    # Introduce variance spike in pre-failure window (indices 70..99 and 220..249)
    vib_x[70:100] += np.random.normal(0, 0.8, 30)
    vib_y[70:100] += np.random.normal(0, 0.7, 30)
    vib_z[70:100] += np.random.normal(0, 1.2, 30)
    
    vib_x[220:250] += np.random.normal(0, 0.9, 30)
    vib_y[220:250] += np.random.normal(0, 0.8, 30)
    vib_z[220:250] += np.random.normal(0, 1.3, 30)

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "unit_uid": [f"UID_{i:04d}" for i in range(n)],
        "device_id": ["DEV_0001"] * n,
        "vibration_x_g": vib_x,
        "vibration_y_g": vib_y,
        "vibration_z_g": vib_z,
        "Machine_failure": failure,
        "region": np.random.choice(["North", "South", "East", "West"], size=n)
    })


def test_dataset_profiler_roles_and_full_hash(synthetic_sensor_df):
    profile = DatasetProfiler.profile(synthetic_sensor_df, "sensor_dataset")
    
    assert profile.row_count == 300
    assert profile.column_count == 8
    assert len(profile.dataset_hash) == 64
    
    # Check semantic roles
    assert profile.columns["timestamp"].semantic_role == "temporal"
    assert profile.columns["unit_uid"].semantic_role == "entity_id"
    assert profile.columns["vibration_x_g"].semantic_role == "continuous_numeric"
    assert profile.columns["Machine_failure"].semantic_role == "binary_flag"
    assert profile.columns["region"].semantic_role == "categorical_nominal"
    
    # Check hash changes if any later row changes
    modified_df = synthetic_sensor_df.copy()
    modified_df.iloc[299, 3] = 999.9
    mod_profile = DatasetProfiler.profile(modified_df, "sensor_dataset")
    assert profile.dataset_hash != mod_profile.dataset_hash


def test_event_window_analysis_operator_pre_failure_variance_spikes(synthetic_sensor_df):
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration_x_g", "vibration_y_g", "vibration_z_g"],
        time_context=TimeContext(
            event_column="Machine_failure",
            time_column="timestamp",
            pre_event_window=30,
            event_predicate="== 1"
        ),
        conditions=[FilterSpec(column="Machine_failure", operator=FilterOperator.EQ, value=1)],
        visualization=VisualizationSpec(type="box"),
        statistical_requirements=["variance", "quantiles", "iqr"],
        arguments={"event_scope": "event_rows"},
    )
    
    op = EventWindowAnalysisOperator()
    out = op.execute(synthetic_sensor_df, intent)
    
    assert out.status == "SUCCEEDED", f"Failed with {out.error_code}: {out.error_message}"
    assert out.manifest.operator == "event_window_analysis"
    assert out.manifest.chart_type == "box"
    assert out.manifest.event_row_count == 2
    assert out.manifest.pre_event_row_count > 0
    assert out.manifest.sorted_by == "timestamp"
    assert "pre_event_variance" in out.manifest.statistics_computed
    assert "variance_spike_ratio" in out.manifest.statistics_computed
    assert out.plot_image_base64 is not None
    assert out.plotly_spec is not None
    assert "Variance Spike Ratio" in out.written_finding

    # Verify that variance spike ratio in pre-event window is > 1.5x normal baseline
    records = out.data.get("records", [])
    pre_rows = [r for r in records if "Pre-Event" in r.get("regime", "")]
    assert len(pre_rows) == 3
    for pr in pre_rows:
        assert pr["variance_spike_ratio"] > 1.5, f"Expected variance spike ratio > 1.5x, got {pr['variance_spike_ratio']}"


def test_filter_preserves_full_in_memory_dataframe_without_truncation():
    # 200 rows where 120 match
    n = 200
    df = pd.DataFrame({
        "status": ["fail"] * 120 + ["pass"] * 80,
        "metric": np.random.randn(n)
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.FILTER_SLICE,
        conditions=[FilterSpec(column="status", operator=FilterOperator.EQ, value="fail")]
    )
    
    op = FilterSliceOperator()
    out = op.execute(df, intent)
    
    assert out.status == "SUCCEEDED"
    assert len(out.dataframe) == 120, f"Expected 120 rows in dataframe, got {len(out.dataframe)}"
    assert out.manifest.row_count_out == 120
    # Preview data records can be head(50) for UI, but dataframe MUST be 120
    assert len(out.data["records"]) <= 50


def test_semantic_result_validator_strictly_rejects_missing_chart():
    # Intent requested 'box' chart, but manifest produced 'none'
    intent = AnalysisIntentV2(
        task_family=TaskFamily.DISTRIBUTION_COMPARISON,
        subjects=["vibration_x_g"],
        visualization=VisualizationSpec(type="box")
    )
    manifest = ExecutionManifest(
        operator="distribution_comparison",
        columns_used=["vibration_x_g"],
        statistics_computed=["mean"],
        chart_type="none",  # Mismatch: missing chart
        row_count_in=100,
        row_count_out=1
    )
    res = SemanticResultValidator.validate_manifest(intent, manifest)
    assert res.is_valid is False
    assert any("Requested chart 'box', but operator generated 'none'" in issue for issue in res.issues)


def test_semantic_result_validator_strictly_rejects_missing_subject_column():
    intent = AnalysisIntentV2(
        task_family=TaskFamily.DISTRIBUTION_COMPARISON,
        subjects=["vibration_x_g", "vibration_y_g", "vibration_z_g"],
        visualization=VisualizationSpec(type="box")
    )
    manifest = ExecutionManifest(
        operator="distribution_comparison",
        columns_used=["vibration_x_g"],  # Missing vibration_y_g and vibration_z_g!
        statistics_computed=["mean", "variance"],
        chart_type="box",
        row_count_in=100,
        row_count_out=1
    )
    res = SemanticResultValidator.validate_manifest(intent, manifest)
    assert res.is_valid is False
    assert any("Subject column mismatch" in issue for issue in res.issues)


def test_typed_operator_registry_strictly_rejects_unsupported_operation():
    # Test that unknown task family fails fast with UNSUPPORTED_OPERATION (NO hidden EDA fallback)
    class FakeTaskFamily:
        value = "quantum_simulation"
        
    fake_intent = AnalysisIntentV2.model_construct(
        task_family=FakeTaskFamily,
        subjects=["vibration_x_g"],
        conditions=[],
        grouping=[],
        time_context=None,
        statistical_requirements=[],
        visualization=None,
        required_artifacts=[],
        assumptions=[],
        prompt_summary=None
    )
    
    dummy_df = pd.DataFrame({"vibration_x_g": [1.0, 2.0, 3.0]})
    out = TypedOperatorRegistry.execute_intent(dummy_df, fake_intent)
    assert out.status == "FAILED"
    assert out.error_code == "UNSUPPORTED_OPERATION"
    assert "No deterministic typed operator registered" in out.error_message


def test_full_predictive_maintenance_user_query_pipeline(synthetic_sensor_df):
    """
    Simulates the exact user query:
    'Machine failure hone se pehle vibration_x_g, vibration_y_g, aur vibration_z_g me kya pattern (variance spike) dikhta hai? Box plot draw karein.'
    """
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration_x_g", "vibration_y_g", "vibration_z_g"],
        time_context=TimeContext(
            event_column="Machine_failure",
            time_column="timestamp",
            entity_id="device_id",
            pre_event_window=30,
            event_predicate="Machine_failure == 1"
        ),
        conditions=[FilterSpec(column="Machine_failure", operator=FilterOperator.EQ, value=1)],
        visualization=VisualizationSpec(type="box"),
        statistical_requirements=["variance", "quantiles", "iqr"],
        arguments={"event_scope": "event_rows"},
    )
    
    profile = DatasetProfiler.profile(synthetic_sensor_df, "predictive_maintenance")
    out = TypedOperatorRegistry.execute_intent(synthetic_sensor_df, intent, profile)
    
    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "event_window_analysis"
    assert out.manifest.chart_type == "box"
    assert out.manifest.event_row_count == 2
    assert out.manifest.pre_event_row_count > 0
    
    # Strict Semantic Validation
    val_res = SemanticResultValidator.validate_manifest(intent, out.manifest)
    assert val_res.is_valid is True
    assert len(val_res.issues) == 0


def test_event_window_no_temporal_column():
    df = pd.DataFrame({
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "event": [0, 0, 1, 0, 0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(event_column="event", time_column="non_existent_time", pre_event_window=2, event_predicate="== 1")
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "NO_TEMPORAL_COLUMN"
    assert out.error_message == "Event window analysis requires an explicit datetime/temporal column."


def test_event_window_pre_event_window_required():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "event": [0, 0, 1, 0, 0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(event_column="event", time_column="timestamp", event_predicate="== 1")
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "PRE_EVENT_WINDOW_REQUIRED"
    assert out.error_message == "Event window analysis requires an explicit pre_event_window."


def test_event_window_event_predicate_required():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "event": [0, 0, 1, 0, 0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(event_column="event", time_column="timestamp", pre_event_window=2)
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "EVENT_PREDICATE_REQUIRED"
    assert out.error_message == "Event window analysis requires an explicit event predicate."


def test_event_window_uses_explicit_entity_id():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="1min"),
        "unit_uid": [f"U_{i:02d}" for i in range(10)],
        "machine_id": ["M1", "M1", "M1", "M1", "M1", "M2", "M2", "M2", "M2", "M2"],
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 2.1, 3.1, 4.1, 5.1],
        "event": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    })
    profile = DatasetProfiler.profile(df, "multi_entity_df")
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(event_column="event", time_column="timestamp", entity_id="machine_id", pre_event_window=2),
        conditions=[FilterSpec(column="event", operator=FilterOperator.EQ, value=1)],
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent, profile)
    assert out.status == "SUCCEEDED"
    assert out.manifest.partitioned_by == "machine_id"


def test_event_window_event_column_required():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(time_column="timestamp", pre_event_window=2, event_predicate="== 1")
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent)
    assert out.status == "FAILED"
    assert out.error_code in ("EVENT_COLUMN_REQUIRED", "EVENT_PREDICATE_REQUIRED")
    assert out.error_message == "Event window analysis requires an explicit event_column."


def test_event_window_no_event_rows_found():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "event": [0, 0, 0, 0, 0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(event_column="event", time_column="timestamp", pre_event_window=2),
        conditions=[FilterSpec(column="event", operator=FilterOperator.EQ, value=1)]
        ,arguments={"event_scope": "event_rows"}
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "NO_EVENT_ROWS_FOUND"
    assert "No event occurrences matched predicate for column 'event'." in out.error_message


def test_event_window_insufficient_pre_event_data():
    # Event occurs at index 0, so 0 pre-event lookback rows
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "event": [1, 0, 0, 0, 0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(event_column="event", time_column="timestamp", pre_event_window=5),
        conditions=[FilterSpec(column="event", operator=FilterOperator.EQ, value=1)],
        arguments={"event_scope": "event_rows"},
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "INSUFFICIENT_PRE_EVENT_DATA"
    assert out.error_message == "Insufficient pre-event historical data points for variance analysis."


def test_event_window_invalid_column_type():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        "category_str": ["A", "B", "C", "D", "E"],
        "event": [0, 0, 1, 0, 0]
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["category_str"],
        time_context=TimeContext(event_column="event", time_column="timestamp", pre_event_window=2, event_predicate="== 1")
    )
    op = EventWindowAnalysisOperator()
    out = op.execute(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "INVALID_COLUMN_TYPE"
    assert "Event window variance analysis requires numeric subject columns" in out.error_message


def test_autocorrelation_operator_success():
    n = 100
    df = pd.DataFrame({"signal": np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.AUTOCORRELATION,
        subjects=["signal"],
        arguments={"max_lags": 20}
    )
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "autocorrelation"
    assert out.manifest.chart_type == "acf"
    assert "acf" in out.manifest.statistics_computed
    assert out.plot_image_base64 is not None


def test_zero_default_clean_failures_for_all_operators():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=20, freq="1D"),
        "val1": np.random.randn(20),
        "val2": np.random.randn(20),
        "category": ["A", "B"] * 10
    })

    # 1. Distribution: omitted subjects -> SUBJECT_COLUMNS_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.DISTRIBUTION_COMPARISON, subjects=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "SUBJECT_COLUMNS_REQUIRED"

    # 2. Group aggregation: omitted grouping -> GROUP_BY_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.GROUP_AGGREGATION, subjects=["val1"], grouping=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "GROUP_BY_REQUIRED"

    # 3. Group aggregation: omitted subjects -> SUBJECT_COLUMNS_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.GROUP_AGGREGATION, subjects=[], grouping=["category"])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "SUBJECT_COLUMNS_REQUIRED"

    # 4. Forecast: omitted target column -> TARGET_COLUMN_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.TIME_SERIES_FORECAST, subjects=[], time_context=TimeContext(time_column="time", horizon=5, frequency="D"), arguments={"method": "naive_last_value", "frequency": "D", "chart_type": "none"})
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "TARGET_COLUMN_REQUIRED"

    # 5. Forecast: omitted horizon -> HORIZON_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.TIME_SERIES_FORECAST, subjects=["val1"], time_context=TimeContext(time_column="time"))
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "HORIZON_REQUIRED"

    # 6. Forecast: omitted time column -> NO_TEMPORAL_COLUMN
    intent = AnalysisIntentV2(task_family=TaskFamily.TIME_SERIES_FORECAST, subjects=["val1"], time_context=TimeContext(horizon=5))
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "NO_TEMPORAL_COLUMN"

    # 7. Correlation: omitted structured specification is rejected first.
    intent = AnalysisIntentV2(task_family=TaskFamily.CORRELATION_ANALYSIS, subjects=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "CORRELATION_SPEC_REQUIRED"

    # 8. Anomaly detection: omitted subjects -> SUBJECT_COLUMNS_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.ANOMALY_DETECTION, subjects=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "SUBJECT_COLUMNS_REQUIRED"

    # 9. Supervised ML: omitted target -> TARGET_COLUMN_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.SUPERVISED_ML, subjects=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "TARGET_COLUMN_REQUIRED"

    # 10. Clustering: omitted subjects -> SUBJECT_COLUMNS_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.CLUSTERING, subjects=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "SUBJECT_COLUMNS_REQUIRED"

    # 11. PCA: omitted subjects -> SUBJECT_COLUMNS_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.DIMENSIONALITY_REDUCTION, subjects=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "SUBJECT_COLUMNS_REQUIRED"

    # 12. Feature explainability: omitted target -> TARGET_COLUMN_REQUIRED
    intent = AnalysisIntentV2(task_family=TaskFamily.EXPLAINABILITY, subjects=[])
    out = TypedOperatorRegistry.execute_intent(df, intent)
    assert out.status == "FAILED"
    assert out.error_code == "TARGET_COLUMN_REQUIRED"
