"""Regression tests for formerly silent analytical assumptions."""
import numpy as np
import pandas as pd

from backend.ds_engine.intent_schema import AnalysisIntentV2, FilterOperator, FilterSpec, TaskFamily, TimeContext
from backend.ds_engine.typed_operators import (
    AnomalyDetectionOperator,
    DataCleaningOperator,
    EventWindowAnalysisOperator,
    FeatureExplainabilityOperator,
)
from modules.semantic_planner import SemanticPlanner


def test_cleaning_applies_only_declared_operation():
    df = pd.DataFrame({"x": [1.0, 1.0, None], "tag": ["same", "same", "other"]})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        subjects=["x"],
        arguments={"method": "drop_missing", "columns": ["x"]},
    )
    result = DataCleaningOperator().execute(df, intent)
    assert result.status == "SUCCEEDED"
    assert len(result.dataframe) == 2
    assert result.dataframe.duplicated().sum() == 1


def test_anomaly_operator_honors_explicit_method_and_threshold():
    df = pd.DataFrame({"signal": [0.0] * 9 + [10.0]})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.ANOMALY_DETECTION,
        subjects=["signal"],
        arguments={"method": "zscore", "sensitivity": 2.0},
    )
    result = AnomalyDetectionOperator().execute(df, intent)
    assert result.status == "SUCCEEDED"
    assert result.dataframe["anomaly_flag"].sum() == 1
    assert "zscore" in result.written_finding
    assert "2" in result.written_finding


def test_event_rows_and_onsets_are_distinct_semantics():
    df = pd.DataFrame({
        "step": range(8),
        "event": [0, 0, 0, 1, 1, 0, 0, 0],
        "signal": [1.0, 1.1, 1.2, 4.0, 4.2, 1.1, 1.0, 1.2],
    })
    base = dict(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["signal"],
        time_context=TimeContext(
            sequence_column="step", event_column="event", pre_event_window=2,
            event_predicate={"column": "event", "operator": "==", "value": 1},
        ),
        conditions=[FilterSpec(column="event", operator=FilterOperator.EQ, value=1)],
    )
    rows = EventWindowAnalysisOperator().execute(df, AnalysisIntentV2(**base, arguments={"event_scope": "event_rows"}))
    onsets = EventWindowAnalysisOperator().execute(df, AnalysisIntentV2(**base, arguments={"event_scope": "event_onsets"}))
    assert rows.status == onsets.status == "SUCCEEDED"
    assert rows.manifest.event_row_count == 2
    assert onsets.manifest.event_row_count == 1

    generated_scope = {}
    exec(onsets.reproducible_code, {"df": df.copy()}, generated_scope)
    assert int(generated_scope["is_event"].sum()) == onsets.manifest.event_row_count
    assert generated_scope["event_scope"] == "event_onsets"


def test_explainability_uses_declared_target_and_features_on_holdout():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "target": np.tile([0, 1], 40),
        "numeric": rng.normal(size=80),
        "category": np.tile(["a", "b", "c", "d"], 20),
        "unused": rng.normal(size=80),
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EXPLAINABILITY,
        subjects=["numeric", "category"],
        target_column="target",
        arguments={
            "target_column": "target",
            "feature_columns": ["numeric", "category"],
            "task_type": "classification",
            "method": "permutation_importance",
            "prediction_context": "retrospective_explanation",
        },
    )
    result = FeatureExplainabilityOperator().execute(df, intent)
    assert result.status == "SUCCEEDED"
    assert result.manifest.columns_used == ["target", "numeric", "category"]
    assert "held_out_score" in result.manifest.statistics_computed
    assert "unused" not in result.dataframe["feature"].tolist()


def test_planner_rejects_columns_absent_from_user_authored_evidence():
    df = pd.DataFrame({"temperature": [1.0], "humidity": [2.0]})
    parsed = {"commands": [{
        "tool": "group_data",
        "arguments": {
            "group_by": ["humidity"], "metric": "temperature", "aggregation": "mean",
            "datetime_parses": [], "chart_type": "line",
        },
    }]}
    error = SemanticPlanner._dataset_column_grounding_error(parsed, df, "Analyze this data")
    assert error and error.startswith("UNGROUNDED_COLUMN_SELECTION")


def test_planner_accepts_schema_columns_from_prior_user_turn_or_ui_selection():
    df = pd.DataFrame({"attendance_rate_pct": [90.0], "grade_band": ["A"]})
    parsed = {"tool": "group_data", "arguments": {
        "group_by": ["grade_band"], "metric": "attendance_rate_pct", "aggregation": "mean",
        "datetime_parses": [], "chart_type": "none",
    }}
    evidence = SemanticPlanner._trusted_grounding_text(
        "average it by that group",
        [{"role": "user", "content": "Use attendance_rate_pct"}, {"role": "assistant", "content": "grade_band"}],
        {"selected_columns": ["grade_band"]},
    )
    assert SemanticPlanner._dataset_column_grounding_error(parsed, df, evidence) is None


def test_planner_rejects_ungrounded_column_even_when_another_column_is_grounded():
    df = pd.DataFrame({"attendance_rate_pct": [90.0], "grade_band": ["A"]})
    parsed = {"tool": "group_data", "arguments": {
        "group_by": ["grade_band"], "metric": "attendance_rate_pct", "aggregation": "mean",
        "datetime_parses": [], "chart_type": "none",
    }}
    error = SemanticPlanner._dataset_column_grounding_error(
        parsed, df, "Show average attendance_rate_pct"
    )
    assert error and "grade_band" in error
