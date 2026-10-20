"""Regression tests for execution paths that must never infer user intent."""

import numpy as np
import pandas as pd

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily, TimeContext
from backend.ds_engine.typed_operators import EventWindowAnalysisOperator, TypedOperatorRegistry


def test_event_operator_rejects_implicit_binary_event_column():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=8, freq="D"),
        "Machine_failure": [0, 0, 0, 1, 0, 0, 0, 0],
        "vibration": [1.0, 1.1, 1.2, 5.0, 1.3, 1.1, 1.0, 1.2],
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibration"],
        time_context=TimeContext(
            time_column="timestamp",
            event_predicate={"operator": "==", "value": 1},
            pre_event_window=2,
        ),
        arguments={"event_scope": "event_rows"},
    )
    result = EventWindowAnalysisOperator().execute(df, intent, DatasetProfiler.profile(df, "audit"))
    assert result.status == "FAILED"
    assert result.error_code == "EVENT_COLUMN_REQUIRED"


def test_sequence_order_is_not_coerced_to_datetime_and_executes():
    df = pd.DataFrame({
        "opaque_counter": list(range(1, 9)),
        "incident": [0, 0, 0, 1, 0, 0, 0, 0],
        "signal": [1.0, 1.1, 1.2, 5.0, 1.3, 1.1, 1.0, 1.2],
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["signal"],
        time_context=TimeContext(
            sequence_column="opaque_counter",
            event_column="incident",
            event_predicate={"column": "incident", "operator": "==", "value": 1},
            pre_event_window=2,
        ),
        arguments={"event_scope": "event_rows"},
    )
    result = EventWindowAnalysisOperator().execute(df, intent, DatasetProfiler.profile(df, "audit"))
    assert result.status == "SUCCEEDED"
    assert result.manifest.sorted_by == "opaque_counter"
    assert "to_datetime" not in result.reproducible_code


def test_decomposition_cannot_reuse_event_window_or_frequency_as_period():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
        "signal": np.arange(30, dtype=float),
    })
    for context in (
        TimeContext(time_column="timestamp", pre_event_window=7),
        TimeContext(time_column="timestamp", frequency="daily"),
    ):
        intent = AnalysisIntentV2(
            task_family=TaskFamily.TIME_SERIES_DECOMPOSITION,
            target_column="signal",
            time_context=context,
            arguments={"model": "additive"},
        )
        result = TypedOperatorRegistry.execute_intent(df, intent)
        assert result.status == "FAILED"
        assert result.error_code == "PERIOD_REQUIRED"
