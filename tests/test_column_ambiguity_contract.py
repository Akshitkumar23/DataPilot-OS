"""
DataPilot OS — Phase 3 Column Ambiguity and No-Silent-Defaults Contract Suite.
Enforces Rule 4: Zero silent defaults (never pick first numeric/date/target column,
never infer dimensions/metrics when multiple exist).
Enforces Rule 1: Zero hardcoded column keywords.
"""

import pandas as pd
import pytest

from backend.ds_engine.intent_schema import SemanticIntent, IntentType, AggregationType
from backend.ds_engine.intent_resolver import IntentResolver, AmbiguousIntentError, IntentResolutionError
from backend.ds_engine.command_resolver import CommandResolver
from backend.ds_engine.semantic_command import ToolCall


@pytest.fixture
def multi_ambiguity_df():
    """Dataset with multiple dates, multiple metrics, and multiple dimensions."""
    return pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "ship_date": pd.date_range("2024-01-05", periods=10, freq="D"),
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 350.0, 500.0, 450.0, 600.0],
        "profit": [20.0, 40.0, 30.0, 60.0, 50.0, 80.0, 70.0, 100.0, 90.0, 120.0],
        "cost": [80.0, 160.0, 120.0, 240.0, 200.0, 320.0, 280.0, 400.0, 360.0, 480.0],
        "region": ["North", "South"] * 5,
        "category": ["Electronics", "Furniture"] * 5,
        "segment": ["Enterprise", "Consumer"] * 5,
    })


def test_multiple_date_columns_without_explicit_selection_raises_ambiguity(multi_ambiguity_df):
    """Never pick first date column when multiple dates exist."""
    intent = SemanticIntent(
        intent_type=IntentType.TIME_SERIES_FORECAST,
        primary_intent=IntentType.TIME_SERIES_FORECAST,
        target_column="revenue",
        raw_query="Forecast revenue for next 7 days",
    )
    resolver = IntentResolver()
    with pytest.raises(AmbiguousIntentError) as exc_info:
        resolver.resolve(intent, df=multi_ambiguity_df)

    assert exc_info.value.code in ["AMBIGUOUS_COLUMN", "AMBIGUOUS_INTENT"]
    assert "order_date" in exc_info.value.candidate_columns
    assert "ship_date" in exc_info.value.candidate_columns


def test_multiple_metrics_without_explicit_selection_raises_ambiguity(multi_ambiguity_df):
    """Never pick first numeric metric when multiple exist for group aggregation."""
    intent = SemanticIntent(
        intent_type=IntentType.GROUP_AGGREGATE,
        primary_intent=IntentType.GROUP_AGGREGATE,
        aggregation=AggregationType.SUM,
        dimensions=["region"],
        raw_query="Total by region",
    )
    resolver = IntentResolver()
    with pytest.raises(AmbiguousIntentError) as exc_info:
        resolver.resolve(intent, df=multi_ambiguity_df)

    assert exc_info.value.code == "AMBIGUOUS_COLUMN"
    # Should surface all numeric metric candidates
    candidates = exc_info.value.candidate_columns
    assert "revenue" in candidates
    assert "profit" in candidates
    assert "cost" in candidates


def test_multiple_dimensions_without_explicit_selection_raises_ambiguity(multi_ambiguity_df):
    """Never pick first dimension when multiple exist for group aggregation."""
    intent = SemanticIntent(
        intent_type=IntentType.GROUP_AGGREGATE,
        primary_intent=IntentType.GROUP_AGGREGATE,
        aggregation=AggregationType.SUM,
        metrics=["revenue"],
        raw_query="Total revenue by group",
    )
    resolver = IntentResolver()
    with pytest.raises(AmbiguousIntentError) as exc_info:
        resolver.resolve(intent, df=multi_ambiguity_df)

    assert exc_info.value.code == "AMBIGUOUS_COLUMN"
    assert len(exc_info.value.candidate_columns) >= 2


def test_statistical_test_multiple_columns_raises_ambiguity_not_silent_slice(multi_ambiguity_df):
    """Never silently slice [:2] for statistical test when 3+ columns exist."""
    intent = SemanticIntent(
        intent_type=IntentType.STATISTICAL_TEST,
        primary_intent=IntentType.STATISTICAL_TEST,
        raw_query="Run statistical hypothesis test",
    )
    resolver = IntentResolver()
    with pytest.raises(AmbiguousIntentError) as exc_info:
        resolver.resolve(intent, df=multi_ambiguity_df)

    assert exc_info.value.code == "AMBIGUOUS_COLUMN"
    assert "Statistical test requires explicit columns" in exc_info.value.message


def test_command_resolver_rejects_nonexistent_column_with_typo_suggestions(multi_ambiguity_df):
    """CommandResolver rejects unknown columns and surfaces typo suggestions without guessing."""
    command_with_typo = [
        {
            "tool": "group_data",
            "arguments": {
                "group_by": ["regin"],  # typo for 'region'
                "metric": "revenue",
                "aggregation": "sum",
                "chart_type": "bar",
            },
        }
    ]
    res = CommandResolver.resolve_command(command_with_typo, df=multi_ambiguity_df)
    assert not res.is_valid
    assert res.error_code == "COLUMN_NOT_FOUND"
    assert "regin" in res.error_message
    assert "region" in res.error_message  # suggested correct column


def test_zero_hardcoded_target_columns_on_arbitrary_dataset():
    """Verify resolver does NOT auto-bind 'machine_failure', 'churn', etc. on arbitrary datasets."""
    df_unseen = pd.DataFrame({
        "device_uuid": ["d1", "d2", "d3"],
        "operating_temperature": [75.2, 80.1, 95.4],
        "abnormal_shutdown_count": [0, 1, 5],
    })

    # Asking for 'failure_flag' when it doesn't exist should NOT guess or auto-substitute
    intent = SemanticIntent(
        intent_type=IntentType.CLASSIFICATION,
        primary_intent=IntentType.CLASSIFICATION,
        target_column="failure_flag",
        raw_query="Train classification model for failure_flag",
    )
    resolver = IntentResolver()
    with pytest.raises(IntentResolutionError) as exc_info:
        resolver.resolve(intent, df=df_unseen)

    assert exc_info.value.code == "COLUMN_NOT_FOUND"
    assert "failure_flag" in exc_info.value.invalid_columns
