"""
DataPilot OS — Error UX & Granular Error Codes Diagnostics Test Suite.
Verifies:
1. Nonexistent column query returns COLUMN_NOT_FOUND with similar column suggestions.
2. Ambiguous date/metric returns AMBIGUOUS_COLUMN.
3. Forecasting on non-date dataset returns NO_TEMPORAL_COLUMN.
4. Unsupported operations return UNSUPPORTED_OPERATION.
5. Invalid column types return INVALID_COLUMN_TYPE.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.ds_engine.intent_schema import SemanticIntent, IntentType, AggregationType, FilterSpec, FilterOperator
from backend.ds_engine.intent_resolver import IntentResolver, IntentResolutionError, AmbiguousIntentError
from backend.ds_engine.command_resolver import CommandResolver, command_resolver
from modules.query_orchestrator import QueryOrchestrator
from modules.semantic_planner import SemanticPlanner


@pytest.fixture
def ecommerce_df():
    return pd.DataFrame({
        "revenue": [100.5, 200.0, 350.75, 420.10, 510.0],
        "profit": [20.5, 45.0, 80.0, 95.0, 110.0],
        "region": ["North", "South", "East", "West", "North"],
        "customer_segment": ["Consumer", "Corporate", "Home Office", "Consumer", "Corporate"],
        "order_date": pd.date_range("2024-01-01", periods=5, freq="D")
    })


@pytest.fixture
def no_date_df():
    return pd.DataFrame({
        "item_id": [1, 2, 3, 4, 5],
        "category": ["Electronics", "Furniture", "Clothing", "Groceries", "Automotive"],
        "unit_price": [1200.50, 350.00, 45.99, 2.50, 110.00]
    })


@pytest.fixture
def multi_date_df():
    return pd.DataFrame({
        "created_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "shipped_date": pd.date_range("2024-01-03", periods=10, freq="D"),
        "delivery_date": pd.date_range("2024-01-07", periods=10, freq="D"),
        "sales_volume": [50, 65, 70, 85, 90, 110, 125, 140, 160, 175]
    })


# ==============================================================================
# 1. NONEXISTENT COLUMN & SIMILAR SUGGESTIONS (COLUMN_NOT_FOUND)
# ==============================================================================

def test_missing_column_suggestion_format(ecommerce_df):
    """Test that IntentResolver suggests close column matches with clear formatting."""
    resolver = IntentResolver()
    
    # Typo 'revanue' -> should match 'revenue'
    intent = SemanticIntent(
        intent_type=IntentType.GROUP_AGGREGATE,
        primary_intent=IntentType.GROUP_AGGREGATE,
        target_column="revanue",
        dimensions=["region"],
        aggregation=AggregationType.SUM,
        raw_query="Calculate sum of revanue by region"
    )

    with pytest.raises(IntentResolutionError) as exc_info:
        resolver.resolve(intent, df=ecommerce_df)

    err = exc_info.value
    assert err.code == "COLUMN_NOT_FOUND"
    assert "revanue" in err.invalid_columns
    assert "I couldn't find 'revanue' in the current dataset." in err.message
    assert "Available similar columns:" in err.message
    assert "• revenue" in err.message


def test_missing_column_orchestrator_pipeline(ecommerce_df):
    """Test full QueryOrchestrator pipeline returns COLUMN_NOT_FOUND at PLAN_VALIDATION."""
    mock_plan = {
        "intent_type": "GROUP_AGGREGATE",
        "primary_intent": "GROUP_AGGREGATE",
        "target_column": "proffit",  # Typo for profit
        "dimensions": ["region"],
        "aggregation": "mean",
        "raw_query": "Average proffit by region"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_plan)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="Average proffit by region",
            df=ecommerce_df
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "FAILED"
    assert res.phase == "PLAN_VALIDATION"
    assert res.error is not None
    assert res.error.code == "COLUMN_NOT_FOUND"
    assert "proffit" in res.error.message
    assert "profit" in res.error.message


# ==============================================================================
# 2. AMBIGUOUS SCHEMA RESOLUTION (AMBIGUOUS_COLUMN)
# ==============================================================================

def test_ambiguous_date_column(multi_date_df):
    """Test that multiple date columns without user selection raises AMBIGUOUS_COLUMN."""
    resolver = IntentResolver()
    intent = SemanticIntent(
        intent_type=IntentType.TIME_SERIES_FORECAST,
        primary_intent=IntentType.TIME_SERIES_FORECAST,
        target_column="sales_volume",
        raw_query="Forecast sales_volume for next 7 days"
    )

    with pytest.raises(AmbiguousIntentError) as exc_info:
        resolver.resolve(intent, df=multi_date_df)

    err = exc_info.value
    assert err.code == "AMBIGUOUS_COLUMN"
    assert len(err.candidate_columns) >= 2
    assert "created_date" in err.candidate_columns
    assert "shipped_date" in err.candidate_columns


def test_ambiguous_date_orchestrator_pipeline(multi_date_df):
    """Test full QueryOrchestrator pipeline returns AMBIGUOUS_COLUMN for ambiguous dates."""
    mock_plan = {
        "intent_type": "TIME_SERIES_FORECAST",
        "primary_intent": "TIME_SERIES_FORECAST",
        "target_column": "sales_volume",
        "time_column": None,
        "horizon": 7,
        "raw_query": "Forecast sales_volume for next 7 days"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_plan)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="Forecast sales_volume for next 7 days",
            df=multi_date_df
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "NEEDS_CLARIFICATION"
    assert res.phase == "PLAN_VALIDATION"
    assert res.error is not None
    assert res.error.code == "CLARIFICATION_REQUIRED"


def test_ambiguous_metric_column(ecommerce_df):
    """Test that multiple numeric metric columns without selection raises AMBIGUOUS_COLUMN."""
    resolver = IntentResolver()
    # Group aggregate with dimension specified but no metric specified
    intent = SemanticIntent(
        intent_type=IntentType.GROUP_AGGREGATE,
        primary_intent=IntentType.GROUP_AGGREGATE,
        dimensions=["region"],
        raw_query="Group by region"
    )

    with pytest.raises(AmbiguousIntentError) as exc_info:
        resolver.resolve(intent, df=ecommerce_df)

    err = exc_info.value
    assert err.code == "AMBIGUOUS_COLUMN"
    assert len(err.candidate_columns) >= 2
    assert "revenue" in err.candidate_columns
    assert "profit" in err.candidate_columns


# ==============================================================================
# 3. FORECASTING WITHOUT DATE COLUMN (NO_TEMPORAL_COLUMN)
# ==============================================================================

def test_forecasting_on_non_date_dataset(no_date_df):
    """Test that time-series forecast on a dataset without temporal columns raises NO_TEMPORAL_COLUMN."""
    resolver = IntentResolver()
    intent = SemanticIntent(
        intent_type=IntentType.TIME_SERIES_FORECAST,
        primary_intent=IntentType.TIME_SERIES_FORECAST,
        target_column="unit_price",
        raw_query="Forecast unit_price for next 30 days"
    )

    with pytest.raises(IntentResolutionError) as exc_info:
        resolver.resolve(intent, df=no_date_df)

    err = exc_info.value
    assert err.code == "NO_TEMPORAL_COLUMN"
    assert "date/time column" in err.message.lower() or "temporal" in err.message.lower()


def test_forecasting_on_non_date_orchestrator_pipeline(no_date_df):
    """Test full QueryOrchestrator pipeline returns NO_TEMPORAL_COLUMN."""
    mock_plan = {
        "intent_type": "TIME_SERIES_FORECAST",
        "primary_intent": "TIME_SERIES_FORECAST",
        "target_column": "unit_price",
        "time_column": None,
        "horizon": 30,
        "raw_query": "Forecast unit_price for next 30 days"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_plan)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="Forecast unit_price for next 30 days",
            df=no_date_df
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "FAILED"
    assert res.phase == "PLAN_VALIDATION"
    assert res.error is not None
    assert res.error.code == "NO_TEMPORAL_COLUMN"


# ==============================================================================
# 4. UNSUPPORTED OPERATIONS (UNSUPPORTED_OPERATION)
# ==============================================================================

def test_unsupported_tool_in_command_resolver(ecommerce_df):
    """Test that CommandResolver returns UNSUPPORTED_OPERATION for unknown semantic tools."""
    unknown_command = {
        "tool": "quantum_neural_simulator",
        "arguments": {"qubits": 8}
    }

    res = command_resolver.resolve_command(unknown_command, df=ecommerce_df, raw_query="run quantum simulator")
    assert res.is_valid is False
    assert res.error_code == "UNSUPPORTED_OPERATION"
    assert "quantum_neural_simulator" in res.error_message


def test_unsupported_tool_orchestrator_pipeline(ecommerce_df):
    """Unknown structured tools are rejected during plan validation."""
    mock_command = {
        "tool": "unsupported_crypto_miner",
        "arguments": {}
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_command)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="mine crypto from dataset",
            df=ecommerce_df
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "FAILED"
    assert res.phase == "PLAN_VALIDATION"
    assert res.error is not None
    assert res.error.code == "UNSUPPORTED_OPERATION"
    assert "unsupported_crypto_miner" in res.error.message


# ==============================================================================
# 5. INVALID COLUMN TYPE GUARDRAILS (INVALID_COLUMN_TYPE)
# ==============================================================================

def test_invalid_target_column_type(ecommerce_df):
    """Test that passing a non-numeric column as target for forecast raises INVALID_COLUMN_TYPE."""
    resolver = IntentResolver()
    intent = SemanticIntent(
        intent_type=IntentType.TIME_SERIES_FORECAST,
        primary_intent=IntentType.TIME_SERIES_FORECAST,
        target_column="region",  # 'region' is string/categorical
        time_column="order_date",
        raw_query="Forecast region over time"
    )

    with pytest.raises(IntentResolutionError) as exc_info:
        resolver.resolve(intent, df=ecommerce_df)

    err = exc_info.value
    assert err.code == "INVALID_COLUMN_TYPE"
    assert "region" in err.invalid_columns
    assert "numeric" in err.message.lower()
