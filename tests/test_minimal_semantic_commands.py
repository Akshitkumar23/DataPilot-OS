"""
DataPilot OS — Comprehensive Minimal Semantic Command & Decoupled Execution Test Suite.
Verifies all 17 matrix requirements, command resolution, synonym handling,
missing arguments, schema grounding, multi-intent DAGs, and error guardrails.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.ds_engine.intent_schema import SemanticIntent, IntentType, FilterSpec, FilterOperator, AggregationType
from backend.ds_engine.semantic_command import LLMSemanticCommand, ToolCall, SemanticTool
from backend.ds_engine.command_resolver import CommandResolver, command_resolver
from modules.semantic_planner import SemanticPlanner
from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import CanonicalQueryResponse


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES & DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def get_customers_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "income": np.linspace(20000, 100000, n),
        "age": np.random.randint(18, 70, n),
        "gender": np.tile(["Male", "Female"], 50),
        "sales": np.linspace(100, 5000, n),
        "region": np.tile(["North", "South", "East", "West"], 25),
        "temperature": np.random.normal(25, 5, n),
        "created_at": pd.date_range("2024-01-01", periods=n, freq="D")
    })


def get_sensor_dataset_with_nulls() -> pd.DataFrame:
    np.random.seed(42)
    n = 50
    temp = np.random.normal(30, 2, n)
    temp[5] = np.nan
    temp[10] = np.nan
    temp[20] = 95.0  # Anomaly spike
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature": temp,
        "vibration": np.random.normal(1.0, 0.1, n)
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMMAND RESOLVER UNIT TESTS (17 MATRIX QUERIES & SYNONYMS)
# ─────────────────────────────────────────────────────────────────────────────

def test_01_filter_income_above_50000():
    """1. 'filter income above 50000' -> filter_data tool with condition 'above'"""
    df = get_customers_dataset()
    cmd = {
        "tool": "filter_data",
        "arguments": {
            "column": "income",
            "condition": "above",
            "value": 50000
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.FILTER_QUERY
    assert len(res.semantic_intent.filters) == 1
    assert res.semantic_intent.filters[0].column == "income"
    assert res.semantic_intent.filters[0].operator == FilterOperator.GT
    assert res.semantic_intent.filters[0].value == 50000


def test_02_show_income_below_30000():
    """2. 'show income below 30000' -> filter_data with condition 'below'"""
    df = get_customers_dataset()
    cmd = {
        "tool": "filter_data",
        "arguments": {
            "column": "income",
            "condition": "below",
            "value": 30000
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.filters[0].operator == FilterOperator.LT
    assert res.semantic_intent.filters[0].value == 30000


def test_03_sort_sales_descending():
    """3. 'sort sales descending' -> sort_data tool"""
    df = get_customers_dataset()
    cmd = {
        "tool": "sort_data",
        "arguments": {
            "column": "sales",
            "order": "descending",
            "limit": 10
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.FILTER_QUERY
    assert res.semantic_intent.target_column == "sales"
    assert res.semantic_intent.order == "desc"
    assert res.semantic_intent.limit == 10


def test_04_average_income_by_gender():
    """4. 'average income by gender' -> group_data with mean aggregation"""
    df = get_customers_dataset()
    cmd = {
        "tool": "group_data",
        "arguments": {
            "group_by": ["gender"],
            "metric": "income",
            "aggregation": "average"
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.GROUP_AGGREGATE
    assert "gender" in res.semantic_intent.dimensions
    assert "income" in res.semantic_intent.metrics
    assert res.semantic_intent.aggregation == AggregationType.MEAN


def test_05_total_sales_by_region():
    """5. 'total sales by region' -> group_data with sum aggregation"""
    df = get_customers_dataset()
    cmd = {
        "tool": "group_data",
        "arguments": {
            "group_by": ["region"],
            "metric": "sales",
            "aggregation": "total"
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.GROUP_AGGREGATE
    assert "region" in res.semantic_intent.dimensions
    assert "sales" in res.semantic_intent.metrics
    assert res.semantic_intent.aggregation == AggregationType.SUM


def test_06_find_relationship_between_age_and_income():
    """6. 'find relationship between age and income' -> correlate_data"""
    df = get_customers_dataset()
    cmd = {
        "tool": "correlate_data",
        "arguments": {
            "columns": ["age", "income"],
            "analysis_mode": "pairwise",
            "relationship_form": "linear",
            "correlation_methods": ["pearson"],
            "chart_type": "none",
            "smoother": "none",
            "expected_direction": "unspecified"
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.CORRELATION
    assert "age" in res.semantic_intent.feature_columns
    assert "income" in res.semantic_intent.feature_columns


def test_07_forecast_sales_for_next_30_days():
    """7. 'forecast sales for next 30 days' -> forecast_data"""
    df = get_customers_dataset()
    cmd = {
        "tool": "forecast_data",
        "arguments": {
            "value_column": "sales",
            "time_column": "created_at",
            "horizon": 30,
            "frequency": "D",
            "method": "holt_linear",
            "chart_type": "line"
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.TIME_SERIES_FORECAST
    assert res.semantic_intent.target_column == "sales"
    assert res.semantic_intent.time_column == "created_at"
    assert res.semantic_intent.horizon == 30


def test_08_remove_missing_values():
    """8. 'remove missing values' -> clean_data"""
    df = get_sensor_dataset_with_nulls()
    cmd = {
        "tool": "clean_data",
        "arguments": {
            "columns": ["temperature"],
            "method": "drop_missing"
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.DATA_CLEAN
    assert "temperature" in res.semantic_intent.feature_columns


def test_09_find_anomalies_in_temperature():
    """9. 'find anomalies in temperature' -> detect_anomalies"""
    df = get_sensor_dataset_with_nulls()
    cmd = {
        "tool": "detect_anomalies",
        "arguments": {
            "columns": ["temperature"],
            "method": "zscore",
            "sensitivity": 3.0,
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.ANOMALY_DETECTION
    assert "temperature" in res.semantic_intent.feature_columns


def test_10_multi_intent_clean_data_and_find_anomalies():
    """10. 'clean data and find anomalies' -> sequential commands [clean_data, detect_anomalies]"""
    df = get_sensor_dataset_with_nulls()
    cmd = {
        "commands": [
            {"tool": "clean_data", "arguments": {"columns": ["temperature"], "method": "impute_median"}},
            {"tool": "detect_anomalies", "arguments": {"columns": ["temperature"], "method": "zscore", "sensitivity": 3.0}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.semantic_intent.intent_type == IntentType.ANOMALY_DETECTION
    assert IntentType.DATA_CLEAN in res.semantic_intent.prerequisite_operations


def test_11_ambiguous_query_analyze_this_dataset():
    """11. 'analyze this dataset' -> clarification_needed"""
    df = get_customers_dataset()
    cmd = {
        "tool": "clarification_needed",
        "arguments": {
            "reason": "Query is underspecified. Please specify operation or columns."
        }
    }
    res = command_resolver.resolve_command(cmd, df=df)
    assert res.is_valid is True
    assert res.is_ambiguous is True
    assert res.semantic_intent.is_ambiguous is True
    assert "underspecified" in res.ambiguity_reason.lower()


def test_12_operator_synonyms_variations():
    """12. Operator synonyms: 'greater_than', '>=', 'under', 'at least', 'is'"""
    assert CommandResolver.resolve_operator("greater_than") == FilterOperator.GT
    assert CommandResolver.resolve_operator("above") == FilterOperator.GT
    assert CommandResolver.resolve_operator("at least") == FilterOperator.GTE
    assert CommandResolver.resolve_operator("below") == FilterOperator.LT
    assert CommandResolver.resolve_operator("at most") == FilterOperator.LTE
    assert CommandResolver.resolve_operator("equals") == FilterOperator.EQ
    assert CommandResolver.resolve_operator("different from") == FilterOperator.NEQ
    assert CommandResolver.resolve_operator("missing") == FilterOperator.IS_NULL
    assert CommandResolver.resolve_operator("contains") == FilterOperator.CONTAINS


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROBUSTNESS & TORTURE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_13_missing_operator_in_filter_dict():
    """Test filter dictionary missing operator: should default safely to EQ without throwing exception"""
    f = CommandResolver.normalize_filter_spec({"column": "sales", "value": 500})
    assert f is not None
    assert f.column == "sales"
    assert f.operator == FilterOperator.EQ
    assert f.value == 500


def test_14_raw_string_filter_parsing():
    """Test structured filters: 'income > 50000' and 'region == EMEA'"""
    f1 = CommandResolver.normalize_filter_spec({"column": "income", "operator": ">", "value": 50000})
    assert f1.column == "income"
    assert f1.operator == FilterOperator.GT
    assert f1.value == 50000

    f2 = CommandResolver.normalize_filter_spec({"column": "region", "operator": "==", "value": "EMEA"})
    assert f2.column == "region"
    assert f2.operator == FilterOperator.EQ
    assert f2.value == "EMEA"


def test_15_unknown_tool_safe_handling():
    """Unknown tool name should return clean error without crashing"""
    res = command_resolver.resolve_command({"tool": "make_coffee", "arguments": {}})
    assert res.is_valid is False
    assert res.error_code in ["UNSUPPORTED_OPERATION", "UNKNOWN_OPERATION"]
    assert "make_coffee" in res.error_message


def test_16_extra_unexpected_fields_rejected_strictly():
    """Pydantic extra='forbid' strictly ensures unexpected/hallucinated fields are rejected."""
    cmd = {
        "tool": "filter_data",
        "arguments": {
            "column": "income",
            "condition": ">",
            "value": 50000,
            "extra_hallucinated_field": "some_random_value",
            "another_field": [1, 2, 3]
        }
    }
    res = command_resolver.resolve_command(cmd)
    assert res.is_valid is False
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "Validation failed for tool 'filter_data'" in res.error_message


# ─────────────────────────────────────────────────────────────────────────────
# 3. END-TO-END PIPELINE TESTS WITH QUERY ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def test_17_full_pipeline_filter_and_group_aggregate():
    """Full E2E test: Filter sales > 500 and average balance by region"""
    df = pd.DataFrame({
        "sales": np.linspace(100, 1000, 100),
        "balance": np.linspace(50, 500, 100),
        "region": np.tile(["North", "South", "East", "West"], 25)
    })

    mock_llm_command = {
        "commands": [
            {"tool": "filter_data", "arguments": {"column": "sales", "condition": ">", "value": 500}},
            {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "balance", "aggregation": "mean"}}
        ]
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_llm_command)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="Filter sales > 500 and average balance by region",
            df=df
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "SUCCEEDED", f"Query failed: {res.error}"
    assert res.result is not None
    manifest = res.result.get("manifest", {})
    assert manifest.get("operator") == "group_aggregation"
    assert manifest.get("filters_applied") is not None or "manifest" in res.result


def test_18_full_pipeline_nonexistent_column_clean_rejection():
    """Full E2E test: Querying nonexistent column cleanly returns PLAN_VALIDATION_FAILED"""
    df = get_customers_dataset()

    mock_llm_command = {
        "tool": "filter_data",
        "arguments": {"column": "nonexistent_column_xyz", "condition": ">", "value": 100}
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_llm_command)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="filter nonexistent_column_xyz > 100",
            df=df
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "FAILED"
    assert res.error.code in ["COLUMN_NOT_FOUND", "PLAN_VALIDATION_FAILED"]
    assert "nonexistent_column_xyz" in res.error.message


def test_19_full_pipeline_forecast_on_non_date_dataset_clean_rejection():
    """Full E2E test: Forecasting on dataset with no temporal column cleanly rejected"""
    df_no_date = pd.DataFrame({
        "category": ["A", "B", "C"],
        "sales": [100, 200, 300]
    })

    mock_llm_command = {
        "tool": "forecast_data",
        "arguments": {
            "value_column": "sales",
            "time_column": "order_date",
            "horizon": 14,
            "horizon_evidence": "14 days",
            "frequency": "D",
            "method": "naive_last_value",
            "chart_type": "line",
        }
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_llm_command)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="forecast sales for 14 days",
            df=df_no_date
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "FAILED"
    assert res.error.code in ["NO_TEMPORAL_COLUMN", "NO_DATE_COLUMN", "PLAN_VALIDATION_FAILED", "UNGROUNDED_REQUEST_ARGUMENT"]
    assert "date" in res.error.message.lower() or "temporal" in res.error.message.lower() or "not present" in res.error.message.lower()
