"""
Comprehensive End-to-End Verification Test for True Zero-Silent-Fallback & Architectural Hardening.
Tests:
1. Strict IntentType validation (No silent fallback to EDA)
2. Strict FilterSpec operator validation (No silent fallback to EQ)
3. Zero Filter Dropping (Invalid filter fails command resolution instead of running unfiltered)
4. Dynamic Single Source of Truth in CapabilityRegistry
5. Ambiguity Candidate Columns Error Contract
6. Safe Repr string filtering (quotes & special characters)
7. Pure General Chat conversational execution
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from backend.ds_engine.intent_schema import SemanticIntent, IntentType, FilterSpec, FilterOperator
from backend.ds_engine.command_resolver import CommandResolver, command_resolver
from modules.capability_registry import capability_registry
from modules.models.execution_result import QueryErrorDetail, LifecycleState
from modules.query_orchestrator import QueryOrchestrator
from modules.execution_engine import ModularExecutionEngine
from backend.ds_engine.intent_resolver import intent_resolver
from backend.ds_engine.code_generator import code_generator


def test_intent_type_strict_validation():
    """Verify SemanticIntent raises ValueError when intent_type is unrecognized (NO SILENT EDA)."""
    with pytest.raises(ValueError, match="Invalid intent_type"):
        SemanticIntent.model_validate({"intent_type": "completely_unknown_intent_xyz"})


def test_filter_spec_operator_strict_validation():
    """Verify FilterSpec raises ValueError when operator is unrecognized (NO SILENT EQ)."""
    with pytest.raises(ValueError, match="Invalid filter operator"):
        FilterSpec.model_validate({"column": "sales", "operator": "invalid_op_123", "value": 100})


def test_zero_filter_dropping_in_command_resolver():
    """
    Verify CommandResolver fails resolution with INVALID_FILTER_SPEC when a filter operator is invalid,
    instead of dropping the filter and running unfiltered queries.
    """
    cmd = {
        "commands": [
            {"tool": "filter_data", "arguments": {"column": "sales", "condition": "invalid_op_xyz", "value": 100}},
            {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "sales", "aggregation": "sum"}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=None, raw_query="filter sales invalid_op_xyz 100 and group by region")
    assert res.is_valid is False
    assert res.error_code == "INVALID_FILTER_SPEC"


def test_capability_registry_dynamic_single_source_of_truth():
    """Verify that registering a new capability dynamically updates definitions, prompts, and intent map."""
    test_tool_name = "test_custom_anomaly_tool"
    capability_registry.register_tool(
        tool_name=test_tool_name,
        schema_signature="{column: col, threshold: 2.5}",
        intent_type=IntentType.ANOMALY_DETECTION,
        synonyms=["custom_anomaly"],
        description="(for custom z-score anomalies)"
    )

    # 1. Check definitions export
    defs = capability_registry.get_planner_tool_definitions()
    assert test_tool_name in defs

    # 2. Check prompt export
    prompt = capability_registry.get_planner_tools_prompt()
    assert test_tool_name in prompt
    assert "{column: col, threshold: 2.5}" in prompt

    # 3. Check tool intent map
    tool_map = capability_registry.get_tool_intent_map()
    assert tool_map.get(test_tool_name) == IntentType.ANOMALY_DETECTION
    assert tool_map.get("custom_anomaly") == IntentType.ANOMALY_DETECTION


def test_ambiguity_candidate_columns_contract():
    """Verify QueryErrorDetail preserves candidate_columns for AMBIGUOUS_COLUMN."""
    err = QueryErrorDetail(
        code="AMBIGUOUS_COLUMN",
        message="Multiple date columns found",
        candidate_columns=["order_date", "ship_date", "delivery_date"]
    )
    dumped = err.model_dump()
    assert dumped["code"] == "AMBIGUOUS_COLUMN"
    assert dumped["candidate_columns"] == ["order_date", "ship_date", "delivery_date"]


def test_safe_repr_string_filter_execution():
    """Verify filter execution handles strings with quotes safely without syntax errors."""
    df = pd.DataFrame({
        "customer_name": ["O'Connor", "Smith", "D'Souza", "Taylor"],
        "revenue": [500, 300, 700, 200]
    })
    cmd = {
        "commands": [
            {"tool": "filter_data", "arguments": {"column": "customer_name", "condition": "==", "value": "O'Connor"}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=df, raw_query="filter customer_name is O'Connor")
    assert res.is_valid is True
    assert len(res.semantic_intent.filters) == 1
    assert res.semantic_intent.filters[0].value == "O'Connor"

    # Compile and execute code
    resolved = intent_resolver.resolve(res.semantic_intent, df)
    code_ver = code_generator.generate_code(resolved)
    exec_out = ModularExecutionEngine().execute_code(code_ver.code, df)
    assert exec_out["status"] == "success"
    assert exec_out["dataframe"] is not None
    assert len(exec_out["dataframe"]) == 1
    assert exec_out["dataframe"]["customer_name"].iloc[0] == "O'Connor"


def test_pure_general_chat_execution():
    """Verify pure general chat execution generates markdown answers without dataset."""
    query = "What is the difference between supervised and unsupervised learning?"
    res = QueryOrchestrator.process_query(query_raw=query, df=None)
    assert res.status == "SUCCEEDED"
    assert res.result["type"] == "text"
    assert len(res.result["response"]) > 20


def test_invalid_aggregation_strict_validation():
    """Verify SemanticIntent raises ValueError when aggregation is unrecognized (NO SILENT SUM)."""
    with pytest.raises(ValueError, match="Invalid aggregation"):
        SemanticIntent.model_validate({
            "intent_type": "GROUP_AGGREGATE",
            "dimensions": ["region"],
            "metrics": ["sales"],
            "aggregation": "completely_invalid_aggregation_xyz"
        })


def test_filters_exception_not_swallowed():
    """Verify SemanticIntent raises ValueError when an invalid filter is inside the filters list."""
    with pytest.raises(ValueError, match="Invalid filter operator"):
        SemanticIntent.model_validate({
            "intent_type": "FILTER_QUERY",
            "filters": [
                {"column": "sales", "operator": "invalid_op_xyz", "value": 500}
            ]
        })


def test_empty_filter_data_command_rejected():
    """Verify CommandResolver rejects empty filter_data tool calls with INVALID_FILTER_SPEC."""
    cmd = {
        "commands": [
            {"tool": "filter_data", "arguments": {}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=None, raw_query="filter data")
    assert res.is_valid is False
    assert res.error_code == "INVALID_FILTER_SPEC"


def test_contains_filter_literal_matching():
    """Verify str.contains executes with regex=False for literal pattern matching."""
    df = pd.DataFrame({
        "product_code": ["Item (A)", "Item [B]", "Item C", "Item (A)+1"],
        "stock": [10, 20, 30, 40]
    })
    cmd = {
        "commands": [
            {"tool": "filter_data", "arguments": {"column": "product_code", "condition": "contains", "value": "(A)"}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=df, raw_query="filter product_code contains (A)")
    assert res.is_valid is True
    assert res.semantic_intent.filters[0].operator == FilterOperator.CONTAINS

    resolved = intent_resolver.resolve(res.semantic_intent, df)
    code_ver = code_generator.generate_code(resolved)
    exec_out = ModularExecutionEngine().execute_code(code_ver.code, df)
    assert exec_out["status"] == "success"
    assert exec_out["dataframe"] is not None
    assert len(exec_out["dataframe"]) == 2
    assert "regex=False" in code_ver.code
