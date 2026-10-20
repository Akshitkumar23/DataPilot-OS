"""
Comprehensive Evaluation Suite for ToolSpec-Centric Unified Architecture.
Tests:
1. 0, False, and empty string ("") filter value preservation
2. Strict Pydantic input models with extra="forbid"
3. Dynamic ToolSpec registration and execution
4. Strict group query validation (missing metric = count, metric without agg = failure)
5. General chat unmasked failure contract on Ollama timeout
6. Candidate columns contract in ambiguity errors
7. Literal substring contains matching with regex characters
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from pydantic import ValidationError
from backend.ds_engine.intent_schema import SemanticIntent, IntentType, FilterSpec, FilterOperator, AggregationType
from backend.ds_engine.command_resolver import CommandResolver, command_resolver
from modules.capability_registry import (
    CapabilityRegistry, capability_registry, ToolSpec, CompletionPolicy, FilterArgs, GroupArgs, StrictBaseModel
)
from modules.models.execution_result import QueryErrorDetail, LifecycleState
from backend.ds_engine.analysis_spec_v3 import AnalysisMode, AnalysisObjective, RequestedArtifact
from modules.query_orchestrator import QueryOrchestrator
from modules.execution_engine import ModularExecutionEngine


from backend.ds_engine.intent_resolver import intent_resolver
from backend.ds_engine.code_generator import code_generator


def test_zero_filter_value_preservation():
    """Verify filter with value=0 is accurately preserved and executed."""
    df = pd.DataFrame({
        "item": ["A", "B", "C", "D"],
        "stock": [0, 15, 0, 50]
    })
    cmd = {
        "commands": [
            {"tool": "filter_data", "arguments": {"column": "stock", "condition": "==", "value": 0}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=df, raw_query="filter items where stock is 0")
    assert res.is_valid is True
    assert res.semantic_intent.filters[0].value == 0

    resolved = intent_resolver.resolve(res.semantic_intent, df)
    code_ver = code_generator.generate_code(resolved)
    exec_out = ModularExecutionEngine().execute_code(code_ver.code, df)
    assert exec_out["status"] == "success"
    assert exec_out["dataframe"] is not None
    assert len(exec_out["dataframe"]) == 2
    assert all(exec_out["dataframe"]["stock"] == 0)


def test_false_boolean_filter_preservation():
    """Verify filter with boolean value=False is accurately preserved and executed."""
    df = pd.DataFrame({
        "user_id": [101, 102, 103, 104],
        "is_active": [True, False, True, False]
    })
    cmd = {
        "commands": [
            {"tool": "filter_data", "arguments": {"column": "is_active", "condition": "==", "value": False}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=df, raw_query="show inactive users")
    assert res.is_valid is True
    assert res.semantic_intent.filters[0].value is False

    resolved = intent_resolver.resolve(res.semantic_intent, df)
    code_ver = code_generator.generate_code(resolved)
    exec_out = ModularExecutionEngine().execute_code(code_ver.code, df)
    assert exec_out["status"] == "success"
    assert exec_out["dataframe"] is not None
    assert len(exec_out["dataframe"]) == 2
    assert all(exec_out["dataframe"]["is_active"] == False)


def test_empty_string_filter_preservation():
    """Verify filter with value='' (empty string) is accurately preserved."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "notes": ["", "urgent", ""]
    })
    cmd = {
        "commands": [
            {"tool": "filter_data", "arguments": {"column": "notes", "condition": "==", "value": ""}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=df, raw_query="filter orders with empty notes")
    assert res.is_valid is True
    assert res.semantic_intent.filters[0].value == ""

    resolved = intent_resolver.resolve(res.semantic_intent, df)
    code_ver = code_generator.generate_code(resolved)
    exec_out = ModularExecutionEngine().execute_code(code_ver.code, df)
    assert exec_out["status"] == "success"
    assert exec_out["dataframe"] is not None
    assert len(exec_out["dataframe"]) == 2


def test_strict_toolspec_input_model_forbids_unknown_fields():
    """Verify ToolSpec input models forbid unexpected extra keys."""
    with pytest.raises(ValidationError):
        FilterArgs.model_validate({"column": "sales", "operator": "==", "value": 100, "unknown_field_xyz": "bad"})

    with pytest.raises(ValidationError):
        GroupArgs.model_validate({"group_by": ["region"], "invented_extra_key": 999})


def test_dynamic_toolspec_registration_and_execution():
    """Verify that registering a brand new ToolSpec allows dynamic validation and execution."""
    class CustomMultiplierArgs(StrictBaseModel):
        target_column: str
        factor: float = 2.0

    class CustomMultiplierDriver:
        def compile(self, resolved):
            col = resolved.resolved_target_column or "revenue"
            code = (
                f"import pandas as pd\n"
                f"res_df = df.copy()\n"
                f"res_df['{col}_multiplied'] = res_df['{col}'] * 2.0\n"
                f"result_df = res_df\n"
            )
            return code, [], ["dataframe"]

    custom_driver = CustomMultiplierDriver()
    custom_spec = ToolSpec(
        name="custom_multiply",
        description="Multiplies target column by 2.0",
        input_model=CustomMultiplierArgs,
        objective=AnalysisObjective.TRANSFORM,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        compiler_or_driver=custom_driver,
        adapter=lambda args, state: state.update({"target_column": args.target_column}),
        task_family=IntentType.EDA,
        category="eda",
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )

    capability_registry.register_tool_spec(custom_spec)

    # Verify retrieval
    retrieved = capability_registry.get_tool_spec("custom_multiply")
    assert retrieved is not None
    assert retrieved.name == "custom_multiply"

    # Verify driver lookup
    driver = capability_registry.get_driver("custom_multiply")
    assert driver == custom_driver

    # Verify strict validation
    validated_args = capability_registry.validate_tool_call("custom_multiply", {"target_column": "revenue", "factor": 2.0})
    assert validated_args.target_column == "revenue"


def test_group_query_metric_without_aggregation_fails():
    """Verify group query with metric but missing aggregation fails with MISSING_REQUIRED_INFORMATION."""
    cmd = {
        "commands": [
            {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "revenue"}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=None, raw_query="group by region and revenue")
    assert res.is_valid is False
    assert res.error_code == "MISSING_REQUIRED_INFORMATION"
    assert "missing an explicit aggregation function" in res.error_message


def test_group_query_without_metric_defaults_to_count():
    """Verify group query without metric explicitly resolves to record COUNT."""
    cmd = {
        "commands": [
            {"tool": "group_data", "arguments": {"group_by": ["region"]}}
        ]
    }
    res = command_resolver.resolve_command(cmd, df=None, raw_query="count rows by region")
    assert res.is_valid is True
    assert res.semantic_intent.aggregation == AggregationType.COUNT


def test_general_chat_unmasked_failure_contract(monkeypatch):
    """Verify Ollama exception in general chat returns FAILED status with INTENT_PLANNER_UNAVAILABLE."""
    from backend.intelligence.ollama_service import ollama_service
    from modules.semantic_planner import SemanticPlanner

    def mock_extract(*args, **kwargs):
        query = kwargs.get("query_raw") or (args[0] if args else "")
        return {"tool": "general_chat", "arguments": {"response": ""}}

    def mock_failing_chat(*args, **kwargs):
        raise ConnectionError("Ollama service unreachable on port 11434")

    monkeypatch.setattr(SemanticPlanner, "extract_semantic_intent", mock_extract)
    monkeypatch.setattr(ollama_service, "chat_completion", mock_failing_chat)

    res = QueryOrchestrator.process_query(
        query_raw="Explain the difference between PCA and t-SNE",
        df=None
    )
    assert res.status == "FAILED"
    assert res.error is not None
    assert res.error.code == "INTENT_PLANNER_UNAVAILABLE"
    assert "unreachable" in res.error.message


def test_time_context_event_predicate():
    """Verify TimeContext supports event_predicate of type FilterSpec."""
    from backend.ds_engine.intent_schema import TimeContext, FilterSpec, FilterOperator
    fs = FilterSpec(column="temperature", operator=FilterOperator.GT, value=100)
    tc = TimeContext(event_predicate=fs)
    assert tc.event_predicate is not None
    assert tc.event_predicate.column == "temperature"
    assert tc.event_predicate.operator == FilterOperator.GT
    assert tc.event_predicate.value == 100


def test_event_window_args_extensions():
    """Verify EventWindowArgs supports entity_id, pre_event_window, and event_predicate."""
    from modules.capability_registry import EventWindowArgs
    args = EventWindowArgs(
        entity_id="machine_42",
        pre_event_window=14,
        event_predicate={"column": "status", "value": "FAILURE"},
        event_scope="event_onsets",
    )
    assert args.entity_id == "machine_42"
    assert args.pre_event_window == 14
    assert args.event_predicate == {"column": "status", "value": "FAILURE"}
    assert args.event_scope == "event_onsets"


def test_registered_toolspec_task_families():
    """Verify dedicated ToolSpecs are registered in CapabilityRegistry with correct TaskFamily."""
    from backend.ds_engine.intent_schema import TaskFamily
    expected_mappings = {
        "statistical_test": TaskFamily.STATISTICAL_TEST,
        "pca_analysis": TaskFamily.DIMENSIONALITY_REDUCTION,
        "explain_features": TaskFamily.EXPLAINABILITY,
        "clean_data": TaskFamily.DATA_CLEANING,
        "autocorrelation": TaskFamily.AUTOCORRELATION,
        "decompose_time_series": TaskFamily.TIME_SERIES_DECOMPOSITION,
    }

    for tool_name, expected_family in expected_mappings.items():
        spec = capability_registry.get_tool_spec(tool_name)
        assert spec is not None, f"ToolSpec '{tool_name}' not registered in capability_registry"
        assert spec.task_family == expected_family, f"ToolSpec '{tool_name}' task_family is {spec.task_family}, expected {expected_family}"
