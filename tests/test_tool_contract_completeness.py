"""Comprehensive verification suite for Phase 2: Strict Capability Registry.

Validates:
1. Every canonical tool satisfies ToolSpec.validate_contract() with complete contracts.
2. Every dataset tool declares explicit allowed_predecessors and task_family.
3. Every conversational tool has requires_dataset=False and empty allowed_predecessors.
4. Fake model persistence is strictly banned (RequestedArtifact.MODEL cannot be claimed or produced).
5. Chart contract parity: RequestedArtifact.CHART is bidirectional with physical chart types.
6. Execution DAG predecessor rules are enforced deterministically by ExecutionGraphCompiler.
"""

import pytest
from pydantic import BaseModel, ConfigDict
from backend.ds_engine.analysis_spec_v3 import (
    AnalysisDecision,
    AnalysisMode,
    AnalysisObjective,
    RequestedArtifact,
)
from backend.ds_engine.execution_graph import ExecutionGraphCompiler
from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily
from modules.capability_registry import (
    CapabilityRegistry,
    ToolSpec,
    CompletionPolicy,
    RegistryContractError,
    ToolPrerequisites,
    capability_registry,
)


def test_all_canonical_tools_pass_contract_validation():
    """Verify all 22 registered tools pass strict contract validation."""
    specs = capability_registry.list_tool_specs()
    assert len(specs) >= 22

    canonical_names = {
        "filter_data", "sort_data", "conditional_rate", "group_data", "aggregate_data",
        "audit_metric_distribution", "correlate_data", "forecast_data", "clean_data",
        "detect_anomalies", "eda_data", "train_model", "cluster_data", "statistical_test",
        "categorical_outcome_association", "pca_analysis", "explain_features",
        "autocorrelation", "decompose_time_series", "event_window_analysis",
        "general_chat", "clarification_needed"
    }

    registered_names = {s.name for s in specs}
    assert canonical_names.issubset(registered_names)

    for spec in specs:
        # Must pass without raising ValueError
        spec.validate_contract()
        assert spec.name.strip()
        assert spec.description.strip()
        assert issubclass(spec.input_model, BaseModel)
        assert spec.objective in AnalysisObjective
        assert len(spec.analysis_modes) >= 1
        assert len(spec.supported_artifacts) >= 1
        assert spec.supported_chart_types is not None and len(spec.supported_chart_types) >= 1
        assert callable(spec.adapter)
        assert spec.result_contract in {"dataframe", "text", "plot", "metric"}
        assert spec.side_effect_level in {"read_only", "modifies_data", "trains_model"}


def test_dataset_tools_declare_explicit_predecessors_and_task_family():
    """Every dataset tool must declare task_family and non-null allowed_predecessors."""
    specs = capability_registry.list_tool_specs()
    dataset_tools = [s for s in specs if s.requires_dataset]
    assert len(dataset_tools) == 22

    for tool in dataset_tools:
        assert tool.task_family is not None, f"Tool '{tool.name}' missing task_family"
        assert tool.allowed_predecessors is not None, f"Tool '{tool.name}' missing allowed_predecessors"
        assert isinstance(tool.allowed_predecessors, list), f"Tool '{tool.name}' predecessors must be list"
        # Pure preprocessing tools only allow preprocessing predecessors
        if tool.name == "clean_data":
            assert set(tool.allowed_predecessors) == {"clean_data"}
        elif tool.name == "filter_data":
            assert "clean_data" in tool.allowed_predecessors
            assert "filter_data" in tool.allowed_predecessors


def test_conversational_tools_disallow_dag_execution():
    """Conversational tools must have requires_dataset=False and empty predecessors."""
    for tool_name in ["general_chat", "clarification_needed"]:
        spec = capability_registry.get_tool_spec(tool_name)
        assert spec is not None
        assert spec.requires_dataset is False
        assert spec.allowed_predecessors == []
        assert spec.result_contract == "text"


def test_banned_fake_model_persistence():
    """Zero tools may declare RequestedArtifact.MODEL in supported_artifacts."""
    specs = capability_registry.list_tool_specs()
    for spec in specs:
        assert RequestedArtifact.MODEL not in spec.supported_artifacts, (
            f"Tool '{spec.name}' claims RequestedArtifact.MODEL; fake model persistence is banned."
        )

    # Attempting to register a tool with RequestedArtifact.MODEL must fail
    class DummyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    bad_spec = ToolSpec(
        name="fake_model_tool",
        description="Claims to save a model file",
        input_model=DummyArgs,
        objective=AnalysisObjective.PREDICT,
        analysis_modes=[AnalysisMode.PROSPECTIVE],
        supported_artifacts=[RequestedArtifact.MODEL, RequestedArtifact.TABLE],
        requires_dataset=True,
        adapter=lambda args, state: None,
        task_family=TaskFamily.SUPERVISED_ML,
        allowed_predecessors=["filter_data"],
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )

    with pytest.raises(ValueError, match="cannot declare RequestedArtifact.MODEL"):
        bad_spec.validate_contract()


def test_toolspec_missing_completion_policy_fails_registry_validation():
    """A ToolSpec without an explicit CompletionPolicy must fail contract validation."""
    class DummyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    spec_without_policy = ToolSpec(
        name="no_policy_tool",
        description="Tool lacking CompletionPolicy",
        input_model=DummyArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        adapter=lambda args, state: None,
        task_family=TaskFamily.GENERAL_EDA,
        allowed_predecessors=["clean_data"],
        supported_chart_types=["none"],
        completion_policy=None,
    )
    with pytest.raises(RegistryContractError, match="must declare a valid CompletionPolicy"):
        spec_without_policy.validate_contract()


def test_compiling_plan_requesting_model_artifact_is_rejected():
    """An analytical intent requesting 'model' artifact must be rejected by compiler."""
    v2 = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        required_artifacts=["model"],
        raw_query="Train a model and save it as an artifact.",
    )
    with pytest.raises(ValueError, match="cannot produce requested artifacts"):
        capability_registry.compile_v3_from_v2(
            tool_name="train_model",
            intent=v2,
            dataset_id="churn_data",
            arguments={
                "target_column": "churn",
                "feature_columns": ["tenure", "monthly_charges"],
                "task_type": "classification",
                "evaluation_artifact": "confusion_matrix",
                "prediction_context": "prospective_prediction",
            },
            analysis_mode=AnalysisMode.PROSPECTIVE,
        )


def test_chart_contract_parity_validation():
    """RequestedArtifact.CHART and supported_chart_types must be strictly consistent."""
    class DummyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    # 1. Declares chart types but no CHART artifact -> fails
    spec_missing_artifact = ToolSpec(
        name="dummy_chart_tool_1",
        description="Dummy tool",
        input_model=DummyArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        adapter=lambda args, state: None,
        task_family=TaskFamily.GENERAL_EDA,
        allowed_predecessors=["clean_data"],
        supported_chart_types=["bar"],
        completion_policy=CompletionPolicy(),
    )
    with pytest.raises(ValueError, match="declares chart types but not the chart artifact"):
        spec_missing_artifact.validate_contract()

    # 2. Declares CHART artifact but only ['none'] -> fails
    spec_missing_chart_types = ToolSpec(
        name="dummy_chart_tool_2",
        description="Dummy tool",
        input_model=DummyArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.CHART],
        requires_dataset=True,
        adapter=lambda args, state: None,
        task_family=TaskFamily.GENERAL_EDA,
        allowed_predecessors=["clean_data"],
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )
    with pytest.raises(ValueError, match="declares RequestedArtifact.CHART but no specific chart types"):
        spec_missing_chart_types.validate_contract()


def test_dag_predecessor_allowed_chains():
    """Verify valid multi-step DAG chains compile cleanly."""
    # 1. clean_data -> filter_data -> group_data
    commands = [
        {"tool": "clean_data", "arguments": {"method": "drop_missing", "column": "amount"}},
        {"tool": "filter_data", "arguments": {"column": "amount", "operator": ">", "value": 0}},
        {"tool": "group_data", "arguments": {
            "group_by": ["category"], "metric": "amount", "aggregation": "sum", "chart_type": "bar"
        }},
    ]
    graph = ExecutionGraphCompiler.compile(commands, capability_registry)
    assert len(graph.nodes) == 3
    assert [node.tool for node in graph.nodes] == ["clean_data", "filter_data", "group_data"]
    assert graph.nodes[0].depends_on == []
    assert graph.nodes[1].depends_on == ["op_1"]
    assert graph.nodes[2].depends_on == ["op_2"]

    # 2. filter_data -> train_model (terminal)
    commands_ml = [
        {"tool": "filter_data", "arguments": {"column": "tenure", "operator": ">=", "value": 1}},
        {
            "tool": "train_model",
            "arguments": {
                "target_column": "churn",
                "feature_columns": ["tenure", "monthly_charges"],
                "task_type": "classification",
                "evaluation_artifact": "confusion_matrix",
                "prediction_context": "prospective_prediction",
            },
        },
    ]
    graph_ml = ExecutionGraphCompiler.compile(commands_ml, capability_registry)
    assert len(graph_ml.nodes) == 2
    assert [node.tool for node in graph_ml.nodes] == ["filter_data", "train_model"]

    # 3. group_data -> sort_data
    commands_sort = [
        {"tool": "group_data", "arguments": {
            "group_by": ["region"], "metric": "sales", "aggregation": "mean", "chart_type": "none"
        }},
        {"tool": "sort_data", "arguments": {"column": "sales", "ascending": False}},
    ]
    graph_sort = ExecutionGraphCompiler.compile(commands_sort, capability_registry)
    assert len(graph_sort.nodes) == 2
    assert [node.tool for node in graph_sort.nodes] == ["group_data", "sort_data"]


def test_dag_predecessor_disallowed_chains():
    """Verify invalid DAG chains are deterministically rejected with clear errors."""
    # 1. train_model cannot precede correlate_data (side effect + predecessor mismatch)
    commands_invalid_1 = [
        {
            "tool": "train_model",
            "arguments": {
                "target_column": "churn",
                "feature_columns": ["tenure"],
                "task_type": "classification",
                "evaluation_artifact": "none",
                "prediction_context": "prospective_prediction",
            },
        },
        {
            "tool": "correlate_data",
            "arguments": {
                "columns": ["tenure", "monthly_charges"],
                "analysis_mode": "pairwise",
                "methods": ["pearson"],
                "chart_type": "none",
            },
        },
    ]
    with pytest.raises(ValueError, match="INVALID_PLAN_SEQUENCE.*must be the final plan node"):
        ExecutionGraphCompiler.compile(commands_invalid_1, capability_registry)

    # 2. forecast_data cannot precede clean_data
    commands_invalid_2 = [
        {
            "tool": "forecast_data",
            "arguments": {
                "target_column": "sales",
                "time_column": "date",
                "horizon": 12,
                "horizon_evidence": "next 12 months",
                "frequency": "MS",
                "method": "holt_linear",
                "chart_type": "none",
            },
        },
        {"tool": "clean_data", "arguments": {"method": "drop_missing", "column": "sales"}},
    ]
    with pytest.raises(ValueError, match="INVALID_PLAN_SEQUENCE.*does not allow predecessor 'forecast_data'"):
        ExecutionGraphCompiler.compile(commands_invalid_2, capability_registry)

    # 3. general_chat cannot be in an execution graph
    commands_invalid_3 = [
        {"tool": "general_chat", "arguments": {"message": "hello"}},
        {"tool": "filter_data", "arguments": {"column": "sales", "operator": ">", "value": 0}},
    ]
    with pytest.raises(ValueError, match="Conversational/clarification tools cannot be execution nodes"):
        ExecutionGraphCompiler.compile(commands_invalid_3, capability_registry)
