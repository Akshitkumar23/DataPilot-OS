import pytest
from pydantic import ValidationError

from backend.ds_engine.analysis_spec_v3 import (
    AnalysisDecision,
    AnalysisMode,
    AnalysisObjective,
    AnalysisOperationV3,
    AnalysisSpecV3,
    ClarificationRequestV3,
    DatasetBindingV3,
)
from backend.ds_engine.intent_schema import (
    AnalysisIntentV2,
    FilterSpec,
    TaskFamily,
    TimeContext,
    VisualizationSpec,
)
from modules.capability_registry import capability_registry


def _executable(**updates):
    payload = {
        "decision": AnalysisDecision.EXECUTE,
        "dataset": DatasetBindingV3(dataset_id="dataset-1"),
        "objective": AnalysisObjective.DESCRIBE,
        "analysis_mode": AnalysisMode.DESCRIPTIVE,
        "operations": [AnalysisOperationV3(operation_id="op_1", tool="group_data", arguments={})],
        "raw_query": "Show a grouped summary.",
    }
    payload.update(updates)
    return AnalysisSpecV3(**payload)


def test_executable_contract_is_strict_and_versioned():
    spec = _executable()
    assert spec.schema_version == "3.0"
    with pytest.raises(ValidationError):
        AnalysisSpecV3(**{**spec.model_dump(), "unexpected": True})


@pytest.mark.parametrize("field", ["dataset", "objective", "analysis_mode"])
def test_execute_requires_grounding_fields(field):
    with pytest.raises(ValidationError, match=field):
        _executable(**{field: None})


def test_execute_and_clarification_are_mutually_exclusive():
    clarification = ClarificationRequestV3(
        question="Which metric?", missing_fields=["metric"]
    )
    with pytest.raises(ValidationError, match="cannot also request clarification"):
        _executable(clarification=clarification)


def test_clarification_cannot_hide_an_executable_operation():
    with pytest.raises(ValidationError, match="Non-executable"):
        AnalysisSpecV3(
            decision=AnalysisDecision.CLARIFY,
            dataset=DatasetBindingV3(dataset_id="dataset-1"),
            objective=None,
            analysis_mode=None,
            operations=[AnalysisOperationV3(operation_id="op_1", tool="group_data", arguments={})],
            clarification=ClarificationRequestV3(
                question="Which metric?", missing_fields=["metric"]
            ),
            raw_query="Summarize it.",
        )


def test_operation_dag_rejects_unknown_dependencies_and_cycles():
    with pytest.raises(ValidationError, match="unknown dependencies"):
        _executable(
            operations=[
                AnalysisOperationV3(
                    operation_id="op_1", tool="group_data", arguments={}, depends_on=["missing"]
                )
            ]
        )
    with pytest.raises(ValidationError, match="cycle"):
        _executable(
            operations=[
                AnalysisOperationV3(operation_id="op_1", tool="filter_data", arguments={}, depends_on=["op_2"]),
                AnalysisOperationV3(operation_id="op_2", tool="group_data", arguments={}, depends_on=["op_1"]),
            ]
        )


def test_unsupported_and_unsafe_require_reason():
    for decision in (AnalysisDecision.UNSUPPORTED, AnalysisDecision.UNSAFE):
        with pytest.raises(ValidationError, match="require a reason"):
            AnalysisSpecV3(
                decision=decision,
                dataset=None,
                objective=None,
                analysis_mode=None,
                raw_query="Do something unavailable.",
            )


def test_v2_migration_preserves_explicit_semantic_evidence():
    v2 = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["revenue"],
        grouping=["region"],
        time_column="ordered_at",
        time_context=TimeContext(time_column="ordered_at", frequency="D"),
        conditions=[FilterSpec(column="status", operator="eq", value="complete")],
        visualization=VisualizationSpec(type="bar", x="region", y="revenue"),
        required_artifacts=["table", "chart", "written_finding"],
        assumptions=["Null metric rows are excluded."],
        raw_query="Show mean revenue by region as a bar chart.",
    )
    arguments = {
        "group_by": ["region"],
        "metric": "revenue",
        "aggregation": "mean",
        "filters": [{"column": "status", "operator": "eq", "value": "complete"}],
    }
    spec = AnalysisSpecV3.from_v2(
        v2,
        dataset_id="orders",
        objective=AnalysisObjective.COMPARE,
        analysis_mode=AnalysisMode.RETROSPECTIVE,
        tool="group_data",
        arguments=arguments,
    )
    assert spec.dataset.dataset_id == "orders"
    assert spec.operations[0].arguments == arguments
    assert spec.columns.metrics == ["revenue"]
    assert spec.columns.dimensions == ["region"]
    assert spec.filters[0].column == "status"
    assert spec.time_axis.column == "ordered_at"
    assert spec.visualizations[0].chart_type == "bar"
    assert spec.raw_query == v2.raw_query


def test_v2_migration_rejects_unknown_artifact_instead_of_defaulting():
    v2 = AnalysisIntentV2(
        task_family=TaskFamily.GENERAL_EDA,
        required_artifacts=["mystery_artifact"],
        raw_query="Inspect data.",
    )
    with pytest.raises(ValueError, match="unsupported artifact"):
        AnalysisSpecV3.from_v2(
            v2,
            dataset_id="dataset-1",
            objective=AnalysisObjective.DESCRIBE,
            analysis_mode=AnalysisMode.DESCRIPTIVE,
            tool="eda_data",
            arguments={},
        )


def test_registry_manifest_is_complete_for_every_canonical_tool():
    manifest = capability_registry.capability_manifest()
    assert len(manifest) == 24
    assert len({entry["name"] for entry in manifest}) == len(manifest)
    for entry in manifest:
        assert entry["objective"]
        assert entry["analysis_modes"]
        assert entry["supported_artifacts"]
        assert entry["input_schema"]["type"] == "object"


def test_registry_compiler_validates_arguments_and_preserves_tool_identity():
    v2 = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["revenue"],
        grouping=["region"],
        visualization=VisualizationSpec(type="bar", x="region", y="revenue"),
        required_artifacts=["table", "chart", "written_finding"],
        raw_query="Show mean revenue by region as a bar chart.",
    )
    compiled = capability_registry.compile_v3_from_v2(
        tool_name="group_data",
        intent=v2,
        dataset_id="orders",
        arguments={
            "group_by": ["region"],
            "metric": "revenue",
            "aggregation": "mean",
            "chart_type": "bar",
        },
        analysis_mode=AnalysisMode.RETROSPECTIVE,
    )
    assert compiled.objective == AnalysisObjective.COMPARE
    assert compiled.operations[0].tool == "group_data"
    assert compiled.operations[0].arguments["aggregation"] == "mean"


def test_registry_compiler_rejects_mode_and_artifact_mismatch():
    v2 = AnalysisIntentV2(
        task_family=TaskFamily.FILTER_SLICE,
        required_artifacts=["model"],
        raw_query="Filter rows and train a model.",
    )
    with pytest.raises(ValueError, match="does not support analysis mode"):
        capability_registry.compile_v3_from_v2(
            tool_name="filter_data",
            intent=v2,
            dataset_id="orders",
            arguments={"column": "status", "condition": "==", "value": "complete"},
            analysis_mode=AnalysisMode.PROSPECTIVE,
        )
    with pytest.raises(ValueError, match="cannot produce requested artifacts"):
        capability_registry.compile_v3_from_v2(
            tool_name="filter_data",
            intent=v2,
            dataset_id="orders",
            arguments={"column": "status", "condition": "==", "value": "complete"},
            analysis_mode=AnalysisMode.RETROSPECTIVE,
        )
