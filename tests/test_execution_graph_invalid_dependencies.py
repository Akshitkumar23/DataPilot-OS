"""
DataPilot OS — Execution Graph Dependency & Predecessor Validation Test Suite.
Verifies DAG invariants:
1. Validated operation IDs (op_1, op_2, ...) with explicit dependency links.
2. Cycle detection and unknown dependency rejection.
3. Strict enforcement of declared predecessor combinations.
4. Model-training operations are strictly terminal (cannot have successors).
5. Conversational / clarification tools are forbidden inside an execution DAG.
6. Explicit dataframe input/output contract enforcement across nodes.
"""

import pytest
from backend.ds_engine.execution_graph import (
    ExecutionGraph,
    ExecutionGraphNode,
    ExecutionGraphCompiler,
)
from modules.capability_registry import capability_registry


def test_valid_operation_ids_and_dependency_links():
    """Verify linear DAG generates op_1 -> op_2 -> op_3 with explicit depends_on."""
    commands = [
        {"tool": "clean_data", "arguments": {"method": "drop_missing", "column": "sales"}},
        {"tool": "filter_data", "arguments": {"column": "sales", "operator": ">", "value": 100}},
        {"tool": "group_data", "arguments": {
            "group_by": ["region"], "metric": "sales", "aggregation": "sum", "chart_type": "bar"
        }},
    ]
    graph = ExecutionGraphCompiler.compile(commands, capability_registry)

    assert len(graph.nodes) == 3
    assert graph.nodes[0].operation_id == "op_1"
    assert graph.nodes[0].depends_on == []
    assert graph.nodes[1].operation_id == "op_2"
    assert graph.nodes[1].depends_on == ["op_1"]
    assert graph.nodes[2].operation_id == "op_3"
    assert graph.nodes[2].depends_on == ["op_2"]


def test_graph_rejects_unknown_dependency_ids():
    """Directly validating an ExecutionGraph with unknown depends_on ID raises ValueError."""
    node1 = ExecutionGraphNode(
        operation_id="op_1",
        step_index=1,
        tool="filter_data",
        arguments={"column": "x", "operator": ">", "value": 0},
        depends_on=["nonexistent_op"],
        input_contract="dataframe",
        output_contract="dataframe",
        task_family="filtering",
    )
    with pytest.raises(ValueError, match="depends on unavailable prior nodes"):
        ExecutionGraph(nodes=[node1])


def test_graph_rejects_duplicate_operation_ids():
    """Duplicate operation IDs in DAG nodes are strictly rejected."""
    node1 = ExecutionGraphNode(
        operation_id="op_1",
        step_index=1,
        tool="filter_data",
        arguments={"column": "x", "operator": ">", "value": 0},
        depends_on=[],
        input_contract="dataframe",
        output_contract="dataframe",
        task_family="filtering",
    )
    node2 = ExecutionGraphNode(
        operation_id="op_1",  # duplicate ID
        step_index=2,
        tool="filter_data",
        arguments={"column": "y", "operator": ">", "value": 1},
        depends_on=["op_1"],
        input_contract="dataframe",
        output_contract="dataframe",
        task_family="filtering",
    )
    with pytest.raises(ValueError, match="operation IDs must be unique"):
        ExecutionGraph(nodes=[node1, node2])


def test_train_model_as_intermediate_node_is_rejected():
    """Train model cannot be followed by another operator; it must be terminal."""
    commands = [
        {
            "tool": "train_model",
            "arguments": {
                "target_column": "churn",
                "feature_columns": ["tenure", "monthly_charges"],
                "task_type": "classification",
                "evaluation_artifact": "confusion_matrix",
                "prediction_context": "prospective_prediction",
                "data_split": "train_test_split",
            },
        },
        {"tool": "filter_data", "arguments": {"column": "tenure", "operator": ">", "value": 10}},
    ]
    with pytest.raises(ValueError, match="must be the final plan node"):
        ExecutionGraphCompiler.compile(commands, capability_registry)


def test_conversational_tool_inside_graph_is_rejected():
    """Clarification and general chat cannot be nodes in an execution DAG."""
    commands = [
        {"tool": "filter_data", "arguments": {"column": "x", "operator": ">", "value": 0}},
        {"tool": "clarification_needed", "arguments": {"reason": "Which region?"}},
    ]
    with pytest.raises(ValueError, match="Conversational/clarification tools cannot be execution nodes"):
        ExecutionGraphCompiler.compile(commands, capability_registry)


def test_disallowed_predecessor_combination_is_rejected():
    """Clean data cannot follow forecast data."""
    commands = [
        {
            "tool": "forecast_data",
            "arguments": {
                "target_column": "sales",
                "time_column": "date",
                "horizon": 10,
                "horizon_evidence": "10 days",
                "frequency": "D",
                "method": "holt_linear",
                "chart_type": "none",
            },
        },
        {"tool": "clean_data", "arguments": {"method": "drop_missing", "column": "sales"}},
    ]
    with pytest.raises(ValueError, match="does not allow predecessor"):
        ExecutionGraphCompiler.compile(commands, capability_registry)


def test_clean_train_pipeline_is_permitted():
    """Clean data followed by train model is a valid predecessor combination."""
    commands = [
        {"tool": "clean_data", "arguments": {"method": "drop_missing", "column": "churn"}},
        {
            "tool": "train_model",
            "arguments": {
                "target_column": "churn",
                "feature_columns": ["tenure"],
                "task_type": "classification",
                "evaluation_artifact": "confusion_matrix",
                "prediction_context": "prospective_prediction",
            },
        },
    ]
    graph = ExecutionGraphCompiler.compile(commands, capability_registry)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].tool == "clean_data"
    assert graph.nodes[1].tool == "train_model"
    assert graph.nodes[1].depends_on == ["op_1"]
