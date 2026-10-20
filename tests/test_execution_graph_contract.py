import pytest

from backend.ds_engine.execution_graph import ExecutionGraphCompiler
from modules.capability_registry import capability_registry


def test_ordered_commands_compile_to_explicit_dependency_graph():
    graph = ExecutionGraphCompiler.compile(
        [
            {"tool": "filter_data", "arguments": {"column": "value", "operator": ">", "value": 0}},
            {"tool": "group_data", "arguments": {
                "group_by": ["segment"], "metric": "value", "aggregation": "mean", "chart_type": "none"
            }},
        ],
        capability_registry,
    )
    assert graph.nodes[0].operation_id == "op_1"
    assert graph.nodes[0].depends_on == []
    assert graph.nodes[1].depends_on == ["op_1"]
    assert graph.nodes[1].arguments["aggregation"] == "mean"
    assert all(node.input_contract == "dataframe" for node in graph.nodes)


@pytest.mark.parametrize("commands", [[], [{}], [{"tool": "filter_data"}]])
def test_invalid_sequence_shape_fails_before_execution(commands):
    with pytest.raises(ValueError, match="INVALID_PLAN_SEQUENCE"):
        ExecutionGraphCompiler.compile(commands, capability_registry)


def test_conversation_tool_cannot_be_hidden_inside_execution_graph():
    with pytest.raises(ValueError, match="Conversational/clarification"):
        ExecutionGraphCompiler.compile(
            [{"tool": "general_chat", "arguments": {"message": "hello"}}],
            capability_registry,
        )


def test_graph_compiler_does_not_default_missing_tool_arguments():
    graph = ExecutionGraphCompiler.compile(
        [{"tool": "group_data", "arguments": {"group_by": ["segment"], "metric": "value"}}],
        capability_registry,
    )
    assert graph.nodes[0].arguments["aggregation"] is None
