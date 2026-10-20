"""Registry contracts must describe artifacts the typed operators can emit."""

from backend.ds_engine.analysis_spec_v3 import RequestedArtifact
from backend.ds_engine.execution_graph import ExecutionGraphCompiler
from modules.capability_registry import capability_registry


def test_every_canonical_capability_declares_a_physical_chart_contract():
    specs = capability_registry.list_tool_specs()
    assert specs
    for spec in specs:
        assert spec.supported_chart_types, spec.name
        if set(spec.supported_chart_types) - {"none"}:
            assert RequestedArtifact.CHART in spec.supported_artifacts, spec.name


def test_ephemeral_model_training_is_only_allowed_as_a_terminal_node():
    train = capability_registry.get_tool_spec("train_model")
    assert train is not None
    assert train.side_effect_level == "trains_model"

    commands = [
        {
            "tool": "filter_data",
            "arguments": {"column": "age", "operator": ">", "value": 18},
        },
        {
            "tool": "train_model",
            "arguments": {
                "target_column": "outcome",
                "feature_columns": ["age"],
                "task_type": "classification",
                "evaluation_artifact": "confusion_matrix",
                "prediction_context": "prospective_prediction",
            },
        },
    ]
    graph = ExecutionGraphCompiler.compile(commands, capability_registry)
    assert [node.tool for node in graph.nodes] == ["filter_data", "train_model"]
