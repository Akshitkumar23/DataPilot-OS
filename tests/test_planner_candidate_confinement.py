import pandas as pd

from modules.capability_registry import capability_registry
from modules.planner_prompt_contract import render_user_prompt
from modules.semantic_planner import SemanticPlanner


def _schema_tool_names(schema):
    names = set()
    for variant in schema["oneOf"]:
        direct = variant.get("properties", {}).get("tool", {}).get("const")
        if direct:
            names.add(direct)
        commands = variant.get("properties", {}).get("commands", {})
        for call in commands.get("items", {}).get("oneOf", []):
            name = call.get("properties", {}).get("tool", {}).get("const")
            if name:
                names.add(name)
    return names


def test_generation_schema_contains_only_retrieved_candidates():
    selected = capability_registry.select_planner_tools(
        "Train a classifier and show a confusion matrix.",
        has_dataset=True,
        limit=2,
    )
    schema = capability_registry.get_dynamic_command_json_schema(
        selected, allow_command_sequence=True
    )
    assert _schema_tool_names(schema) == {spec.name for spec in selected}
    assert len(_schema_tool_names(schema)) <= 3  # two retrieved plus clarification


def test_without_dataset_only_non_dataset_tools_are_retrieved():
    selected = capability_registry.select_planner_tools(
        "Explain what a confidence interval is.", has_dataset=False
    )
    assert selected
    assert all(not spec.requires_dataset for spec in selected)
    assert "general_chat" in {spec.name for spec in selected}


def test_prompt_history_excludes_assistant_generated_column_claims():
    frame = pd.DataFrame({"actual_column": [1, 2, 3]})
    prompt = render_user_prompt(
        "sum actual_column",
        frame,
        conversation_history=[
            {"role": "assistant", "content": "Use invented_column."},
            {"role": "user", "content": "Earlier I selected actual_column."},
        ],
        context={"selected_columns": ["actual_column"], "untrusted": "invented_column"},
    )
    assert "Earlier I selected actual_column" in prompt
    assert "invented_column" not in prompt


def test_protocol_gate_rejects_tools_outside_retrieved_set():
    parsed = {"tool": "correlate_data", "arguments": {}}
    assert not SemanticPlanner._uses_only_selected_tools(parsed, {"train_model"})
