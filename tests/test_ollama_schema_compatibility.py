"""Compatibility checks for the combined registry schema sent to Ollama."""

import json

import jsonschema

from modules.capability_registry import capability_registry


def test_dynamic_command_schema_is_self_contained_for_nested_tool_arguments():
    """Pydantic-local refs must not escape into the outer command schema.

    Ollama validates `format` as one schema document.  A ToolSpec schema nested
    under `arguments` cannot retain a `#/$defs/...` reference because that path
    now points at the command envelope, not the original ToolSpec document.
    """
    schema = capability_registry.get_dynamic_command_json_schema(
        capability_registry.select_planner_tools(
            "show a chart of a column distribution",
            has_dataset=True,
        )
    )

    serialized = json.dumps(schema)
    assert '"$ref": "#/$defs/' not in serialized
    jsonschema.Draft202012Validator.check_schema(schema)
