import pytest
import jsonschema
from pydantic import BaseModel, Field
from typing import Dict, Any, List

from modules.capability_registry import (
    capability_registry, CapabilityRegistry, ToolSpec, CompletionPolicy, StrictBaseModel,
    GeneralChatArgs, ClarificationArgs
)
from backend.ds_engine.command_resolver import CommandResolver, CommandResolutionResult
from backend.ds_engine.intent_schema import IntentType, AggregationType
from backend.ds_engine.analysis_spec_v3 import AnalysisMode, AnalysisObjective, RequestedArtifact


def test_legacy_dict_input_strictly_rejected():
    """Invariant 1: Unstructured legacy dictionaries are completely rejected without fallback."""
    legacy_payload = {
        "primary_intent": "FILTER_QUERY",
        "target_column": "sales",
        "aggregation": "sum",
        "horizon": 30
    }
    res = CommandResolver.resolve_command(legacy_payload)
    assert res.is_valid is False
    assert res.error_code == "UNSUPPORTED_OPERATION"
    assert "Legacy unstructured intent dictionary is deprecated and rejected" in res.error_message


def test_invalid_category_in_toolspec_raises_contract_invalid():
    """Invariant 2: ToolSpec with invalid category raises REGISTRY_CONTRACT_INVALID upon intent mapping."""
    test_reg = CapabilityRegistry()
    
    class DummyArgs(StrictBaseModel):
        x: int

    invalid_spec = ToolSpec(
        name="dummy_invalid_cat",
        description="Dummy tool with nonexistent category",
        input_model=DummyArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        adapter=lambda args, state: None,
        task_family=IntentType.EDA,
        category="completely_nonexistent_category",
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )
    test_reg.register_tool_spec(invalid_spec)

    with pytest.raises(ValueError) as exc_info:
        test_reg.get_tool_intent_map()
    assert "REGISTRY_CONTRACT_INVALID" in str(exc_info.value)
    assert "completely_nonexistent_category" in str(exc_info.value)


def test_general_chat_invalid_arguments_returns_error_not_swallowed():
    """Invariant 3a: general_chat invalid args return INVALID_ARGUMENTS, exception not swallowed."""
    invalid_chat_cmd = {
        "tool": "general_chat",
        "arguments": {
            "response": "Hello world",
            "unexpected_extra_field": "forbidden"
        }
    }
    res = CommandResolver.resolve_command(invalid_chat_cmd)
    assert res.is_valid is False
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "Validation failed for tool 'general_chat'" in res.error_message


def test_clarification_invalid_arguments_returns_error_not_swallowed():
    """Invariant 3b: clarification_needed invalid args return INVALID_ARGUMENTS, exception not swallowed."""
    invalid_clarify_cmd = {
        "tool": "clarification_needed",
        "arguments": {
            "reason": "Please specify metric",
            "forbidden_extra": 12345
        }
    }
    res = CommandResolver.resolve_command(invalid_clarify_cmd)
    assert res.is_valid is False
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "Validation failed for tool 'clarification_needed'" in res.error_message


def test_dynamic_schema_commands_variant_requires_nonempty_array():
    """Invariant 4: commands requires minItems: 1 and dynamic tool names."""
    schema = capability_registry.get_dynamic_command_json_schema()
    variants = schema.get("oneOf", [schema])
    sequence_variant = next(
        variant for variant in variants
        if "commands" in variant.get("properties", {})
    )

    assert sequence_variant["properties"]["commands"].get("minItems") == 1
    tool_names = {
        variant.get("properties", {}).get("tool", {}).get("const")
        for variant in variants
    }
    assert "filter_data" in tool_names

    # Empty commands list must fail validation
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"commands": []}, schema=schema)

    # Valid commands list must pass validation
    valid_payload = {
        "commands": [
            {
                "tool": "filter_data",
                "arguments": {
                    "column": "sales",
                    "condition": ">",
                    "value": 100
                }
            }
        ]
    }
    jsonschema.validate(instance=valid_payload, schema=schema)
