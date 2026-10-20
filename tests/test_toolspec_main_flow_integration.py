import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, ConfigDict, Field

from modules.capability_registry import (
    capability_registry, ToolSpec, StrictBaseModel, CompletionPolicy,
    FilterArgs, GroupArgs, SortArgs, ForecastArgs
)
from backend.ds_engine.command_resolver import CommandResolver
from backend.ds_engine.intent_resolver import IntentResolver, ResolvedIntent
from backend.ds_engine.code_generator import CodeGenerator
from backend.ds_engine.capability_drivers import CapabilityDriver, FilterQueryDriver, GroupAggregateDriver
from backend.ds_engine.semantic_command import ToolCall, LLMSemanticCommand
from backend.ds_engine.intent_schema import IntentType, AggregationType
from backend.ds_engine.analysis_spec_v3 import AnalysisMode, AnalysisObjective, RequestedArtifact


def test_canonical_toolspec_bootstrap_integrity():
    specs = capability_registry.list_tool_specs()
    assert len(specs) >= 14, f"Expected >= 14 ToolSpecs, found {len(specs)}"

    tool_names = [s.name for s in specs]
    assert "filter_data" in tool_names
    assert "group_data" in tool_names
    assert "sort_data" in tool_names
    assert "aggregate_data" in tool_names
    assert "correlate_data" in tool_names
    assert "forecast_data" in tool_names
    assert "clean_data" in tool_names
    assert "detect_anomalies" in tool_names
    assert "eda_data" in tool_names
    assert "train_model" in tool_names
    assert "cluster_data" in tool_names
    assert "autocorrelation" in tool_names
    assert "general_chat" in tool_names
    assert "clarification_needed" in tool_names

    filter_driver = capability_registry.get_driver("filter_data")
    assert filter_driver is not None
    assert isinstance(filter_driver, FilterQueryDriver)

    group_driver = capability_registry.get_driver("group_data")
    assert group_driver is not None
    assert isinstance(group_driver, GroupAggregateDriver)

    tool_map = capability_registry.get_tool_intent_map()
    assert "filter_data" in tool_map
    assert "group_data" in tool_map
    assert "forecast_data" in tool_map


def test_dynamic_toolspec_full_pipeline_reachability():
    class CustomProbeArgs(StrictBaseModel):
        target_col: str
        multiplier: float = 2.0

    class CustomProbeDriver(CapabilityDriver):
        def compile(self, resolved: ResolvedIntent) -> Tuple[str, List[str], List[Dict[str, Any]]]:
            col = resolved.resolved_metrics[0] if resolved.resolved_metrics else "val"
            code = f"# Custom Probe Driver Execution\nresult_df = df.copy()\nresult_df['{col}_probed'] = df['{col}'] * 2.0"
            return code, ["pandas"], [{"type": "table", "name": "probed"}]

    def adapt_probe(args: Any, state: Dict[str, Any]):
        if getattr(args, "target_col", None):
            state["target_column"] = args.target_col
            mets = state.setdefault("metrics", [])
            if args.target_col not in mets:
                mets.append(args.target_col)

    probe_driver = CustomProbeDriver()
    probe_spec = ToolSpec(
        name="probe_dynamic",
        description="Custom dynamic analytical probing tool",
        input_model=CustomProbeArgs,
        objective=AnalysisObjective.TRANSFORM,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        compiler_or_driver=probe_driver,
        adapter=adapt_probe,
        task_family=IntentType.EDA,
        result_contract="dataframe",
        category="eda",
        synonyms=["probe"],
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )
    capability_registry.register_tool_spec(probe_spec)

    eff_map = CommandResolver.get_effective_tool_intent_map()
    assert "probe_dynamic" in eff_map
    assert "probe" in eff_map

    cmd_data = {
        "tool": "probe_dynamic",
        "arguments": {"target_col": "revenue", "multiplier": 3.0}
    }
    df = pd.DataFrame({"revenue": [100, 200, 300], "region": ["North", "South", "East"]})
    res = CommandResolver.resolve_command(cmd_data, df=df, raw_query="Probe revenue")
    assert res.is_valid is True, f"Failed resolution: {res.error_message}"
    assert res.semantic_intent.tool_name == "probe_dynamic"

    intent_res = IntentResolver().resolve(res.semantic_intent, df=df)
    assert intent_res.is_valid is True
    assert intent_res.tool_name == "probe_dynamic"

    code_gen = CodeGenerator()
    code_ver = code_gen.generate_from_resolved_intent(intent_res)
    assert "Custom Probe Driver Execution" in code_ver.code
    assert "result_df['revenue_probed'] = df['revenue'] * 2.0" in code_ver.code


def test_dynamic_tool_without_adapter_fails_with_unsupported_operation():
    class MissingAdapterArgs(StrictBaseModel):
        val: int

    missing_adapter_spec = ToolSpec(
        name="tool_without_adapter",
        description="Tool intentionally missing adapter",
        input_model=MissingAdapterArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        adapter=None,
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )
    with pytest.raises(ValueError, match="must declare an execution adapter"):
        capability_registry.register_tool_spec(missing_adapter_spec)


def test_empty_registry_fails_fast_with_no_schema_fallback():
    from modules.capability_registry import CapabilityRegistry
    empty_reg = CapabilityRegistry()
    
    with pytest.raises(RuntimeError) as exc_info:
        empty_reg.get_dynamic_command_json_schema()
    assert "CAPABILITY_REGISTRY_UNAVAILABLE" in str(exc_info.value)


def test_unknown_field_in_main_resolver_fails_strict_validation():
    cmd_data = {
        "tool": "filter_data",
        "arguments": {
            "column": "sales",
            "condition": ">",
            "value": 500,
            "hallucinated_param": "should_be_forbidden"
        }
    }
    res = CommandResolver.resolve_command(cmd_data, raw_query="Filter sales > 500")
    assert res.is_valid is False
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "Validation failed for tool 'filter_data'" in res.error_message


def test_codegenerator_fails_when_driver_missing_from_registry():
    from backend.core.error_states import DataPilotError
    from backend.ds_engine.intent_schema import SemanticIntent

    unregistered_intent = IntentType.EDA
    fake_semantic = SemanticIntent(
        intent_type=unregistered_intent,
        primary_intent=unregistered_intent,
        tool_name="completely_unregistered_tool"
    )
    resolved = ResolvedIntent(
        is_valid=True,
        intent=fake_semantic,
        tool_name="completely_unregistered_tool",
        dataset_name="Test",
        row_count=10,
        column_count=2
    )

    code_gen = CodeGenerator()
    with pytest.raises(DataPilotError) as exc_info:
        code_gen.generate_from_resolved_intent(resolved)
    assert "completely_unregistered_tool" in str(exc_info.value)


def test_health_check_verifies_toolspec_readiness():
    from modules.capability_registry import capability_registry
    count = len(capability_registry.list_tool_specs())
    assert count >= 12
    toolspec_ready = count >= 12
    assert toolspec_ready is True


def test_dynamic_command_json_schema_generation():
    from modules.capability_registry import capability_registry, ToolSpec, StrictBaseModel

    def tool_names(command_schema):
        # The production schema is a discriminated oneOf contract: every tool
        # carries its own typed arguments model.  A global string enum would
        # weaken that invariant by allowing arguments from another tool.
        return {
            variant.get("properties", {}).get("tool", {}).get("const")
            for variant in command_schema.get("oneOf", [])
            if variant.get("properties", {}).get("tool", {}).get("const")
        }

    schema = capability_registry.get_dynamic_command_json_schema()
    registered_tools = tool_names(schema)
    
    assert "filter_data" in registered_tools
    assert "group_data" in registered_tools
    assert "general_chat" in registered_tools
    assert "clarification_needed" in registered_tools
    
    # Register dynamic custom tool
    class CustomSchemaProbeArgs(StrictBaseModel):
        target: str

    probe_spec = ToolSpec(
        name="probe_dynamic_schema",
        description="Probe tool for testing schema",
        input_model=CustomSchemaProbeArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        adapter=lambda args, state: None,
        task_family=IntentType.EDA,
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )
    capability_registry.register_tool_spec(probe_spec)
    
    updated_schema = capability_registry.get_dynamic_command_json_schema()
    assert "probe_dynamic_schema" in tool_names(updated_schema)


def test_dynamic_command_json_schema_rejects_empty_commands():
    import jsonschema
    from modules.capability_registry import capability_registry
    
    schema = capability_registry.get_dynamic_command_json_schema()

    commands_variant = next(
        variant for variant in schema["oneOf"]
        if "commands" in variant.get("properties", {})
    )
    assert commands_variant["properties"]["commands"].get("minItems") == 1
    assert any(
        variant.get("properties", {}).get("tool", {}).get("const") == "filter_data"
        for variant in schema["oneOf"]
    )

    # Rejection of empty commands array
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"commands": []}, schema=schema)

    # Acceptance of valid single tool
    valid_single = {"tool": "filter_data", "arguments": {"column": "age", "condition": "==", "value": 25}}
    jsonschema.validate(instance=valid_single, schema=schema)

    # Acceptance of valid multi-command
    valid_multi = {"commands": [{"tool": "filter_data", "arguments": {"column": "age", "condition": "==", "value": 25}}]}
    jsonschema.validate(instance=valid_multi, schema=schema)


def test_invalid_toolspec_category_raises_registry_contract_invalid():
    from modules.capability_registry import CapabilityRegistry, ToolSpec, StrictBaseModel
    reg = CapabilityRegistry()

    class DummyArgs(StrictBaseModel):
        x: int = 1

    spec = ToolSpec(
        name="bad_tool",
        description="Tool with invalid category",
        input_model=DummyArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        adapter=lambda args, state: None,
        task_family=IntentType.EDA,
        category="invalid_category_xyz",
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )
    reg.register_tool_spec(spec)

    with pytest.raises(ValueError) as exc_info:
        reg.get_tool_intent_map()

    assert "REGISTRY_CONTRACT_INVALID: ToolSpec 'bad_tool' has invalid category 'invalid_category_xyz'. Must match a valid IntentType member." in str(exc_info.value)


def test_unmapped_tool_in_resolve_command_raises_runtime_error(monkeypatch):
    from modules.capability_registry import capability_registry, ToolSpec, StrictBaseModel

    class DummyProbeArgs(StrictBaseModel):
        target: str = "val"

    def dummy_adapt(args, state):
        pass

    probe_spec = ToolSpec(
        name="isolated_probe",
        description="Isolated probe",
        input_model=DummyProbeArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        adapter=dummy_adapt,
        task_family=IntentType.EDA,
        category="eda",
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(),
    )
    capability_registry.register_tool_spec(probe_spec)

    # Monkeypatch get_effective_tool_intent_map to omit 'isolated_probe'
    original_get_map = CommandResolver.get_effective_tool_intent_map
    def mock_map():
        m = original_get_map()
        m.pop("isolated_probe", None)
        return m

    monkeypatch.setattr(CommandResolver, "get_effective_tool_intent_map", mock_map)

    cmd = {
        "commands": [
            {"tool": "isolated_probe", "arguments": {"target": "x"}}
        ]
    }
    with pytest.raises(RuntimeError) as exc_info:
        CommandResolver.resolve_command(cmd)

    assert "REGISTRY_CONTRACT_INVALID: Tool 'isolated_probe' is not mapped to an IntentType in tool_intent_map." in str(exc_info.value)


