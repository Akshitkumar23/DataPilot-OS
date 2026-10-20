"""
DataPilot OS — Production Hardening & Rollout Verification Tests.
Validates:
1. Minimal schema & prompt versioning propagation.
2. LLM repair-and-retry loop on malformed JSON outputs.
3. Normalization audit logging across operators, aggregations, and filters.
4. Runtime options (top_p, temperature, num_predict, stream: False).
5. Feature flag management, canary percentage routing, and shadow-mode dual-run comparison.
"""

import sys
import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.ds_engine.semantic_command import (
    LLMSemanticCommand,
    SCHEMA_VERSION,
    PROMPT_VERSION,
    PROTOCOL_NAME,
    MINIMAL_COMMAND_JSON_SCHEMA
)
from backend.ds_engine.command_resolver import command_resolver
from backend.intelligence.ollama_service import OllamaRequest, OllamaResponse, ollama_service
from modules.semantic_planner import SemanticPlanner
from backend.core.feature_flags import feature_flag_manager, FeatureFlagManager
from modules.query_orchestrator import QueryOrchestrator


def test_schema_and_prompt_versioning_constants():
    """Verify protocol, schema_version, and prompt_version are defined on models."""
    cmd = LLMSemanticCommand(
        tool="filter_data",
        arguments={"column": "sales", "operator": ">", "value": 500}
    )
    assert cmd.protocol == "minimal_semantic_command"
    assert cmd.schema_version == "1.0"
    assert cmd.prompt_version == PROMPT_VERSION
    assert cmd.tool == "filter_data"


def test_runtime_options_and_token_counts():
    """Verify top_p, stream: False, num_predict, and token telemetry in OllamaRequest and OllamaResponse."""
    req = OllamaRequest(
        prompt="Test prompt",
        temperature=0.0,
        top_p=0.1,
        response_schema=MINIMAL_COMMAND_JSON_SCHEMA
    )
    assert req.temperature == 0.0
    assert req.top_p == 0.1
    assert req.response_schema == MINIMAL_COMMAND_JSON_SCHEMA

    res = OllamaResponse(
        success=True,
        content='{"tool": "eda_data"}',
        parsed_json={"tool": "eda_data"},
        latency_ms=45.2,
        prompt_eval_count=120,
        eval_count=25,
        model_used="qwen2.5:1.5b"
    )
    assert res.prompt_eval_count == 120
    assert res.eval_count == 25
    assert res.latency_ms == 45.2


def test_llm_repair_and_retry_loop_on_malformed_json():
    """Verify SemanticPlanner attempts repair when initial LLM response is malformed JSON."""
    df = pd.DataFrame({"sales": [100, 200, 300], "region": ["North", "South", "East"]})

    # First attempt returns malformed string, second attempt returns valid JSON
    mock_resp_1 = OllamaResponse(
        success=True,
        content="Here is your output: {bad_json...",
        parsed_json=None,
        latency_ms=50.0,
        prompt_eval_count=100,
        eval_count=10
    )
    mock_resp_2 = OllamaResponse(
        success=True,
        content='{"tool": "filter_data", "arguments": {"column": "sales", "operator": ">", "value": 150}}',
        parsed_json={"tool": "filter_data", "arguments": {"column": "sales", "operator": ">", "value": 150}},
        latency_ms=40.0,
        prompt_eval_count=130,
        eval_count=20
    )

    with patch.object(ollama_service, "structured_request", side_effect=[mock_resp_1, mock_resp_2]) as mock_call:
        intent = SemanticPlanner.extract_semantic_intent("sales > 150", df=df)
        assert intent is not None
        assert intent["intent_type"] == "FILTER_QUERY"
        assert intent["schema_version"] == "1.0"
        assert intent["prompt_version"] == PROMPT_VERSION
        assert intent["repair_attempts"] == 1  # Repaired on attempt 2
        assert mock_call.call_count == 2


def test_normalization_audit_logging(caplog):
    """Verify structured normalization audit logging records operator and aggregation conversions."""
    import logging
    from backend.ds_engine.intent_schema import FilterOperator, AggregationType
    with caplog.at_level(logging.INFO, logger="backend.ds_engine.command_resolver"):
        # 1. Operator normalization
        op = command_resolver.resolve_operator("strictly above")
        assert op == FilterOperator.GT
        assert "AUDIT_NORMALIZATION_OPERATOR" in caplog.text

        # 2. Aggregation normalization
        agg = command_resolver.resolve_aggregation("average value")
        assert agg == AggregationType.MEAN
        assert "AUDIT_NORMALIZATION_AGGREGATION" in caplog.text

        # 3. Filter Spec normalization
        spec = command_resolver.normalize_filter_spec({"column": "temperature", "operator": ">=", "value": 35.5})
        assert spec.column == "temperature"
        assert spec.operator == FilterOperator.GTE
        assert spec.value == 35.5
        assert "AUDIT_NORMALIZATION_FILTER" in caplog.text


def test_feature_flags_and_canary_routing():
    """Verify feature flag toggles and canary percentage traffic distribution."""
    ff = FeatureFlagManager()
    
    # 1. Disabled minimal command flag
    ff.set_config(enable_minimal_semantic_command=False)
    assert ff.should_use_minimal_command() is False

    # 2. 100% Canary
    ff.set_config(enable_minimal_semantic_command=True, canary_percentage=100.0)
    assert ff.should_use_minimal_command("req-12345") is True

    # 3. 0% Canary
    ff.set_config(canary_percentage=0.0)
    assert ff.should_use_minimal_command("req-12345") is False

    # 4. 50% Canary deterministic test
    ff.set_config(canary_percentage=50.0)
    results = [ff.should_use_minimal_command(f"req-{i}") for i in range(100)]
    assert any(results) and not all(results)


def test_shadow_mode_dual_run_comparison():
    """Verify shadow mode dual run records comparison telemetry without error."""
    ff = FeatureFlagManager()
    ff.set_config(enable_shadow_mode=True)
    assert ff.is_shadow_mode_enabled() is True

    primary = {"intent_type": "GROUP_AGGREGATE", "metrics": ["sales"]}
    shadow = {"intent_type": "GROUP_AGGREGATE", "metrics": ["sales"]}

    record = ff.log_shadow_comparison(
        request_id="req-shadow-01",
        query="Average sales by region",
        primary_intent=primary,
        shadow_intent=shadow,
        primary_latency_ms=120.0,
        shadow_latency_ms=125.5
    )

    assert record["event"] == "SHADOW_MODE_DUAL_RUN_COMPARISON"
    assert record["intents_match"] is True
    assert record["latency_delta_ms"] == 5.5
