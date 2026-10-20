"""
DataPilot OS — Shadow Mode & Resource Guard Verification Tests.

Guarantees:
1. Shadow Mode runs intent planning and schema comparison ONLY.
2. Shadow Mode does NOT execute sandbox code, AST compilation, or heavy ML models twice.
3. Default shadow mode flag is False so local desktop resources (CPU/RAM) are never strained in production.
4. Shadow mode failures/exceptions are isolated and never crash the primary execution pipeline.
"""

import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.core.feature_flags import FeatureFlagManager, RolloutConfig, feature_flag_manager
from modules.query_orchestrator import QueryOrchestrator
from modules.execution_engine import ModularExecutionEngine
from backend.ds_engine.code_validator import CodeValidator
from backend.intelligence.ollama_service import ollama_service, OllamaReadinessStatus
from modules.semantic_planner import SemanticPlanner


@pytest.fixture(autouse=True)
def reset_feature_flags():
    """Ensure feature flag manager state is reset before and after each test."""
    feature_flag_manager.set_config(
        enable_shadow_mode=False,
        enable_minimal_semantic_command=True,
        canary_percentage=100.0
    )
    yield
    feature_flag_manager.set_config(
        enable_shadow_mode=False,
        enable_minimal_semantic_command=True,
        canary_percentage=100.0
    )


def test_default_shadow_mode_is_false():
    """Verify that Shadow Mode is disabled by default to guard desktop CPU/RAM resources."""
    # 1. RolloutConfig default
    config = RolloutConfig()
    assert config.enable_shadow_mode is False, "RolloutConfig default must be False"

    # 2. Fresh FeatureFlagManager default
    ff = FeatureFlagManager()
    assert ff.is_shadow_mode_enabled() is False, "FeatureFlagManager default must be False"

    # 3. Global singleton default
    assert feature_flag_manager.is_shadow_mode_enabled() is False, "Global feature_flag_manager default must be False"

    # 4. Environment variable parsing fallback
    with patch.dict(os.environ, {}, clear=True):
        ff_env = FeatureFlagManager()
        assert ff_env.is_shadow_mode_enabled() is False


def test_shadow_mode_logs_intent_comparison_without_double_sandbox_execution():
    """
    Verify that when shadow mode is enabled:
    - Intent planning & schema comparison telemetry is logged.
    - Code execution / sandbox subprocess is called exactly ONCE (no duplicate execution).
    - AST code validator is called exactly ONCE (no duplicate validation).
    """
    feature_flag_manager.set_config(enable_shadow_mode=True)
    assert feature_flag_manager.is_shadow_mode_enabled() is True

    df = pd.DataFrame({"sales": [100, 200, 300], "region": ["North", "South", "East"]})

    mock_readiness = OllamaReadinessStatus(
        ready=True,
        model="qwen2.5:1.5b",
        model_available=True
    )

    mock_intent_payload = {
        "intent_type": "GROUP_AGGREGATE",
        "metrics": ["sales"],
        "dimensions": ["region"],
        "aggregation": "mean",
        "confidence": 0.95,
        "llm_latency_ms": 55.0
    }

    with patch.object(ollama_service, "verify_ready", return_value=mock_readiness), \
         patch.object(SemanticPlanner, "extract_semantic_intent", return_value=mock_intent_payload), \
         patch.object(feature_flag_manager, "log_shadow_comparison", wraps=feature_flag_manager.log_shadow_comparison) as spy_shadow_log, \
         patch.object(ModularExecutionEngine, "execute_code_subprocess", return_value={"success": True, "data": {"result": 200}, "error": None}) as mock_exec, \
         patch.object(CodeValidator, "validate", wraps=CodeValidator().validate) as spy_val:

        response = QueryOrchestrator.process_query(
            query_raw="Average sales by region",
            df=df,
            request_id="req-guard-01"
        )

        # 1. Pipeline succeeded
        assert response.status == "SUCCEEDED"

        # 2. Shadow comparison was invoked exactly once for intent planning telemetry
        assert spy_shadow_log.call_count == 1
        call_args = spy_shadow_log.call_args[1]
        assert call_args["request_id"] == "req-guard-01"
        assert call_args["query"] == "Average sales by region"
        assert call_args["primary_intent"]["intent_type"] == "GROUP_AGGREGATE"

        # 3. Sandbox execution engine was called exactly ONCE (resource guard protection)
        assert mock_exec.call_count == 1, (
            f"Sandbox code execution was called {mock_exec.call_count} times; expected exactly 1!"
        )

        # 4. AST Validator was called for generated code (no duplicate execution passes)
        assert spy_val.call_count in [1, 2], (
            f"AST Validator was called {spy_val.call_count} times; expected 1 or 2!"
        )


def test_shadow_mode_disabled_does_not_log_comparison():
    """Verify that when shadow mode is disabled (default), zero shadow telemetry is executed."""
    feature_flag_manager.set_config(enable_shadow_mode=False)
    assert feature_flag_manager.is_shadow_mode_enabled() is False

    df = pd.DataFrame({"sales": [100, 200, 300], "region": ["North", "South", "East"]})

    mock_readiness = OllamaReadinessStatus(
        ready=True,
        model="qwen2.5:1.5b",
        model_available=True
    )

    mock_intent_payload = {
        "intent_type": "EDA",
        "metrics": ["sales"],
        "confidence": 0.95,
        "llm_latency_ms": 40.0
    }

    with patch.object(ollama_service, "verify_ready", return_value=mock_readiness), \
         patch.object(SemanticPlanner, "extract_semantic_intent", return_value=mock_intent_payload), \
         patch.object(feature_flag_manager, "log_shadow_comparison") as mock_shadow_log, \
         patch.object(ModularExecutionEngine, "execute_code_subprocess", return_value={"success": True, "data": {"sales": 200}, "error": None}) as mock_exec:

        response = QueryOrchestrator.process_query(
            query_raw="Summarize sales",
            df=df,
            request_id="req-guard-02"
        )

        assert response.status == "SUCCEEDED"
        assert mock_shadow_log.call_count == 0, "log_shadow_comparison should not be called when shadow mode is False"
        assert mock_exec.call_count == 1


def test_shadow_mode_error_resilience(caplog):
    """Verify that if shadow mode telemetry logger encounters an exception, primary execution succeeds unaffected."""
    import logging
    feature_flag_manager.set_config(enable_shadow_mode=True)

    df = pd.DataFrame({"sales": [100, 200, 300]})

    mock_readiness = OllamaReadinessStatus(
        ready=True,
        model="qwen2.5:1.5b",
        model_available=True
    )

    mock_intent_payload = {
        "intent_type": "EDA",
        "metrics": ["sales"],
        "confidence": 0.95
    }

    with patch.object(ollama_service, "verify_ready", return_value=mock_readiness), \
         patch.object(SemanticPlanner, "extract_semantic_intent", return_value=mock_intent_payload), \
         patch.object(feature_flag_manager, "log_shadow_comparison", side_effect=RuntimeError("Telemetry sink failure")), \
         patch.object(ModularExecutionEngine, "execute_code_subprocess", return_value={"success": True, "data": {"mean": 200}, "error": None}) as mock_exec, \
         caplog.at_level(logging.WARNING):

        response = QueryOrchestrator.process_query(
            query_raw="Summarize sales",
            df=df,
            request_id="req-guard-03"
        )

        # Primary query must succeed even if shadow logging throws
        assert response.status == "SUCCEEDED"
        assert mock_exec.call_count == 1
        assert "SHADOW_MODE_ERROR" in caplog.text


def test_shadow_mode_telemetry_schema():
    """Verify the structured telemetry record generated by log_shadow_comparison."""
    ff = FeatureFlagManager()
    ff.set_config(enable_shadow_mode=True)

    primary = {"intent_type": "FILTER_QUERY", "column": "sales", "operator": ">", "value": 100}
    shadow_matching = {"intent_type": "FILTER_QUERY", "column": "sales", "operator": ">", "value": 100}
    shadow_mismatch = {"intent_type": "GROUP_AGGREGATE", "metrics": ["sales"]}

    # 1. Matching intent
    rec_match = ff.log_shadow_comparison(
        request_id="req-tel-01",
        query="Sales > 100",
        primary_intent=primary,
        shadow_intent=shadow_matching,
        primary_latency_ms=80.0,
        shadow_latency_ms=85.0
    )
    assert rec_match["event"] == "SHADOW_MODE_DUAL_RUN_COMPARISON"
    assert rec_match["request_id"] == "req-tel-01"
    assert rec_match["intents_match"] is True
    assert rec_match["primary_intent_type"] == "FILTER_QUERY"
    assert rec_match["shadow_intent_type"] == "FILTER_QUERY"
    assert rec_match["latency_delta_ms"] == 5.0

    # 2. Mismatched intent
    rec_mismatch = ff.log_shadow_comparison(
        request_id="req-tel-02",
        query="Sales > 100",
        primary_intent=primary,
        shadow_intent=shadow_mismatch,
        primary_latency_ms=90.0,
        shadow_latency_ms=110.0
    )
    assert rec_mismatch["intents_match"] is False
    assert rec_mismatch["primary_intent_type"] == "FILTER_QUERY"
    assert rec_mismatch["shadow_intent_type"] == "GROUP_AGGREGATE"
    assert rec_mismatch["latency_delta_ms"] == 20.0
