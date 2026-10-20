"""
DataPilot OS — Phase 13B Test Suite: Data-Aware Intent Completion & Conversational Slot Filling.

Guarantees:
1. No first-column blind picking or arbitrary defaults.
2. DatasetProfile ranking accurately categorizes numeric measures, categorical dimensions,
   temporal columns, and target candidates.
3. IntentCompletionService safely distinguishes between:
   - safe dataset-wide execution (eda_data, detect_anomalies, correlate_data),
   - high-confidence statistical inference (>= 0.85 confidence, >= 0.15 ambiguity margin),
   - interactive slot-filling clarification (card_type: target_selector, forecast_selector, etc.),
   - explicit rejection for scientifically unsupported/unsafe operations.
4. Provenance tracking: every inferred binding records source, confidence, and evidence.
5. End-to-end query orchestrator execution for natural English and Hindi/Hinglish queries.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.ds_engine.dataset_profiler import DatasetProfiler, DatasetProfile, RoleCandidate
from backend.ds_engine.intent_completion import (
    IntentCompletionService,
    intent_completion_service,
    IntentCompletionDecision,
    SlotFillingCard,
    InferredBinding
)
from backend.ds_engine.command_resolver import command_resolver, CommandResolutionResult
from backend.ds_engine.semantic_command import ToolCall
from backend.ds_engine.intent_schema import IntentType
from modules.query_orchestrator import QueryOrchestrator, humanize_clarification_message
from modules.models.execution_result import LifecycleState


# ── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def multi_numeric_sales_df():
    """Dataset with 1 date, 1 categorical dimension, and 3 numeric measures."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "order_date": dates,
        "region": np.random.choice(["North", "South", "East", "West"], size=n),
        "sales_amount": np.random.uniform(100, 1000, size=n),
        "profit": np.random.uniform(10, 200, size=n),
        "discount": np.random.uniform(0.01, 0.3, size=n),
    })


@pytest.fixture
def single_series_df():
    """Dataset with exactly 1 date and exactly 1 numeric measure."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "metric_value": np.random.normal(50, 5, size=n),
    })


@pytest.fixture
def classification_df():
    """Dataset with clear features and a binary target."""
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "monthly_spend": np.random.exponential(100, size=n),
        "support_tickets": np.random.poisson(2, size=n),
        "churn": np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
    })


@pytest.fixture
def multi_target_df():
    """Dataset with multiple ambiguous binary target candidates."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, size=n),
        "feature_b": np.random.normal(0, 1, size=n),
        "is_active": np.random.choice([0, 1], size=n),
        "is_subscribed": np.random.choice([0, 1], size=n),
        "is_churned": np.random.choice([0, 1], size=n),
    })


# ── TEST SUITE 1: DATASET PROFILER ROLE CANDIDATE RANKING ───────────────────

def test_dataset_profiler_ranks_candidates(multi_numeric_sales_df):
    """Profiler must compute statistical role candidates with confidence and evidence."""
    profile = DatasetProfiler.profile(multi_numeric_sales_df)
    
    assert profile is not None
    assert len(profile.ranked_numeric_measures) >= 3
    assert len(profile.ranked_temporal_columns) == 1
    assert profile.ranked_temporal_columns[0].column == "order_date"
    assert profile.ranked_temporal_columns[0].confidence >= 0.85

    # Categorical dimension check
    cat_names = [c.column for c in profile.ranked_categorical_dimensions]
    assert "region" in cat_names

    # Check top candidate helper
    top_time, _ = profile.get_top_candidate("temporal", threshold=0.85, ambiguity_margin=0.15)
    assert top_time is not None
    assert top_time.column == "order_date"


def test_dataset_profiler_detects_ambiguity_margin(multi_numeric_sales_df):
    """When multiple numeric columns have similar scores, get_top_candidate returns None."""
    profile = DatasetProfiler.profile(multi_numeric_sales_df)
    
    # 3 numeric measures with close properties should not have a decisive top candidate
    top_measure, _ = profile.get_top_candidate("numeric_measure", threshold=0.85, ambiguity_margin=0.25)
    # Because profit and sales_amount both have high score, ambiguity margin is not cleared
    assert top_measure is None


# ── TEST SUITE 2: INTENT COMPLETION SERVICE SAFE BEHAVIORS ──────────────────

def test_intent_completion_eda_dataset_wide(multi_numeric_sales_df):
    """EDA with missing columns executes dataset-wide safely."""
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(multi_numeric_sales_df)
    
    tool_call = ToolCall(tool="eda_data", arguments={})
    decision = service.complete_intent(tool_call, profile=profile, query_raw="Dataset ka overview do")
    
    assert decision.action == "execute"
    assert decision.is_dataset_wide is True
    assert decision.is_ambiguous is False


def test_intent_completion_anomaly_detection_dataset_wide(multi_numeric_sales_df):
    """Anomaly detection with explicit method/sensitivity runs dataset-wide over all numeric measures."""
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(multi_numeric_sales_df)
    
    # Missing method/sensitivity prompts for clarification (no silent defaults)
    empty_call = ToolCall(tool="detect_anomalies", arguments={})
    clarify_dec = service.complete_intent(empty_call, profile=profile, query_raw="Missing values aur outliers dikhao")
    assert clarify_dec.action == "clarify"
    assert clarify_dec.slot_filling is not None
    assert clarify_dec.slot_filling.card_type == "method_selector"

    # Explicit method + sensitivity executes dataset-wide safely
    tool_call = ToolCall(tool="detect_anomalies", arguments={"method": "iqr", "sensitivity": 1.5})
    decision = service.complete_intent(tool_call, profile=profile, query_raw="Missing values aur outliers dikhao")
    
    assert decision.action == "execute"
    assert decision.is_dataset_wide is True
    assert "columns" in decision.completed_arguments
    assert set(decision.completed_arguments["columns"]) == {"sales_amount", "profit", "discount"}
    assert all(b.source == "dataset_wide_default" for b in decision.inferred_bindings)


def test_intent_completion_correlation_matrix_dataset_wide(multi_numeric_sales_df):
    """Correlation query executes dataset-wide correlation matrix only when matrix mode is requested."""
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(multi_numeric_sales_df)
    
    # Without matrix mode requested, prompts for pairwise clarification
    call_no_mode = ToolCall(tool="correlate_data", arguments={})
    clarify_dec = service.complete_intent(call_no_mode, profile=profile, query_raw="Correlation dikhao")
    assert clarify_dec.action == "clarify"
    assert clarify_dec.slot_filling.card_type == "correlation_selector"

    # With explicit matrix mode, executes dataset-wide correlation matrix
    tool_call = ToolCall(tool="correlate_data", arguments={"analysis_mode": "matrix"})
    decision = service.complete_intent(tool_call, profile=profile, query_raw="Correlation matrix dikhao")
    
    assert decision.action == "execute"
    assert decision.is_dataset_wide is True
    assert "columns" in decision.completed_arguments
    assert len(decision.completed_arguments["columns"]) >= 3


def test_intent_completion_forecast_ambiguous_emits_slot_card(multi_numeric_sales_df):
    """Forecast on multi-numeric dataset without target emits forecast slot card."""
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(multi_numeric_sales_df)
    
    tool_call = ToolCall(tool="forecast_data", arguments={})
    decision = service.complete_intent(tool_call, profile=profile, query_raw="Sales forecast karo")
    
    assert decision.action == "clarify"
    assert decision.is_ambiguous is True
    assert decision.slot_filling is not None
    assert decision.slot_filling.card_type == "forecast_selector"
    assert len(decision.slot_filling.slots) >= 1
    assert len(decision.slot_filling.suggested_queries) > 0


def test_intent_completion_forecast_single_series_auto_binds(single_series_df):
    """Forecast auto-binds unambiguous temporal column while requiring explicit target selection."""
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(single_series_df)
    
    # With explicit target, time column auto-binds with high confidence
    tool_call = ToolCall(tool="forecast_data", arguments={"target_column": "metric_value"})
    decision = service.complete_intent(tool_call, profile=profile, query_raw="Forecast metric_value for next 10 periods")
    
    assert decision.action == "execute"
    assert decision.is_ambiguous is False
    assert decision.completed_arguments["time_column"] == "timestamp"
    assert decision.completed_arguments["target_column"] == "metric_value"
    time_bindings = [b for b in decision.inferred_bindings if b.slot == "time_column"]
    assert len(time_bindings) == 1
    assert time_bindings[0].source == "profile_inference"
    assert time_bindings[0].confidence >= 0.85
    policy_bindings = [b for b in decision.inferred_bindings if b.source == "policy_default"]
    assert len(policy_bindings) >= 4


def test_intent_completion_train_model_ambiguous_target(multi_target_df):
    """Train model query without target on ambiguous dataset produces target selector card."""
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(multi_target_df)
    
    tool_call = ToolCall(tool="train_model", arguments={})
    decision = service.complete_intent(tool_call, profile=profile, query_raw="Model train karo")
    
    assert decision.action == "clarify"
    assert decision.is_ambiguous is True
    assert decision.slot_filling is not None
    assert decision.slot_filling.card_type == "target_selector"
    assert len(decision.candidate_columns) >= 3


def test_intent_completion_rejects_unsafe_forecasting_no_temporal():
    """Forecasting on dataset without temporal columns is safely rejected."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(df)
    
    tool_call = ToolCall(tool="forecast_data", arguments={})
    decision = service.complete_intent(tool_call, profile=profile, query_raw="Forecast values")
    
    assert decision.action == "reject"
    assert decision.error_code == "NO_TEMPORAL_COLUMN"


# ── TEST SUITE 3: COMMAND RESOLVER INTEGRATION ───────────────────────────────

def test_command_resolver_resolves_dataset_wide_eda(multi_numeric_sales_df):
    """CommandResolver cleanly accepts dataset-wide eda_data command."""
    cmd = {
        "tool_calls": [
            {"tool": "eda_data", "arguments": {}}
        ]
    }
    resolution = command_resolver.resolve_command(cmd, df=multi_numeric_sales_df, raw_query="Dataset ka overview do")
    
    assert resolution.is_valid is True
    assert resolution.semantic_intent is not None
    assert resolution.is_dataset_wide is True
    assert resolution.semantic_intent.is_dataset_wide is True


def test_command_resolver_resolves_dataset_wide_anomalies(multi_numeric_sales_df):
    """CommandResolver completes arguments for detect_anomalies dataset-wide when method is explicit."""
    cmd = {
        "tool_calls": [
            {"tool": "detect_anomalies", "arguments": {"method": "iqr", "sensitivity": 1.5}}
        ]
    }
    resolution = command_resolver.resolve_command(cmd, df=multi_numeric_sales_df, raw_query="Missing values aur outliers dikhao")
    
    assert resolution.is_valid is True
    assert resolution.semantic_intent is not None
    assert resolution.is_dataset_wide is True
    assert len(resolution.inferred_bindings) > 0


def test_command_resolver_clarifies_ambiguous_forecast(multi_numeric_sales_df):
    """CommandResolver emits clarification result with slot filling for ambiguous forecast."""
    cmd = {
        "tool_calls": [
            {"tool": "forecast_data", "arguments": {}}
        ]
    }
    resolution = command_resolver.resolve_command(cmd, df=multi_numeric_sales_df, raw_query="Sales forecast karo")
    
    assert resolution.is_valid is True
    assert resolution.is_ambiguous is True
    assert resolution.slot_filling is not None
    assert resolution.slot_filling["card_type"] == "forecast_selector"


# ── TEST SUITE 4: END-TO-END PIPELINE & HINDI / HINGLISH QUERIES ─────────────

def test_query_orchestrator_eda_hinglish(multi_numeric_sales_df):
    """End-to-end execution of 'Dataset ka overview do'."""
    cmd = {
        "tool_calls": [
            {"tool": "eda_data", "arguments": {}}
        ]
    }
    resp = QueryOrchestrator._process_intent(
        query_raw="Dataset ka overview do",
        df=multi_numeric_sales_df,
        intent_dict=cmd,
        profile=DatasetProfiler.profile(multi_numeric_sales_df),
        conversation_history=[],
        context={},
        request_id="test-hinglish-eda",
        t_start=0.0
    )
    
    assert resp.status == LifecycleState.SUCCEEDED.value
    assert resp.decision == "execute"
    assert resp.result is not None
    assert resp.result["is_dataset_wide"] is True


def test_query_orchestrator_anomalies_hinglish(multi_numeric_sales_df):
    """End-to-end execution of 'Missing values aur outliers dikhao' with explicit method."""
    cmd = {
        "tool_calls": [
            {"tool": "detect_anomalies", "arguments": {"method": "iqr", "sensitivity": 1.5}}
        ]
    }
    resp = QueryOrchestrator._process_intent(
        query_raw="Missing values aur outliers dikhao",
        df=multi_numeric_sales_df,
        intent_dict=cmd,
        profile=DatasetProfiler.profile(multi_numeric_sales_df),
        conversation_history=[],
        context={},
        request_id="test-hinglish-anomalies",
        t_start=0.0
    )
    
    assert resp.status == LifecycleState.SUCCEEDED.value
    assert resp.decision == "execute"
    assert resp.result is not None
    assert len(resp.inferred_bindings) > 0


def test_query_orchestrator_correlation_hinglish(multi_numeric_sales_df):
    """End-to-end execution of 'Correlation dikhao' with matrix mode."""
    cmd = {
        "tool_calls": [
            {"tool": "correlate_data", "arguments": {"analysis_mode": "matrix"}}
        ]
    }
    resp = QueryOrchestrator._process_intent(
        query_raw="Correlation matrix dikhao",
        df=multi_numeric_sales_df,
        intent_dict=cmd,
        profile=DatasetProfiler.profile(multi_numeric_sales_df),
        conversation_history=[],
        context={},
        request_id="test-hinglish-corr",
        t_start=0.0
    )
    
    assert resp.status == LifecycleState.SUCCEEDED.value
    assert resp.decision == "execute"
    assert resp.result is not None


def test_query_orchestrator_ambiguous_forecast_clarification(multi_numeric_sales_df):
    """Ambiguous forecast query returns NEEDS_CLARIFICATION with slot_filling card."""
    cmd = {
        "tool_calls": [
            {"tool": "forecast_data", "arguments": {}}
        ]
    }
    resp = QueryOrchestrator._process_intent(
        query_raw="Sales forecast karo",
        df=multi_numeric_sales_df,
        intent_dict=cmd,
        profile=DatasetProfiler.profile(multi_numeric_sales_df),
        conversation_history=[],
        context={},
        request_id="test-hinglish-forecast-clarify",
        t_start=0.0
    )
    
    assert resp.status == LifecycleState.NEEDS_CLARIFICATION.value
    assert resp.decision == "clarify"
    assert resp.slot_filling is not None
    assert resp.slot_filling["card_type"] == "forecast_selector"
    assert len(resp.candidate_columns) > 0


def test_humanize_clarification_message():
    """Verify humanize_clarification_message transforms technical strings into friendly questions."""
    raw = "Ambiguous column reference 'val': matches multiple dataset columns (value_a, value_b). Please specify exact column name."
    humanized = humanize_clarification_message(raw, candidate_columns=["value_a", "value_b"])
    assert "Multiple columns match 'val'" in humanized
    assert "`value_a`" in humanized or "value_a" in humanized

    cat_raw = "Statistical test requires explicit columns to compare; multiple categorical columns found (region, status). Please specify columns."
    humanized_cat = humanize_clarification_message(cat_raw, candidate_columns=["region", "status"])
    assert "Which category would you like to compare across?" in humanized_cat
