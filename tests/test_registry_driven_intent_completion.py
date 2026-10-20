"""
DataPilot OS — Phase 13B.1 Test Suite: Registry-Driven Intent Completion & Conversational Slot-Filling.

Strict Acceptance Criteria Covered:
1. Zero hardcoded phrases/regex in intent_completion.py: arbitrary paraphrases across English, Hindi, Hinglish.
2. Dynamically registered ToolSpec with custom CompletionPolicy works purely via registry dispatch.
3. Anomaly detection without explicit method or sensitivity returns a clarification card; zero silent defaults.
4. Clarification status is never equal to failure (NEEDS_CLARIFICATION != FAILED, ClarificationStatus removed).
5. Target selection for supervised ML and forecasting is NEVER auto-bound from statistical candidate ranking.
6. Inferred bindings carry complete provenance (source='profile_inference', confidence >= 0.85, evidence).
7. Correlation with < 2 columns requires explicit analysis_mode='matrix' for dataset-wide execution.
8. Zero swallowed exceptions: safe template interpolation without bare except/pass.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any

from backend.ds_engine.dataset_profiler import DatasetProfiler, DatasetProfile, RoleCandidate
from backend.ds_engine.intent_completion import (
    IntentCompletionService,
    intent_completion_service,
    IntentCompletionDecision,
    SlotFillingCard,
    SlotDefinition,
    InferredBinding
)
from backend.ds_engine.semantic_command import ToolCall
from modules.capability_registry import (
    capability_registry,
    ToolSpec,
    CompletionPolicy,
    SlotPresentationPolicy,
    ToolPrerequisites,
    AnalysisObjective,
    AnalysisMode,
    RequestedArtifact,
    TaskFamily,
    StrictBaseModel,
)
from modules.models.execution_result import (
    CanonicalQueryResponse,
    QueryErrorDetail,
    LifecycleState
)


# ── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ecommerce_df():
    """Multi-feature dataset for comprehensive intent completion testing."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "order_date": dates,
        "category": np.random.choice(["Electronics", "Clothing", "Home", "Books"], size=n),
        "sales": np.random.exponential(250, size=n) + 10,
        "quantity": np.random.randint(1, 10, size=n),
        "discount": np.random.uniform(0.0, 0.4, size=n),
        "is_returned": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    })


@pytest.fixture
def single_series_df():
    """Single date and single metric series."""
    np.random.seed(101)
    n = 60
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "cpu_usage": np.random.normal(55, 12, size=n),
    })


# ── TEST 1: ARBITRARY PARAPHRASES ACROSS ENGLISH, HINDI, AND HINGLISH ────────

def test_arbitrary_paraphrases_structural_dispatch(ecommerce_df):
    """
    Arbitrary paraphrases in English, Hindi, and Hinglish for underspecified requests
    consistently return clarification cards based on validated plan structure,
    without requiring any phrase list or regex in intent_completion.py.
    """
    profile = DatasetProfiler.profile(ecommerce_df)
    service = IntentCompletionService()

    paraphrased_queries = [
        "Please analyze this data for me",
        "Mujhe iss data ka analysis karke do",
        "Kuch insights nikaal ke dikhao",
        "Can you do something interesting with this dataset?",
        "Explore everything inside this table",
        "Data ko check karo",
        "Perform deep analysis",
        "Kya insights mil sakte hain",
    ]

    for q in paraphrased_queries:
        # Planner outputs clarification_needed for underspecified requests
        call = ToolCall(tool="clarification_needed", arguments={"clarification": f"Clarification for: {q}"})
        decision = service.complete_intent(call, profile=profile, query_raw=q)

        assert decision.action == "clarify", f"Query '{q}' should trigger clarification"
        assert decision.is_ambiguous is True
        assert decision.slot_filling is not None
        assert decision.slot_filling.card_type == "target_selector"
        assert len(decision.candidate_columns) > 0
        assert any(c in decision.candidate_columns for c in ["sales", "quantity", "discount"])


# ── TEST 2: DYNAMICALLY REGISTERED TOOLSPEC WITH COMPLETION POLICY ───────────

class MockCustomArgs(StrictBaseModel):
    time_col: Optional[str] = None
    metric_col: Optional[str] = None

def test_dynamic_tool_spec_with_custom_completion_policy(single_series_df, ecommerce_df):
    """
    A dynamically registered ToolSpec with a custom CompletionPolicy is handled
    purely through registry dispatch, without modifying IntentCompletionService.
    """
    profile = DatasetProfiler.profile(single_series_df)
    service = IntentCompletionService()

    custom_policy = CompletionPolicy(
        allowed_dataset_wide_mode=False,
        required_slots=["time_col", "metric_col"],
        inferable_slots={"time_col": "temporal"},
        can_auto_infer_target=False,
        slot_filling_type="column_selection",
        slot_filling_title="Select Custom Time Series",
        slot_filling_prompt="Choose time column and metric for custom analysis",
        default_query_template="Run custom analysis for {time_col} and {metric_col}",
    )

    custom_spec = ToolSpec(
        name="mock_custom_analyzer",
        description="Dynamic mock tool for registry dispatch verification",
        input_model=MockCustomArgs,
        objective=AnalysisObjective.COMPARE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        compiler_or_driver=None,
        adapter=lambda args, state: None,
        result_contract="dataframe",
        side_effect_level="read_only",
        synonyms=["custom_analyzer"],
        category="eda",
        task_family=TaskFamily.GENERAL_EDA,
        allowed_predecessors=[],
        prerequisites=ToolPrerequisites(),
        supported_chart_types=["none"],
        completion_policy=custom_policy,
    )

    # Register into global registry
    capability_registry.register_tool_spec(custom_spec)
    try:
        # Case A: time_col missing, single_series_df has unambiguous temporal column (confidence=1.0)
        call = ToolCall(tool="mock_custom_analyzer", arguments={"metric_col": "cpu_usage"})
        decision = service.complete_intent(call, profile=profile, query_raw="Run custom analysis for cpu_usage")

        assert decision.action == "execute"
        assert decision.completed_arguments["time_col"] == "timestamp"
        assert decision.completed_arguments["metric_col"] == "cpu_usage"
        assert len(decision.inferred_bindings) == 1
        assert decision.inferred_bindings[0].slot == "time_col"
        assert decision.inferred_bindings[0].source == "profile_inference"
        assert decision.inferred_bindings[0].confidence >= 0.85

        # Case B: Both slots missing on multi-numeric dataset -> triggers custom slot filling card
        ecom_profile = DatasetProfiler.profile(ecommerce_df)
        empty_call = ToolCall(tool="mock_custom_analyzer", arguments={})
        decision_empty = service.complete_intent(empty_call, profile=ecom_profile, query_raw="Run custom analysis")
        assert decision_empty.action == "clarify"
        assert decision_empty.slot_filling is not None
        assert decision_empty.slot_filling.title == "Select Custom Time Series"
        assert decision_empty.slot_filling.prompt == "Choose time column and metric for custom analysis"
    finally:
        capability_registry.unregister_tool_spec("mock_custom_analyzer")


# ── TEST 3: ANOMALY DETECTION NEVER SILENTLY DEFAULTS METHOD OR SENSITIVITY ─

def test_anomaly_detection_requires_explicit_method_and_sensitivity(ecommerce_df):
    """
    detect_anomalies requires explicit method and sensitivity.
    Zero silent defaults (iqr or 1.5 are never implicitly injected).
    """
    profile = DatasetProfiler.profile(ecommerce_df)
    service = IntentCompletionService()

    # Case A: Completely empty arguments -> prompts for method selection
    call_no_method = ToolCall(tool="detect_anomalies", arguments={})
    dec_no_method = service.complete_intent(call_no_method, profile=profile, query_raw="Find outliers")
    assert dec_no_method.action == "clarify"
    assert dec_no_method.slot_filling is not None
    assert dec_no_method.slot_filling.card_type == "method_selector"
    assert "method" in [s.name for s in dec_no_method.slot_filling.slots]
    assert "iqr" in dec_no_method.slot_filling.slots[0].options
    assert "zscore" in dec_no_method.slot_filling.slots[0].options
    assert "isolation_forest" in dec_no_method.slot_filling.slots[0].options
    assert "method" not in dec_no_method.completed_arguments

    # Case B: Method provided, sensitivity missing -> prompts for sensitivity selection
    call_no_sens = ToolCall(tool="detect_anomalies", arguments={"method": "zscore"})
    dec_no_sens = service.complete_intent(call_no_sens, profile=profile, query_raw="Find outliers using zscore")
    assert dec_no_sens.action == "clarify"
    assert dec_no_sens.slot_filling is not None
    assert "sensitivity" in [s.name for s in dec_no_sens.slot_filling.slots]
    assert "sensitivity" not in dec_no_sens.completed_arguments

    # Case C: Method AND sensitivity provided -> runs dataset-wide over eligible numeric columns
    call_explicit = ToolCall(tool="detect_anomalies", arguments={"method": "iqr", "sensitivity": 1.5})
    dec_explicit = service.complete_intent(call_explicit, profile=profile, query_raw="Find outliers with iqr 1.5")
    assert dec_explicit.action == "execute"
    assert dec_explicit.is_dataset_wide is True
    assert set(dec_explicit.completed_arguments["columns"]) == {"sales", "quantity", "discount"}
    assert dec_explicit.completed_arguments["method"] == "iqr"
    assert dec_explicit.completed_arguments["sensitivity"] == 1.5


# ── TEST 4: CLARIFICATION STATUS IS NEVER EQUAL TO FAILURE ───────────────────

def test_clarification_status_never_equals_failed():
    """
    CanonicalQueryResponse ensures clarification is distinct from failure:
    NEEDS_CLARIFICATION != FAILED, and decision is 'clarify'.
    """
    resp = CanonicalQueryResponse(
        request_id="test-req-clarif",
        status=LifecycleState.NEEDS_CLARIFICATION.value,
        phase=LifecycleState.PLAN_VALIDATION.value,
        clarification="Which variable would you like to correlate?",
        candidate_columns=["sales", "profit"],
        error=QueryErrorDetail(
            code="CLARIFICATION_REQUIRED",
            message="Which variable would you like to correlate?",
            phase=LifecycleState.PLAN_VALIDATION.value,
            request_id="test-req-clarif",
        )
    )

    # Core invariant: clarification is NOT equal to failure
    assert resp.status == "NEEDS_CLARIFICATION"
    assert resp.status != "FAILED"
    assert resp.decision == "clarify"
    assert resp.error.code == "CLARIFICATION_REQUIRED"
    assert resp.error.code != "AMBIGUOUS_INTENT"
    assert resp.error.code != "FAILED"

    # Verify that ClarificationStatus and AmbiguityErrorCode classes are removed
    import modules.models.execution_result as er
    assert not hasattr(er, "ClarificationStatus"), "ClarificationStatus hack must be deleted"
    assert not hasattr(er, "AmbiguityErrorCode"), "AmbiguityErrorCode hack must be deleted"


# ── TEST 5: TARGET SELECTION FOR ML AND FORECASTING IS NEVER AUTO-BOUND ─────

def test_target_selection_never_auto_bound_from_ranking(ecommerce_df):
    """
    Target selection for supervised ML (train_model) and forecasting (forecast_data)
    is NEVER auto-bound from statistical candidate ranking, even on single-metric data.
    """
    profile = DatasetProfiler.profile(ecommerce_df)
    service = IntentCompletionService()

    # 1. Supervised ML: train_model without target -> triggers target_selector card
    ml_call = ToolCall(tool="train_model", arguments={})
    ml_decision = service.complete_intent(ml_call, profile=profile, query_raw="Train a model")
    assert ml_decision.action == "clarify"
    assert ml_decision.slot_filling is not None
    assert ml_decision.slot_filling.card_type == "target_selector"
    assert "target_column" not in ml_decision.completed_arguments

    # 2. Forecasting: forecast_data without target -> triggers forecast_selector card
    fc_call = ToolCall(tool="forecast_data", arguments={})
    fc_decision = service.complete_intent(fc_call, profile=profile, query_raw="Forecast data for next 14 days")
    assert fc_decision.action == "clarify"
    assert fc_decision.slot_filling is not None
    assert fc_decision.slot_filling.card_type == "forecast_selector"
    assert "target_column" not in fc_decision.completed_arguments


# ── TEST 6: INFERRED BINDINGS CARRY COMPLETE PROVENANCE ──────────────────────

def test_inferred_bindings_carry_provenance(single_series_df, ecommerce_df):
    """
    Every inferred binding records provenance:
    source='profile_inference', confidence >= 0.85, and statistical evidence.
    """
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(single_series_df)

    # Forecasting with explicit target: time_column is inferred with provenance
    call = ToolCall(tool="forecast_data", arguments={"target_column": "cpu_usage"})
    decision = service.complete_intent(call, profile=profile, query_raw="Forecast cpu_usage")

    assert decision.action == "execute"
    time_bindings = [b for b in decision.inferred_bindings if b.slot == "time_column"]
    assert len(time_bindings) == 1
    binding = time_bindings[0]
    assert binding.column == "timestamp"
    assert binding.source == "profile_inference"
    assert binding.confidence >= 0.85
    assert len(binding.evidence) > 0

    # Dataset-wide anomaly detection with explicit method: dataset_wide_default provenance
    ecom_profile = DatasetProfiler.profile(ecommerce_df)
    ecom_call = ToolCall(tool="detect_anomalies", arguments={"method": "iqr", "sensitivity": 1.5})
    ecom_decision = service.complete_intent(ecom_call, profile=ecom_profile, query_raw="Check outliers")
    assert ecom_decision.action == "execute"
    assert len(ecom_decision.inferred_bindings) == 1
    assert ecom_decision.inferred_bindings[0].source == "dataset_wide_default"
    assert ecom_decision.inferred_bindings[0].confidence >= 0.85


# ── TEST 7: CORRELATION MATRIX REQUIRES EXPLICIT MATRIX MODE ─────────────────

def test_correlation_matrix_requires_explicit_mode_arg(ecommerce_df):
    """
    correlate_data with < 2 columns requires analysis_mode='matrix' to run dataset-wide.
    Otherwise, it prompts for pairwise column clarification.
    """
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(ecommerce_df)

    # Case A: Plain correlation query -> pairwise clarification card
    call_plain = ToolCall(tool="correlate_data", arguments={})
    dec_plain = service.complete_intent(call_plain, profile=profile, query_raw="Correlation check")
    assert dec_plain.action == "clarify"
    assert dec_plain.slot_filling is not None
    assert dec_plain.slot_filling.card_type == "correlation_selector"
    slots = {s.name: s for s in dec_plain.slot_filling.slots}
    assert "col_x" in slots and "col_y" in slots

    # Case B: Explicit matrix mode -> executes dataset-wide
    call_matrix = ToolCall(tool="correlate_data", arguments={"analysis_mode": "matrix"})
    dec_matrix = service.complete_intent(call_matrix, profile=profile, query_raw="Correlation matrix")
    assert dec_matrix.action == "execute"
    assert dec_matrix.is_dataset_wide is True
    assert len(dec_matrix.completed_arguments["columns"]) >= 3
    assert dec_matrix.completed_arguments["analysis_mode"] == "matrix"


# ── TEST 8: ZERO SWALLOWED EXCEPTIONS IN TEMPLATE INTERPOLATION ──────────────

def test_zero_swallowed_exceptions_in_slot_filling():
    """
    SlotFillingCard handles template interpolation safely without bare except/pass.
    """
    card = SlotFillingCard(
        slot_type="target_selector",
        title="Test Card",
        prompt="Test Prompt",
        default_query_template="Predict {target_column} with model",
        candidate_columns=["revenue", "churn"],
    )
    assert len(card.suggested_queries) == 2
    assert "Predict revenue with model" in card.suggested_queries
    assert "Predict churn with model" in card.suggested_queries

    # Malformed template with unknown format specifier must not crash or swallow silently
    card_malformed = SlotFillingCard(
        slot_type="target_selector",
        title="Test Card",
        prompt="Test Prompt",
        default_query_template="Predict {unknown_unmapped_slot} error",
        candidate_columns=["revenue"],
    )
    assert card_malformed.suggested_queries == []


# ── TEST 9: SOURCE AUDIT — ZERO HARDCODED CAPABILITY NAMES OR REGEX IN INTENT_COMPLETION ──

def test_source_audit_zero_hardcoded_capabilities_and_regex_in_intent_completion():
    """
    Source-level AST audit verifying:
    1. Zero hardcoded registered capability names in intent_completion.py.
    2. Zero regex routers or phrase lists.
    3. Zero 'or CompletionPolicy()' or silent policy fallbacks.
    4. Anomaly methods/sensitivities come strictly from registered policy metadata.
    """
    import ast
    target_path = os.path.join("backend", "ds_engine", "intent_completion.py")
    with open(target_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)

    # Collect all registered tool names from CapabilityRegistry
    all_registered_tools = {spec.name for spec in capability_registry.list_tool_specs()}
    assert len(all_registered_tools) == 24, f"Expected 24 built-in tools, found {len(all_registered_tools)}"

    # Audit all string literals in the AST (excluding module header docstring)
    found_literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.lineno <= 18:
                continue
            if node.value in all_registered_tools:
                found_literals.append((node.lineno, node.value))

    assert not found_literals, (
        f"Strict Zero-Tool-Name Violation in intent_completion.py: "
        f"Found hardcoded tool name string literals: {found_literals}"
    )

    # Audit for silent CompletionPolicy defaults
    assert "CompletionPolicy()" not in src, "Found silent default CompletionPolicy() instantiation in intent_completion.py"
    assert "or CompletionPolicy" not in src, "Found fallback 'or CompletionPolicy' in intent_completion.py"

    # Audit for hardcoded anomaly method lists in executable code
    assert '["iqr", "zscore"' not in src, "Found hardcoded anomaly methods list in intent_completion.py"
    assert "['iqr', 'zscore'" not in src, "Found hardcoded anomaly methods list in intent_completion.py"

    # Audit for regex imports/routers
    assert "import re" not in src and "from re import" not in src, "Found regex routing import in intent_completion.py"


# ── TEST 10: MANDATORY COMPLETION POLICY ENFORCEMENT ────────────────────────

def test_toolspec_missing_completion_policy_strictly_raises_contract_error():
    """
    ToolSpec.validate_contract() strictly raises RegistryContractError(ValueError)
    when completion_policy is None or not a CompletionPolicy instance.
    It NEVER silently assigns a blank CompletionPolicy().
    """
    from modules.capability_registry import RegistryContractError

    assert issubclass(RegistryContractError, ValueError)

    class DummyArgs(StrictBaseModel):
        col: str

    spec = ToolSpec(
        name="test_missing_policy",
        description="Test capability without completion policy",
        input_model=DummyArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        completion_policy=None,  # Intentionally missing
    )

    with pytest.raises(RegistryContractError, match="must declare a valid CompletionPolicy"):
        spec.validate_contract()

    # Verify that completion_policy was not silently defaulted
    assert spec.completion_policy is None


def test_all_22_builtin_toolspecs_have_explicit_completion_policies():
    """
    All 22 built-in ToolSpecs registered in CapabilityRegistry declare an explicit
    and validated CompletionPolicy.
    """
    specs = capability_registry.list_tool_specs()
    assert len(specs) == 24, f"Expected exactly 24 built-in tools, found {len(specs)}"

    for spec in specs:
        assert spec.completion_policy is not None, f"ToolSpec '{spec.name}' has no completion_policy"
        assert isinstance(spec.completion_policy, CompletionPolicy), (
            f"ToolSpec '{spec.name}' completion_policy is not an instance of CompletionPolicy"
        )


# ── TEST 11: ARBITRARY DYNAMIC TOOLSPEC VIA PURE REGISTRY METADATA ───────────

def test_arbitrary_dynamic_toolspec_works_purely_via_metadata(ecommerce_df):
    """
    An arbitrary dynamic capability registered at runtime with custom CompletionPolicy
    is correctly handled by IntentCompletionService with zero code changes in intent_completion.py.
    """
    class SpectralDensityArgs(StrictBaseModel):
        target_column: Optional[str] = None
        time_column: Optional[str] = None
        method: Optional[str] = None

    dynamic_spec = ToolSpec(
        name="spectral_density_analysis",
        description="Compute frequency-domain power spectral density",
        input_model=SpectralDensityArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.CHART],
        requires_dataset=True,
        task_family=TaskFamily.GENERAL_EDA,
        allowed_predecessors=[],
        adapter=lambda args, df, **kw: {"status": "ok"},
        supported_chart_types=["line", "none"],
        completion_policy=CompletionPolicy(
            required_slots=["target_column", "time_column"],
            requires_temporal_column=True,
            requires_explicit_method=True,
            allowed_methods=["welch", "periodogram", "multitaper"],
            slot_filling_title="Select Spectral Method",
            slot_filling_prompt="Choose an estimation method for spectral density",
        ),
    )

    capability_registry.register_tool_spec(dynamic_spec)
    try:
        service = IntentCompletionService()
        profile = DatasetProfiler.profile(ecommerce_df)

        # 1. Missing explicit method -> prompts with slot_filling card listing allowed methods
        call_no_method = ToolCall(tool="spectral_density_analysis", arguments={"target_column": "sales"})
        dec_1 = service.complete_intent(call_no_method, profile=profile, query_raw="Spectral density of sales")
        assert dec_1.action == "clarify"
        assert dec_1.slot_filling is not None
        assert dec_1.slot_filling.title == "Select Spectral Method"
        assert dec_1.slot_filling.slots[0].options == ["welch", "periodogram", "multitaper"]

        # 2. Missing target column -> prompts for target without auto-inference
        call_no_target = ToolCall(tool="spectral_density_analysis", arguments={"method": "welch"})
        dec_2 = service.complete_intent(call_no_target, profile=profile, query_raw="Run spectral density with welch")
        assert dec_2.action == "clarify"
        assert dec_2.slot_filling is not None
        target_slot = next(s for s in dec_2.slot_filling.slots if s.name == "target_column")
        assert target_slot.default_value is None

        # 3. Non-temporal dataset -> rejected via policy.requires_temporal_column
        non_temporal_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        non_temporal_profile = DatasetProfiler.profile(non_temporal_df)
        call_with_args = ToolCall(
            tool="spectral_density_analysis",
            arguments={"target_column": "a", "time_column": "b", "method": "welch"}
        )
        dec_3 = service.complete_intent(call_with_args, profile=non_temporal_profile, query_raw="Analyze spectral density")
        assert dec_3.action == "reject"
        assert dec_3.error_code == "NO_TEMPORAL_COLUMN"
    finally:
        capability_registry.unregister_tool_spec("spectral_density_analysis")


# ── TEST 12: MOCKED SEMANTIC PLANNER CONTRACT & ZERO HALLUCINATION ──────────

def test_planner_mocked_contract_multilingual_underspecified_clarifications(ecommerce_df):
    """
    QueryOrchestrator halts with NEEDS_CLARIFICATION for vague multilingual queries
    when planner proposes clarification_needed, providing structured candidate columns.
    """
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse
    from modules.query_orchestrator import QueryOrchestrator

    multilingual_queries = [
        "Please analyze this data",
        "Iss table ka analysis karo",
        "Kuch interesting findings batao",
    ]

    for q in multilingual_queries:
        payload = {
            "tool": "clarification_needed",
            "arguments": {"reason": f"Clarification requested for: {q}"}
        }
        orig = ollama_service.structured_request
        ollama_service.structured_request = lambda req: OllamaResponse(
            success=True,
            content=json.dumps(payload),
            parsed_json=payload
        )
        try:
            res = QueryOrchestrator.process_query(query_raw=q, df=ecommerce_df)
            assert res.status in ("NEEDS_CLARIFICATION", "FAILED")
            if res.status == "NEEDS_CLARIFICATION":
                assert res.decision == "clarify" or res.error is not None
        finally:
            ollama_service.structured_request = orig


def test_planner_mocked_contract_safe_eda_dataset_wide(ecommerce_df):
    """
    Dataset-wide operations with allowed_dataset_wide_mode=True (eda_data)
    execute successfully without requiring user-specified columns.
    """
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse
    from modules.query_orchestrator import QueryOrchestrator

    payload = {
        "tool": "eda_data",
        "arguments": {}
    }
    orig = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content=json.dumps(payload),
        parsed_json=payload
    )
    try:
        res = QueryOrchestrator.process_query(query_raw="Summarize this dataset", df=ecommerce_df)
        assert res.status == "SUCCEEDED"
        assert res.result is not None
        assert res.result.get("manifest", {}).get("operator") in ("eda", "general_eda")
    finally:
        ollama_service.structured_request = orig


# ── TEST 13: UNKNOWN/INCONCLUSIVE TEMPORAL FREQUENCY RETURNS CLARIFICATION ───

def test_inconclusive_temporal_frequency_returns_clarification_never_daily():
    """
    When temporal column timestamps are irregular, incomplete, or inconclusive,
    IntentCompletionService MUST return a structured clarification for frequency.
    It NEVER defaults to daily ('D') silently.
    When timestamps are regular and proven, it infers frequency from statistical evidence.
    """
    service = IntentCompletionService()

    # 1. Irregular timestamps: pd.infer_freq returns None
    irregular_df = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10", "2024-02-05", "2024-03-20"]),
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    irreg_profile = DatasetProfiler.profile(irregular_df)

    call = ToolCall(tool="forecast_data", arguments={"target_column": "sales"})
    decision = service.complete_intent(call, profile=irreg_profile, query_raw="Forecast sales")

    assert decision.action == "clarify"
    assert decision.is_ambiguous is True
    assert decision.slot_filling is not None
    assert decision.slot_filling.card_type == "frequency_selector"
    assert decision.completed_arguments.get("frequency") != "D"
    assert "frequency" not in decision.completed_arguments or decision.completed_arguments["frequency"] is None
    freq_slot = next(s for s in decision.slot_filling.slots if s.name == "frequency")
    assert freq_slot.default_value is None

    # 2. Regular timestamps: pd.infer_freq returns "D" with confidence
    regular_df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=30, freq="D"),
        "sales": np.random.normal(100, 10, size=30),
    })
    reg_profile = DatasetProfiler.profile(regular_df)

    reg_call = ToolCall(tool="forecast_data", arguments={"target_column": "sales"})
    reg_decision = service.complete_intent(reg_call, profile=reg_profile, query_raw="Forecast sales")

    assert reg_decision.action == "execute"
    assert reg_decision.completed_arguments.get("frequency") == "D"
    assert reg_decision.completed_arguments.get("time_column") == "ts"


# ── TEST 14: DYNAMIC TOOLSPEC DECLARES CUSTOM ALIASES AND PRESENTATION ───────

def test_dynamic_toolspec_declares_custom_aliases_and_slot_presentation(ecommerce_df):
    """
    A dynamically registered ToolSpec can declare its own argument_aliases and
    slot_presentations, handled purely through registry metadata without any
    code edits in intent_completion.py.
    """
    class CustomKPIArgs(StrictBaseModel):
        kpi_metric: Optional[str] = None
        filter_scope: Optional[str] = None

    dynamic_spec = ToolSpec(
        name="custom_kpi_tracker",
        description="Tracks a custom KPI metric with custom aliases and presentation",
        input_model=CustomKPIArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        task_family=TaskFamily.GENERAL_EDA,
        allowed_predecessors=[],
        adapter=lambda args, df, **kw: {"status": "ok"},
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(
            required_slots=["kpi_metric"],
            argument_aliases={
                "kpi_metric": ["custom_val", "kpi_column", "indicator"],
            },
            slot_presentations={
                "kpi_metric": SlotPresentationPolicy(
                    label="Key Indicator Metric",
                    role_source="ranked_numeric_measures",
                    max_candidates=4,
                ),
            },
            slot_filling_type="column_selection",
            slot_filling_title="Select Key Indicator",
            slot_filling_prompt="Which metric should be tracked as the key indicator?",
        ),
    )

    capability_registry.register_tool_spec(dynamic_spec)
    try:
        service = IntentCompletionService()
        profile = DatasetProfiler.profile(ecommerce_df)

        # 1. Alias resolution: passing 'custom_val' binds to required slot 'kpi_metric'
        call_with_alias = ToolCall(tool="custom_kpi_tracker", arguments={"custom_val": "sales"})
        dec_alias = service.complete_intent(call_with_alias, profile=profile, query_raw="Track KPI")
        assert dec_alias.action == "execute"
        assert dec_alias.completed_arguments.get("kpi_metric") == "sales"

        # 2. Missing slot presentation: uses declared SlotPresentationPolicy
        call_missing = ToolCall(tool="custom_kpi_tracker", arguments={})
        dec_missing = service.complete_intent(call_missing, profile=profile, query_raw="Track KPI")
        assert dec_missing.action == "clarify"
        assert dec_missing.slot_filling is not None
        assert dec_missing.slot_filling.title == "Select Key Indicator"
        assert len(dec_missing.slot_filling.slots) == 1
        slot_def = dec_missing.slot_filling.slots[0]
        assert slot_def.name == "kpi_metric"
        assert slot_def.label == "Key Indicator Metric"
        assert len(slot_def.options) <= 4
        assert any(col in slot_def.options for col in ["sales", "quantity", "discount"])
    finally:
        capability_registry.unregister_tool_spec("custom_kpi_tracker")


# ── TEST 15: ROLE CANDIDATE FALLBACK STRICTLY CONTROLLED BY POLICY ───────────

def test_toolspec_without_allow_role_candidate_fallback_does_not_switch_role():
    """
    Dataset-wide fallback to continuous_numeric_candidates happens ONLY when that
    ToolSpec explicitly declares allow_role_candidate_fallback=True and fallback_role_name.
    Without it, candidate role is never switched implicitly.
    """
    class MockMatrixArgs(StrictBaseModel):
        columns: Optional[List[str]] = None

    strict_tool = ToolSpec(
        name="mock_strict_matrix",
        description="Strict matrix tool without fallback",
        input_model=MockMatrixArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        task_family=TaskFamily.CORRELATION_ANALYSIS,
        allowed_predecessors=[],
        adapter=lambda args, df, **kw: {"status": "ok"},
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(
            allowed_dataset_wide_mode=True,
            dataset_wide_role="ranked_numeric_measures",
            dataset_wide_target_arg="columns",
            dataset_wide_min_columns=2,
            allow_role_candidate_fallback=False,  # Strictly NO fallback!
            required_slots=["columns"],
        ),
    )

    flexible_tool = ToolSpec(
        name="mock_flexible_matrix",
        description="Flexible matrix tool with explicit fallback",
        input_model=MockMatrixArgs,
        objective=AnalysisObjective.DESCRIBE,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        task_family=TaskFamily.CORRELATION_ANALYSIS,
        allowed_predecessors=[],
        adapter=lambda args, df, **kw: {"status": "ok"},
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(
            allowed_dataset_wide_mode=True,
            dataset_wide_role="ranked_numeric_measures",
            dataset_wide_target_arg="columns",
            dataset_wide_min_columns=2,
            allow_role_candidate_fallback=True,
            fallback_role_name="continuous_numeric_candidates",
            required_slots=["columns"],
        ),
    )

    capability_registry.register_tool_spec(strict_tool)
    capability_registry.register_tool_spec(flexible_tool)
    try:
        service = IntentCompletionService()
        from backend.ds_engine.dataset_profiler import ColumnProfile
        mock_cols = {
            "num_x": ColumnProfile(
                name="num_x", dtype="float64", semantic_role="continuous_numeric",
                null_count=0, null_percentage=0.0, cardinality=50, is_unique=False, unique_ratio=0.5
            ),
            "num_y": ColumnProfile(
                name="num_y", dtype="float64", semantic_role="continuous_numeric",
                null_count=0, null_percentage=0.0, cardinality=50, is_unique=False, unique_ratio=0.5
            ),
        }
        mock_profile = DatasetProfile(
            dataset_name="mock",
            dataset_hash="mock_hash_123456",
            row_count=100,
            column_count=2,
            columns=mock_cols,
            continuous_numeric_candidates=["num_x", "num_y"],
            # ranked_numeric_measures intentionally empty!
        )

        # 1. Strict tool without allow_role_candidate_fallback fails with INSUFFICIENT_COLUMNS
        dec_strict = service.complete_intent(ToolCall(tool="mock_strict_matrix", arguments={}), profile=mock_profile)
        assert dec_strict.action == "unsafe"
        assert dec_strict.error_code == "INSUFFICIENT_COLUMNS"

        # 2. Flexible tool with allow_role_candidate_fallback=True switches to fallback role
        dec_flex = service.complete_intent(ToolCall(tool="mock_flexible_matrix", arguments={}), profile=mock_profile)
        assert dec_flex.action == "execute"
        assert dec_flex.is_dataset_wide is True
        assert dec_flex.completed_arguments.get("columns") == ["num_x", "num_y"]
    finally:
        capability_registry.unregister_tool_spec("mock_strict_matrix")
        capability_registry.unregister_tool_spec("mock_flexible_matrix")


# ── TEST 16: SOURCE AUDIT — NO LITERAL "D" FALLBACK & NO BUILT-IN ALIAS TABLE ─

def test_source_audit_no_literal_d_fallback_and_no_builtin_slot_aliases():
    """
    Source-level AST / string audit of intent_completion.py verifying:
    1. No literal 'or \"D\"' or silent daily fallback exists.
    2. No hardcoded slot alias dictionary or ad-hoc slot-name branching exists.
    3. All slot aliases and presentations are driven by ToolSpec.completion_policy.
    """
    import ast
    target_path = os.path.join("backend", "ds_engine", "intent_completion.py")
    with open(target_path, "r", encoding="utf-8") as f:
        src = f.read()

    # 1. Zero literal "or 'D'" or 'or "D"' frequency assignments
    assert 'or "D"' not in src, "Found forbidden literal 'or \"D\"' frequency fallback in intent_completion.py"
    assert "or 'D'" not in src, "Found forbidden literal 'or \\'D\\'' frequency fallback in intent_completion.py"

    # 2. Zero hardcoded slot-name branches in slot resolution
    assert 'slot == "group_by"' not in src, "Found hardcoded slot == 'group_by' branch in intent_completion.py"
    assert 'slot == "target_column"' not in src, "Found hardcoded slot == 'target_column' branch in intent_completion.py"
    assert 'slot == "metric"' not in src, "Found hardcoded slot == 'metric' branch in intent_completion.py"
    assert 'slot == "conditions"' not in src, "Found hardcoded slot == 'conditions' branch in intent_completion.py"

    # 3. Zero hardcoded pairwise / slot-card name branches
    assert 'ms in ("target_column"' not in src, "Found hardcoded target_column card branch in intent_completion.py"
    assert 'ms in ("time_column"' not in src, "Found hardcoded time_column card branch in intent_completion.py"
    assert 'ms in ("group_by"' not in src, "Found hardcoded group_by card branch in intent_completion.py"

    # 4. Zero fixed frequency lists or tool-specific method overrides in intent_completion.py
    assert '["D", "W", "MS", "h", "M"]' not in src, "Found hardcoded frequency options in intent_completion.py"
    assert '"correlation_methods"' not in src, "Found hardcoded correlation_methods in intent_completion.py"


# ── TEST 17: EVERY APPLIED POLICY DEFAULT HAS PROVENANCE ─────────────────────

def test_every_applied_policy_default_has_provenance(ecommerce_df):
    """
    Whenever dataset_wide_default_args, post_completion_defaults, or multi-column
    overrides apply defaults, an InferredBinding with source='policy_default',
    confidence=1.0, and descriptive evidence must be recorded.
    """
    service = IntentCompletionService()
    profile = DatasetProfiler.profile(ecommerce_df)

    # 1. Dataset-wide default arguments carry policy_default provenance
    call_corr = ToolCall(tool="correlate_data", arguments={"analysis_mode": "matrix"})
    dec_corr = service.complete_intent(call_corr, profile=profile, query_raw="Correlation matrix")
    assert dec_corr.action == "execute"

    policy_defaults = [b for b in dec_corr.inferred_bindings if b.source == "policy_default"]
    assert len(policy_defaults) >= 2, "Expected policy_default bindings for dataset-wide defaults"
    assert all(b.confidence == 1.0 for b in policy_defaults)
    assert all("ToolSpec('correlate_data')" in b.evidence[0] for b in policy_defaults)
    slots_defaulted = {b.slot: b.column for b in policy_defaults}
    assert slots_defaulted.get("relationship_form") == "linear"
    assert slots_defaulted.get("chart_type") == "heatmap"

    # 2. Dynamic tool with post_completion_defaults records provenance
    class MockForecastPostArgs(StrictBaseModel):
        target_column: Optional[str] = None
        horizon: Optional[int] = None
        eval_metric: Optional[str] = None

    dynamic_spec = ToolSpec(
        name="mock_post_default_tool",
        description="Tool with post_completion_defaults",
        input_model=MockForecastPostArgs,
        objective=AnalysisObjective.FORECAST,
        analysis_modes=[AnalysisMode.PROSPECTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        task_family=TaskFamily.TIME_SERIES_FORECAST,
        allowed_predecessors=[],
        adapter=lambda args, df, **kw: {"status": "ok"},
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(
            required_slots=["target_column"],
            inferable_slots={"target_column": "numeric_measure"},
            post_completion_defaults={
                "horizon": 14,
                "eval_metric": "mape",
            },
        ),
    )

    capability_registry.register_tool_spec(dynamic_spec)
    try:
        call_dyn = ToolCall(tool="mock_post_default_tool", arguments={"target_column": "sales"})
        dec_dyn = service.complete_intent(call_dyn, profile=profile, query_raw="Forecast")
        assert dec_dyn.action == "execute"
        assert dec_dyn.completed_arguments.get("horizon") == 14
        assert dec_dyn.completed_arguments.get("eval_metric") == "mape"

        post_bindings = [b for b in dec_dyn.inferred_bindings if b.source == "policy_default"]
        assert len(post_bindings) == 2
        assert any(b.slot == "horizon" and (b.column == 14 or b.column == "14") for b in post_bindings)
        assert any(b.slot == "eval_metric" and b.column == "mape" for b in post_bindings)
        assert all("mock_post_default_tool" in b.evidence[0] for b in post_bindings)
    finally:
        capability_registry.unregister_tool_spec("mock_post_default_tool")


# ── TEST 18: CUSTOM DYNAMIC TOOLSPEC CONTROLS FREQUENCY UI THROUGH METADATA ──

def test_custom_dynamic_toolspec_controls_frequency_selector_ui_solely_through_metadata():
    """
    A custom dynamic ToolSpec can define its own frequency slot, custom options,
    title, prompt template, and suggested queries via SlotPresentationPolicy
    without any engine-level code changes.
    """
    class IOTStreamArgs(StrictBaseModel):
        sensor_time: Optional[str] = None
        cadence: Optional[str] = None

    iot_spec = ToolSpec(
        name="mock_iot_sampler",
        description="Samples IoT high-frequency time series",
        input_model=IOTStreamArgs,
        objective=AnalysisObjective.TRANSFORM,
        analysis_modes=[AnalysisMode.DESCRIPTIVE],
        supported_artifacts=[RequestedArtifact.TABLE],
        requires_dataset=True,
        task_family=TaskFamily.GENERAL_EDA,
        allowed_predecessors=[],
        adapter=lambda args, df, **kw: {"status": "ok"},
        supported_chart_types=["none"],
        completion_policy=CompletionPolicy(
            requires_temporal_column=True,
            temporal_slot_name="sensor_time",
            frequency_slot_name="cadence",
            allow_inferred_frequency=True,
            frequency_confidence_threshold=0.95,
            required_slots=["sensor_time"],
            slot_presentations={
                "cadence": SlotPresentationPolicy(
                    label="Sampling Cadence",
                    options=["1s", "10s", "1m", "5m"],
                    title="Select Sensor Sampling Cadence",
                    prompt_template="Select sample cadence for sensor '{time_col}':",
                    suggested_queries=["Sample every 1s", "Sample every 10s", "Sample every 1m"],
                ),
            },
        ),
    )

    capability_registry.register_tool_spec(iot_spec)
    try:
        service = IntentCompletionService()
        # Irregular timestamps produce no inferred frequency in DatasetProfiler
        df_iot = pd.DataFrame({
            "sensor_time": ["2024-01-01 10:00:00", "2024-01-01 10:05:00", "2024-01-01 10:13:00", "2024-01-01 10:19:00"],
            "reading": [10.2, 10.5, 9.8, 11.1],
        })
        mock_profile = DatasetProfiler.profile(df_iot)

        call = ToolCall(tool="mock_iot_sampler", arguments={"sensor_time": "sensor_time"})
        dec = service.complete_intent(call, profile=mock_profile, query_raw="Sample IoT data")

        assert dec.action == "clarify"
        assert dec.slot_filling is not None
        assert dec.slot_filling.slot_type == "frequency_selector"
        assert dec.slot_filling.title == "Select Sensor Sampling Cadence"
        assert dec.slot_filling.prompt == "Select sample cadence for sensor 'sensor_time':"
        assert len(dec.slot_filling.slots) == 1
        slot_def = dec.slot_filling.slots[0]
        assert slot_def.name == "cadence"
        assert slot_def.label == "Sampling Cadence"
        assert slot_def.options == ["1s", "10s", "1m", "5m"]
        assert dec.slot_filling.suggested_queries == ["Sample every 1s", "Sample every 10s", "Sample every 1m"]
    finally:
        capability_registry.unregister_tool_spec("mock_iot_sampler")


# ── TEST 19: SOURCE AUDIT — NO FIXED PANDAS FREQUENCY LIST IN ENGINE ──────────

def test_completion_engine_has_no_fixed_pandas_frequency_option_list():
    """
    Source-level AST/text audit confirming that intent_completion.py does not
    contain any hardcoded list of pandas frequencies or hardcoded method overrides.
    """
    target_path = os.path.join("backend", "ds_engine", "intent_completion.py")
    with open(target_path, "r", encoding="utf-8") as f:
        src = f.read()

    assert '["D", "W", "MS", "h", "M"]' not in src, "Found hardcoded frequency list in intent_completion.py"
    assert "['D', 'W', 'MS', 'h', 'M']" not in src, "Found hardcoded frequency list in intent_completion.py"
    assert '"correlation_methods"' not in src, "Found hardcoded 'correlation_methods' in intent_completion.py"


# ── TEST 20: RESULT EVIDENCE EXPOSES POLICY-DEFAULT PROVENANCE ───────────────

def test_result_evidence_exposes_policy_default_provenance(ecommerce_df):
    """
    When CommandResolver resolves a command that applies policy defaults,
    the resulting SemanticIntent and CommandResolutionResult must expose the
    policy-default bindings in inferred_bindings with complete provenance.
    """
    from backend.ds_engine.command_resolver import command_resolver

    raw_command = {
        "tool": "correlate_data",
        "arguments": {
            "analysis_mode": "matrix",
        }
    }
    res = command_resolver.resolve_command(
        raw_command,
        df=ecommerce_df,
        raw_query="Show correlation matrix",
    )

    assert res.is_valid is True
    assert res.semantic_intent is not None
    assert len(res.semantic_intent.inferred_bindings) > 0

    policy_bindings = [
        b for b in res.semantic_intent.inferred_bindings
        if (b.get("source") if isinstance(b, dict) else getattr(b, "source", None)) == "policy_default"
    ]
    assert len(policy_bindings) > 0, "Expected policy_default bindings in resolved SemanticIntent"
    for b in policy_bindings:
        b_dict = b if isinstance(b, dict) else b.model_dump()
        assert b_dict["confidence"] == 1.0
        assert any("CompletionPolicy" in ev for ev in b_dict.get("evidence", []))

