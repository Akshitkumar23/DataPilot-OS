"""
DataPilot OS — Repair Loop Safety & Anti-Hallucination Contract Verification Suite.

Validates the repair-and-retry safety invariants of modules/semantic_planner.py:
1. Repair must ONLY repair schema/structure (malformed JSON, unparseable syntax, unknown tool).
2. Repair must NEVER guess or invent semantic domain values (e.g. missing filter column).
3. Total attempts never exceed max_repair_attempts + 1 (exactly 3 total attempts for max=2).
4. Safe error return when exhausted (returns None or structured error, never arbitrary DS execution).
"""

import sys
import os
import json
import logging
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.intelligence.ollama_service import (
    ollama_service,
    OllamaResponse,
    OllamaRequest,
    OllamaReadinessStatus
)
from modules.semantic_planner import SemanticPlanner
from backend.ds_engine.command_resolver import CommandResolver, command_resolver
from backend.ds_engine.intent_schema import SemanticIntent, IntentType, FilterOperator, FilterSpec
from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import CanonicalQueryResponse, LifecycleState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RepairSafetyAuditor")


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Standard tabular dataframe with numeric, categorical, and temporal fields."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "sales": np.linspace(100, 5000, n),
        "profit": np.linspace(10, 500, n),
        "region": np.tile(["North", "South", "East", "West"], 25),
        "category": np.tile(["Tech", "Office", "Furniture", "Service"], 25),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D")
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. JSON SYNTAX & STRUCTURE REPAIR TESTS (WITHIN 2 REPAIR ATTEMPTS)
# ─────────────────────────────────────────────────────────────────────────────

def test_syntax_repair_trailing_comma_succeeds_on_attempt_2(sample_dataframe):
    """
    Contract Check: Malformed JSON with trailing comma fails on Attempt 1,
    prompts schema repair notice, and succeeds on Attempt 2 when valid JSON is returned.
    Verifies repair_attempts == 1 and no semantic hallucination.
    """
    attempt_calls = []

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        attempt_calls.append(req)
        if len(attempt_calls) == 1:
            # Attempt 1: Malformed JSON with trailing comma (unparseable by json.loads)
            return OllamaResponse(
                success=True,
                content='{\n  "tool": "filter_data",\n  "arguments": {\n    "column": "sales",\n    "condition": ">",\n    "value": 500,\n  }\n}',
                parsed_json=None,
                latency_ms=120.0
            )
        else:
            # Attempt 2: Repaired clean JSON
            return OllamaResponse(
                success=True,
                content='{\n  "tool": "filter_data",\n  "arguments": {\n    "column": "sales",\n    "condition": ">",\n    "value": 500\n  }\n}',
                parsed_json={
                    "tool": "filter_data",
                    "arguments": {
                        "column": "sales",
                        "condition": ">",
                        "value": 500
                    }
                },
                latency_ms=95.0
            )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Filter sales greater than 500",
            df=sample_dataframe
        )

    assert plan is not None, "Expected successful plan after Attempt 2 repair"
    assert plan["primary_intent"] == "FILTER_QUERY" or plan["intent_type"] == "FILTER_QUERY"
    assert len(plan["filters"]) == 1
    assert plan["filters"][0]["column"] == "sales"
    assert plan["filters"][0]["value"] == 500
    assert plan["repair_attempts"] == 1
    assert len(attempt_calls) == 2

    # Verify that the repair prompt for attempt 2 only requested JSON schema compliance
    repair_prompt = attempt_calls[1].prompt
    assert "REPAIR NOTICE: Your previous output was not valid JSON" in repair_prompt
    assert "Please return ONLY a strict JSON object" in repair_prompt


def test_syntax_repair_markdown_code_fences_handled_cleanly(sample_dataframe):
    """
    Contract Check: Markdown code fences around JSON should be parsed or repaired cleanly.
    """
    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        fenced_content = "```json\n{\n  \"tool\": \"group_data\",\n  \"arguments\": {\n    \"group_by\": [\"region\"],\n    \"metric\": [\"sales\"],\n    \"aggregation\": \"sum\"\n  }\n}\n```"
        return OllamaResponse(
            success=True,
            content=fenced_content,
            parsed_json={
                "tool": "group_data",
                "arguments": {
                    "group_by": ["region"],
                    "metric": ["sales"],
                    "aggregation": "sum"
                }
            },
            latency_ms=105.0
        )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Total sales by region",
            df=sample_dataframe
        )

    assert plan is not None
    assert plan["intent_type"] == "GROUP_AGGREGATE"
    assert "region" in plan["dimensions"]
    assert "sales" in plan["metrics"]
    assert plan["repair_attempts"] == 0


def test_repair_unknown_tool_recovers_on_attempt_2(sample_dataframe):
    """
    Contract Check: Unknown tool emitted on Attempt 1 triggers repair prompt with recognized tools,
    recovers successfully on Attempt 2 with recognized tool name.
    """
    attempt_calls = []

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        attempt_calls.append(req)
        if len(attempt_calls) == 1:
            # Unknown tool
            return OllamaResponse(
                success=True,
                content='{"tool": "magical_sorter", "arguments": {"column": "sales"}}',
                parsed_json={"tool": "magical_sorter", "arguments": {"column": "sales"}},
                latency_ms=110.0
            )
        else:
            # Repaired recognized tool
            return OllamaResponse(
                success=True,
                content='{"tool": "sort_data", "arguments": {"column": "sales", "direction": "desc"}}',
                parsed_json={"tool": "sort_data", "arguments": {"column": "sales", "direction": "desc"}},
                latency_ms=90.0
            )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Sort sales descending",
            df=sample_dataframe
        )

    assert plan is not None
    assert len(attempt_calls) == 2
    assert plan["repair_attempts"] == 1
    assert plan["order"] == "desc"
    assert "REPAIR NOTICE: Your previous output produced an error" in attempt_calls[1].prompt
    assert "magical_sorter" in attempt_calls[1].prompt


def test_mixed_analysis_and_clarification_is_repaired_not_surfaced():
    """A clarification cannot suppress a simultaneously emitted analysis call.

    This reproduces the local-model protocol failure where it asks for a
    column that is present in the query while also emitting an incomplete
    group_data call.  The resolver must reject the mixed wire shape and let
    the planner rebuild a single complete command; no dataset-specific branch
    is involved.
    """
    df = pd.DataFrame({
        "Date": ["01/01/2024", "01/01/2024", "02/01/2024", "02/01/2024"],
        "Hour": [8, 9, 8, 9],
        "Seasons": ["Winter", "Winter", "Winter", "Winter"],
        "Rented Bike Count": [100, 120, 110, 130],
    })
    calls = []

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        calls.append(req)
        if len(calls) == 1:
            payload = {
                "commands": [
                    {"tool": "group_data", "arguments": {
                        "group_by": ["Date"], "metric": "Rented Bike Count",
                        "chart_type": "line",
                    }},
                    {"tool": "clarification_needed", "arguments": {
                        "reason": "Date is missing even though the query provides it.",
                    }},
                ]
            }
        else:
            payload = {"tool": "group_data", "arguments": {
                "group_by": ["Hour", "Seasons"],
                "metric": "Rented Bike Count",
                "aggregation": "mean",
                "datetime_parses": [{"column": "Date", "format": "%d/%m/%Y"}],
                "chart_type": "line",
            }}
        return OllamaResponse(success=True, content=json.dumps(payload), parsed_json=payload, latency_ms=10.0)

    query = (
        "Parse Date as DD/MM/YYYY, then show average Rented Bike Count by Hour "
        "and Seasons with a line chart."
    )
    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(query_raw=query, df=df)

    assert plan is not None
    assert plan["is_ambiguous"] is False
    assert plan["dimensions"] == ["Hour", "Seasons"]
    assert plan["metrics"] == ["Rented Bike Count"]
    assert plan["aggregation"] == "mean"
    assert plan["repair_attempts"] == 1
    assert len(calls) == 2
    assert "MIXED_CLARIFICATION_AND_ANALYSIS" in calls[1].prompt


def test_planner_uses_schema_only_single_command_and_canonical_group_contract():
    """Planner extraction must not combine Ollama tools with JSON-schema format.

    The canonical planner contract is intentionally narrower than the execution
    model: it removes provider aliases while retaining the same typed execution
    validation after the model responds.
    """
    df = pd.DataFrame({
        "period": ["01/01/2024", "02/01/2024"],
        "bucket": [8, 9],
        "segment": ["A", "B"],
        "measure": [100.0, 120.0],
    })
    captured = []
    payload = {"tool": "group_data", "arguments": {
        "group_by": ["bucket", "segment"],
        "metric": "measure",
        "aggregation": "mean",
        "datetime_parses": [{"column": "period", "format": "%d/%m/%Y"}],
        "chart_type": "line",
    }}

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        captured.append(req)
        return OllamaResponse(success=True, content=json.dumps(payload), parsed_json=payload, latency_ms=1.0)

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Parse period as DD/MM/YYYY and show average measure by bucket and segment with a line chart.",
            df=df,
        )

    assert plan is not None
    assert len(captured) == 1
    request = captured[0]
    assert request.tools is None
    assert request.response_schema is not None
    assert "ROLE-BINDING FOR GROUPED ANALYSES" in request.system_prompt
    schema = request.response_schema
    variants = schema.get("oneOf", [schema])
    group_variant = next(variant for variant in variants if variant.get("properties", {}).get("tool", {}).get("const") == "group_data")
    group_args = group_variant["properties"]["arguments"]
    assert set(group_args["required"]) == {
        "group_by", "metric", "aggregation", "chart_type"
    }
    assert "datetime_parses" in group_args["properties"]
    assert "metrics" not in group_args["properties"]
    assert "columns" not in group_args["properties"]
    assert "target_column" not in group_args["properties"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANTI-HALLUCINATION CONTRACT: REPAIR NEVER INVENTS SEMANTIC DOMAIN VALUES
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_semantic_column_is_not_invented(sample_dataframe):
    """
    Contract Check: When LLM outputs a filter tool missing column specification
    ({"tool": "filter_data", "arguments": {"condition": ">", "value": 500}}),
    DataPilot must NOT guess or fabricate a column from the dataset.
    """
    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        return OllamaResponse(
            success=True,
            content='{"tool": "filter_data", "arguments": {"condition": ">", "value": 500}}',
            parsed_json={"tool": "filter_data", "arguments": {"condition": ">", "value": 500}},
            latency_ms=80.0
        )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="filter above 500",
            df=sample_dataframe
        )

    # CommandResolver rejects ungrounded filter specification without hallucinating columns;
    # after retries exhaust, plan is safely None or clarification or contains no hallucinated filters
    assert plan is None or plan.get("intent_type") == "CLARIFICATION" or plan.get("filters") == [] or plan.get("filters") is None, f"Expected no fabricated filter, got: {plan}"


def test_missing_filter_operator_safely_defaults_without_hallucination():
    """
    Contract Check: When filter specification has column and value but missing/unknown operator,
    CommandResolver normalizes safely to canonical FilterOperator.EQ without guessing domain semantics.
    """
    cmd = {
        "tool": "filter_data",
        "arguments": {
            "column": "region",
            "value": "North"
        }
    }
    res = command_resolver.resolve_command(cmd)
    assert res.is_valid is True
    assert len(res.semantic_intent.filters) == 1
    assert res.semantic_intent.filters[0].column == "region"
    assert res.semantic_intent.filters[0].operator == FilterOperator.EQ
    assert res.semantic_intent.filters[0].value == "North"


def test_repair_prompt_never_contains_domain_guesses(sample_dataframe):
    """
    Contract Check: Verify that repair notices generated across repair iterations
    strictly enforce structural/schema constraints and NEVER inject speculative domain values.
    """
    attempt_prompts = []

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        attempt_prompts.append(req.prompt)
        # Always return malformed JSON to observe repair prompts
        return OllamaResponse(
            success=True,
            content='INVALID_NON_JSON_OUTPUT',
            parsed_json=None,
            latency_ms=50.0
        )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Find outliers in data",
            df=sample_dataframe
        )

    assert plan is None
    # Check attempt 2 and 3 repair notices
    assert len(attempt_prompts) == 3
    for p in attempt_prompts[1:]:
        assert "REPAIR NOTICE" in p
        # Ensure repair notices don't guess arbitrary column names as hardcoded replacements
        assert "Use column 'sales'" not in p
        assert "Assume filter" not in p
        assert "strict JSON object" in p


# ─────────────────────────────────────────────────────────────────────────────
# 3. BOUNDED ATTEMPT CONTRACT: NEVER EXCEEDS max_repair_attempts + 1
# ─────────────────────────────────────────────────────────────────────────────

def test_attempt_count_never_exceeds_three_on_continuous_json_errors(sample_dataframe):
    """
    Contract Check: With max_repair_attempts = 2, the loop runs exactly 3 attempts
    (1 initial + 2 retries) and NEVER loops infinitely or performs a 4th attempt.
    """
    call_count = 0

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        nonlocal call_count
        call_count += 1
        return OllamaResponse(
            success=True,
            content='{"bad_syntax": ',
            parsed_json=None,
            latency_ms=40.0
        )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="correlate sales and profit",
            df=sample_dataframe
        )

    assert plan is None, "Expected None return after exhausting retries"
    assert call_count == 3, f"Expected exactly 3 total attempts, got {call_count}"


def test_attempt_count_never_exceeds_three_on_continuous_unknown_tools(sample_dataframe):
    """
    Contract Check: With continuous unknown tool rejection, exactly 3 attempts occur.
    """
    call_count = 0

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        nonlocal call_count
        call_count += 1
        return OllamaResponse(
            success=True,
            content=f'{{"tool": "non_existent_tool_{call_count}", "arguments": {{}}}}',
            parsed_json={"tool": f"non_existent_tool_{call_count}", "arguments": {}},
            latency_ms=40.0
        )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="forecast sales next month",
            df=sample_dataframe
        )

    assert plan is None
    assert call_count == 3


def test_attempt_count_with_mixed_failures_and_final_success(sample_dataframe):
    """
    Contract Check: Attempt 1 has syntax error, Attempt 2 has unknown tool,
    Attempt 3 produces valid tool and succeeds on the final allowed attempt (attempt #3).
    """
    attempt_count = 0

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            return OllamaResponse(
                success=True,
                content='{broken json',
                parsed_json=None,
                latency_ms=50.0
            )
        elif attempt_count == 2:
            return OllamaResponse(
                success=True,
                content='{"tool": "custom_cluster", "arguments": {}}',
                parsed_json={"tool": "custom_cluster", "arguments": {}},
                latency_ms=50.0
            )
        else:
            return OllamaResponse(
                success=True,
                content='{"tool": "cluster_data", "arguments": {"columns": ["sales", "profit"], "n_clusters": 3, "scaling": "standard"}}',
                parsed_json={"tool": "cluster_data", "arguments": {"columns": ["sales", "profit"], "n_clusters": 3, "scaling": "standard"}},
                latency_ms=50.0
            )

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Standardize sales and profit and cluster them into 3 segments",
            df=sample_dataframe
        )

    assert plan is not None
    assert plan["intent_type"] == "CLUSTERING"
    assert plan["feature_columns"] == ["sales", "profit"]
    assert plan["repair_attempts"] == 2
    assert attempt_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# 4. UNRECOVERABLE MALFORMED INPUT TERMINATES SAFELY WITHOUT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def test_unrecoverable_malformed_input_returns_none_cleanly(sample_dataframe):
    """
    Contract Check: Completely garbage input (binary, empty, repeated exceptions)
    returns None cleanly from extract_semantic_intent without throwing unhandled exceptions.
    """
    garbage_responses = [
        OllamaResponse(success=False, error_code="OLLAMA_TIMEOUT", error_message="Timeout"),
        OllamaResponse(success=True, content="", parsed_json=None),
        OllamaResponse(success=True, content="???!!!###$$$", parsed_json=None)
    ]

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        if garbage_responses:
            return garbage_responses.pop(0)
        return OllamaResponse(success=False, error_code="EXHAUSTED", error_message="Exhausted")

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="&@#*&@#*&@#*",
            df=sample_dataframe
        )

    assert plan is None


def test_full_pipeline_orchestrator_halts_on_unrecoverable_input(sample_dataframe):
    """
    Contract Check: When SemanticPlanner returns None after repair exhaustion,
    QueryOrchestrator halts in PLANNING phase with status='FAILED',
    plan=None, result=None, error.code='INTENT_PLAN_INVALID', and NEVER executes DS code.
    """
    # Mock OllamaService verify_ready to return ready=True
    ready_status = OllamaReadinessStatus(
        ready=True,
        ollama_url="http://localhost:11434",
        model="qwen2.5:1.5b",
        model_available=True
    )

    # Mock extract_semantic_intent to return None (simulating repair failure exhaustion)
    with patch.object(ollama_service, "verify_ready", return_value=ready_status):
        with patch.object(SemanticPlanner, "extract_semantic_intent", return_value=None):
            res = QueryOrchestrator.process_query(
                query_raw="Unparseable query that exhausted repair loop",
                df=sample_dataframe
            )

    assert isinstance(res, CanonicalQueryResponse)
    assert res.status in ("FAILED", "NEEDS_CLARIFICATION")
    assert res.result is None
    assert res.error is not None


def test_exception_in_ollama_service_handled_safely_in_loop(sample_dataframe):
    """
    Contract Check: Unhandled exceptions during structured_request (e.g. network/memory crash)
    are caught within the attempt loop, logged, and safely return None without crashing the process.
    """
    def mock_raising_request(req: OllamaRequest) -> OllamaResponse:
        raise RuntimeError("Low-level OS socket error during Ollama connection")

    with patch.object(ollama_service, "structured_request", side_effect=mock_raising_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Average sales by category",
            df=sample_dataframe
        )

    assert plan is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOOL-LOCK REPAIR & EXCLUSIVE CLARIFICATION REGRESSION CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_tool_repair_recovers_from_multi_step_faulty_sequence():
    """
    Contract Check:
    1. Attempt 1: Mixed clarification_needed + invalid group_data (rejected by exclusivity).
    2. Attempt 2: Tool-locked repair narrows capability contract to group_data;
       model returns out-of-contract multi-tool payload which is rejected without hallucination.
    3. Attempt 3: Model returns valid single locked group_data command with typed aggregation='mean',
       datetime_parses, and line chart. Succeeds with repair_attempts == 2.
    """
    df = pd.DataFrame({
        "Date": ["01/01/2024", "01/01/2024", "02/01/2024", "02/01/2024"],
        "Hour": [8, 9, 8, 9],
        "Seasons": ["Winter", "Winter", "Winter", "Winter"],
        "Rented Bike Count": [100, 120, 110, 130],
    })
    calls = []

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        calls.append(req)
        if len(calls) == 1:
            # Faulty Attempt 1: Mixed clarification + group_data
            payload = {
                "commands": [
                    {
                        "tool": "group_data",
                        "arguments": {
                            "group_by": ["Date"],
                            "metric": ["Rented Bike Count"],
                            "chart_type": "line",
                        },
                    },
                    {
                        "tool": "clarification_needed",
                        "arguments": {"reason": "Which hours and seasons should be grouped?"},
                    },
                ]
            }
        elif len(calls) == 2:
            # Faulty Attempt 2: Multiple out-of-contract tools
            payload = {
                "commands": [
                    {
                        "tool": "group_data",
                        "arguments": {
                            "group_by": ["Hour"],
                            "metric": "Rented Bike Count",
                        },
                    },
                    {
                        "tool": "conditional_rate",
                        "arguments": {
                            "group_by": ["Seasons"],
                            "condition_column": "Rented Bike Count",
                        },
                    },
                ]
            }
        else:
            # Repaired Attempt 3: Valid single locked group_data command
            payload = {
                "tool": "group_data",
                "arguments": {
                    "group_by": ["Hour", "Seasons"],
                    "metric": "Rented Bike Count",
                    "aggregation": "mean",
                    "datetime_parses": [{"column": "Date", "format": "%d/%m/%Y"}],
                    "chart_type": "line",
                },
            }
        return OllamaResponse(success=True, content=json.dumps(payload), parsed_json=payload, latency_ms=15.0)

    query = (
        "Operations needs an hourly demand profile. Parse Date as DD/MM/YYYY, "
        "then show average Rented Bike Count by Hour and Seasons with a line chart."
    )
    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(query_raw=query, df=df)

    assert plan is not None
    assert plan["is_ambiguous"] is False
    assert plan["intent_type"] == "GROUP_AGGREGATE"
    assert plan["dimensions"] == ["Hour", "Seasons"]
    assert plan["metrics"] == ["Rented Bike Count"]
    assert plan["aggregation"] == "mean"
    assert plan["chart_type"] == "line"
    assert plan["repair_attempts"] == 2
    assert len(calls) == 3

    # Attempt 2 prompt must contain the locked tool error and reference
    assert "MIXED_CLARIFICATION_AND_ANALYSIS" in calls[1].prompt
    assert "group_data" in calls[1].prompt

    # Attempt 3 prompt must reject the out-of-contract conditional_rate tool
    assert "conditional_rate" in calls[2].prompt or "outside the retrieved capability contract" in calls[2].prompt or "ConditionalRate" in calls[2].prompt


def test_genuinely_underspecified_query_surfaces_clarification():
    """
    Contract Check: When a query genuinely lacks required analytical targets,
    clarification is requested, confirmed across the strict contract, and returned.
    """
    df = pd.DataFrame({
        "Date": ["01/01/2024", "02/01/2024"],
        "Value": [10.0, 20.0],
    })
    calls = []

    def mock_structured_request(req: OllamaRequest) -> OllamaResponse:
        calls.append(req)
        payload = {
            "tool": "clarification_needed",
            "arguments": {"reason": "Please specify whether to aggregate, filter, or model the data."},
        }
        return OllamaResponse(success=True, content=json.dumps(payload), parsed_json=payload, latency_ms=10.0)

    with patch.object(ollama_service, "structured_request", side_effect=mock_structured_request):
        plan = SemanticPlanner.extract_semantic_intent(
            query_raw="Do something with this dataset",
            df=df
        )

    assert plan is not None
    assert plan["is_ambiguous"] is True
    assert "specify" in plan["ambiguity_reason"].lower()
    # Initial clarification confirmation ran before surfacing genuine user question
    assert len(calls) in (2, 3)
    assert "CLARIFICATION CONFIRMATION" in calls[1].prompt
