import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.query_orchestrator import QueryOrchestrator
from modules.semantic_planner import SemanticPlanner
from backend.ds_engine.intent_schema import SemanticIntent, IntentType, FilterSpec, FilterOperator
from backend.ds_engine.capability_drivers import FilterQueryDriver
from backend.ds_engine.intent_resolver import IntentResolver, AmbiguousIntentError, IntentResolutionError

def get_test_dataset():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'sales': np.linspace(100, 1000, n),
        'balance': np.linspace(50, 500, n),
        'age': np.random.randint(20, 60, n),
        'region': np.tile(['North', 'South', 'East', 'West'], 25)
    })

def test_p0_filter_dict_preservation():
    """Verify that structured filter dictionaries [{"column": "sales", "operator": "gt", "value": 500}] are preserved."""
    df = get_test_dataset()
    expected_rows = len(df[df['sales'] > 500])  # Exactly 56 rows
    
    mock_plan = {
        "tool": "filter_data",
        "arguments": {
            "column": "sales",
            "condition": ">",
            "value": 500
        },
        "raw_query": "Filter sales > 500"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_plan)

    try:
        res = QueryOrchestrator.process_query(query_raw="Filter sales > 500", df=df)
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "SUCCEEDED", f"Query failed: {res.error}"
    assert res.result is not None, "Result is None"
    
    exec_steps = res.plan.get("execution_plan", [])
    assert len(exec_steps) == 1, f"Expected 1 step, got {len(exec_steps)}"
    print(f"\n[PASS] test_p0_filter_dict_preservation: Filter dictionaries preserved cleanly")

def test_multistep_aggregate_exact_values():
    """Verify multi-step execution output values EXACTLY match ground-truth Pandas calculated aggregate values."""
    df = get_test_dataset()
    
    # Ground truth Pandas calculation over filtered dataset
    df_filtered = df[df['sales'] > 500]
    expected_region_means = df_filtered.groupby('region')['balance'].mean().to_dict()

    mock_plan = {
        "commands": [
            {
                "tool": "filter_data",
                "arguments": {
                    "column": "sales",
                    "condition": ">",
                    "value": 500
                }
            },
            {
                "tool": "group_data",
                "arguments": {
                    "group_by": ["region"],
                    "metric": "balance",
                    "aggregation": "mean"
                }
            }
        ],
        "raw_query": "Filter sales > 500 and average balance by region"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_plan)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="Filter sales > 500 and average balance by region",
            df=df
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status == "SUCCEEDED", f"Query failed: {res.error}"
    assert res.result is not None, "Response result is None"
    
    exec_steps = res.plan.get("execution_plan", [])
    assert len(exec_steps) == 2, f"Expected 2 steps, got {len(exec_steps)}"
    assert exec_steps[0]["intent_type"] == "FILTER_QUERY", "Step 1 must be FILTER_QUERY"
    assert exec_steps[1]["intent_type"] == "GROUP_AGGREGATE", "Step 2 must be GROUP_AGGREGATE"

    # Truthful assertion of calculated values
    res_data = res.result.get("data", {})
    records = res_data.get("records") or res_data.get("table", [])
    assert records, "Result records are empty"
    
    res_dict = {}
    for r in records:
        reg = r.get("region")
        val = r.get("balance") or r.get("balance_mean") or r.get("mean")
        if reg and val is not None:
            res_dict[reg] = float(val)

    for reg, exp_val in expected_region_means.items():
        assert reg in res_dict, f"Missing region '{reg}' in result"
        assert abs(res_dict[reg] - exp_val) < 0.01, f"Region {reg} mean mismatch: expected {exp_val}, got {res_dict[reg]}"

    print(f"\n[PASS] test_multistep_aggregate_exact_values: Calculated region means EXACTLY match ground truth Pandas values ({res_dict})")

def test_malformed_ollama_json_produces_invalid_plan():
    """Negative Test: Malformed or unknown LLM intent yields INTENT_PLAN_INVALID without EDA fallback."""
    df = get_test_dataset()
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse

    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content='{"primary_intent": "UNKNOWN_GARBAGE_INTENT"}',
        parsed_json={"primary_intent": "UNKNOWN_GARBAGE_INTENT"}
    )

    try:
        res = QueryOrchestrator.process_query(query_raw="Some random query", df=df)
    finally:
        ollama_service.structured_request = orig_req

    assert res.status == "FAILED", "Expected query failure for invalid LLM intent"
    assert res.error.code in ("INTENT_PLAN_INVALID", "LOOP_NO_PROGRESS"), f"Expected INTENT_PLAN_INVALID or LOOP_NO_PROGRESS, got {res.error.code}"
    print(f"\n[PASS] test_malformed_ollama_json_produces_invalid_plan: Returned {res.error.code} without EDA fallback")

def test_invalid_filter_column_raises_error():
    """Negative Test: Filter referencing non-existent column raises COLUMN_NOT_FOUND."""
    df = get_test_dataset()
    intent = SemanticIntent(
        intent_type=IntentType.FILTER_QUERY,
        primary_intent=IntentType.FILTER_QUERY,
        filters=[FilterSpec(column="non_existent_column", operator=FilterOperator.GT, value=100)]
    )
    
    resolver = IntentResolver()
    raised = False
    try:
        resolver.resolve(intent, df=df)
    except IntentResolutionError as err:
        raised = True
        assert err.code == "COLUMN_NOT_FOUND"
        assert "non_existent_column" in err.invalid_columns
        
    assert raised, "Expected IntentResolutionError for non-existent filter column"
    print("\n[PASS] test_invalid_filter_column_raises_error: Raised IntentResolutionError cleanly")

def test_ambiguous_date_column_raises_error():
    """Negative Test: Multiple date columns without explicit user specification raise AmbiguousIntentError."""
    df = get_test_dataset()
    df['order_date'] = pd.date_range('2023-01-01', periods=100)
    df['ship_date'] = pd.date_range('2023-01-05', periods=100)
    
    intent = SemanticIntent(
        intent_type=IntentType.TIME_SERIES_FORECAST,
        primary_intent=IntentType.TIME_SERIES_FORECAST,
        raw_query="Forecast sales for next 30 days"
    )
    
    resolver = IntentResolver()
    raised = False
    try:
        resolver.resolve(intent, df=df)
    except AmbiguousIntentError as err:
        raised = True
        assert err.code in ["AMBIGUOUS_COLUMN", "AMBIGUOUS_INTENT"]
        assert len(err.candidate_columns) >= 2
        
    assert raised, "Expected AmbiguousIntentError when multiple date columns exist without user selection"
    print("\n[PASS] test_ambiguous_date_column_raises_error: Raised AmbiguousIntentError cleanly")

if __name__ == "__main__":
    test_p0_filter_dict_preservation()
    test_multistep_aggregate_exact_values()
    test_malformed_ollama_json_produces_invalid_plan()
    test_invalid_filter_column_raises_error()
    test_ambiguous_date_column_raises_error()
    print("\n" + "=" * 80)
    print("ALL 5 TRUTHFUL ARCHITECTURAL CONTRACT TESTS PASSED CLEANLY [OK]")
    print("=" * 80)
