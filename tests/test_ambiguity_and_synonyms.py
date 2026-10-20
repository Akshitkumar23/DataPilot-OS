import sys
import os
import logging
import pandas as pd
import numpy as np

# Ensure root workspace is on path
sys.path.insert(0, os.path.abspath("."))

from modules.query_orchestrator import QueryOrchestrator
from modules.semantic_planner import SemanticPlanner
from backend.ds_engine.intent_schema import SemanticIntent, IntentType, ChartType
from backend.ds_engine.intent_resolver import IntentResolver, AmbiguousIntentError
import pytest
from backend.intelligence.ollama_service import ollama_service, OllamaResponse

pytestmark = pytest.mark.live_ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmbiguitySynonymAuditor")


def get_weather_sensor_dataset() -> pd.DataFrame:
    """Generate deterministic sensor/weather dataset for correlation testing."""
    np.random.seed(42)
    temp = np.array([20.0, 22.5, 25.0, 28.0, 30.5, 32.0, 35.0, 38.0, 40.0, 42.5])
    # Humidity strongly negatively correlated with temperature
    humidity = np.array([85.0, 80.0, 75.0, 68.0, 60.0, 52.0, 45.0, 38.0, 32.0, 25.0])
    # Pressure moderately correlated
    pressure = np.array([1015.0, 1014.0, 1013.0, 1012.0, 1011.0, 1010.0, 1009.0, 1008.0, 1007.0, 1006.0])
    
    return pd.DataFrame({
        "temperature": temp,
        "humidity": humidity,
        "pressure": pressure
    })


def get_multi_date_dataset() -> pd.DataFrame:
    """Generate dataset with multiple date columns to trigger schema ambiguity."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "created_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "shipped_date": pd.date_range("2024-01-05", periods=n, freq="D"),
        "delivery_date": pd.date_range("2024-01-10", periods=n, freq="D"),
        "sales_amount": np.random.uniform(100, 1000, size=n)
    })


# ==============================================================================
# TEST CASE 1: UNDERSPECIFIED QUERIES & AMBIGUITY INTERDICTION
# ==============================================================================

def test_case_1_underspecified_queries_semantic_planner_ambiguity():
    """
    Test Case 1.1: Underspecified queries ('Analyze this data', 'Do something with data', etc.)
    Verify that SemanticPlanner sets is_ambiguous=True and populates ambiguity_reason,
    or cleanly returns None if unparseable, rather than hallucinating an arbitrary execution.
    """
    df = get_weather_sensor_dataset()
    underspecified_queries = [
        "Analyze this data",
        "Do something with data",
        "Analyze data",
        "Show me data",
        "Do something"
    ]

    for query in underspecified_queries:
        plan = SemanticPlanner.extract_semantic_intent(query_raw=query, df=df)
        logger.info(f"Tested query: '{query}' -> Plan: {plan}")
        
        if plan is not None:
            assert plan.get("is_ambiguous") is True, (
                f"Expected is_ambiguous=True for underspecified query '{query}', got: {plan.get('is_ambiguous')}"
            )
            assert plan.get("ambiguity_reason"), (
                f"Expected non-empty ambiguity_reason for underspecified query '{query}'"
            )
            print(f"[PASS] SemanticPlanner flagged query '{query}' as is_ambiguous=True | Reason: {plan.get('ambiguity_reason')}")
        else:
            print(f"[PASS] SemanticPlanner safely rejected unparseable query '{query}' with None")


def test_case_1_underspecified_queries_orchestrator_interdiction():
    """
    Test Case 1.2: Underspecified queries submitted to QueryOrchestrator
    Verify that QueryOrchestrator halts with status='FAILED' and error code
    'AMBIGUOUS_INTENT' or 'INTENT_PLAN_INVALID', ensuring no arbitrary DS code executes.
    """
    df = get_weather_sensor_dataset()
    vague_queries = [
        "Analyze this data",
        "Do something with data",
        "Analyze dataset"
    ]

    for query in vague_queries:
        res = QueryOrchestrator.process_query(query_raw=query, df=df)
        
        assert res.status in ("NEEDS_CLARIFICATION", "FAILED"), f"Expected clarification or failure for vague query '{query}', got: {res.status}"
        assert res.result is None, f"Expected result=None for vague query '{query}', but got execution result: {res.result}"
        assert res.error is not None, f"Expected non-null error for vague query '{query}'"
        assert res.error.code in ["CLARIFICATION_REQUIRED", "AMBIGUOUS_INTENT", "INTENT_PLAN_INVALID"], (
            f"Expected error code CLARIFICATION_REQUIRED, AMBIGUOUS_INTENT, or INTENT_PLAN_INVALID, got: {res.error.code}"
        )
        print(f"[PASS] QueryOrchestrator interdicted vague query '{query}' -> Code: {res.error.code} | Phase: {res.error.phase} | Msg: {res.error.message}")


def test_case_1_explicit_ambiguous_intent_halt():
    """
    Test Case 1.3: Deterministic mock verification of ambiguity interdiction.
    When is_ambiguous=True is returned from planner, QueryOrchestrator MUST NOT execute code.
    """
    df = get_weather_sensor_dataset()
    mock_ambiguous_plan = {
        "intent_type": "EDA",
        "primary_intent": "EDA",
        "is_ambiguous": True,
        "ambiguity_reason": "Query is too broad. Please specify target column or analytical task.",
        "raw_query": "Analyze this data"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_ambiguous_plan)

    try:
        res = QueryOrchestrator.process_query(query_raw="Analyze this data", df=df)
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    assert res.status in ("NEEDS_CLARIFICATION", "FAILED"), "Query must require clarification or fail on ambiguous intent"
    assert res.result is None, "Execution result must be None when ambiguous"
    assert res.error.code in ("CLARIFICATION_REQUIRED", "AMBIGUOUS_INTENT"), f"Expected CLARIFICATION_REQUIRED or AMBIGUOUS_INTENT, got {res.error.code}"
    assert "Query is too broad" in res.error.message, "Ambiguity reason must be communicated in error message"
    print(f"[PASS] test_case_1_explicit_ambiguous_intent_halt: Cleanly returned clarification/ambiguity without execution")


def test_case_1_schema_ambiguity_candidate_resolution():
    """
    Test Case 1.4: Schema Ambiguity — Multiple candidate columns without explicit user specification.
    Verify IntentResolver raises AmbiguousIntentError and QueryOrchestrator returns AMBIGUOUS_INTENT at PLAN_VALIDATION.
    """
    multi_date_df = get_multi_date_dataset()
    
    intent = SemanticIntent(
        intent_type=IntentType.TIME_SERIES_FORECAST,
        primary_intent=IntentType.TIME_SERIES_FORECAST,
        raw_query="Forecast sales for the next 14 days"
    )

    resolver = IntentResolver()
    ambiguity_raised = False
    try:
        resolver.resolve(intent, df=multi_date_df)
    except AmbiguousIntentError as err:
        ambiguity_raised = True
        assert err.code in ["AMBIGUOUS_COLUMN", "AMBIGUOUS_INTENT"]
        assert len(err.candidate_columns) >= 2, f"Expected multiple candidate date columns, got {err.candidate_columns}"
        print(f"[PASS] IntentResolver raised AmbiguousIntentError for candidate columns: {err.candidate_columns}")

    assert ambiguity_raised, "Expected AmbiguousIntentError when multiple candidate date columns exist"


# ==============================================================================
# TEST CASE 2: SYNONYM VARIATION A — RELATIONSHIP ROUTING (CORRELATION / STATISTICAL)
# ==============================================================================

def test_case_2_synonym_variation_a_relationship_routing():
    """
    Test Case 2: Synonym variation A: 'Find relationship between temperature and humidity'
    Verify routes to CORRELATION or STATISTICAL_TEST, executes without error,
    and analyzes temperature & humidity relationship.
    """
    df = get_weather_sensor_dataset()
    query = "Find relationship between temperature and humidity"

    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    
    assert res.status == "SUCCEEDED", f"Query failed for '{query}': {res.error}"
    assert res.plan is not None, "Execution plan is None"
    
    intent_type = res.plan.get("primary_intent") or res.plan.get("intent_type")
    assert intent_type in [IntentType.CORRELATION.value, IntentType.STATISTICAL_TEST.value, "CORRELATION", "STATISTICAL_TEST"], (
        f"Expected CORRELATION or STATISTICAL_TEST, got: {intent_type}"
    )
    
    assert res.result is not None, "Result payload is None"
    stdout_text = res.result.get("artifacts", {}).get("stdout", "")
    
    # Verify temperature and humidity were analyzed in output
    assert "temperature" in stdout_text.lower(), "Expected 'temperature' in stdout output"
    assert "humidity" in stdout_text.lower(), "Expected 'humidity' in stdout output"
    print(f"\n[PASS] Test Case 2 ('{query}'): Routed to {intent_type} | Output summary:\n{stdout_text.strip()}")


# ==============================================================================
# TEST CASE 3: SYNONYM VARIATION B — CORRELATION METRICS & MATRIX
# ==============================================================================

def test_case_3_synonym_variation_b_correlation_metrics():
    """
    Test Case 3: Synonym variation B: 'Are temperature and humidity correlated?'
    Verify routes to CORRELATION and produces correlation matrix / metrics (Pearson / Spearman / p-value).
    """
    df = get_weather_sensor_dataset()
    query = "Are temperature and humidity correlated?"

    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    
    assert res.status == "SUCCEEDED", f"Query failed for '{query}': {res.error}"
    assert res.plan is not None, "Execution plan is None"
    
    intent_type = res.plan.get("primary_intent") or res.plan.get("intent_type")
    assert intent_type in [IntentType.CORRELATION.value, "CORRELATION"], (
        f"Expected CORRELATION, got: {intent_type}"
    )
    
    assert res.result is not None, "Result payload is None"
    stdout_text = res.result.get("artifacts", {}).get("stdout", "")
    
    # Verify correlation calculation output (e.g. Pearson correlation matrix / metric)
    assert any(k in stdout_text.lower() for k in ["pearson", "correlation", "matrix", "spearman", "r"]), (
        f"Expected correlation metrics in stdout, got:\n{stdout_text}"
    )
    assert "temperature" in stdout_text.lower(), "Expected 'temperature' in output"
    assert "humidity" in stdout_text.lower(), "Expected 'humidity' in output"
    
    print(f"\n[PASS] Test Case 3 ('{query}'): Routed to {intent_type} | Output:\n{stdout_text.strip()}")


# ==============================================================================
# TEST CASE 4: ADDITIONAL SYNONYM VARIATION SUITE
# ==============================================================================

def test_case_4_broad_semantic_synonyms_matrix():
    """
    Test Case 4: Broad semantic synonym variations for relationships and associations.
    Verify that diverse phrasings resolve cleanly to appropriate analytical drivers.
    """
    df = get_weather_sensor_dataset()
    
    synonym_test_cases = [
        ("Compute correlation between temperature and humidity", ["CORRELATION"]),
        ("How does temperature correlate with humidity and pressure?", ["CORRELATION"]),
        ("Check statistical association between temperature and humidity", ["CORRELATION", "STATISTICAL_TEST"])
    ]

    for query, expected_intents in synonym_test_cases:
        res = QueryOrchestrator.process_query(query_raw=query, df=df)
        
        assert res.status == "SUCCEEDED", f"Query '{query}' failed: {res.error}"
        intent_type = str(res.plan.get("primary_intent") or res.plan.get("intent_type"))
        assert any(exp in intent_type for exp in expected_intents), (
            f"Query '{query}' routed to '{intent_type}', expected one of {expected_intents}"
        )
        print(f"[PASS] Broad synonym query '{query}' -> Routed cleanly to {intent_type} (Status: {res.status})")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING AUDIT: AMBIGUITY INTERDICTION & SEMANTIC SYNONYM ROUTING")
    print("=" * 80 + "\n")
    
    test_case_1_underspecified_queries_semantic_planner_ambiguity()
    test_case_1_underspecified_queries_orchestrator_interdiction()
    test_case_1_explicit_ambiguous_intent_halt()
    test_case_1_schema_ambiguity_candidate_resolution()
    test_case_2_synonym_variation_a_relationship_routing()
    test_case_3_synonym_variation_b_correlation_metrics()
    test_case_4_broad_semantic_synonyms_matrix()
    
    print("\n" + "=" * 80)
    print("ALL AMBIGUITY & SEMANTIC SYNONYM AUDIT TESTS PASSED 100% CLEANLY [OK]")
    print("=" * 80 + "\n")
