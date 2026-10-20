"""
DataPilot OS — Live Ollama E2E Integration & Structured Output Verification.
Tests live requests against local Ollama server (http://localhost:11434).
Verifies JSON schema enforcement, minimal semantic command generation,
natural language normalization, and execution pipeline integrity.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

pytestmark = pytest.mark.live_ollama

sys.path.insert(0, os.path.abspath("."))

from backend.intelligence.ollama_service import ollama_service
from modules.semantic_planner import SemanticPlanner
from modules.query_orchestrator import QueryOrchestrator


def check_ollama_available():
    status = ollama_service.detect_server()
    return status.is_running and len(status.available_models) > 0


@pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
def test_live_ollama_structured_filtering():
    """Live Ollama test: natural language filter query."""
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "sales_amount": [120.0, 450.0, 850.0, 920.0, 310.0],
        "region": ["North", "South", "East", "West", "North"]
    })

    query = "Show only records where sales_amount is strictly above 500"
    intent = SemanticPlanner.extract_semantic_intent(query, df=df)

    assert intent is not None, "Live Ollama should return a valid structured intent"
    assert "schema_version" in intent
    assert "prompt_version" in intent
    assert "raw_llm_output" in intent
    assert intent.get("intent_type") == "FILTER_QUERY"
    
    # Run full query through orchestrator
    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    assert res.status == "SUCCEEDED"
    assert len(res.result["data"]["records"]) == 2  # 850.0 and 920.0


@pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
def test_live_ollama_structured_group_aggregation():
    """Live Ollama test: group aggregation query."""
    df = pd.DataFrame({
        "category": ["Electronics", "Electronics", "Clothing", "Clothing", "Home"],
        "revenue": [500, 700, 200, 300, 400]
    })

    query = "Calculate average revenue by category"
    intent = SemanticPlanner.extract_semantic_intent(query, df=df)

    assert intent is not None
    assert intent.get("intent_type") == "GROUP_AGGREGATE"

    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    assert res.status == "SUCCEEDED"
    assert len(res.result["data"]["records"]) == 3


@pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
def test_live_ollama_multistep_compound_query():
    """Live Ollama test: compound multi-step query (clean missing then detect anomalies)."""
    np.random.seed(42)
    df = pd.DataFrame({
        "reading_1": [10.0, 11.2, np.nan, 10.8, 9.9, 10.5, 99.0, 10.2, 10.4, 11.0],
        "reading_2": [20.0, 21.0, 20.5, np.nan, 19.8, 20.2, 88.0, 20.1, 20.3, 21.1]
    })

    query = "First clean missing values then detect anomalies on reading_1 and reading_2"
    res = QueryOrchestrator.process_query(query_raw=query, df=df)

    assert res.status == "SUCCEEDED"
    assert res.plan is not None
