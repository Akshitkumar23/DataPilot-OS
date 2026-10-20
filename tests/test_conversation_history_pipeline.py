"""
Tests for Conversation History Pipeline in DataPilot OS.
Verifies that conversation history is properly passed to QueryOrchestrator and used during planning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from modules.query_orchestrator import QueryOrchestrator


@pytest.fixture
def df_sample():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "sales": [1000, 1500, 1200, 1800]
    })


def test_conversation_history_passed_to_orchestrator(df_sample, monkeypatch):
    """Verify QueryOrchestrator accepts conversation history without errors."""
    from modules.semantic_planner import SemanticPlanner

    def mock_extract(*args, **kwargs):
        return {
            "commands": [
                {
                    "tool": "group_data",
                    "arguments": {
                        "group_by": ["region"],
                        "metric": "sales",
                        "aggregation": "mean"
                    }
                }
            ]
        }

    monkeypatch.setattr(SemanticPlanner, "extract_semantic_intent", mock_extract)

    history = [
        {"role": "user", "content": "What is the total sales?"},
        {"role": "assistant", "content": "Total sales is 5,500."}
    ]
    query = "Now calculate average sales by region"
    res = QueryOrchestrator.process_query(
        query_raw=query,
        df=df_sample,
        conversation_history=history
    )
    assert res.status == "SUCCEEDED"
    assert res.result is not None


def test_general_chat_with_conversation_history(monkeypatch):
    """Verify general chat uses conversation history."""
    from modules.semantic_planner import SemanticPlanner
    from backend.intelligence.ollama_service import ollama_service

    def mock_extract(*args, **kwargs):
        return {
            "tool": "general_chat",
            "arguments": {"response": ""},
            "raw_query": kwargs.get("query_raw") or (args[0] if args else ""),
            "is_ambiguous": False
        }

    def mock_chat_completion(messages, **kwargs):
        assert len(messages) >= 3
        return {"content": "Linear regression is an example of supervised learning."}

    monkeypatch.setattr(SemanticPlanner, "extract_semantic_intent", mock_extract)
    monkeypatch.setattr(ollama_service, "chat_completion", mock_chat_completion)

    history = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a field of AI that enables systems to learn from data."}
    ]
    query = "Can you give me an example of supervised learning?"
    res = QueryOrchestrator.process_query(
        query_raw=query,
        df=None,
        conversation_history=history
    )
    assert res.status == "SUCCEEDED"
    assert res.result.get("type") == "text"
    assert len(res.result.get("response", "")) > 0
