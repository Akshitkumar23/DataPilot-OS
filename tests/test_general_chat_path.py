"""
Tests for DataPilot OS General Chat, Conversational, and Educational Query Path.
Verifies that general conceptual questions (e.g. recursion, machine learning concepts)
are answered directly via Ollama without requiring an active dataset or executing sandbox code.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from modules.query_orchestrator import QueryOrchestrator
from backend.ds_engine.command_resolver import command_resolver
from backend.ds_engine.intent_schema import IntentType


def test_command_resolver_general_chat():
    """Verify CommandResolver parses general_chat as non-ambiguous GENERAL_CHAT intent."""
    raw_cmd = {
        "tool": "general_chat",
        "arguments": {
            "response": "Recursion is a method where the solution to a problem depends on solutions to smaller instances of the same problem."
        }
    }
    resolution = command_resolver.resolve_command(raw_cmd, df=None, raw_query="Explain recursion")
    assert resolution.is_valid is True
    assert resolution.is_ambiguous is False
    assert resolution.semantic_intent is not None
    assert resolution.semantic_intent.intent_type == IntentType.GENERAL_CHAT
    assert resolution.semantic_intent.primary_intent == IntentType.GENERAL_CHAT
    assert "Recursion" in resolution.error_message


def test_general_chat_orchestrator_without_dataset():
    """Verify QueryOrchestrator answers general questions when df=None without failing."""
    query = "Explain what is recursion in programming"
    res = QueryOrchestrator.process_query(query_raw=query, df=None)
    
    assert res.status == "SUCCEEDED"
    assert res.phase == "SUCCEEDED"
    assert res.result is not None
    assert res.result.get("type") == "text"
    assert len(res.result.get("response", "")) > 0
    assert res.error is None


def test_conversational_greeting_without_dataset():
    """Verify conversational queries like 'Hello' succeed cleanly."""
    query = "Hello DataPilot, what can you do?"
    res = QueryOrchestrator.process_query(query_raw=query, df=None)
    
    assert res.status == "SUCCEEDED"
    assert res.result is not None
    assert len(res.result.get("response", "")) > 0
