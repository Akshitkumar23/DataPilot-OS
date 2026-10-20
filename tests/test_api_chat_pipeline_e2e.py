import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import app
from modules.semantic_planner import SemanticPlanner
from backend.intelligence.ollama_service import ollama_service, OllamaResponse
from backend.core.dataset_registry import dataset_registry

def test_api_chat_full_pipeline():
    """Verify POST /api/chat payload processing with conversation_history and context."""
    client = TestClient(app)
    
    # Register test dataframe into registry
    df = pd.DataFrame({
        'sales': [100, 200, 600, 800, 1000],
        'region': ['North', 'North', 'South', 'South', 'South']
    })
    dataset_registry.register("test_ds_api", df)
    
    # Mock Ollama plan response conforming to CapabilityRegistry ToolSpec
    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content='{"tool": "filter_data", "arguments": {"column": "sales", "condition": ">", "value": 500}}',
        parsed_json={"tool": "filter_data", "arguments": {"column": "sales", "condition": ">", "value": 500}}
    )

    token = os.getenv('DATAPILOT_API_TOKEN', '')
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        payload = {
            "query": "Filter sales > 500",
            "dataset_id": "test_ds_api",
            "conversation_history": [{"role": "user", "content": "Hello"}],
            "context": {"active_module": "data_grid", "selected_columns": ["sales"]}
        }
        
        response = client.post("/api/chat", json=payload, headers=headers)
    finally:
        ollama_service.structured_request = orig_req

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    assert data["status"] == "SUCCEEDED", f"Status not SUCCEEDED: {data}"
    assert data["result"] is not None, "Result is None"
    assert "request_id" in data, "Missing request_id in CanonicalQueryResponse"
    print("\n[PASS] test_api_chat_full_pipeline: POST /api/chat pipeline executed successfully with 200 OK")


def test_api_returns_clarification_contract_http_422_for_missing_target():
    """A non-existent requested target must return HTTP 422 with clarification contract."""
    client = TestClient(app)
    dataset_registry.register("test_ds_missing_target", pd.DataFrame({
        "exam_score": [41, 63, 89, 72],
        "study_hours": [2.0, 4.5, 8.0, 6.0],
    }))

    original = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **_kwargs: {
        "intent_type": "EDA",
        "primary_intent": "EDA",
        "raw_query": "Train a classifier for grade_band",
        "is_ambiguous": True,
        "ambiguity_reason": "The dataset does not contain the requested target column 'grade_band'.",
    })
    try:
        response = client.post("/api/chat", json={
            "query": "Train a classifier for grade_band",
            "dataset_id": "test_ds_missing_target",
        })
    finally:
        SemanticPlanner.extract_semantic_intent = original

    assert response.status_code == 422, response.text
    body = response.json()
    payload = body.get("detail", body)
    assert payload["status"] == "NEEDS_CLARIFICATION"
    assert payload["error"]["code"] == "CLARIFICATION_REQUIRED"
    assert "grade_band" in payload["clarification"]

if __name__ == "__main__":
    test_api_chat_full_pipeline()
    print("\n" + "=" * 80)
    print("API CHAT E2E PIPELINE TEST PASSED CLEANLY [OK]")
    print("=" * 80)
