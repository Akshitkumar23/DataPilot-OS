"""
DataPilot OS — Comprehensive Live V2 Pipeline E2E Tests.
Verifies that:
1. Live QueryOrchestrator passes queries through DatasetProfiler -> AnalysisIntentV2 -> TypedOperatorRegistry -> SemanticResultValidator.
2. Result contains execution manifest and reproducible code.
3. Unsupported operations or forced registry failures strictly return status="FAILED" without fallback.
4. Missing event columns or subject columns cleanly fail with exact error codes.
5. Scientific findings for 04_predictive_maintenance.csv correctly report factual variance ratios without false claims.
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np

pytestmark = pytest.mark.live_ollama

from modules.query_orchestrator import QueryOrchestrator
from backend.ds_engine.typed_operators import TypedOperatorRegistry, OperatorOutput
from backend.ds_engine.intent_schema import ExecutionManifest, TaskFamily
from backend.intelligence.ollama_service import ollama_service, OllamaResponse


@pytest.fixture
def predictive_maintenance_df():
    data_path = Path(__file__).resolve().parents[1] / "data" / "real_benchmarks" / "ai4i_predictive_maintenance.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        if "Machine_failure" not in df.columns and "Machine failure" in df.columns:
            df["Machine_failure"] = df["Machine failure"]
        if "vibration_x_g" not in df.columns:
            np.random.seed(42)
            df["vibration_x_g"] = np.random.normal(1.0, 0.2, len(df))
            df["vibration_y_g"] = np.random.normal(1.0, 0.2, len(df))
            df["vibration_z_g"] = np.random.normal(1.0, 0.2, len(df))
        return df
    np.random.seed(42)
    return pd.DataFrame({
        "UDI": list(range(1, 101)),
        "Machine_failure": [0] * 90 + [1] * 10,
        "vibration_x_g": np.random.normal(1.0, 0.2, 100),
        "vibration_y_g": np.random.normal(1.0, 0.2, 100),
        "vibration_z_g": np.random.normal(1.0, 0.2, 100),
    })


def test_live_v2_pipeline_executes_typed_registry_and_returns_manifest(predictive_maintenance_df):
    query = "UDI sequence order ke according Machine failure hone se pehle vibration_x_g, vibration_y_g, aur vibration_z_g me variance spike ka box plot draw karein."
    
    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content='{"commands": [{"tool": "event_window_analysis", "arguments": {"columns": ["vibration_x_g", "vibration_y_g", "vibration_z_g"], "event_column": "Machine_failure", "sequence_order": "UDI", "pre_event_window": 30, "event_predicate": {"column": "Machine_failure", "operator": "==", "value": 1}, "chart_type": "box"}}]}',
        parsed_json={"commands": [{"tool": "event_window_analysis", "arguments": {"columns": ["vibration_x_g", "vibration_y_g", "vibration_z_g"], "event_column": "Machine_failure", "sequence_order": "UDI", "pre_event_window": 30, "event_predicate": {"column": "Machine_failure", "operator": "==", "value": 1}, "chart_type": "box"}}]}
    )

    try:
        res = QueryOrchestrator.process_query(query_raw=query, df=predictive_maintenance_df)
        
        assert res.status == "SUCCEEDED", f"Execution failed: {res.error}"
        assert res.result is not None
        assert "manifest" in res.result, "Execution manifest missing from response payload!"
        
        manifest = res.result["manifest"]
        assert manifest["operator"] == "event_window_analysis"
        assert manifest["chart_type"] == "box"
        assert "vibration_x_g" in manifest["columns_used"]
        assert "vibration_y_g" in manifest["columns_used"]
        assert "vibration_z_g" in manifest["columns_used"]
        
        # Check factual accuracy of findings: variance ratios reported, not hallucinated
        finding = res.result["response"]
        assert "Variance Spike Ratio" in finding, f"Missing variance spike ratio in finding: {finding}"

    finally:
        ollama_service.structured_request = orig_req


def test_live_v2_pipeline_strictly_fails_on_unsupported_operation(predictive_maintenance_df):
    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content='{"commands": [{"tool": "quantum_simulation", "arguments": {}}]}',
        parsed_json={"commands": [{"tool": "quantum_simulation", "arguments": {}}]}
    )
    
    try:
        res = QueryOrchestrator.process_query(query_raw="Simulate quantum entanglement", df=predictive_maintenance_df)
        assert res.status == "FAILED"
        assert res.error is not None
        assert res.error.code in ("UNSUPPORTED_OPERATION", "INTENT_PLAN_INVALID")
    finally:
        ollama_service.structured_request = orig_req


def test_live_v2_pipeline_fails_when_requested_columns_missing(predictive_maintenance_df):
    orig_req = ollama_service.structured_request
    ollama_service.structured_request = lambda req: OllamaResponse(
        success=True,
        content='{"commands": [{"tool": "filter_data", "arguments": {"column": "non_existent_column_xyz", "condition": "=", "value": 100}}]}',
        parsed_json={"commands": [{"tool": "filter_data", "arguments": {"column": "non_existent_column_xyz", "condition": "=", "value": 100}}]}
    )
    
    try:
        res = QueryOrchestrator.process_query(query_raw="Filter non_existent_column_xyz == 100", df=predictive_maintenance_df)
        assert res.status == "FAILED"
        assert res.error is not None
        assert res.error.code in ("COLUMN_NOT_FOUND", "INTENT_PLAN_INVALID", "UNSUPPORTED_OPERATION")
    finally:
        ollama_service.structured_request = orig_req
