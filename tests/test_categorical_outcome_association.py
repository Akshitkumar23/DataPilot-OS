"""Regression coverage for generic retrospective numeric-versus-outcome analysis."""

import json

import numpy as np
import pandas as pd

from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily
from backend.ds_engine.typed_operators import CategoricalOutcomeAssociationOperator
from backend.intelligence.ollama_service import OllamaResponse, ollama_service
from modules.capability_registry import capability_registry
from modules.query_orchestrator import QueryOrchestrator


def test_toolspec_and_operator_are_dataset_agnostic():
    args = capability_registry.validate_tool_call(
        "categorical_outcome_association",
        {"numeric_predictor": "cycle_time", "outcome_column": "resolution_status"},
    )
    assert args.analysis_context == "retrospective_association"
    assert args.chart_type == "box"

    df = pd.DataFrame({
        "cycle_time": [3.2, 4.1, 3.7, 8.4, 9.0, 7.8],
        "resolution_status": ["resolved", "resolved", "resolved", "escalated", "escalated", "escalated"],
    })
    output = CategoricalOutcomeAssociationOperator().execute(
        df,
        AnalysisIntentV2(
            task_family=TaskFamily.CATEGORICAL_OUTCOME_ASSOCIATION,
            arguments=args.model_dump(mode="json"),
        ),
    )
    assert output.status == "SUCCEEDED"
    assert output.manifest.columns_used == ["cycle_time", "resolution_status"]
    assert output.manifest.chart_type == "box"
    assert "prospective targeting" in output.written_finding


def test_end_to_end_planned_association_never_trains_a_model(monkeypatch):
    df = pd.DataFrame({
        "numeric_measure": np.linspace(1.0, 12.0, 24),
        "binary_outcome": ["no"] * 12 + ["yes"] * 12,
    })
    payload = {
        "tool": "categorical_outcome_association",
        "arguments": {
            "numeric_predictor": "numeric_measure",
            "outcome_column": "binary_outcome",
            "chart_type": "none",
            "analysis_context": "retrospective_association",
        },
    }
    monkeypatch.setattr(
        ollama_service,
        "structured_request",
        lambda _request: OllamaResponse(success=True, content=json.dumps(payload), parsed_json=payload),
    )
    monkeypatch.setattr(ollama_service, "verify_ready", lambda: type("Ready", (), {"ready": True})())
    monkeypatch.setattr(ollama_service, "embed_texts", lambda *_args, **_kwargs: None)

    result = QueryOrchestrator.process_query("Compare numeric_measure across historical binary_outcome groups.", df=df)
    assert result.status == "SUCCEEDED", result.error
    assert result.result["manifest"]["operator"] == "categorical_outcome_association"
    assert result.result["manifest"]["chart_type"] == "none"
    assert "RandomForest" not in result.result["code"]
