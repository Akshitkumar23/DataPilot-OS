"""Deterministic safety regressions, not training examples or live-model evals."""
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException

from backend.api import chat
from backend.ds_engine.intent_schema import AnalysisIntentV2, ExecutionManifest, TaskFamily, VisualizationSpec
from backend.ds_engine.result_validator import SemanticResultValidator
from backend.ds_engine.typed_operators import OperatorOutput
from modules.models.execution_result import CanonicalQueryResponse
from modules.query_orchestrator import QueryOrchestrator


@pytest.mark.parametrize("mode", ["missing", "exception", "unavailable"])
def test_explicit_dataset_never_falls_back(monkeypatch, mode):
    def load(_):
        if mode == "exception":
            raise OSError("unreadable")
        return None
    def forbidden(*args, **kwargs):
        pytest.fail("Another dataset or the planner must not be used")
    registry = SimpleNamespace(get_dataframe=load, get_active=forbidden)
    monkeypatch.setattr(chat, "dataset_registry", None if mode == "unavailable" else registry)
    monkeypatch.setattr(QueryOrchestrator, "process_query", forbidden)
    with pytest.raises(HTTPException) as error:
        chat.chat_copilot(chat.ChatRequest(query="mean value", dataset_id="requested"))
    assert error.value.status_code in (404, 503)
    assert error.value.detail["phase"] == "PLAN_VALIDATION"


def test_conflicting_context_rejected(monkeypatch):
    with pytest.raises(HTTPException) as error:
        chat.chat_copilot(chat.ChatRequest(query="mean value", dataset_id="one", context={"dataset_id": "two"}))
    assert error.value.detail["error"]["code"] == "DATASET_CONTEXT_MISMATCH"


def test_request_frame_and_persistence_bound_to_same_dataset(monkeypatch):
    frame = pd.DataFrame({"value": [1, 3]})
    registry = SimpleNamespace(get_active=lambda: SimpleNamespace(dataset_id="active"),
                               get_dataframe=lambda dataset_id: frame)
    monkeypatch.setattr(chat, "dataset_registry", registry)
    saved = []
    monkeypatch.setattr(chat, "_persist_workflow", lambda req, *args: saved.append(req.dataset_id))
    def execute(**kwargs):
        assert kwargs["context"]["dataset_id"] == "active"
        kwargs["df"].iloc[0, 0] = 99
        return CanonicalQueryResponse(status="SUCCEEDED")
    monkeypatch.setattr(QueryOrchestrator, "process_query", execute)
    assert chat.chat_copilot(chat.ChatRequest(query="mean value")).status == "SUCCEEDED"
    assert saved == ["active"]
    assert frame.iloc[0, 0] == 1


def test_planner_failure_does_not_invent_missing_user_arguments(monkeypatch):
    monkeypatch.setattr("modules.query_orchestrator.ollama_service.verify_ready", lambda: SimpleNamespace(ready=True))
    monkeypatch.setattr("modules.query_orchestrator.SemanticPlanner.extract_semantic_intent", lambda **kwargs: None)
    response = QueryOrchestrator.process_query("Mean value by category as a bar chart", pd.DataFrame({"value": [1, 2], "category": ["a", "b"]}))
    assert response.status == "FAILED"
    assert response.error.code == "INTENT_PLAN_INVALID"
    assert response.clarification is None


def make_output():
    intent = AnalysisIntentV2(task_family=TaskFamily.GROUP_AGGREGATION, subjects=["value"],
                              visualization=VisualizationSpec(type="bar"))
    output = OperatorOutput(manifest=ExecutionManifest(operator="group_aggregation", columns_used=["value"],
                            chart_type="bar", row_count_in=2, row_count_out=1),
                            dataframe=pd.DataFrame({"value": [2]}))
    return intent, output


def test_manifest_claim_cannot_replace_actual_chart():
    intent, output = make_output()
    result = SemanticResultValidator.validate_output(intent, output, input_row_count=2)
    assert not result.is_valid
    assert any("materialized artifact" in issue for issue in result.issues)


def test_empty_column_manifest_cannot_bypass_validation():
    intent, output = make_output()
    output.manifest.columns_used = []
    assert not SemanticResultValidator.validate_manifest(intent, output.manifest).is_valid


def test_std_does_not_satisfy_variance_requirement():
    intent, output = make_output()
    intent.statistical_requirements = ["variance"]
    output.manifest.statistics_computed = ["std"]
    result = SemanticResultValidator.validate_manifest(intent, output.manifest)
    assert any("variance" in issue for issue in result.issues)


def test_row_count_checks_full_frame_not_capped_preview():
    intent, output = make_output()
    intent.visualization = VisualizationSpec(type="none")
    output.manifest.chart_type = "none"
    output.data = {"records": []}
    assert SemanticResultValidator.validate_output(intent, output, input_row_count=2).is_valid
    output.manifest.row_count_out = 10
    assert not SemanticResultValidator.validate_output(intent, output, input_row_count=2).is_valid


def test_unicode_column_aliases_preserve_hindi_vowels():
    from backend.ds_engine.semantic_feasibility import _normalise_column_name, _resolve_explicit_column
    profile = SimpleNamespace(columns={"मूल्य": None, "मिल": None})
    assert _normalise_column_name("मूल्य") != _normalise_column_name("मिल")
    assert _resolve_explicit_column(profile, " मूल्य ") == "मूल्य"
    assert _resolve_explicit_column(profile, "---") is None
