"""Regression tests for generic, evidence-based target leakage protection."""

import pandas as pd

from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily, VisualizationSpec
from backend.ds_engine.target_leakage import TargetLeakageDetector
from backend.ds_engine.typed_operators import SupervisedMLOperator
from backend.intelligence.ollama_service import OllamaReadinessStatus, OllamaResponse, ollama_service
from modules.query_orchestrator import QueryOrchestrator


def _grade_data():
    rows = []
    for grade, score in (("F", 5), ("D", 10), ("C", 12), ("B", 14), ("A", 17)):
        for offset in range(40):
            rows.append({
                "grade_band": grade,
                "final_score": score,
                "study_hours": float((offset * 7) % 11 + 1),
            })
    return pd.DataFrame(rows)


def test_detector_flags_only_a_held_out_deterministic_target_encoding():
    assessment = TargetLeakageDetector.assess_classification(
        _grade_data(),
        target_column="grade_band",
        feature_columns=["final_score", "study_hours"],
    )

    assert assessment.has_deterministic_signal
    assert [signal.feature for signal in assessment.signals] == ["final_score"]
    assert assessment.signals[0].held_out_balanced_accuracy == 1.0


def test_retrospective_execution_marks_leaky_score_and_generated_code_is_runnable():
    df = _grade_data()
    assessment = TargetLeakageDetector.assess_classification(
        df, target_column="grade_band", feature_columns=["final_score", "study_hours"]
    )
    intent = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        subjects=["grade_band", "final_score", "study_hours"],
        target_column="grade_band",
        visualization=VisualizationSpec(type="confusion_matrix"),
        arguments={
            "task_type": "classification",
            "evaluation_artifact": "confusion_matrix",
            "prediction_context": "retrospective_explanation",
            "target_leakage_assessment": assessment.to_dict(),
        },
    )

    output = SupervisedMLOperator().execute(df, intent)

    assert output.status == "SUCCEEDED"
    assert output.manifest.warnings
    assert "final_score" in output.manifest.warnings[0]
    assert "model_evaluation" not in output.reproducible_code
    namespace = {"df": df.copy()}
    exec(output.reproducible_code, namespace)
    assert "result_df" in namespace


def test_pipeline_excludes_leaky_feature_and_returns_safe_prospective_baseline(monkeypatch):
    payload = {
        "tool": "train_model",
        "arguments": {
            "target_column": "grade_band",
            "feature_columns": ["final_score", "study_hours"],
            "task_type": "classification",
            "evaluation_artifact": "confusion_matrix",
            "prediction_context": "prospective_prediction",
        },
    }
    monkeypatch.setattr(
        ollama_service,
        "verify_ready",
        lambda: OllamaReadinessStatus(ready=True, model="test", model_available=True),
    )
    monkeypatch.setattr(
        ollama_service,
        "structured_request",
        lambda _request: OllamaResponse(success=True, parsed_json=payload),
    )

    response = QueryOrchestrator.process_query(
        "Predict grade_band using final_score and study_hours and plot a confusion matrix.", df=_grade_data()
    )

    assert response.status == "SUCCEEDED"
    assert response.result["manifest"]["columns_used"] == ["grade_band", "study_hours"]
    assert response.plan["safety_audit"]["excluded_features"] == ["final_score"]
    assert "Prospective safe baseline" in response.result["manifest"]["warnings"][0]
