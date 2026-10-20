"""Regression coverage for typed supervised-ML role and feature preservation."""

import pandas as pd

from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily, VisualizationSpec
from backend.ds_engine.typed_operators import SupervisedMLOperator


def test_classifier_encodes_selected_categorical_features_and_returns_confusion_matrix():
    rows = []
    for index in range(60):
        rows.append({
            "study_mode": "online" if index % 2 else "campus",
            "hours": float(index % 10 + 1),
            "grade_band": "A" if index % 3 == 0 else ("B" if index % 3 == 1 else "C"),
        })
    df = pd.DataFrame(rows)
    intent = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        subjects=["grade_band", "study_mode", "hours"],
        target_column="grade_band",
        visualization=VisualizationSpec(type="confusion_matrix"),
        arguments={
            "task_type": "classification",
            "evaluation_artifact": "confusion_matrix",
            "prediction_context": "prospective_prediction",
            "feature_columns": ["study_mode", "hours"],
        },
    )

    output = SupervisedMLOperator().execute(df, intent)

    assert output.status == "SUCCEEDED"
    assert output.manifest.chart_type == "confusion_matrix"
    assert {"grade_band", "study_mode", "hours"}.issubset(output.manifest.columns_used)
    assert "confusion_matrix" in output.manifest.statistics_computed
    assert output.plot_image_base64


def test_supervised_ml_never_substitutes_a_predictor_for_the_explicit_target():
    df = pd.DataFrame({"target": ["yes", "no"] * 4, "predictor": list(range(8))})
    intent = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        subjects=["predictor"],
        target_column="missing_target",
        arguments={"task_type": "classification", "evaluation_artifact": "confusion_matrix", "prediction_context": "prospective_prediction"},
    )

    output = SupervisedMLOperator().execute(df, intent)

    assert output.status == "FAILED"
    assert output.error_code == "TARGET_COLUMN_REQUIRED"
