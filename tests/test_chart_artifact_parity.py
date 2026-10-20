"""
DataPilot OS — Chart Artifact Parity Tests (Phase 8).
Enforces Non-Negotiable Rule 4:
"chart requested ho aur generated na ho to success nahi return karna."
If a chart is requested, an image or plotly artifact must be materialized;
otherwise, result verification strictly fails.
"""

import pandas as pd
import pytest

from backend.ds_engine.result_verification_registry import ResultVerificationRegistry
from backend.ds_engine.result_validator import SemanticResultValidator
from backend.ds_engine.intent_schema import (
    AnalysisIntentV2,
    ExecutionManifest,
    TaskFamily,
    VisualizationSpec,
)
from backend.ds_engine.typed_operators import OperatorOutput


@pytest.fixture
def mock_output():
    return OperatorOutput(
        status="SUCCEEDED",
        dataframe=pd.DataFrame({"cat": ["A", "B"], "val": [10, 20]}),
        manifest=ExecutionManifest(
            operator="group_aggregation",
            columns_used=["cat", "val"],
            statistics_computed=["mean"],
            chart_type="bar",
            row_count_in=100,
            row_count_out=2,
        ),
        plot_image_base64=None,
        plotly_spec=None,
    )


def test_missing_chart_artifact_fails_verification_registry(mock_output):
    """Chart requested in intent with chart_type='bar' but output has neither plot_image nor plotly_spec."""
    intent = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["val"],
        grouping=["cat"],
        visualization=VisualizationSpec(type="bar"),
    )
    report = ResultVerificationRegistry.verify_operator_result(intent, mock_output)

    assert not report.is_valid
    assert any("CHART_ARTIFACT_MISSING" in d for d in report.discrepancies)


def test_missing_chart_artifact_fails_semantic_result_validator(mock_output):
    """SemanticResultValidator also detects missing materialized chart artifact."""
    intent = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["val"],
        grouping=["cat"],
        visualization=VisualizationSpec(type="bar"),
    )
    res = SemanticResultValidator.validate_output(intent, mock_output)

    assert not res.is_valid
    assert any("Declared chart has no materialized artifact" in issue for issue in res.issues)


def test_present_chart_artifact_passes_verification(mock_output):
    """Providing valid plot_image_base64 or plotly_spec satisfies the chart parity contract."""
    mock_output.plot_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    intent = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["val"],
        grouping=["cat"],
        visualization=VisualizationSpec(type="bar"),
    )
    report = ResultVerificationRegistry.verify_operator_result(intent, mock_output)

    assert report.is_valid
    assert len(report.discrepancies) == 0


def test_table_only_operation_requires_no_chart():
    """When chart is 'none', missing plot image is expected and valid."""
    output = OperatorOutput(
        status="SUCCEEDED",
        dataframe=pd.DataFrame({"cat": ["A", "B"], "val": [10, 20]}),
        manifest=ExecutionManifest(
            operator="group_aggregation",
            columns_used=["cat", "val"],
            statistics_computed=["mean"],
            chart_type="none",
            row_count_in=100,
            row_count_out=2,
        ),
        plot_image_base64=None,
        plotly_spec=None,
    )
    intent = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["val"],
        grouping=["cat"],
        visualization=VisualizationSpec(type="none"),
    )
    report = ResultVerificationRegistry.verify_operator_result(intent, output)

    assert report.is_valid
    assert len(report.discrepancies) == 0
