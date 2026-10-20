"""
DataPilot OS — Result Verification Registry Tests (Phase 8).
Validates that ResultVerificationRegistry rigorously catches status failures,
row count mismatches, oracle discrepancies, and mathematical invariant violations.
"""

import numpy as np
import pandas as pd
import pytest

from backend.ds_engine.result_verification_registry import (
    ResultVerificationRegistry,
    VerificationReport,
)
from backend.ds_engine.intent_schema import (
    AnalysisIntentV2,
    CorrelationSpec,
    ExecutionManifest,
    TaskFamily,
    VisualizationSpec,
)
from backend.ds_engine.typed_operators import OperatorOutput, CorrelationOperator


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    return pd.DataFrame({
        "num_a": np.random.normal(10, 2, 50),
        "num_b": np.random.normal(20, 5, 50),
        "group": np.random.choice(["X", "Y"], 50),
    })


def test_verification_passes_on_valid_correlation(sample_dataset):
    op = CorrelationOperator()
    intent = AnalysisIntentV2(
        task_family=TaskFamily.CORRELATION_ANALYSIS,
        subjects=["num_a", "num_b"],
        correlation_spec=CorrelationSpec(
            analysis_mode="matrix",
            relationship_form="linear",
            correlation_methods=["pearson"],
            chart_type="none",
            smoother="none",
            expected_direction="unspecified",
        ),
    )
    output = op.execute(sample_dataset, intent)
    report = ResultVerificationRegistry.verify_operator_result(intent, output, sample_dataset)

    assert report.is_valid
    assert len(report.discrepancies) == 0
    assert "correlation_matrix_bounds_symmetry_diagonal" in report.invariants_verified


def test_verification_fails_when_status_is_failed():
    intent = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["num_a"],
        grouping=["group"],
    )
    failed_output = OperatorOutput(
        status="FAILED",
        error_code="COLUMN_NOT_FOUND",
        error_message="Column not found",
        manifest=ExecutionManifest(
            operator="group_aggregation",
            columns_used=["num_a"],
            statistics_computed=[],
            chart_type="none",
            row_count_in=50,
            row_count_out=0,
        ),
    )
    report = ResultVerificationRegistry.verify_operator_result(intent, failed_output)

    assert not report.is_valid
    assert any("failed with code: COLUMN_NOT_FOUND" in d for d in report.discrepancies)


def test_verification_catches_row_count_mismatch(sample_dataset):
    intent = AnalysisIntentV2(
        task_family=TaskFamily.GROUP_AGGREGATION,
        subjects=["num_a"],
        grouping=["group"],
    )
    # Manifest claims 100 rows in, but sample_dataset only has 50 rows
    tampered_output = OperatorOutput(
        status="SUCCEEDED",
        dataframe=pd.DataFrame({"group": ["X", "Y"], "num_a_mean": [10.0, 11.0]}),
        manifest=ExecutionManifest(
            operator="group_aggregation",
            columns_used=["num_a", "group"],
            statistics_computed=["mean"],
            chart_type="none",
            row_count_in=100,  # Mismatch!
            row_count_out=2,
        ),
    )
    report = ResultVerificationRegistry.verify_operator_result(intent, tampered_output, sample_dataset)

    assert not report.is_valid
    assert any("ROW_COUNT_MISMATCH" in d for d in report.discrepancies)


def test_verification_catches_tampered_correlation_values(sample_dataset):
    intent = AnalysisIntentV2(
        task_family=TaskFamily.CORRELATION_ANALYSIS,
        subjects=["num_a", "num_b"],
        correlation_spec=CorrelationSpec(
            analysis_mode="matrix",
            relationship_form="linear",
            correlation_methods=["pearson"],
            chart_type="none",
            smoother="none",
            expected_direction="unspecified",
        ),
    )
    # Fabricate a correlation output that contradicts the raw data
    tampered_df = pd.DataFrame({
        "feature": ["num_a", "num_b"],
        "num_a": [1.0, 0.99],  # Real correlation is near 0
        "num_b": [0.99, 1.0],
    })
    tampered_output = OperatorOutput(
        status="SUCCEEDED",
        dataframe=tampered_df,
        manifest=ExecutionManifest(
            operator="correlation_analysis",
            columns_used=["num_a", "num_b"],
            statistics_computed=["pearson_correlation"],
            chart_type="none",
            row_count_in=len(sample_dataset),
            row_count_out=len(tampered_df),
        ),
    )
    report = ResultVerificationRegistry.verify_operator_result(intent, tampered_output, sample_dataset)

    assert not report.is_valid
    assert any("ORACLE_DISCREPANCY" in d for d in report.discrepancies)
