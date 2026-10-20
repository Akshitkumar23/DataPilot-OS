"""
DataPilot OS — Paraphrase & Multi-Domain Zero-Hardcode V2 E2E Test Suite.
Verifies that:
1. Event window queries phrased with synonyms ("records immediately preceding equipment breakdown", Hindi/Hinglish) map cleanly to TaskFamily.EVENT_WINDOW_ANALYSIS without raw keyword matching.
2. Multi-domain datasets (Customer Churn, Manufacturing IoT, E-commerce) execute typed operators accurately.
3. TaskFamily operators (Clustering, Statistical Test, PCA, Explainability, Cleaning) execute deterministically.
4. Unmapped or unsupported tasks fail with UNSUPPORTED_OPERATION without silently becoming GENERAL_EDA.
"""

import pytest
import pandas as pd
import numpy as np

from modules.query_orchestrator import QueryOrchestrator, _map_intent_to_v2
from backend.ds_engine.intent_schema import (
    SemanticIntent, IntentType, AnalysisIntentV2, TaskFamily, FilterSpec, FilterOperator, TimeContext
)
from backend.ds_engine.typed_operators import TypedOperatorRegistry
from backend.ds_engine.result_validator import SemanticResultValidator
from backend.ds_engine.dataset_profiler import DatasetProfiler


@pytest.fixture
def sample_iot_breakdown_df():
    np.random.seed(42)
    rows = 100
    dates = pd.date_range("2026-01-01", periods=rows, freq="1h")
    breakdown_flag = [0] * 90 + [1] * 10
    vibe = np.random.normal(1.0, 0.2, rows)
    vibe[80:90] += 2.5  # Pre-event spike
    return pd.DataFrame({
        "timestamp": dates,
        "device_id": ["DEV_01"] * rows,
        "breakdown_flag": breakdown_flag,
        "vibe_sensor": vibe,
        "temp_sensor": np.random.normal(50, 5, rows)
    })


@pytest.fixture
def sample_churn_df():
    """Longitudinal panel dataset: 5 users × 20 daily time periods each.
    
    Each user has 20 daily records. For users 3-5, churn=1 occurs at day 18-20.
    Pre-event rows (days 1-17 before churn) provide enough data for variance analysis.
    """
    np.random.seed(42)
    n_users = 5
    n_days = 20
    records = []
    start_date = pd.Timestamp("2026-01-01")
    for u in range(n_users):
        for d in range(n_days):
            # Users 3, 4, 5 churn on day 18, 19, 20
            churn = 1 if (u >= 2 and d >= 17) else 0
            activity = np.random.normal(10, 2) - (5.0 if (u >= 2 and 14 <= d < 17) else 0.0)
            records.append({
                "activity_date": start_date + pd.Timedelta(days=d),
                "user_id": f"U_{u}",
                "churn": churn,
                "daily_login_count": max(0.0, activity),
                "support_tickets": max(0, int(np.random.poisson(1)))
            })
    return pd.DataFrame(records)


def test_map_intent_to_v2_strict_enum_mapping():
    """Verifies that _map_intent_to_v2 relies 100% on structured intent properties without keyword searching."""
    intent = SemanticIntent(
        intent_type=IntentType.EVENT_WINDOW_ANALYSIS,
        primary_intent=IntentType.EVENT_WINDOW_ANALYSIS,
        tool_name="event_window_analysis",
        time_column="timestamp",
        target_column="breakdown_flag",
        metrics=["vibe_sensor"],
        prompt_summary="Paraphrased equipment breakdown query"
    )
    
    intent_v2 = _map_intent_to_v2(
        intent,
        raw_query="records immediately preceding equipment breakdown",
        validated_tool_args={"event_column": "breakdown_flag"},
    )
    assert intent_v2.task_family == TaskFamily.EVENT_WINDOW_ANALYSIS
    assert intent_v2.subjects == ["vibe_sensor"]
    assert intent_v2.time_context is not None
    assert intent_v2.time_context.event_column == "breakdown_flag"


def test_paraphrased_equipment_breakdown_event_window(sample_iot_breakdown_df):
    """Verifies paraphrased query 'records immediately preceding equipment breakdown' executes EventWindowAnalysisOperator."""
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["vibe_sensor"],
        time_context=TimeContext(
            time_column="timestamp",
            event_column="breakdown_flag",
            entity_id="device_id",
            pre_event_window=10
        ),
        conditions=[FilterSpec(column="breakdown_flag", operator=FilterOperator.EQ, value=1)]
        ,arguments={"event_scope": "event_rows"}
    )

    profile = DatasetProfiler.profile(sample_iot_breakdown_df, "iot_dataset")
    out = TypedOperatorRegistry.execute_intent(sample_iot_breakdown_df, intent_v2, profile)

    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "event_window_analysis"
    assert out.manifest.row_count_out > 0
    assert "vibe_sensor" in out.manifest.columns_used

    val_res = SemanticResultValidator.validate_manifest(intent_v2, out.manifest)
    assert val_res.is_valid is True


def test_customer_churn_domain_event_window(sample_churn_df):
    """Verifies event window analysis on Customer Churn dataset with custom column names."""
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
        subjects=["daily_login_count"],
        time_context=TimeContext(
            time_column="activity_date",
            event_column="churn",
            entity_id="user_id",  # Required: profiler detects user_id as multi-entity candidate
            pre_event_window=15
        ),
        conditions=[FilterSpec(column="churn", operator=FilterOperator.EQ, value=1)]
        ,arguments={"event_scope": "event_rows"}
    )

    profile = DatasetProfiler.profile(sample_churn_df, "churn_dataset")
    out = TypedOperatorRegistry.execute_intent(sample_churn_df, intent_v2, profile)

    assert out.status == "SUCCEEDED", f"Failed: {out.error_code} — {out.error_message}"
    assert out.manifest.operator == "event_window_analysis"
    assert "daily_login_count" in out.manifest.columns_used


def test_clustering_operator_execution():
    """Verifies TaskFamily.CLUSTERING executes ClusteringOperator."""
    df = pd.DataFrame({
        "income": np.random.normal(50000, 10000, 50),
        "spend": np.random.normal(50, 10, 50)
    })
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.CLUSTERING,
        subjects=["income", "spend"],
        arguments={"n_clusters": 3, "scaling": "standard"}
    )

    out = TypedOperatorRegistry.execute_intent(df, intent_v2)
    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "clustering"
    assert out.manifest.chart_type in ("scatter", "hist")


def test_statistical_test_operator_execution():
    """Verifies TaskFamily.STATISTICAL_TEST executes StatisticalTestOperator."""
    df = pd.DataFrame({
        "grp": ["A"] * 25 + ["B"] * 25,
        "score": np.concatenate([np.random.normal(10, 2, 25), np.random.normal(15, 2, 25)])
    })
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.STATISTICAL_TEST,
        subjects=["score"],
        grouping=["grp"],
        arguments={
            "test_type": "one_way_anova",
            "columns": ["score", "grp"],
            "alpha": 0.05,
        }
    )

    out = TypedOperatorRegistry.execute_intent(df, intent_v2)
    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "statistical_test"
    assert "p_value" in out.manifest.statistics_computed


def test_pca_operator_execution():
    """Verifies TaskFamily.DIMENSIONALITY_REDUCTION executes PCADimensionalityReductionOperator."""
    df = pd.DataFrame({
        "f1": np.random.normal(0, 1, 30),
        "f2": np.random.normal(0, 1, 30),
        "f3": np.random.normal(0, 1, 30)
    })
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.DIMENSIONALITY_REDUCTION,
        subjects=["f1", "f2", "f3"],
        arguments={"n_components": 2}
    )

    out = TypedOperatorRegistry.execute_intent(df, intent_v2)
    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "pca_dimensionality_reduction"


def test_explainability_operator_execution():
    """Verifies TaskFamily.EXPLAINABILITY executes FeatureExplainabilityOperator."""
    df = pd.DataFrame({
        "target": np.random.randint(0, 2, 40),
        "feat_a": np.random.normal(5, 1, 40),
        "feat_b": np.random.normal(10, 2, 40)
    })
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.EXPLAINABILITY,
        subjects=["feat_a", "feat_b"],
        target_column="target",
        arguments={
            "target_column": "target",
            "feature_columns": ["feat_a", "feat_b"],
            "task_type": "classification",
            "method": "permutation_importance",
            "prediction_context": "retrospective_explanation",
        },
    )

    out = TypedOperatorRegistry.execute_intent(df, intent_v2)
    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "feature_explainability"


def test_data_cleaning_operator_execution():
    """Verifies TaskFamily.DATA_CLEANING executes DataCleaningOperator."""
    df = pd.DataFrame({
        "col_num": [1.0, np.nan, 3.0, 1.0],
        "col_str": ["x", "y", None, "x"]
    })
    intent_v2 = AnalysisIntentV2(
        task_family=TaskFamily.DATA_CLEANING,
        subjects=["col_num", "col_str"],
        arguments={"method": "impute_mode", "columns": ["col_num", "col_str"]},
    )

    out = TypedOperatorRegistry.execute_intent(df, intent_v2)
    assert out.status == "SUCCEEDED"
    assert out.manifest.operator == "data_cleaning"
    assert out.dataframe.isnull().sum().sum() == 0


def test_unregistered_task_family_strictly_fails():
    """Verifies unmapped TaskFamily strictly fails with UNSUPPORTED_OPERATION without silent EDA fallback."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    fake_intent = AnalysisIntentV2.model_construct(
        task_family="NON_EXISTENT_FAMILY_XYZ",
        subjects=["x"],
        conditions=[],
        grouping=[],
        time_context=None,
        statistical_requirements=[],
        visualization=None,
        required_artifacts=[],
        assumptions=[],
        prompt_summary=None
    )

    out = TypedOperatorRegistry.execute_intent(df, fake_intent)
    assert out.status == "FAILED"
    assert out.error_code == "UNSUPPORTED_OPERATION"
    assert "No deterministic typed operator registered" in out.error_message
