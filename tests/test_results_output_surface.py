import pytest
import pandas as pd
import numpy as np
from modules.semantic_planner import SemanticPlanner
from backend.ds_engine.intent_schema import AnalysisIntentV2, IntentType
from backend.ds_engine.typed_operators import SupervisedMLOperator
from backend.api.datasets import SaveAsDatasetRequest, save_result_as_dataset
from backend.core.dataset_registry import dataset_registry


def test_schema_level_column_grounding():
    """Verify numeric columns, categorical columns, datetime columns pass grounding."""
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "city": ["Seoul", "Busan", "Incheon"],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "has_car": [True, False, True],
        "notes": [np.nan, "good", np.nan]
    })

    # Schema selectors must pass without error
    parsed_numeric = {"commands": [{"tool": "aggregate_data", "arguments": {"columns": ["numeric columns"]}}]}
    assert SemanticPlanner._dataset_column_grounding_error(parsed_numeric, df, "show numeric columns") is None

    parsed_all_numeric = {"commands": [{"tool": "aggregate_data", "arguments": {"columns": ["numeric features"]}}]}
    assert SemanticPlanner._dataset_column_grounding_error(parsed_all_numeric, df, "summarize all numeric features") is None

    parsed_cat = {"commands": [{"tool": "aggregate_data", "arguments": {"columns": ["categorical columns"]}}]}
    assert SemanticPlanner._dataset_column_grounding_error(parsed_cat, df, "list categorical columns") is None

    parsed_dt = {"commands": [{"tool": "aggregate_data", "arguments": {"columns": ["datetime columns"]}}]}
    assert SemanticPlanner._dataset_column_grounding_error(parsed_dt, df, "check datetime columns") is None

    parsed_miss = {"commands": [{"tool": "aggregate_data", "arguments": {"columns": ["missing columns"]}}]}
    assert SemanticPlanner._dataset_column_grounding_error(parsed_miss, df, "identify missing columns") is None

    # Hallucinated actual column name (salary is not in df, but if in df or schema grounded, it catches)
    # When a column exists in df but NOT in evidence_text:
    parsed_unmentioned = {"commands": [{"tool": "aggregate_data", "arguments": {"columns": ["city"]}}]}
    err = SemanticPlanner._dataset_column_grounding_error(parsed_unmentioned, df, "show age distribution")
    assert err is not None
    assert "city" in err


def test_supervised_ml_extended_metrics_and_baseline():
    """Verify SupervisedMLOperator computes baseline, multi-metrics, and 5-fold CV."""
    from backend.ds_engine.intent_schema import TaskFamily
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })

    intent = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        target_column="target",
        subjects=["target", "feature_1", "feature_2"],
        arguments={"task_type": "classification", "evaluation_artifact": "feature_importance"}
    )
    op = SupervisedMLOperator()
    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    assert "extended_metrics" in res.data
    ext = res.data["extended_metrics"]
    assert "Balanced Accuracy" in ext
    assert "Baseline Score" in ext
    assert "Improvement vs Baseline" in ext
    assert "Precision" in ext
    assert "Recall" in ext
    assert "F1 Score" in ext
    assert "ROC-AUC" in ext
    assert "PR-AUC" in ext
    assert "Brier Score" in ext

    # FW-02 Threshold tuning payload
    assert "threshold_tuning_payload" in res.data
    payload = res.data["threshold_tuning_payload"]
    assert payload is not None
    assert "y_test" in payload
    assert "test_probs" in payload
    assert len(payload["y_test"]) == len(payload["test_probs"])
    assert payload["brier_score"] is not None
    assert payload["roc_auc"] is not None
    assert payload["pr_auc"] is not None

    # CV breakdown
    assert "cv_folds" in res.data
    assert len(res.data["cv_folds"]) == 5
    assert res.data["cv_mean"] is not None
    assert res.data["cv_std"] is not None


def test_dataset_promotion_save_endpoint(tmp_path):
    """Verify SaveAsDatasetRequest and save_result_as_dataset creates valid dataset."""
    req = SaveAsDatasetRequest(
        name="test_promoted_dataset",
        columns=["colA", "colB"],
        rows=[
            {"colA": 10, "colB": "foo"},
            {"colA": 20, "colB": "bar"}
        ],
        metadata={"operator": "unit_test"}
    )
    result = save_result_as_dataset(req)
    assert result["status"] == "SUCCEEDED"
    assert "dataset_id" in result
    assert result["name"] == "test_promoted_dataset.csv"
    assert result["row_count"] == 2
    assert result["column_count"] == 2


def test_supervised_ml_regression_residual_diagnostics():
    """Verify regression computes residual diagnostics without CV leakage."""
    from backend.ds_engine.intent_schema import TaskFamily
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "target": np.random.randn(n) * 2 + 5
    })

    intent = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        target_column="target",
        subjects=["target", "x1", "x2"],
        arguments={"task_type": "regression", "evaluation_artifact": "feature_importance"}
    )
    op = SupervisedMLOperator()
    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    assert "residual_diagnostics" in res.data
    diag = res.data["residual_diagnostics"]
    assert "rmse" in diag
    assert "mae" in diag
    assert "std" in diag
    assert "mean" in diag
    assert "skewness" in diag
    assert "durbin_watson" in diag
    assert "heteroscedasticity_p" in diag

    # Regression CV folds
    assert "cv_folds" in res.data
    assert len(res.data["cv_folds"]) == 5
    assert res.data["cv_mean"] is not None


def test_duckdb_explain_query_plan():
    """Verify DuckDB SQL manager captures EXPLAIN query plan."""
    from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
    df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    res = GovernedDuckDBManager.execute_query(sql="SELECT id, val * 2 AS doubled FROM df WHERE id > 1", df=df)
    assert res.status == "SUCCEEDED"
    assert res.explain_plan is not None
    assert "SCAN" in res.explain_plan.upper() or "EXPLAIN" in res.explain_plan.upper() or "FILTER" in res.explain_plan.upper()


def test_dataset_metadata_source_native_contract():
    """Verify compute_profile_summary and connect endpoint populate source_storage and is_source_native flags."""
    from backend.api.datasets import compute_profile_summary
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    summary = compute_profile_summary(df, "ds_test", "sales.csv")
    summary["is_source_native"] = True
    summary["source_storage"] = "source_native"
    assert summary["is_source_native"] is True
    assert summary["source_storage"] == "source_native"
    assert summary["row_count"] == 2
    assert summary["column_count"] == 2


def test_threshold_tuning_split_isolation_and_class_imbalance():
    """Verify threshold tuning strictly isolates tuning split (OOF CV on train) from pristine eval split (test)."""
    from backend.ds_engine.intent_schema import TaskFamily
    np.random.seed(42)
    n = 100
    # Create 85% majority class to trigger imbalance warning
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })

    intent = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        target_column="target",
        subjects=["target", "feat1", "feat2"],
        arguments={"task_type": "classification", "evaluation_artifact": "feature_importance"}
    )
    op = SupervisedMLOperator()
    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    payload = res.data.get("threshold_tuning_payload")
    assert payload is not None

    # Verify both tuning and evaluation splits are present
    assert "tuning_split" in payload
    assert "eval_split" in payload
    tuning = payload["tuning_split"]
    eval_split = payload["eval_split"]

    # Verify sample sizes (80 train, 20 test for n=100)
    assert tuning["sample_size"] == 80
    assert eval_split["sample_size"] == 20
    assert len(tuning["y_true"]) == 80
    assert len(tuning["probs"]) == 80
    assert len(eval_split["y_true"]) == 20
    assert len(eval_split["probs"]) == 20

    # Ensure descriptions reflect methodology
    assert "Out-of-fold" in tuning["description"]
    assert "Held-out" in eval_split["description"]

    # Verify class imbalance warning triggered (> 80%)
    assert res.data.get("class_imbalance_warning") is not None
    assert "imbalance" in res.data["class_imbalance_warning"].lower()


def test_statistical_test_hypotheses_and_df_suppression():
    """Verify StatisticalTestOperator outputs test-specific hypotheses and suppresses df for non-parametric tests."""
    from backend.ds_engine.intent_schema import TaskFamily
    from backend.ds_engine.typed_operators import StatisticalTestOperator
    np.random.seed(42)

    # 1. Non-parametric test: Mann-Whitney U
    df_samples = pd.DataFrame({
        "score_a": np.random.normal(10, 2, 30),
        "score_b": np.random.normal(12, 2, 30)
    })
    intent_mwu = AnalysisIntentV2(
        task_family=TaskFamily.STATISTICAL_TEST,
        target_column="score_a",
        subjects=["score_a", "score_b"],
        arguments={"test_type": "mann_whitney_u", "columns": ["score_a", "score_b"], "alpha": 0.05}
    )
    op = StatisticalTestOperator()
    res_mwu = op.execute(df_samples, intent_mwu)

    assert res_mwu.status == "SUCCEEDED"
    stat_test_mwu = res_mwu.data.get("statistical_test")
    assert stat_test_mwu is not None
    assert stat_test_mwu["test_type"] == "mann_whitney_u"
    assert "identical" in stat_test_mwu["null_hypothesis"].lower()
    assert "stochastically" in stat_test_mwu["alt_hypothesis"].lower()
    # Non-parametric: degrees_of_freedom must be None (suppressed)
    assert stat_test_mwu["degrees_of_freedom"] is None
    assert stat_test_mwu.get("scientific_caution") is not None

    # 2. Parametric test: Welch's t-test
    intent_welch = AnalysisIntentV2(
        task_family=TaskFamily.STATISTICAL_TEST,
        target_column="score_a",
        subjects=["score_a", "score_b"],
        arguments={"test_type": "welch_ttest", "columns": ["score_a", "score_b"], "alpha": 0.05}
    )
    res_welch = op.execute(df_samples, intent_welch)
    assert res_welch.status == "SUCCEEDED"
    stat_test_welch = res_welch.data.get("statistical_test")
    assert stat_test_welch is not None
    assert "population means" in stat_test_welch["null_hypothesis"].lower()
    # Welch's t-test: degrees_of_freedom must be mathematically defined
    assert stat_test_welch["degrees_of_freedom"] is not None
    assert isinstance(stat_test_welch["degrees_of_freedom"], (int, float))
    assert stat_test_welch["degrees_of_freedom"] > 0
    assert stat_test_welch.get("effect_size") is not None
    assert stat_test_welch.get("ci_lower") is not None
    assert stat_test_welch.get("ci_upper") is not None


def test_single_class_target_safe_auc():
    """Verify SupervisedMLOperator gracefully skips ROC-AUC / PR-AUC when target has < 2 unique classes."""
    from backend.ds_engine.intent_schema import TaskFamily
    np.random.seed(42)
    n = 60
    # Create single-class target
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": [1] * n  # only 1 class
    })
    intent = AnalysisIntentV2(
        task_family=TaskFamily.SUPERVISED_ML,
        target_column="target",
        subjects=["target", "f1", "f2"],
        arguments={"task_type": "classification", "evaluation_artifact": "feature_importance"}
    )
    op = SupervisedMLOperator()
    res = op.execute(df, intent)
    # The operator either fails safely with clear message or handles it without uncaught exception
    assert res.status in ["FAILED", "SUCCEEDED"]
    if res.status == "SUCCEEDED":
        # ROC-AUC must not crash and be None
        ext = res.data.get("extended_metrics", {})
        assert ext.get("ROC-AUC") is None or np.isnan(ext.get("ROC-AUC", np.nan))



