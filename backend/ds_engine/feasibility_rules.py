"""
DataPilot OS — Scientific Feasibility Rules Engine.
Generic rules enforcing statistical assumptions, time-series grain,
target leakage prevention, prospective/retrospective ML separation,
and event-window feasibility.
Zero domain-specific heuristics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.ds_engine.dataset_profiler import DatasetProfile
from backend.ds_engine.time_series_grain import analyze_time_series_grain

logger = logging.getLogger(__name__)


class FeasibilityRuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_feasible: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    clarification_prompt: Optional[str] = None
    candidate_columns: List[str] = Field(default_factory=list)
    suggested_adjustments: Dict[str, Any] = Field(default_factory=dict)


class FeasibilityRulesEngine:
    """
    Executes domain-agnostic scientific feasibility gates before analysis runs.
    """

    @classmethod
    def check_time_series_grain(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame],
    ) -> Optional[FeasibilityRuleResult]:
        """Verify time-series operations have unique timestamps and regular temporal grid."""
        if tool_name not in {"forecast_data", "autocorrelation", "decompose_time_series"}:
            return None

        time_col = arguments.get("time_column")
        if not time_col or df is None or time_col not in df.columns:
            return None

        diag = analyze_time_series_grain(df, time_col)
        if diag.requires_clarification:
            return FeasibilityRuleResult(
                is_feasible=False,
                error_code="TIME_SERIES_GRAIN_CLARIFICATION",
                error_message=diag.clarification_reason,
                clarification_prompt=diag.clarification_reason,
                suggested_adjustments={"suggested_grains": diag.suggested_grains},
            )
        return None

    @classmethod
    def check_supervised_ml_contract(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame],
    ) -> Optional[FeasibilityRuleResult]:
        """Validate ML task context, target separation, and target leakage prevention."""
        if tool_name != "train_model":
            return None

        target_col = arguments.get("target_column") or arguments.get("target")
        feature_cols = arguments.get("feature_columns") or arguments.get("features") or []

        if not target_col:
            return FeasibilityRuleResult(
                is_feasible=False,
                error_code="TARGET_COLUMN_REQUIRED",
                error_message="Supervised ML model training requires an explicit target_column.",
            )

        if not feature_cols:
            return FeasibilityRuleResult(
                is_feasible=False,
                error_code="FEATURE_COLUMNS_REQUIRED",
                error_message="Supervised ML model training requires explicit feature_columns.",
            )

        # Target Leakage: Target must not be in feature columns
        if target_col in feature_cols:
            return FeasibilityRuleResult(
                is_feasible=False,
                error_code="TARGET_LEAKAGE_DETECTED",
                error_message=(
                    f"Target leakage detected: target column '{target_col}' cannot be included in "
                    "predictor feature_columns."
                ),
            )

        prediction_context = arguments.get("prediction_context")
        if not prediction_context or prediction_context not in {"prospective_prediction", "retrospective_explanation"}:
            return FeasibilityRuleResult(
                is_feasible=False,
                error_code="PREDICTION_CONTEXT_REQUIRED",
                error_message=(
                    "Supervised ML model training requires an explicit prediction_context: "
                    "'prospective_prediction' (features temporally precede target) or "
                    "'retrospective_explanation' (features explain historical outcome)."
                ),
            )

        # If prospective prediction, verify features do not deterministically leak target
        if df is not None and target_col in df.columns and prediction_context == "prospective_prediction":
            from backend.ds_engine.target_leakage import TargetLeakageDetector
            assessment = TargetLeakageDetector.assess_classification(
                df=df,
                target_column=target_col,
                feature_columns=[c for c in feature_cols if c in df.columns],
            )
            leaky_features = [s.feature for s in assessment.signals]
            remaining_features = [f for f in feature_cols if f not in leaky_features]
            if leaky_features and not remaining_features:
                return FeasibilityRuleResult(
                    is_feasible=False,
                    error_code="TARGET_LEAKAGE_DETECTED",
                    error_message=(
                        f"Target leakage detected: all requested features {leaky_features} "
                        f"deterministically encode target '{target_col}' in prospective prediction context."
                    ),
                    suggested_adjustments={"leaky_features": leaky_features},
                )

        return None

    @classmethod
    def check_statistical_assumptions(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame],
    ) -> Optional[FeasibilityRuleResult]:
        """Validate inferential test assumptions (t-test, ANOVA, chi-square)."""
        if tool_name != "statistical_test":
            return None

        test_type = arguments.get("test_type")
        cols = arguments.get("columns") or []

        if not test_type:
            return FeasibilityRuleResult(
                is_feasible=False,
                error_code="TEST_TYPE_REQUIRED",
                error_message=(
                    "Statistical test requires an explicit test_type: "
                    "'independent_ttest', 'one_way_anova', or 'chi_square_independence'."
                ),
            )

        if len(cols) < 2:
            return FeasibilityRuleResult(
                is_feasible=False,
                error_code="INSUFFICIENT_COLUMNS",
                error_message=f"Statistical test '{test_type}' requires at least 2 columns; got {len(cols)}.",
            )

        if df is not None:
            # Check for data existence
            missing = [c for c in cols if c not in df.columns]
            if missing:
                return FeasibilityRuleResult(
                    is_feasible=False,
                    error_code="COLUMN_NOT_FOUND",
                    error_message=f"Statistical test columns {missing} not present in dataset.",
                )

            # Check zero variance or degenerate cases
            if test_type in {"independent_ttest", "welch_ttest", "student_ttest"}:
                for col in cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        valid = df[col].dropna()
                        if len(valid) < 3:
                            return FeasibilityRuleResult(
                                is_feasible=False,
                                error_code="INSUFFICIENT_SAMPLE_SIZE",
                                error_message=(
                                    f"Column '{col}' has fewer than 3 non-null records; "
                                    "independent t-test requires at least 3 records per sample."
                                ),
                            )
                        if valid.std() == 0.0:
                            return FeasibilityRuleResult(
                                is_feasible=False,
                                error_code="ZERO_VARIANCE",
                                error_message=(
                                    f"Column '{col}' has zero variance (constant values); "
                                    "t-test cannot compute a valid t-statistic."
                                ),
                            )
            elif test_type == "chi_square_independence":
                col1, col2 = cols[0], cols[1]
                ct = pd.crosstab(df[col1], df[col2])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    return FeasibilityRuleResult(
                        is_feasible=False,
                        error_code="DEGENERATE_CONTINGENCY_TABLE",
                        error_message=(
                            f"Contingency table for '{col1}' and '{col2}' has shape {ct.shape}. "
                            "Chi-square test of independence requires at least a 2x2 contingency table."
                        ),
                    )

        return None
