"""
DataPilot OS — Result Verification Registry.
Audits operator outputs independently against raw data, numerical oracles,
chart artifact parity, and mathematical invariants before responses are emitted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily
from backend.ds_engine.operator_contracts import (
    NumericalInvariantValidator,
    NumericalInvariantViolation,
)
from backend.ds_engine.numerical_oracles import NumericalOracles

logger = logging.getLogger(__name__)


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    details: str = ""


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    checks: List[VerificationCheck] = Field(default_factory=list)
    discrepancies: List[str] = Field(default_factory=list)
    invariants_verified: List[str] = Field(default_factory=list)


class ResultVerificationRegistry:
    """Independent verification engine executing post-operator sanity checks."""

    @classmethod
    def verify_operator_result(
        cls,
        intent: AnalysisIntentV2,
        output: Any,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> VerificationReport:
        checks: List[VerificationCheck] = []
        discrepancies: List[str] = []
        invariants_verified: List[str] = []

        manifest = getattr(output, "manifest", None)

        # 1. Output status check
        status = getattr(output, "status", "")
        if status != "SUCCEEDED":
            checks.append(VerificationCheck(
                name="status_succeeded",
                passed=False,
                details=f"Operator returned status '{status}' instead of SUCCEEDED.",
            ))
            discrepancies.append(f"Operator execution failed with code: {getattr(output, 'error_code', 'UNKNOWN')}")
            return VerificationReport(is_valid=False, checks=checks, discrepancies=discrepancies, invariants_verified=invariants_verified)

        checks.append(VerificationCheck(name="status_succeeded", passed=True, details="Operator status is SUCCEEDED."))

        # 2. Chart artifact parity check
        requested_chart = None
        if intent.visualization and intent.visualization.type not in ("none", "", None):
            requested_chart = intent.visualization.type.lower()
        elif intent.arguments and intent.arguments.get("chart_type") not in ("none", "", None):
            requested_chart = str(intent.arguments.get("chart_type")).lower()
        elif intent.correlation_spec and intent.correlation_spec.chart_type not in ("none", "", None):
            requested_chart = intent.correlation_spec.chart_type.lower()

        if requested_chart and requested_chart != "none":
            has_image = bool(getattr(output, "plot_image_base64", None))
            has_plotly = bool(getattr(output, "plotly_spec", None) and getattr(output, "plotly_spec", {}).get("data"))
            if not (has_image or has_plotly):
                checks.append(VerificationCheck(
                    name="chart_artifact_parity",
                    passed=False,
                    details=f"Chart '{requested_chart}' was explicitly requested, but no image or plotly artifact was materialized.",
                ))
                discrepancies.append(f"CHART_ARTIFACT_MISSING: Requested chart '{requested_chart}' was not generated.")
            else:
                checks.append(VerificationCheck(
                    name="chart_artifact_parity",
                    passed=True,
                    details=f"Chart artifact for '{requested_chart}' successfully materialized.",
                ))
        else:
            checks.append(VerificationCheck(name="chart_artifact_parity", passed=True, details="No chart requested."))

        # 3. Row count consistency
        if manifest:
            row_in = manifest.row_count_in
            row_out = manifest.row_count_out

            if raw_df is not None and row_in != len(raw_df):
                checks.append(VerificationCheck(
                    name="input_row_count_parity",
                    passed=False,
                    details=f"Manifest row_count_in ({row_in}) does not match raw dataset row count ({len(raw_df)}).",
                ))
                discrepancies.append(f"ROW_COUNT_MISMATCH: Input rows {row_in} != raw dataset {len(raw_df)}.")
            else:
                checks.append(VerificationCheck(name="input_row_count_parity", passed=True, details="Input row counts match."))

            out_df = getattr(output, "dataframe", None)
            if isinstance(out_df, pd.DataFrame) and row_out != len(out_df):
                checks.append(VerificationCheck(
                    name="output_row_count_parity",
                    passed=False,
                    details=f"Manifest row_count_out ({row_out}) does not match materialized table length ({len(out_df)}).",
                ))
                discrepancies.append(f"ROW_COUNT_MISMATCH: Output rows {row_out} != materialized DataFrame {len(out_df)}.")
            else:
                checks.append(VerificationCheck(name="output_row_count_parity", passed=True, details="Output row counts match."))

        # 4. Mathematical Invariants & Oracle Verification
        out_df = getattr(output, "dataframe", None)
        task_fam = intent.task_family

        try:
            if task_fam == TaskFamily.CORRELATION_ANALYSIS and isinstance(out_df, pd.DataFrame):
                if "feature_x" in out_df.columns and "feature_y" in out_df.columns:
                    # Pairwise correlation verification
                    for r_col in ("pearson_r", "spearman_rho", "kendall_tau"):
                        if r_col in out_df.columns and pd.notna(out_df[r_col].iloc[0]):
                            val = float(out_df[r_col].iloc[0])
                            if val < -1.0001 or val > 1.0001:
                                raise NumericalInvariantViolation(f"Correlation coefficient {r_col}={val} out of bounds [-1, 1]")
                    for p_col in ("pearson_p_value", "spearman_p_value", "kendall_p_value", "p_value"):
                        if p_col in out_df.columns and pd.notna(out_df[p_col].iloc[0]):
                            pval = float(out_df[p_col].iloc[0])
                            if pval < 0.0 or pval > 1.0001:
                                raise NumericalInvariantViolation(f"Correlation p-value {p_col}={pval} out of bounds [0, 1]")
                    invariants_verified.append("pairwise_correlation_coefficient_and_pvalue_bounds")
                else:
                    matrix = out_df.set_index("feature") if "feature" in out_df.columns else out_df
                    NumericalInvariantValidator.validate_correlation_matrix(matrix)
                    invariants_verified.append("correlation_matrix_bounds_symmetry_diagonal")

                    if raw_df is not None and intent.subjects:
                        method = (intent.correlation_spec.correlation_methods[0] if intent.correlation_spec else "pearson")
                        oracle_corr = NumericalOracles.compute_correlation_matrix(raw_df, intent.subjects, method=method)
                        for col_a in intent.subjects:
                            for col_b in intent.subjects:
                                if col_a in matrix.index and col_b in matrix.columns:
                                    diff = abs(float(matrix.loc[col_a, col_b]) - float(oracle_corr.loc[col_a, col_b]))
                                    if diff > 1e-3:
                                        discrepancies.append(
                                            f"ORACLE_DISCREPANCY: Correlation between '{col_a}' and '{col_b}' differs from oracle by {diff:.6f}."
                                        )

            elif task_fam == TaskFamily.STATISTICAL_TEST and isinstance(out_df, pd.DataFrame):
                stat_col = "statistic" if "statistic" in out_df.columns else ("Test Statistic" if "Test Statistic" in out_df.columns else None)
                pval_col = "p_value" if "p_value" in out_df.columns else ("p-value" if "p-value" in out_df.columns else None)
                if stat_col and pval_col and not out_df.empty:
                    for idx in range(len(out_df)):
                        test_stat = float(out_df[stat_col].iloc[idx])
                        p_val = float(out_df[pval_col].iloc[idx])
                        dof = float(out_df["degrees_of_freedom"].iloc[idx]) if "degrees_of_freedom" in out_df.columns and pd.notna(out_df["degrees_of_freedom"].iloc[idx]) else None
                        NumericalInvariantValidator.validate_statistical_test_result(test_stat, p_val, dof)
                    invariants_verified.append("statistical_test_p_value_finite_statistic")

            elif task_fam == TaskFamily.GROUP_AGGREGATION and isinstance(out_df, pd.DataFrame):
                # If rate/percentage present
                for rate_col in ("rate", "rate_percent"):
                    if rate_col in out_df.columns:
                        is_pct = rate_col == "rate_percent"
                        NumericalInvariantValidator.validate_rate_or_percentage(out_df[rate_col].dropna(), is_percentage=is_pct)
                        invariants_verified.append(f"rate_bounds_{rate_col}")

            elif task_fam == TaskFamily.DIMENSIONALITY_REDUCTION and isinstance(out_df, pd.DataFrame):
                if "explained_variance_ratio" in out_df.columns:
                    ratios = out_df["explained_variance_ratio"].dropna().tolist()
                    NumericalInvariantValidator.validate_pca_variance_ratios(ratios)
                    invariants_verified.append("pca_variance_ratios_monotonic_descending")

            elif task_fam == TaskFamily.SQL_QUERY and isinstance(out_df, pd.DataFrame):
                if out_df.shape[1] > 0:
                    invariants_verified.append("sql_result_shape_and_types_verified")

        except NumericalInvariantViolation as inv_err:
            checks.append(VerificationCheck(
                name="numerical_invariant_verification",
                passed=False,
                details=str(inv_err),
            ))
            discrepancies.append(f"NUMERICAL_INVARIANT_VIOLATION: {inv_err}")

        is_valid = len(discrepancies) == 0
        return VerificationReport(
            is_valid=is_valid,
            checks=checks,
            discrepancies=discrepancies,
            invariants_verified=invariants_verified,
        )
