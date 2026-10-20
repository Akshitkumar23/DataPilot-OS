import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from backend.ds_engine.intent_schema import (
    IntentType,
    AnalysisIntentV2,
    ExecutionManifest,
)
from backend.ds_engine.code_generator import CodeVersion

logger = logging.getLogger(__name__)


_FILTER_OPERATOR_ALIASES = {
    "=": "eq", "==": "eq", "equals": "eq",
    "!=": "neq", "<>": "neq", "not_equals": "neq",
    ">": "gt", ">=": "gte", "<": "lt", "<=": "lte",
}


def _canonical_filter_operator(operator: Any) -> str:
    """Compare typed filter semantics, never incidental display strings."""
    value = getattr(operator, "value", operator)
    normalised = str(value).strip().lower()
    return _FILTER_OPERATOR_ALIASES.get(normalised, normalised)

class ResultValidationResult(BaseModel):
    is_valid: bool
    issues: List[str]
    metrics: Dict[str, Any]

class ResultValidator:
    def validate_result(
        self,
        code_version: CodeVersion,
        execution_result: Dict[str, Any],
        df: Optional[pd.DataFrame] = None
    ) -> ResultValidationResult:
        is_valid = True
        issues = []
        metrics = {}

        intent = str(code_version.purpose).upper()
        out_df = execution_result.get("dataframe")
        stdout = execution_result.get("stdout", "")
        plot_images = execution_result.get("plot_images", []) or ([execution_result.get("plot_image")] if execution_result.get("plot_image") else [])

        try:
            # 1. CLASSIFICATION & REGRESSION Contract Checks
            if "CLASSIFICATION" in intent or "AUTO_ML" in intent or "REGRESSION" in intent:
                metrics["contract"] = "CLASSIFICATION_REGRESSION_ML"
                target_col = None
                for out in code_version.expected_outputs:
                    if out.get("target"):
                        target_col = out.get("target")

                if df is not None and target_col and target_col in df.columns:
                    unique_classes = df[target_col].dropna().nunique()
                    metrics["unique_classes"] = unique_classes
                    if "CLASSIFICATION" in intent and unique_classes < 2:
                        is_valid = False
                        issues.append(f"EVIDENCE_INSUFFICIENT: Target column '{target_col}' has only {unique_classes} class. Classification requires at least 2 distinct classes.")

                if "NaN" in stdout or "Inf" in stdout:
                    issues.append("Warning: NaN or Inf metrics found in stdout output.")

            # 2. TIME-SERIES FORECAST Contract Checks
            elif "TIME_SERIES_FORECAST" in intent or "FORECAST" in intent:
                metrics["contract"] = "TIME_SERIES_FORECAST"
                expected_horizon = 30
                for out in code_version.expected_outputs:
                    if out.get("horizon"):
                        expected_horizon = out.get("horizon")

                if out_df is not None:
                    actual_rows = len(out_df)
                    metrics["expected_horizon"] = expected_horizon
                    metrics["actual_horizon"] = actual_rows
                    if actual_rows != expected_horizon:
                        is_valid = False
                        issues.append(f"RESULT_CONTRACT_VIOLATION: Forecast horizon mismatch (expected {expected_horizon} periods, got {actual_rows}).")

                    preds = out_df.select_dtypes(include=[np.number])
                    if preds.isna().any().any() or np.isinf(preds).any().any():
                        is_valid = False
                        issues.append("RESULT_CONTRACT_VIOLATION: Forecast predictions contain NaN or Inf values.")

                    lower_cols = [c for c in out_df.columns if 'lower' in c.lower()]
                    upper_cols = [c for c in out_df.columns if 'upper' in c.lower()]
                    if lower_cols and upper_cols:
                        if (out_df[lower_cols[0]] > out_df[upper_cols[0]]).any():
                            is_valid = False
                            issues.append(f"RESULT_CONTRACT_VIOLATION: Lower confidence bound exceeds upper bound.")

            # 3. AUTOCORRELATION Contract Checks
            elif "AUTOCORRELATION" in intent:
                metrics["contract"] = "AUTOCORRELATION"
                if not plot_images and not stdout:
                    is_valid = False
                    issues.append("RESULT_CONTRACT_VIOLATION: ACF/PACF visual plot or summary stdout was not generated.")

            # 4. CORRELATION Contract Checks
            elif "CORRELATION" in intent and "AUTOCORRELATION" not in intent:
                metrics["contract"] = "CORRELATION"
                if out_df is not None:
                    if "Correlation" in out_df.columns:
                        corr_vals = pd.to_numeric(out_df["Correlation"], errors='coerce').dropna()
                        if len(corr_vals) > 0 and ((corr_vals.max() > 1.0 + 1e-5) or (corr_vals.min() < -1.0 - 1e-5)):
                            is_valid = False
                            issues.append("RESULT_CONTRACT_VIOLATION: Correlation coefficients exceed valid [-1.0, 1.0] interval bounds.")
                    else:
                        nums = out_df.select_dtypes(include=[np.number])
                        if len(nums.columns) > 0:
                            if (nums.max().max() > 1.0 + 1e-5) or (nums.min().min() < -1.0 - 1e-5):
                                is_valid = False
                                issues.append("RESULT_CONTRACT_VIOLATION: Correlation coefficients exceed valid [-1.0, 1.0] interval bounds.")

                if not plot_images:
                    is_rejected_non_num = (
                        (out_df is not None and "status" in out_df.columns and (out_df["status"] == "REJECTED_NON_NUMERIC").any()) or
                        any(msg in stdout.lower() for msg in ["insufficient", "non-numeric", "requires at least 2 numeric"])
                    )
                    if not is_rejected_non_num:
                        is_valid = False
                        issues.append("RESULT_CONTRACT_VIOLATION: Heatmap visual artifact was declared but not generated.")

            # 4. STATISTICAL TEST Contract Checks
            elif "STATISTICAL_TEST" in intent:
                metrics["contract"] = "STATISTICAL_TEST"
                stdout_low = stdout.lower()
                if "p-value: -" in stdout_low or ("p-value: 1." in stdout_low and "1.0" not in stdout_low):
                    is_valid = False
                    issues.append("RESULT_CONTRACT_VIOLATION: Calculated p-value falls outside valid [0.0, 1.0] probability interval.")

            # 5. ANOMALY DETECTION & CLUSTERING Contract Checks
            elif "ANOMALY_DETECTION" in intent or "CLUSTERING" in intent or "PCA" in intent:
                metrics["contract"] = "ANOMALY_DETECTION & CLUSTERING"
                if out_df is not None and 'Cluster' in out_df.columns:
                    k = out_df['Cluster'].nunique()
                    metrics["k"] = k
                    if k < 2:
                        is_valid = False
                        issues.append(f"RESULT_CONTRACT_VIOLATION: Clustering produced {k} clusters. Expected K >= 2.")

        except Exception as e:
            logger.error(f"Error in ResultValidator: {e}")
            issues.append(f"Validation Execution Exception: {e}")
            is_valid = False

        return ResultValidationResult(is_valid=is_valid, issues=issues, metrics=metrics)


class SemanticResultValidator:
    """
    Authoritative DS-OS Semantic Result Validator.
    Compares the planned AnalysisIntentV2 against the runtime ExecutionManifest.
    Checks declared semantics and materialized artifacts; this is not a proof
    of arbitrary scientific correctness or of the model's query understanding.
    """

    @classmethod
    def validate_manifest(
        cls,
        intent: AnalysisIntentV2,
        manifest: ExecutionManifest,
        output_data: Optional[Dict[str, Any]] = None
    ) -> ResultValidationResult:
        issues: List[str] = []
        metrics: Dict[str, Any] = {"manifest_operator": manifest.operator}

        # 1. Event Row Count & Pre-Event Row Count Verification for Event Window Analysis
        if manifest.operator == "event_window_analysis":
            if manifest.event_row_count is None or manifest.event_row_count <= 0:
                issues.append(
                    "RESULT_CONTRACT_VIOLATION: Event-window analysis requires event_row_count > 0."
                )
            if manifest.pre_event_row_count is None or manifest.pre_event_row_count < 2:
                issues.append(
                    "RESULT_CONTRACT_VIOLATION: Event-window analysis requires pre_event_row_count >= 2."
                )

        # 2. Time Column Validity Verification
        if manifest.operator in ("event_window_analysis", "time_series_forecast", "time_series_decomposition", "autocorrelation"):
            tc = intent.time_context
            time_col = tc.time_column if tc else None
            if time_col and time_col not in manifest.columns_used:
                issues.append(
                    f"RESULT_CONTRACT_VIOLATION: Temporal analysis missing valid time column '{time_col}' in execution manifest."
                )

        # 3. Entity Boundary Preservation & Partition Verification
        if manifest.operator == "event_window_analysis" and intent.time_context and intent.time_context.entity_id:
            entity_id = intent.time_context.entity_id
            if entity_id not in manifest.columns_used:
                issues.append(
                    f"RESULT_CONTRACT_VIOLATION: Multi-entity dataset analysis did not preserve entity boundaries (entity_id column '{entity_id}' omitted from manifest)."
                )
            if manifest.partitioned_by != entity_id:
                issues.append(
                    f"RESULT_CONTRACT_VIOLATION: Execution manifest partitioned_by '{manifest.partitioned_by}' does not match intent entity_id '{entity_id}'."
                )

        # 4. Filter Predicate Application Verification
        if intent.conditions:
            for cond in intent.conditions:
                expected_operator = _canonical_filter_operator(cond.operator)
                matched = any(
                    applied.get("column") == cond.column
                    and _canonical_filter_operator(applied.get("operator")) == expected_operator
                    and applied.get("value") == cond.value
                    for applied in manifest.filters_applied
                )
                if not matched:
                    issues.append(
                        f"RESULT_CONTRACT_VIOLATION: Intent filter condition on '{cond.column}' was not applied during execution with its requested operator and value."
                    )

        # 5. Strict Visualization Contract Match
        if intent.visualization and intent.visualization.type not in ("none", "", None):
            expected_chart = intent.visualization.type.lower()
            actual_chart = (manifest.chart_type or "none").lower()
            if expected_chart != actual_chart:
                issues.append(
                    f"RESULT_DOES_NOT_MATCH_INTENT: Visualization contract violation. Requested chart '{expected_chart}', but operator generated '{actual_chart}'."
                )

        # 6. Strict Subject Columns Used Match
        if intent.subjects:
            missing_cols = [c for c in intent.subjects if c not in manifest.columns_used]
            if missing_cols:
                issues.append(
                    f"RESULT_DOES_NOT_MATCH_INTENT: Subject column mismatch. Requested column(s) {missing_cols} were not processed in execution manifest (used: {manifest.columns_used})."
                )

        # 7. Statistical Requirements Parity
        if intent.statistical_requirements:
            computed = [s.lower() for s in manifest.statistics_computed]
            for req in intent.statistical_requirements:
                req_low = req.lower()
                if req_low in ("variance", "var") and not any(k in computed for k in ["variance", "var", "pre_event_variance"]):
                    issues.append(f"RESULT_DOES_NOT_MATCH_INTENT: Required statistical metric '{req}' was not computed.")
                elif req_low in ("iqr", "quantiles") and not any(k in computed for k in ["iqr", "25%", "75%", "quantiles"]):
                    issues.append(f"RESULT_DOES_NOT_MATCH_INTENT: Required statistical metric '{req}' was not computed.")

        is_valid = len(issues) == 0
        return ResultValidationResult(is_valid=is_valid, issues=issues, metrics=metrics)

    @classmethod
    def validate_output(cls, intent: AnalysisIntentV2, output: Any,
                        input_row_count: Optional[int] = None) -> ResultValidationResult:
        """Check the real output as well as its self-reported manifest.

        Preview records may be capped, so compare row counts with the complete
        intermediate DataFrame, never with the UI preview length.
        """
        result = cls.validate_manifest(intent, output.manifest)
        issues = list(result.issues)
        manifest = output.manifest
        if output.status != "SUCCEEDED":
            issues.append("RESULT_CONTRACT_VIOLATION: Operator output did not succeed.")
        if manifest.row_count_in < 0 or manifest.row_count_out < 0:
            issues.append("RESULT_CONTRACT_VIOLATION: Negative row count in execution manifest.")
        if input_row_count is not None and manifest.row_count_in != input_row_count:
            issues.append("RESULT_CONTRACT_VIOLATION: Input row count does not match the executed dataset.")
        if isinstance(output.dataframe, pd.DataFrame) and manifest.row_count_out != len(output.dataframe):
            issues.append("RESULT_CONTRACT_VIOLATION: Output row count does not match the materialized table.")
        if (manifest.chart_type or "none").lower() != "none":
            has_plotly = bool(output.plotly_spec and output.plotly_spec.get("data"))
            if not output.plot_image_base64 and not has_plotly:
                issues.append("RESULT_CONTRACT_VIOLATION: Declared chart has no materialized artifact.")

        # Independent verification registry audit
        from backend.ds_engine.result_verification_registry import ResultVerificationRegistry
        verification_report = ResultVerificationRegistry.verify_operator_result(intent, output)
        if not verification_report.is_valid:
            for discrepancy in verification_report.discrepancies:
                if discrepancy not in issues:
                    issues.append(f"VERIFICATION_FAILURE: {discrepancy}")

        return ResultValidationResult(is_valid=not issues, issues=issues, metrics=result.metrics)
