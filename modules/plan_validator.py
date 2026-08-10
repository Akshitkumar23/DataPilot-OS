import logging
import pandas as pd
from typing import Optional
from modules.models.execution_plan import ExecutionPlan
from modules.models.validation_result import ValidationResult

logger = logging.getLogger(__name__)

class PlanValidator:
    """
    Enforces runtime validation contracts on ExecutionPlans:
    1. Verifies column existence in DataFrame.
    2. Verifies data type compatibility (numeric vs categorical).
    3. Verifies capability existence in CapabilityRegistry.
    """

    REGISTERED_CAPABILITIES = {
        "sig_eda_dashboard", "sig_forecast", "sig_ml", "sig_clean",
        "sig_sql", "sig_describe", "sig_hist", "sig_heatmap",
        "sig_scatter", "sig_top", "sig_encode", "sig_scale",
        "sig_skew", "sig_pie", "sig_cluster", "sig_pca", "sig_list"
    }

    @classmethod
    def validate(cls, plan: ExecutionPlan, df: Optional[pd.DataFrame] = None) -> ValidationResult:
        if not plan.steps:
            return ValidationResult(is_valid=False, error_msg="Execution plan contains no steps.", failed_checks=["EMPTY_STEPS"])

        failed_checks = []
        error_msgs = []

        df_cols = list(df.columns) if df is not None and hasattr(df, 'columns') else []

        for step in plan.steps:
            # 1. Capability Registration Check
            if step.capability not in cls.REGISTERED_CAPABILITIES:
                failed_checks.append(f"UNKNOWN_CAPABILITY_{step.capability}")
                error_msgs.append(f"Capability '{step.capability}' is not registered.")

            # 2. Column Existence & Type Compatibility Check
            params = step.parameters
            for param_key in ['column', 'target', 'x_column', 'y_column', 'category_column', 'metric_column']:
                col_name = params.get(param_key)
                if col_name and df_cols:
                    if col_name not in df_cols:
                        failed_checks.append(f"COLUMN_NOT_FOUND_{col_name}")
                        error_msgs.append(f"Column '{col_name}' does not exist in dataset. Available columns: {df_cols}")
                    elif step.capability in ['sig_hist', 'sig_scatter', 'sig_forecast'] and param_key in ['column', 'x_column', 'y_column']:
                        if not pd.api.types.is_numeric_dtype(df[col_name]):
                            failed_checks.append(f"TYPE_MISMATCH_NUMERIC_{col_name}")
                            error_msgs.append(f"Column '{col_name}' is not numeric, required for {step.capability}.")

        if failed_checks:
            err_summary = " | ".join(error_msgs)
            logger.warning(f"PLAN_INVALID | Validation failed: {err_summary}")
            return ValidationResult(is_valid=False, error_msg=err_summary, failed_checks=failed_checks)

        logger.info(f"PLAN_VALID | ExecutionPlan passed all {len(plan.steps)} validation check(s).")
        return ValidationResult(is_valid=True)
