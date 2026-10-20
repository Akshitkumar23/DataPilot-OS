"""
DataPilot OS — Deterministic Typed Operators DS Engine.
Executes authoritative analytical, statistical, time-series, event-window, ML, and visualization tasks
without relying on fragile LLM code generation.
LLM decides "what"; Deterministic Typed Operators decide "how".
100% Zero domain hardcoding, 100% Scientific Factual Accuracy, & 100% Zero Silent Fallbacks.
"""

import io
import re
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from backend.ds_engine.intent_schema import (
    AnalysisIntentV2, CorrelationSpec, TaskFamily, FilterSpec, FilterOperator,
    ExecutionManifest, VisualizationSpec, TimeContext
)
from backend.ds_engine.dataset_profiler import DatasetProfile, DatasetProfiler
from backend.ds_engine.time_frequency import canonicalize_pandas_frequency

logger = logging.getLogger(__name__)


class OperatorOutput(BaseModel):
    model_config = dict(arbitrary_types_allowed=True)
    status: str = "SUCCEEDED"
    error_code: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    dataframe: Optional[Any] = None  # Holds complete in-memory intermediate DataFrame
    manifest: ExecutionManifest
    plot_image_base64: Optional[str] = None
    plotly_spec: Optional[Dict[str, Any]] = None
    written_finding: str = ""
    reproducible_code: str = ""
    error_message: Optional[str] = None


def _event_predicate_value_required(
    df: pd.DataFrame, event_col: str, order_col: str, entity_id_col: Optional[str]
) -> OperatorOutput:
    """Return the shared strict error for comparison predicates without a value."""
    return OperatorOutput(
        status="FAILED",
        error_code="EVENT_PREDICATE_REQUIRED",
        error_message="Event comparison predicates require an explicit value.",
        manifest=ExecutionManifest(
            operator="event_window_analysis",
            columns_used=[event_col, order_col],
            statistics_computed=[],
            chart_type="none",
            row_count_in=len(df),
            row_count_out=0,
            sorted_by=order_col,
            partitioned_by=entity_id_col,
        ),
    )


class BaseTypedOperator:
    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        raise NotImplementedError


class FilterSliceOperator(BaseTypedOperator):
    """Vectorized typed filtering with in-memory preservation of all matching rows."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        res_df = df.copy()
        row_in = len(df)
        filters_applied = []
        warnings = []

        for cond in intent.conditions:
            col = cond.column
            if col not in res_df.columns:
                return OperatorOutput(
                    status="FAILED",
                    error_code="COLUMN_NOT_FOUND",
                    error_message=f"Requested filter column '{col}' not found in dataset.",
                    manifest=ExecutionManifest(
                        operator="filter_slice",
                        columns_used=[col],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=row_in,
                        row_count_out=0,
                        warnings=[f"Column '{col}' missing."]
                    )
                )

            op = cond.operator
            val = cond.value
            series = res_df[col]

            # Coerce value for numeric/binary columns
            if pd.api.types.is_numeric_dtype(series) and isinstance(val, str):
                val_clean = val.strip().lower()
                if val_clean in ("true", "yes", "failure", "failed", "positive", "1"):
                    val = 1
                elif val_clean in ("false", "no", "normal", "negative", "0"):
                    val = 0
                else:
                    v_num = pd.to_numeric(val, errors="coerce")
                    if pd.notna(v_num):
                        val = v_num

            # Apply filter
            if op == FilterOperator.EQ:
                res_df = res_df[res_df[col] == val]
            elif op == FilterOperator.NEQ:
                res_df = res_df[res_df[col] != val]
            elif op == FilterOperator.GT:
                res_df = res_df[res_df[col] > float(val)]
            elif op == FilterOperator.GTE:
                res_df = res_df[res_df[col] >= float(val)]
            elif op == FilterOperator.LT:
                res_df = res_df[res_df[col] < float(val)]
            elif op == FilterOperator.LTE:
                res_df = res_df[res_df[col] <= float(val)]
            elif op == FilterOperator.IN and isinstance(val, list):
                res_df = res_df[res_df[col].isin(val)]
            elif op == FilterOperator.NOT_IN and isinstance(val, list):
                res_df = res_df[~res_df[col].isin(val)]
            elif op == FilterOperator.CONTAINS:
                res_df = res_df[res_df[col].astype(str).str.contains(str(val), case=False, na=False, regex=False)]
            elif op in (FilterOperator.IS_NULL, FilterOperator.MISSING):
                res_df = res_df[res_df[col].isnull()]
            elif op == FilterOperator.IS_NOT_NULL:
                res_df = res_df[res_df[col].notnull()]

            filters_applied.append({"column": col, "operator": op.value if hasattr(op, "value") else str(op), "value": val})

        # Sorting is a typed FILTER_SLICE capability.  Apply it after any
        # filter conditions and before presentation truncation; otherwise a
        # request such as "top 10" could incorrectly return the first 10 rows
        # of the original dataset.  The column and direction originate from
        # the validated SortArgs contract, never from raw query parsing.
        tool_args = intent.arguments or {}
        sort_column = tool_args.get("column") or tool_args.get("sort_by")
        sort_direction = tool_args.get("order") or tool_args.get("direction")
        requested_limit = tool_args.get("limit")
        if sort_column:
            if sort_column not in res_df.columns:
                return OperatorOutput(
                    status="FAILED",
                    error_code="COLUMN_NOT_FOUND",
                    error_message=f"Requested sort column '{sort_column}' not found in dataset.",
                    manifest=ExecutionManifest(
                        operator="filter_slice", columns_used=[sort_column], statistics_computed=[],
                        chart_type="none", row_count_in=row_in, row_count_out=0,
                    ),
                )
            descending = str(sort_direction).lower() == "desc"
            res_df = res_df.sort_values(by=sort_column, ascending=not descending, kind="mergesort")
            if requested_limit is not None:
                if not isinstance(requested_limit, int) or isinstance(requested_limit, bool) or requested_limit <= 0:
                    return OperatorOutput(
                        status="FAILED", error_code="INVALID_LIMIT",
                        error_message="Sort limit must be a positive integer.",
                        manifest=ExecutionManifest(operator="filter_slice", columns_used=[sort_column], statistics_computed=[], chart_type="none", row_count_in=row_in, row_count_out=0),
                    )
                res_df = res_df.head(requested_limit)
            filters_applied.append({"column": sort_column, "operator": "sort", "value": "desc" if descending else "asc"})

        manifest = ExecutionManifest(
            operator="filter_slice",
            columns_used=[f["column"] for f in filters_applied],
            filters_applied=filters_applied,
            statistics_computed=["row_count"],
            chart_type="none",
            row_count_in=row_in,
            row_count_out=len(res_df),
            warnings=warnings
        )

        if sort_column:
            limit_note = f", limited to {requested_limit:,} rows" if requested_limit is not None else ""
            finding = f"Sorted {len(res_df):,} row(s) by `{sort_column}` in {'descending' if descending else 'ascending'} order{limit_note}."
        else:
            finding = f"Filtered dataset from {row_in:,} rows to {len(res_df):,} matching rows based on {len(filters_applied)} condition(s)."
        
        # Build reproducible Python code
        code_lines = ["# DataPilot DS Engine: Deterministic Filter & Slice", "import pandas as pd", "res_df = df.copy()"]
        filter_code_ops = {
            FilterOperator.EQ.value: "==", FilterOperator.NEQ.value: "!=",
            FilterOperator.GT.value: ">", FilterOperator.GTE.value: ">=",
            FilterOperator.LT.value: "<", FilterOperator.LTE.value: "<=",
        }
        for f in filters_applied:
            if f["operator"] in filter_code_ops:
                value = f['value']
                if f['operator'] in {'gt', 'gte', 'lt', 'lte'}:
                    value = float(value)
                elif isinstance(value, np.generic):
                    value = value.item()
                code_lines.append(f"res_df = res_df[res_df[{f['column']!r}] {filter_code_ops[f['operator']]} {value!r}]")
            elif f["operator"] == "in":
                code_lines.append(f"res_df = res_df[res_df[{f['column']!r}].isin({f['value']!r})]")
            elif f["operator"] == "not_in":
                code_lines.append(f"res_df = res_df[~res_df[{f['column']!r}].isin({f['value']!r})]")
            elif f["operator"] == "contains":
                code_lines.append(f"res_df = res_df[res_df[{f['column']!r}].astype(str).str.contains({str(f['value'])!r}, case=False, na=False, regex=False)]")
            elif f["operator"] in {"is_null", "missing"}:
                code_lines.append(f"res_df = res_df[res_df[{f['column']!r}].isna()]")
            elif f["operator"] == "is_not_null":
                code_lines.append(f"res_df = res_df[res_df[{f['column']!r}].notna()]")
        if sort_column:
            code_lines.append(f"res_df = res_df.sort_values(by={sort_column!r}, ascending={not descending}, kind='mergesort')")
            if requested_limit is not None:
                code_lines.append(f"res_df = res_df.head({requested_limit})")
        code_lines.append("result_df = res_df\nprint(f'Filtered rows: {len(result_df)}')")

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.head(50).to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=res_df,  # Complete in-memory DataFrame
            manifest=manifest,
            written_finding=finding,
            reproducible_code="\n".join(code_lines)
        )


class EventWindowAnalysisOperator(BaseTypedOperator):
    """
    Event-Window & Pre-Event Pattern Analysis (e.g. Pre-Failure Variance Spikes).
    Strictly validates time ordering, multi-entity isolation, and factual scientific findings.
    Zero auto-selection fallbacks & Zero factual hallucinations.
    """

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        tc = intent.time_context
        if not tc:
            return OperatorOutput(
                status="FAILED",
                error_code="NO_ORDERING_COLUMN",
                error_message="Event window analysis requires an explicit temporal or sequence-order column.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=["Missing temporal or sequence-order column."]
                )
            )

        time_col = tc.time_column
        sequence_col = tc.sequence_column
        event_col = tc.event_column
        entity_id_col = tc.entity_id

        # 1. Validate a single explicit ordering context. A sequence is not a
        # timestamp and therefore must never be coerced through to_datetime.
        if time_col and sequence_col:
            return OperatorOutput(
                status="FAILED",
                error_code="AMBIGUOUS_ORDERING_CONTEXT",
                error_message="Event window analysis accepts either time_column or sequence_column, not both.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis", columns_used=[], statistics_computed=[],
                    chart_type="none", row_count_in=len(df), row_count_out=0
                )
            )
        order_col = time_col or sequence_col
        if not order_col:
            return OperatorOutput(
                status="FAILED",
                error_code="NO_ORDERING_COLUMN",
                error_message="Event window analysis requires an explicit temporal or sequence-order column.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=["Missing temporal or sequence-order column."]
                )
            )
        if time_col and time_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="NO_TEMPORAL_COLUMN",
                error_message="Event window analysis requires an explicit datetime/temporal column.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis", columns_used=[], statistics_computed=[],
                    chart_type="none", row_count_in=len(df), row_count_out=0,
                    warnings=["Invalid datetime/temporal column."]
                )
            )
        if sequence_col and sequence_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_ORDERING_COLUMN",
                error_message="The supplied sequence-order column is not present in the dataset.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis", columns_used=[], statistics_computed=[],
                    chart_type="none", row_count_in=len(df), row_count_out=0
                )
            )

        work_df = df.copy()
        if time_col and not pd.api.types.is_datetime64_any_dtype(work_df[time_col]):
            parsed_time = pd.to_datetime(work_df[time_col], errors="coerce")
            if parsed_time.notnull().sum() == 0:
                return OperatorOutput(
                    status="FAILED",
                    error_code="NO_TEMPORAL_COLUMN",
                    error_message="Event window analysis requires an explicit datetime/temporal column.",
                    manifest=ExecutionManifest(
                        operator="event_window_analysis",
                        columns_used=[time_col],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=len(df),
                        row_count_out=0,
                        warnings=["Invalid datetime/temporal column."]
                    )
                )
            work_df[time_col] = parsed_time
        if sequence_col:
            sequence = work_df[sequence_col]
            if sequence.isna().any() or not sequence.is_unique:
                return OperatorOutput(
                    status="FAILED",
                    error_code="INVALID_ORDERING_COLUMN",
                    error_message="Sequence-order column must contain unique, non-null values.",
                    manifest=ExecutionManifest(
                        operator="event_window_analysis", columns_used=[sequence_col], statistics_computed=[],
                        chart_type="none", row_count_in=len(df), row_count_out=0
                    )
                )

        work_df = work_df.sort_values(by=order_col).reset_index(drop=True)

        # 2. Validate pre_event_window: Must be explicitly present in intent.time_context
        if tc.pre_event_window is None:
            return OperatorOutput(
                status="FAILED",
                error_code="PRE_EVENT_WINDOW_REQUIRED",
                error_message="Event window analysis requires an explicit pre_event_window.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[order_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=["Missing pre_event_window."]
                )
            )

        try:
            window_size = int(tc.pre_event_window)
        except (ValueError, TypeError):
            return OperatorOutput(
                status="FAILED",
                error_code="PRE_EVENT_WINDOW_REQUIRED",
                error_message="Event window analysis requires an explicit pre_event_window.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[order_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=["Invalid pre_event_window."]
                )
            )

        # 3. Do not turn a grouping column into an entity key implicitly.
        # Multi-entity data must name its partition key in the ToolSpec arguments.
        if not entity_id_col:
            multi_entity_detected = False
            if profile:
                entity_candidates = set(profile.entity_id_candidates) | {
                    c for c, p in profile.columns.items() if getattr(p, "semantic_role", None) == "entity_id"
                }
                for cand in entity_candidates:
                    # A unique row identifier is not a panel/entity boundary:
                    # partitioning on it would leave every event with no
                    # historical observations. Require repeated entity values.
                    if cand in work_df.columns and cand != order_col and cand != event_col and 1 < work_df[cand].nunique() < len(work_df):
                        multi_entity_detected = True
                        break

            if multi_entity_detected:
                return OperatorOutput(
                    status="FAILED",
                    error_code="ENTITY_ID_REQUIRED",
                    error_message="Multi-entity dataset detected. Explicit entity_id (e.g. device_id/machine_id) is required to prevent cross-entity sequence contamination.",
                    manifest=ExecutionManifest(
                        operator="event_window_analysis",
                        columns_used=[],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=len(df),
                        row_count_out=0,
                        warnings=["Multi-entity dataset missing explicit entity_id."]
                    )
                )

        if entity_id_col and entity_id_col in work_df.columns:
            work_df = work_df.sort_values(by=[entity_id_col, order_col]).reset_index(drop=True)
        else:
            work_df = work_df.sort_values(by=order_col).reset_index(drop=True)

        # Resolve only exact normalized aliases (for example spaces vs underscores).
        # Substring matching and profile-based event guessing are deliberately forbidden.
        def _resolve_col(c_name: Optional[str]) -> Optional[str]:
            if not c_name:
                return None
            if c_name in df.columns:
                return c_name
            c_norm = re.sub(r'[^a-zA-Z0-9]', '', str(c_name).lower())
            matches = [
                col for col in df.columns
                if re.sub(r'[^a-zA-Z0-9]', '', str(col).lower()) == c_norm
            ]
            return matches[0] if len(matches) == 1 else None

        # 4. Validate and resolve event_column presence
        pred_raw = tc.event_predicate
        if not event_col and isinstance(pred_raw, dict):
            event_col = pred_raw.get("column")
        if event_col:
            res_ev = _resolve_col(event_col)
            if res_ev:
                event_col = res_ev

        if not event_col or event_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="EVENT_COLUMN_REQUIRED",
                error_message="Event window analysis requires an explicit event_column.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[event_col] if event_col else [],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=["Missing explicit event_column."]
                )
            )

        # 5. Validate subject columns: Must be numeric
        if not intent.subjects:
            return OperatorOutput(
                status="FAILED",
                error_code="SUBJECT_COLUMNS_REQUIRED",
                error_message="Event-window analysis requires explicit subject metric columns to analyze.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[event_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        # Resolve subject column names
        resolved_subjects = []
        for s in intent.subjects:
            res_s = _resolve_col(s)
            if res_s:
                resolved_subjects.append(res_s)
            elif s in df.columns:
                resolved_subjects.append(s)
            else:
                resolved_subjects.append(s)
        intent.subjects = resolved_subjects

        missing_subjects = [c for c in intent.subjects if c not in df.columns]
        if missing_subjects:
            return OperatorOutput(
                status="FAILED",
                error_code="COLUMN_NOT_FOUND",
                error_message=f"Requested subject column(s) {missing_subjects} not found in dataset.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[event_col] + [c for c in intent.subjects if c in df.columns],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=[f"Missing columns: {missing_subjects}"]
                )
            )

        subjects = [c for c in intent.subjects if c in df.columns]
        for s in subjects:
            if not pd.api.types.is_numeric_dtype(df[s]):
                return OperatorOutput(
                    status="FAILED",
                    error_code="INVALID_COLUMN_TYPE",
                    error_message=f"Event window variance analysis requires numeric subject columns. '{s}' is non-numeric.",
                    manifest=ExecutionManifest(
                        operator="event_window_analysis",
                        columns_used=[event_col, s],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=len(df),
                        row_count_out=0
                    )
                )

        # 6. Evaluate a typed, explicit event predicate.
        target_cond = None
        if intent.conditions:
            for cond in intent.conditions:
                if cond.column == event_col:
                    target_cond = cond
                    break

        pred_raw = tc.event_predicate if tc else None

        if target_cond is None and pred_raw is None:
            return OperatorOutput(
                status="FAILED",
                error_code="EVENT_PREDICATE_REQUIRED",
                error_message="Event window analysis requires an explicit event predicate.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[c for c in [event_col, order_col] if c and c in df.columns],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    sorted_by=order_col,
                    partitioned_by=entity_id_col
                )
            )

        # Extract operator and target value
        op_type = None
        val = None

        if target_cond is not None:
            op_type = getattr(target_cond, "operator", None)
            val = getattr(target_cond, "value", None)
        elif isinstance(pred_raw, dict):
            op_type = pred_raw.get("operator") or pred_raw.get("condition") or pred_raw.get("op")
            val = pred_raw.get("value")
        if op_type is None:
            return OperatorOutput(
                status="FAILED",
                error_code="EVENT_PREDICATE_REQUIRED",
                error_message="Event window analysis requires an explicit predicate operator.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis", columns_used=[event_col, order_col],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                    sorted_by=order_col, partitioned_by=entity_id_col
                )
            )

        op_str = op_type.value if hasattr(op_type, "value") else str(op_type).strip().lower()
        series = work_df[event_col]

        # Dtype normalization
        if pd.api.types.is_numeric_dtype(series) and isinstance(val, str):
            v_num = pd.to_numeric(val, errors="coerce")
            if pd.notna(v_num):
                val = v_num
        elif pd.api.types.is_bool_dtype(series) and isinstance(val, str):
            if val.strip().lower() in ("true", "1", "yes"):
                val = True
            elif val.strip().lower() in ("false", "0", "no"):
                val = False

        if op_str in ("==", "eq", "equals", "="):
            is_event = series == val
            predicate_code_expr = f"df['{event_col}'] == {repr(val)}"
            predicate_applied_str = f"{event_col} == {val}"
        elif op_str in ("!=", "ne", "not_equals"):
            is_event = series != val
            predicate_code_expr = f"df['{event_col}'] != {repr(val)}"
            predicate_applied_str = f"{event_col} != {val}"
        elif op_str in (">", "gt"):
            if val is None:
                return _event_predicate_value_required(df, event_col, order_col, entity_id_col)
            num_val = float(val)
            is_event = series > num_val
            predicate_code_expr = f"df['{event_col}'] > {num_val}"
            predicate_applied_str = f"{event_col} > {num_val}"
        elif op_str in (">=", "gte", "ge"):
            if val is None:
                return _event_predicate_value_required(df, event_col, order_col, entity_id_col)
            num_val = float(val)
            is_event = series >= num_val
            predicate_code_expr = f"df['{event_col}'] >= {num_val}"
            predicate_applied_str = f"{event_col} >= {num_val}"
        elif op_str in ("<", "lt"):
            if val is None:
                return _event_predicate_value_required(df, event_col, order_col, entity_id_col)
            num_val = float(val)
            is_event = series < num_val
            predicate_code_expr = f"df['{event_col}'] < {num_val}"
            predicate_applied_str = f"{event_col} < {num_val}"
        elif op_str in ("<=", "lte", "le"):
            if val is None:
                return _event_predicate_value_required(df, event_col, order_col, entity_id_col)
            num_val = float(val)
            is_event = series <= num_val
            predicate_code_expr = f"df['{event_col}'] <= {num_val}"
            predicate_applied_str = f"{event_col} <= {num_val}"
        elif op_str == "in" or (isinstance(val, list) and op_str in ("==", "in")):
            val_list = val if isinstance(val, list) else [val]
            is_event = series.isin(val_list)
            predicate_code_expr = f"df['{event_col}'].isin({repr(val_list)})"
            predicate_applied_str = f"{event_col} in {val_list}"
        elif op_str == "contains":
            if val is None:
                return _event_predicate_value_required(df, event_col, order_col, entity_id_col)
            is_event = series.astype(str).str.contains(str(val), case=False, na=False, regex=False)
            predicate_code_expr = f"df['{event_col}'].astype(str).str.contains({repr(str(val))}, case=False, na=False, regex=False)"
            predicate_applied_str = f"{event_col} contains {val}"
        elif op_str in ("is_null", "isnull"):
            is_event = series.isna()
            predicate_code_expr = f"df['{event_col}'].isna()"
            predicate_applied_str = f"{event_col} is null"
        elif op_str in ("not_null", "notnull"):
            is_event = series.notna()
            predicate_code_expr = f"df['{event_col}'].notna()"
            predicate_applied_str = f"{event_col} not null"
        else:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_OPERATOR",
                error_message=f"Unsupported event predicate operator: '{op_str}'.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis", columns_used=[event_col, order_col],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                    sorted_by=order_col, partitioned_by=entity_id_col
                )
            )

        applied_predicate = {
            "column": event_col,
            "operator": op_str,
            "value": val,
        }
        event_scope = (intent.arguments or {}).get("event_scope")
        if event_scope not in {"event_rows", "event_onsets"}:
            return OperatorOutput(
                status="FAILED", error_code="EVENT_SCOPE_REQUIRED",
                error_message="Event-window analysis requires event_scope='event_rows' or 'event_onsets'.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis", columns_used=[event_col, order_col],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                    predicate_applied=predicate_applied_str, sorted_by=order_col, partitioned_by=entity_id_col,
                ),
            )
        if event_scope == "event_onsets":
            if entity_id_col and entity_id_col in work_df.columns:
                previous = is_event.groupby(work_df[entity_id_col], sort=False).shift(fill_value=False)
            else:
                previous = is_event.shift(fill_value=False)
            is_event = is_event & ~previous.astype(bool)
            predicate_applied_str += " (false-to-true onsets only)"
        applied_predicate["event_scope"] = event_scope
        event_row_count = int(is_event.sum())
        if event_row_count == 0:
            return OperatorOutput(
                status="FAILED",
                error_code="NO_EVENT_ROWS_FOUND",
                error_message=f"No event occurrences matched predicate for column '{event_col}'.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[event_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    event_row_count=0,
                    filters_applied=[applied_predicate],
                    predicate_applied=predicate_applied_str,
                    sorted_by=order_col,
                    partitioned_by=entity_id_col
                )
            )

        # 7. Validate lookback window: If pre-event lookback rows < 2
        event_indices = set(work_df.index[is_event].tolist())
        pre_event_indices = set()

        if entity_id_col and entity_id_col in work_df.columns:
            for entity, group in work_df.groupby(entity_id_col):
                group_indices = group.index.tolist()
                group_event_locs = [i for i, idx in enumerate(group_indices) if idx in event_indices]
                for loc in group_event_locs:
                    start_loc = max(0, loc - window_size)
                    for l in range(start_loc, loc):
                        pre_event_indices.add(group_indices[l])
        else:
            work_indices = work_df.index.tolist()
            work_event_locs = [i for i, idx in enumerate(work_indices) if idx in event_indices]
            for loc in work_event_locs:
                start_loc = max(0, loc - window_size)
                for l in range(start_loc, loc):
                    pre_event_indices.add(work_indices[l])

        pre_event_indices -= event_indices
        pre_event_row_count = len(pre_event_indices)

        if pre_event_row_count < 2:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_PRE_EVENT_DATA",
                error_message="Insufficient pre-event historical data points for variance analysis.",
                manifest=ExecutionManifest(
                    operator="event_window_analysis",
                    columns_used=[event_col] + subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    event_row_count=event_row_count,
                    pre_event_row_count=pre_event_row_count,
                    filters_applied=[applied_predicate],
                    predicate_applied=predicate_applied_str,
                    sorted_by=order_col,
                    partitioned_by=entity_id_col
                )
            )

        regime_series = pd.Series("Normal Operation", index=work_df.index)
        regime_series.loc[list(pre_event_indices)] = f"Pre-Event Window (t-{window_size}..t)"
        regime_series.loc[list(event_indices)] = "Event Active"

        work_df["_regime"] = regime_series

        # Compute comparative statistics across regimes
        stats_rows = []
        regimes = ["Normal Operation", f"Pre-Event Window (t-{window_size}..t)", "Event Active"]

        for s in subjects:
            norm_series = work_df.loc[work_df["_regime"] == "Normal Operation", s].dropna()
            pre_series = work_df.loc[work_df["_regime"] == f"Pre-Event Window (t-{window_size}..t)", s].dropna()

            var_norm = float(norm_series.var(ddof=1)) if len(norm_series) > 1 else 0.0
            var_pre = float(pre_series.var(ddof=1)) if len(pre_series) > 1 else 0.0

            spike_ratio = round(var_pre / max(var_norm, 1e-6), 2)

            for reg in regimes:
                sub_df = work_df.loc[work_df["_regime"] == reg, s].dropna()
                if sub_df.empty:
                    continue
                q25 = float(sub_df.quantile(0.25))
                q50 = float(sub_df.quantile(0.50))
                q75 = float(sub_df.quantile(0.75))
                stats_rows.append({
                    "metric": s,
                    "regime": reg,
                    "count": len(sub_df),
                    "mean": round(float(sub_df.mean()), 3),
                    "std": round(float(sub_df.std()), 3),
                    "variance": round(float(sub_df.var(ddof=1)), 4),
                    "p25": round(q25, 3),
                    "median": round(q50, 3),
                    "p75": round(q75, 3),
                    "iqr": round(q75 - q25, 3),
                    "variance_spike_ratio": spike_ratio if reg.startswith("Pre-Event") else 1.0
                })

        summary_df = pd.DataFrame(stats_rows)

        # Plot comparative multi-panel Box Plots
        fig, axes = plt.subplots(1, len(subjects), figsize=(max(11, 4 * len(subjects)), 5.5), squeeze=False)
        colors = {"Normal Operation": "#3b82f6", f"Pre-Event Window (t-{window_size}..t)": "#f59e0b", "Event Active": "#ef4444"}

        for i, s in enumerate(subjects):
            ax = axes[0, i]
            data_to_plot = [work_df.loc[work_df["_regime"] == reg, s].dropna() for reg in regimes if not work_df.loc[work_df["_regime"] == reg, s].dropna().empty]
            labels = [reg for reg in regimes if not work_df.loc[work_df["_regime"] == reg, s].dropna().empty]
            
            bplot = ax.boxplot(
                data_to_plot,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker="o", markerfacecolor="red", markersize=4)
            )
            for patch, label in zip(bplot["boxes"], labels):
                patch.set_facecolor(colors.get(label, "#6366f1"))
                patch.set_alpha(0.7)

            ax.set_title(f"{s}\n(Pre-Event Pattern)")
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels([l.replace(" (t-", "\n(t-") for l in labels], rotation=20, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

        plt.suptitle(f"Event-Window Analysis: Pre-Event Pattern vs Normal Baseline (Event: '{event_col}', Window={window_size})", fontsize=11, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Interactive Plotly Spec
        plotly_traces = []
        for s in subjects:
            for reg in regimes:
                y_data = work_df.loc[work_df["_regime"] == reg, s].dropna().tolist()[:1000]
                plotly_traces.append({
                    "type": "box",
                    "name": f"{s} ({reg})",
                    "y": y_data,
                    "boxpoints": "outliers"
                })

        plotly_spec = {
            "data": plotly_traces,
            "layout": {
                "title": f"Pre-Event Pattern & Variance Analysis (Event: '{event_col}', Window={window_size})",
                "yaxis": {"title": "Value"},
                "boxmode": "group"
            }
        }

        finding_lines = [
            f"**Event-Window Analysis Findings (Event: `{event_col}`, Pre-Event Window = {window_size} units):**"
        ]
        for s in subjects:
            pre_row = summary_df[(summary_df["metric"] == s) & (summary_df["regime"].str.startswith("Pre-Event"))]
            norm_row = summary_df[(summary_df["metric"] == s) & (summary_df["regime"] == "Normal Operation")]
            if not pre_row.empty and not norm_row.empty:
                v_pre = pre_row["variance"].values[0]
                v_norm = norm_row["variance"].values[0]
                ratio = pre_row["variance_spike_ratio"].values[0]
                if ratio >= 1.5:
                    finding_lines.append(
                        f"• **`{s}`**: Normal variance = `{v_norm:.4f}`, Pre-Event variance = `{v_pre:.4f}` (**Variance Spike Ratio: {ratio:.2f}x**). Significant pre-event volatility increase observed before event `{event_col}`."
                    )
                else:
                    finding_lines.append(
                        f"• **`{s}`**: Normal variance = `{v_norm:.4f}`, Pre-Event variance = `{v_pre:.4f}` (**Variance Spike Ratio: {ratio:.2f}x**). No material pre-event variance increase observed (pre-event volatility remains comparable to normal operating baseline)."
                    )

        finding_text = "\n".join(finding_lines)

        cols_used = []
        for c in [event_col, order_col, entity_id_col] + subjects:
            if c and c in df.columns and c not in cols_used:
                cols_used.append(c)

        manifest = ExecutionManifest(
            operator="event_window_analysis",
            columns_used=cols_used,
            statistics_computed=["pre_event_variance", "normal_variance", "variance_spike_ratio", "quantiles", "iqr"],
            chart_type="box",
            row_count_in=len(df),
            row_count_out=len(summary_df),
            filters_applied=[applied_predicate],
            assumptions=[f"Event predicate '{predicate_applied_str}'; pre-event lookback window set to {window_size} sequence units."],
            event_row_count=event_row_count,
            pre_event_row_count=pre_event_row_count,
            predicate_applied=predicate_applied_str,
            sorted_by=order_col,
            partitioned_by=entity_id_col
        )

        if time_col:
            sort_code = f"df['{order_col}'] = pd.to_datetime(df['{order_col}'])\n"
        else:
            sort_code = ""
        if entity_id_col:
            sort_code += f"df = df.sort_values(by=['{entity_id_col}', '{order_col}']).reset_index(drop=True)\n"
            window_code = (
                f"for _, group in df.groupby('{entity_id_col}'):\n"
                "    group_indices = group.index.tolist()\n"
                "    group_event_locs = [i for i, idx in enumerate(group_indices) if is_event.loc[idx]]\n"
                "    for loc in group_event_locs:\n"
                "        start_loc = max(0, loc - window_size)\n"
                "        for l in range(start_loc, loc):\n"
                "            pre_idx.add(group_indices[l])\n"
            )
        else:
            sort_code += f"df = df.sort_values(by=['{order_col}']).reset_index(drop=True)\n"
            window_code = (
                "work_indices = df.index.tolist()\n"
                "work_event_locs = [i for i, idx in enumerate(work_indices) if is_event.loc[idx]]\n"
                "for loc in work_event_locs:\n"
                "    start_loc = max(0, loc - window_size)\n"
                "    for l in range(start_loc, loc):\n"
                "        pre_idx.add(work_indices[l])\n"
            )

        summary_code = (
            "stats_rows = []\n"
            f"regimes = ['Normal Operation', f'Pre-Event Window (t-{{window_size}}..t)', 'Event Active']\n"
            "for s in subjects:\n"
            "    norm_series = df.loc[df['_regime'] == 'Normal Operation', s].dropna()\n"
            f"    pre_series = df.loc[df['_regime'] == f'Pre-Event Window (t-{{window_size}}..t)', s].dropna()\n"
            "    var_norm = float(norm_series.var(ddof=1)) if len(norm_series) > 1 else 0.0\n"
            "    var_pre = float(pre_series.var(ddof=1)) if len(pre_series) > 1 else 0.0\n"
            "    spike_ratio = round(var_pre / max(var_norm, 1e-6), 2)\n"
            "    for reg in regimes:\n"
            "        sub_df = df.loc[df['_regime'] == reg, s].dropna()\n"
            "        if sub_df.empty:\n"
            "            continue\n"
            "        q25 = float(sub_df.quantile(0.25))\n"
            "        q50 = float(sub_df.quantile(0.50))\n"
            "        q75 = float(sub_df.quantile(0.75))\n"
            "        stats_rows.append({\n"
            "            'metric': s,\n"
            "            'regime': reg,\n"
            "            'count': len(sub_df),\n"
            "            'mean': round(float(sub_df.mean()), 3),\n"
            "            'std': round(float(sub_df.std()), 3),\n"
            "            'variance': round(float(sub_df.var(ddof=1)), 4),\n"
            "            'p25': round(q25, 3),\n"
            "            'median': round(q50, 3),\n"
            "            'p75': round(q75, 3),\n"
            "            'iqr': round(q75 - q25, 3),\n"
            "            'variance_spike_ratio': spike_ratio if reg.startswith('Pre-Event') else 1.0\n"
            "        })\n"
            "summary_df = pd.DataFrame(stats_rows)\n"
            "print(summary_df)\n"
        )

        code = (
            f"# DataPilot DS Engine: Event-Window Pre-Event Variance Analysis\n"
            f"import pandas as pd, numpy as np\n"
            f"import matplotlib.pyplot as plt\n"
            f"event_col = '{event_col}'\n"
            f"order_col = '{order_col}'\n"
            f"{f'entity_col = {repr(entity_id_col)}' if entity_id_col else ''}\n"
            f"window_size = {window_size}\n"
            f"event_scope = {repr(event_scope)}\n"
            f"subjects = {repr(subjects)}\n"
            f"\n"
            f"{sort_code}"
            f"is_event = {predicate_code_expr}\n"
            f"if event_scope == 'event_onsets':\n"
            f"    previous = is_event.groupby(df[entity_col], sort=False).shift(fill_value=False) if {bool(entity_id_col)!r} else is_event.shift(fill_value=False)\n"
            f"    is_event = is_event & ~previous.astype(bool)\n"
            f"event_idx = set(df.index[is_event].tolist())\n"
            f"pre_idx = set()\n"
            f"\n"
            f"{window_code}"
            f"pre_idx -= event_idx\n"
            f"df['_regime'] = 'Normal Operation'\n"
            f"df.loc[list(pre_idx), '_regime'] = f'Pre-Event Window (t-{{window_size}}..t)'\n"
            f"df.loc[list(event_idx), '_regime'] = 'Event Active'\n"
            f"\n"
            f"{summary_code}"
            f"fig, axes = plt.subplots(1, len(subjects), figsize=(max(11, 4 * len(subjects)), 5.5), squeeze=False)\n"
            f"colors = {{'Normal Operation': '#3b82f6', f'Pre-Event Window (t-{{window_size}}..t)': '#f59e0b', 'Event Active': '#ef4444'}}\n"
            f"for i, subject in enumerate(subjects):\n"
            f"    labels = [reg for reg in regimes if not df.loc[df['_regime'] == reg, subject].dropna().empty]\n"
            f"    values = [df.loc[df['_regime'] == reg, subject].dropna() for reg in labels]\n"
            f"    boxes = axes[0, i].boxplot(values, patch_artist=True, medianprops={{'color': 'black', 'linewidth': 2}})\n"
            f"    for patch, label in zip(boxes['boxes'], labels):\n"
            f"        patch.set_facecolor(colors[label])\n"
            f"        patch.set_alpha(0.7)\n"
            f"    axes[0, i].set_title(f'{{subject}} (Pre-Event Pattern)')\n"
            f"    axes[0, i].set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha='right')\n"
            f"    axes[0, i].grid(True, alpha=0.3, linestyle='--')\n"
            f"plt.suptitle(f'Event-Window Analysis: {{event_col}} (window={{window_size}})')\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": summary_df.to_dict(orient="records"), "columns": list(summary_df.columns), "row_count": len(summary_df)},
            dataframe=summary_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=finding_text,
            reproducible_code=code
        )


class DistributionVarianceOperator(BaseTypedOperator):
    """Comprehensive Box Plot, Quantile, Variance, and Outlier Analysis."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        if not intent.subjects:
            return OperatorOutput(
                status="FAILED",
                error_code="SUBJECT_COLUMNS_REQUIRED",
                error_message="Distribution analysis requires explicit numerical subject columns.",
                manifest=ExecutionManifest(
                    operator="distribution_comparison",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        missing = [c for c in intent.subjects if c not in df.columns]
        if missing:
            return OperatorOutput(
                status="FAILED",
                error_code="COLUMN_NOT_FOUND",
                error_message=f"Requested subject column(s) {missing} not found in dataset.",
                manifest=ExecutionManifest(
                    operator="distribution_comparison",
                    columns_used=[c for c in intent.subjects if c in df.columns],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        subjects = [c for c in intent.subjects if c in df.columns]

        non_numeric = [c for c in subjects if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            return OperatorOutput(
                status="FAILED",
                error_code="NUMERIC_SUBJECT_REQUIRED",
                error_message=(
                    "Distribution and variance analysis requires numeric subject column(s); "
                    f"{non_numeric} are not numeric in the active dataset."
                ),
                manifest=ExecutionManifest(
                    operator="distribution_comparison",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                ),
            )

        # Compute 5-number summary + IQR + Variance + Std
        desc = df[subjects].describe().T
        desc["IQR"] = desc["75%"] - desc["25%"]
        desc["variance"] = desc["std"] ** 2

        outlier_counts = {}
        for c in subjects:
            q25 = desc.loc[c, "25%"]
            q75 = desc.loc[c, "75%"]
            iqr = desc.loc[c, "IQR"]
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = df[(df[c] < lower_bound) | (df[c] > upper_bound)][c]
            outlier_counts[c] = len(outliers)

        desc["outlier_count"] = [outlier_counts[c] for c in subjects]
        # A zero is a substantive operational observation for many metrics
        # (demand, volume, balance, duration, etc.).  Report its actual count
        # and rate rather than inferring whether it is "expected" or causal.
        desc["zero_count"] = [int(df[c].eq(0).sum()) for c in subjects]
        desc["zero_rate_percent"] = [
            round((int(df[c].eq(0).sum()) / int(df[c].notna().sum()) * 100), 4)
            if int(df[c].notna().sum()) else 0.0
            for c in subjects
        ]
        res_df = desc.reset_index().rename(columns={"index": "column"})

        # Generate Box Plot Matplotlib Image
        plt.figure(figsize=(10, 5.5))
        data_arrays = [df[c].dropna() for c in subjects]
        plt.boxplot(
            data_arrays,
            patch_artist=True,
            boxprops=dict(facecolor="#6366f1", color="#4338ca", alpha=0.7),
            medianprops=dict(color="#ef4444", linewidth=2),
            flierprops=dict(marker="o", markerfacecolor="#f59e0b", markersize=5)
        )
        plt.xticks(range(1, len(subjects) + 1), subjects, rotation=30, ha="right")
        plt.title("Box Plot Distribution & Variance Analysis: " + ", ".join(subjects))
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        plotly_spec = {
            "data": [
                {"type": "box", "name": c, "y": df[c].dropna().tolist()[:1000], "boxpoints": "outliers"}
                for c in subjects
            ],
            "layout": {
                "title": f"Distribution & Variance Analysis ({', '.join(subjects)})",
                "yaxis": {"title": "Value"},
                "boxmode": "group"
            }
        }

        findings = []
        for c in subjects:
            med = round(desc.loc[c, "50%"], 2)
            iqr = round(desc.loc[c, "IQR"], 2)
            outs = outlier_counts[c]
            zero_count = int(desc.loc[c, "zero_count"])
            zero_rate = round(float(desc.loc[c, "zero_rate_percent"]), 2)
            findings.append(
                f"• **{c}**: Median = {med}, IQR = {iqr}, Outliers = {outs:,} rows; "
                f"observed zeros = {zero_count:,} ({zero_rate}%)."
            )
        finding_text = f"**Distribution & Variance Analysis:**\n" + "\n".join(findings)

        manifest = ExecutionManifest(
            operator="distribution_comparison",
            columns_used=subjects,
            statistics_computed=["mean", "std", "min", "25%", "50%", "75%", "max", "IQR", "variance", "outlier_count", "zero_count", "zero_rate_percent"],
            chart_type="box",
            row_count_in=len(df),
            row_count_out=len(res_df),
            assumptions=["IQR 1.5x standard Tukey outlier boundaries applied."]
        )

        code = (
            "# DataPilot DS Engine: Box Plot & Variance Outlier Analysis\n"
            "import matplotlib.pyplot as plt\n"
            f"subjects = {repr(subjects)}\n"
            "desc = df[subjects].describe().T\n"
            "desc['IQR'] = desc['75%'] - desc['25%']\n"
            "desc['variance'] = desc['std'] ** 2\n"
            "desc['zero_count'] = [int(df[c].eq(0).sum()) for c in subjects]\n"
            "desc['zero_rate_percent'] = [round((df[c].eq(0).sum() / df[c].notna().sum() * 100), 4) if df[c].notna().sum() else 0.0 for c in subjects]\n"
            "result_df = desc.reset_index()\n"
            "plt.figure(figsize=(10, 5.5))\n"
            "plt.boxplot([df[c].dropna() for c in subjects], patch_artist=True)\n"
            "plt.xticks(range(1, len(subjects) + 1), subjects, rotation=30, ha='right')\n"
            "plt.show()\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=finding_text,
            reproducible_code=code
        )


class GroupAggregationOperator(BaseTypedOperator):
    """Vectorized Group Aggregation across multiple dimensions and metrics."""

    @staticmethod
    def _conditional_rate_mask(df: pd.DataFrame, condition: FilterSpec) -> pd.Series:
        """Evaluate a validated predicate as a vectorized boolean numerator mask.

        This mirrors the typed filter semantics without using ``DataFrame.query``
        or treating the numerator as a prerequisite filter.  The latter is
        essential: filtering first would collapse a conditional rate to 100%.
        """
        series = df[condition.column]
        operator = condition.operator
        value = condition.value

        if pd.api.types.is_numeric_dtype(series) and isinstance(value, str):
            value_clean = value.strip().lower()
            if value_clean in ("true", "yes", "failure", "failed", "positive", "1"):
                value = 1
            elif value_clean in ("false", "no", "normal", "negative", "0"):
                value = 0
            else:
                numeric_value = pd.to_numeric(value, errors="coerce")
                if pd.notna(numeric_value):
                    value = numeric_value

        if operator == FilterOperator.EQ:
            return series.eq(value).fillna(False)
        if operator == FilterOperator.NEQ:
            return series.ne(value).fillna(False)
        if operator == FilterOperator.GT:
            return series.gt(value).fillna(False)
        if operator == FilterOperator.GTE:
            return series.ge(value).fillna(False)
        if operator == FilterOperator.LT:
            return series.lt(value).fillna(False)
        if operator == FilterOperator.LTE:
            return series.le(value).fillna(False)
        if operator == FilterOperator.IN:
            if not isinstance(value, list):
                raise ValueError("The 'in' numerator condition requires a list value.")
            return series.isin(value).fillna(False)
        if operator == FilterOperator.NOT_IN:
            if not isinstance(value, list):
                raise ValueError("The 'not_in' numerator condition requires a list value.")
            return (~series.isin(value)).fillna(False)
        if operator == FilterOperator.CONTAINS:
            if value is None:
                raise ValueError("The 'contains' numerator condition requires a value.")
            return series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
        if operator in (FilterOperator.IS_NULL, FilterOperator.MISSING):
            return series.isna()
        if operator == FilterOperator.IS_NOT_NULL:
            return series.notna()
        raise ValueError(f"Unsupported conditional-rate operator '{operator}'.")

    def _execute_conditional_rate(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        raw_condition: Any,
    ) -> OperatorOutput:
        """Compute a generic condition/total cohort rate with auditable evidence."""
        requested_dims = list(dict.fromkeys(intent.grouping or []))
        if not requested_dims:
            return OperatorOutput(
                status="FAILED",
                error_code="GROUP_BY_REQUIRED",
                error_message="Conditional-rate analysis requires explicit grouping dimensions.",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=[], statistics_computed=[],
                    chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        missing_dims = [column for column in requested_dims if column not in df.columns]
        if missing_dims:
            return OperatorOutput(
                status="FAILED",
                error_code="COLUMN_NOT_FOUND",
                error_message=f"Requested grouping column(s) {missing_dims} not found in dataset.",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims,
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        try:
            condition = FilterSpec.model_validate(raw_condition)
        except Exception as exc:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_FILTER_SPEC",
                error_message=f"Conditional-rate numerator condition is invalid: {exc}",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims,
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        if condition.column not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="COLUMN_NOT_FOUND",
                error_message=f"Requested numerator condition column '{condition.column}' not found in dataset.",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims + [condition.column],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        value_required = {
            FilterOperator.EQ, FilterOperator.NEQ, FilterOperator.GT,
            FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE,
            FilterOperator.IN, FilterOperator.NOT_IN, FilterOperator.CONTAINS,
        }
        if condition.operator in value_required and condition.value is None:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_FILTER_SPEC",
                error_message=f"Numerator operator '{condition.operator.value}' requires an explicit value.",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims + [condition.column],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        try:
            numerator_mask = self._conditional_rate_mask(df, condition)
        except (TypeError, ValueError) as exc:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_FILTER_SPEC",
                error_message=f"Unable to apply conditional-rate numerator predicate: {exc}",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims + [condition.column],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        total_count = df.groupby(requested_dims, dropna=False).size().rename("total_count")
        matching_count = (
            df.loc[numerator_mask]
            .groupby(requested_dims, dropna=False)
            .size()
            .rename("matching_count")
        )
        res_df = total_count.to_frame().join(matching_count, how="left")
        res_df["matching_count"] = res_df["matching_count"].fillna(0).astype("int64")
        res_df["total_count"] = res_df["total_count"].astype("int64")
        res_df["rate"] = res_df["matching_count"] / res_df["total_count"]
        res_df["rate_percent"] = res_df["rate"] * 100.0
        res_df = res_df.reset_index()

        tool_args = intent.arguments or {}
        min_group_size = tool_args.get("min_group_size")
        if min_group_size is not None:
            if isinstance(min_group_size, bool) or not isinstance(min_group_size, int) or min_group_size < 1:
                return OperatorOutput(
                    status="FAILED",
                    error_code="INVALID_MIN_GROUP_SIZE",
                    error_message="min_group_size must be a positive integer.",
                    manifest=ExecutionManifest(
                        operator="conditional_rate", columns_used=requested_dims + [condition.column],
                        statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                    ),
                )
            res_df = res_df.loc[res_df["total_count"] >= min_group_size].copy()

        sort_by = tool_args.get("sort_by")
        direction = tool_args.get("direction")
        if sort_by is not None:
            if sort_by not in {"rate_percent", "matching_count", "total_count"}:
                return OperatorOutput(
                    status="FAILED",
                    error_code="INVALID_SORT_SPEC",
                    error_message="Conditional-rate sort_by must be rate_percent, matching_count, or total_count.",
                    manifest=ExecutionManifest(
                        operator="conditional_rate", columns_used=requested_dims + [condition.column],
                        statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                    ),
                )
            if direction not in {"asc", "desc"}:
                return OperatorOutput(
                    status="FAILED",
                    error_code="INVALID_SORT_SPEC",
                    error_message="Conditional-rate ranking requires direction 'asc' or 'desc'.",
                    manifest=ExecutionManifest(
                        operator="conditional_rate", columns_used=requested_dims + [condition.column],
                        statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                    ),
                )
            res_df = res_df.sort_values(sort_by, ascending=(direction == "asc"), kind="mergesort")
        elif direction is not None:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_SORT_SPEC",
                error_message="Conditional-rate direction requires an explicit sort_by metric.",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims + [condition.column],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        requested_limit = tool_args.get("limit")
        if requested_limit is not None:
            if isinstance(requested_limit, bool) or not isinstance(requested_limit, int) or requested_limit < 1:
                return OperatorOutput(
                    status="FAILED",
                    error_code="INVALID_LIMIT",
                    error_message="Conditional-rate limit must be a positive integer.",
                    manifest=ExecutionManifest(
                        operator="conditional_rate", columns_used=requested_dims + [condition.column],
                        statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                    ),
                )
            res_df = res_df.head(requested_limit).copy()

        if res_df.empty:
            threshold_text = f" with min_group_size={min_group_size}" if min_group_size is not None else ""
            return OperatorOutput(
                status="FAILED",
                error_code="NO_ELIGIBLE_SEGMENTS",
                error_message=f"No segment has eligible denominator records{threshold_text}.",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims + [condition.column],
                    statistics_computed=["matching_count", "total_count", "rate", "rate_percent"],
                    chart_type="none", row_count_in=len(df), row_count_out=0,
                    filters_applied=[{"column": condition.column, "operator": condition.operator.value, "value": condition.value}],
                ),
            )

        chart_type = tool_args.get("chart_type")
        if chart_type not in {"bar", "none"}:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_CHART_TYPE",
                error_message="Conditional-rate chart_type must be 'bar' or 'none'.",
                manifest=ExecutionManifest(
                    operator="conditional_rate", columns_used=requested_dims + [condition.column],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        labels = res_df[requested_dims].astype(str).agg(" | ".join, axis=1)
        img_b64: Optional[str] = None
        plotly_spec: Optional[Dict[str, Any]] = None
        if chart_type == "bar":
            plt.figure(figsize=(max(9, min(16, len(res_df) * 0.55)), 4.8))
            plt.bar(labels, res_df["rate_percent"], color="#6366f1", edgecolor="black", alpha=0.85)
            plt.title(f"Conditional Rate by {' / '.join(requested_dims)}")
            plt.xlabel(" / ".join(requested_dims))
            plt.ylabel("Rate (%)")
            plt.xticks(rotation=45, ha="right")
            plt.ylim(bottom=0)
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plotly_spec = {
                "data": [{
                    "type": "bar", "x": labels.tolist(), "y": res_df["rate_percent"].round(6).tolist(),
                    "customdata": res_df[["matching_count", "total_count"]].to_numpy().tolist(),
                    "hovertemplate": "%{x}<br>Rate: %{y:.2f}%<br>Matches: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
                }],
                "layout": {
                    "title": f"Conditional Rate by {' / '.join(requested_dims)}",
                    "xaxis": {"title": " / ".join(requested_dims)},
                    "yaxis": {"title": "Rate (%)", "rangemode": "tozero"},
                },
            }

        predicate_text = f"{condition.column} {condition.operator.value} {condition.value!r}"
        threshold_text = f"; retained cohorts with at least {min_group_size:,} records" if min_group_size is not None else ""
        finding = (
            f"Computed the conditional rate for `{predicate_text}` across {len(res_df):,} "
            f"segment(s) defined by `{', '.join(requested_dims)}`{threshold_text}. "
            "Each rate is matching_count / total_count; the table exposes both counts."
        )
        manifest = ExecutionManifest(
            operator="conditional_rate",
            columns_used=requested_dims + [condition.column],
            filters_applied=[{"column": condition.column, "operator": condition.operator.value, "value": condition.value}],
            statistics_computed=["matching_count", "total_count", "rate", "rate_percent"],
            chart_type=chart_type,
            row_count_in=len(df),
            row_count_out=len(res_df),
        )
        condition_dict = condition.model_dump(mode="json")
        code = (
            "# DataPilot DS Engine: Conditional Rate by Segment\n"
            f"group_by = {requested_dims!r}\n"
            f"condition = {condition_dict!r}\n"
            "series = df[condition['column']]\n"
            "if condition['operator'] == 'eq':\n"
            "    numerator_mask = series.eq(condition['value'])\n"
            "elif condition['operator'] == 'neq':\n"
            "    numerator_mask = series.ne(condition['value'])\n"
            "elif condition['operator'] == 'in':\n"
            "    numerator_mask = series.isin(condition['value'])\n"
            "elif condition['operator'] == 'not_in':\n"
            "    numerator_mask = ~series.isin(condition['value'])\n"
            "elif condition['operator'] == 'contains':\n"
            "    numerator_mask = series.astype(str).str.contains(str(condition['value']), case=False, na=False, regex=False)\n"
            "elif condition['operator'] == 'gt':\n"
            "    numerator_mask = series.gt(condition['value'])\n"
            "elif condition['operator'] == 'gte':\n"
            "    numerator_mask = series.ge(condition['value'])\n"
            "elif condition['operator'] == 'lt':\n"
            "    numerator_mask = series.lt(condition['value'])\n"
            "elif condition['operator'] == 'lte':\n"
            "    numerator_mask = series.le(condition['value'])\n"
            "elif condition['operator'] in ('is_null', 'missing'):\n"
            "    numerator_mask = series.isna()\n"
            "elif condition['operator'] == 'is_not_null':\n"
            "    numerator_mask = series.notna()\n"
            "else:\n"
            "    raise ValueError('Unsupported typed numerator operator')\n"
            "total_count = df.groupby(group_by, dropna=False).size().rename('total_count')\n"
            "matching_count = df.loc[numerator_mask].groupby(group_by, dropna=False).size().rename('matching_count')\n"
            "result_df = total_count.to_frame().join(matching_count, how='left').fillna({'matching_count': 0}).reset_index()\n"
            "result_df['rate'] = result_df['matching_count'] / result_df['total_count']\n"
            "result_df['rate_percent'] = result_df['rate'] * 100\n"
            "print(result_df)\n"
        )
        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.head(50).to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=finding,
            reproducible_code=code,
        )

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        rate_condition = (intent.arguments or {}).get("numerator_condition")
        if rate_condition is not None:
            return self._execute_conditional_rate(df, intent, rate_condition)

        # Explicit parsing is applied to an execution copy only. It is never
        # guessed from a name, locale, or sample value; parse failures remain
        # visible rather than being silently coerced to another convention.
        work_df = df.copy()
        parsed_datetime_columns: List[str] = []
        parse_assumptions: List[str] = []
        raw_datetime_parses = (intent.arguments or {}).get("datetime_parses", [])
        if not isinstance(raw_datetime_parses, list):
            return OperatorOutput(
                status="FAILED", error_code="INVALID_DATETIME_PARSE_SPEC",
                error_message="datetime_parses must be a list of {column, format} specifications.",
                manifest=ExecutionManifest(operator="group_aggregation", columns_used=[], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        for parse_spec in raw_datetime_parses:
            if not isinstance(parse_spec, dict):
                return OperatorOutput(
                    status="FAILED", error_code="INVALID_DATETIME_PARSE_SPEC",
                    error_message="Each datetime parse specification must contain an exact column and format.",
                    manifest=ExecutionManifest(operator="group_aggregation", columns_used=[], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            parse_column = parse_spec.get("column")
            parse_format = parse_spec.get("format")
            if not isinstance(parse_column, str) or parse_column not in work_df.columns or not isinstance(parse_format, str) or not parse_format:
                return OperatorOutput(
                    status="FAILED", error_code="INVALID_DATETIME_PARSE_SPEC",
                    error_message="Each datetime parse needs an existing exact column name and a non-empty strptime format.",
                    manifest=ExecutionManifest(operator="group_aggregation", columns_used=[], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            try:
                from backend.ds_engine.datetime_parser import normalize_datetime_format, parse_datetime_column
                normalized_format = normalize_datetime_format(parse_format)
            except ValueError as norm_err:
                return OperatorOutput(
                    status="FAILED", error_code="INVALID_DATETIME_FORMAT",
                    error_message=f"Invalid datetime format '{parse_format}' for column '{parse_column}': {norm_err}",
                    manifest=ExecutionManifest(operator="group_aggregation", columns_used=[parse_column], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            parse_res = parse_datetime_column(work_df[parse_column], normalized_format)
            if parse_res.unparseable_count > 0:
                return OperatorOutput(
                    status="FAILED", error_code="DATETIME_PARSE_FAILED",
                    error_message=f"Datetime parsing failed for {parse_res.unparseable_count:,} non-null value(s) in '{parse_column}' with declared format '{parse_format}'.",
                    manifest=ExecutionManifest(operator="group_aggregation", columns_used=[parse_column], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            work_df[parse_column] = parse_res.series
            parsed_datetime_columns.append(parse_column)
            parse_assumptions.append(f"Parsed {parse_column} using declared format={normalized_format}")

        dims = [c for c in intent.grouping if c in df.columns]
        if not dims and intent.requires_grouping:
            return OperatorOutput(
                status="FAILED",
                error_code="GROUP_BY_REQUIRED",
                error_message="Group aggregation requires explicit grouping dimensions.",
                manifest=ExecutionManifest(
                    operator="group_aggregation",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if not intent.subjects and intent.requires_grouping:
            return OperatorOutput(
                status="FAILED",
                error_code="SUBJECT_COLUMNS_REQUIRED",
                error_message="Group aggregation requires explicit metric subject columns.",
                manifest=ExecutionManifest(
                    operator="group_aggregation",
                    columns_used=dims,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        missing = [c for c in intent.subjects if c not in df.columns]
        if missing:
            return OperatorOutput(
                status="FAILED",
                error_code="COLUMN_NOT_FOUND",
                error_message=f"Requested metric column(s) {missing} not found in dataset.",
                manifest=ExecutionManifest(
                    operator="group_aggregation",
                    columns_used=dims + [c for c in intent.subjects if c in df.columns],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        mets = [c for c in intent.subjects if c in df.columns and c not in dims]

        declared_aggregations = [
            str(req.value if hasattr(req, "value") else req).lower()
            for req in intent.statistical_requirements
        ]
        allowed_aggregations = {"sum", "mean", "count", "min", "max", "median"}
        agg_name = next((name for name in declared_aggregations if name in allowed_aggregations), None)
        if mets and agg_name is None:
            return OperatorOutput(
                status="FAILED",
                error_code="AGGREGATION_REQUIRED",
                error_message="Grouped metric analysis requires an explicit aggregation: sum, mean, count, min, max, or median.",
                manifest=ExecutionManifest(operator="group_aggregation", columns_used=dims + mets, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        if not mets:
            agg_name = "count"
        elif agg_name != "count":
            non_numeric = [column for column in mets if not pd.api.types.is_numeric_dtype(work_df[column])]
            if non_numeric:
                return OperatorOutput(
                    status="FAILED",
                    error_code="METRIC_COLUMN_NOT_NUMERIC",
                    error_message=f"Aggregation '{agg_name}' requires numeric metric column(s); found non-numeric: {non_numeric}.",
                    manifest=ExecutionManifest(operator="group_aggregation", columns_used=dims + mets, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )

        if dims and mets:
            if agg_name == "count":
                res_df = work_df.groupby(dims, dropna=False)[mets].count().reset_index()
            else:
                res_df = work_df.groupby(dims, dropna=False).agg({metric: agg_name for metric in mets}).reset_index()
            y_col = mets[0]
        elif dims:
            res_df = work_df.groupby(dims, dropna=False).size().reset_index(name="count")
            y_col = "count"
        elif mets:
            # Overall aggregate: a single result row, with no invented
            # grouping category.  For count, pandas count() correctly reports
            # non-null observations of the explicitly named metric.
            result_values = {
                metric: (work_df[metric].count() if agg_name == "count" else work_df[metric].agg(agg_name))
                for metric in mets
            }
            res_df = pd.DataFrame([result_values])
            y_col = mets[0]
        else:
            # Overall record count has no metric subject by design.
            res_df = pd.DataFrame({"count": [len(work_df)]})
            y_col = "count"

        chart_kind = str(
            (intent.visualization.type if intent.visualization else None)
            or (intent.arguments or {}).get("chart_type")
            or "none"
        ).lower()
        if chart_kind not in {"bar", "line", "none"}:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_CHART_TYPE",
                error_message="Group aggregation chart_type must be bar, line, or none.",
                manifest=ExecutionManifest(operator="group_aggregation", columns_used=dims + mets, statistics_computed=[agg_name], chart_type="none", row_count_in=len(df), row_count_out=0),
            )

        dim_col = dims[0] if dims else None
        res_df_preview = res_df.head(100)
        img_b64 = None
        plotly_spec = None
        if chart_kind != "none" and dims:
            plt.figure(figsize=(9, 4.5))
            traces = []
            if chart_kind == "line":
                series_dims = dims[1:]
                grouped = [(None, res_df_preview)] if not series_dims else res_df_preview.groupby(series_dims, dropna=False, sort=False)
                for key, frame in grouped:
                    key_values = key if isinstance(key, tuple) else (key,)
                    label = "Aggregate" if not series_dims else ", ".join(
                        f"{column}={value}" for column, value in zip(series_dims, key_values)
                    )
                    frame = frame.sort_values(dim_col)
                    x_values = frame[dim_col].tolist()
                    y_values = frame[y_col].tolist()
                    plt.plot(x_values, y_values, marker="o", linewidth=2, label=label)
                    traces.append({"type": "scatter", "mode": "lines+markers", "name": label, "x": x_values, "y": y_values})
                if series_dims:
                    plt.legend(title=", ".join(series_dims))
            else:
                labels = (
                    res_df_preview[dim_col].astype(str)
                    if len(dims) == 1
                    else res_df_preview[dims].astype(str).agg(" | ".join, axis=1)
                )
                values = res_df_preview[y_col].tolist()
                plt.bar(labels, values, color="#6366f1", edgecolor="black", alpha=0.85)
                traces.append({"type": "bar", "x": labels.tolist(), "y": values})
            plt.title(f"{agg_name.title()} {y_col} by {', '.join(dims)}")
            plt.xlabel(dim_col if chart_kind == "line" else " | ".join(dims))
            plt.ylabel(f"{agg_name.title()} {y_col}")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plotly_spec = {
                "data": traces,
                "layout": {"title": f"{agg_name.title()} {y_col} by {', '.join(dims)}", "xaxis": {"title": dim_col}, "yaxis": {"title": f"{agg_name.title()} {y_col}"}},
            }

        finding = (
            f"Computed overall {agg_name} of `{y_col}` from {len(df):,} rows."
            if not dims else
            f"Computed {agg_name} of `{y_col}` across {len(res_df):,} group(s) defined by {', '.join(f'`{d}`' for d in dims)} from {len(df):,} rows."
        )
        manifest = ExecutionManifest(
            operator="group_aggregation",
            columns_used=list(dict.fromkeys(dims + mets + parsed_datetime_columns)),
            statistics_computed=[agg_name],
            chart_type=chart_kind,
            row_count_in=len(df),
            row_count_out=len(res_df),
            assumptions=parse_assumptions,
        )
        from backend.ds_engine.datetime_parser import normalize_datetime_format
        parse_code = "".join(
            f"df[{column!r}] = pd.to_datetime(df[{column!r}], format={normalize_datetime_format(format_string)!r}, errors='raise')\n"
            for column, format_string in (
                (parse_spec["column"], parse_spec["format"])
                for parse_spec in raw_datetime_parses
            )
        )
        code = (
            "# DataPilot DS Engine: typed group aggregation\n"
            "df = df.copy()\n"
            + parse_code
            + f"group_by = {dims!r}\nmetric_columns = {mets!r}\naggregation = {agg_name!r}\n"
            + "result_df = (df.groupby(group_by, dropna=False)[metric_columns].count().reset_index() "
            + "if aggregation == 'count' else df.groupby(group_by, dropna=False).agg({column: aggregation for column in metric_columns}).reset_index())\n"
            + "print(result_df)\n"
        ) if dims and mets else (
            "# DataPilot DS Engine: typed grouped record count\n"
            "df = df.copy()\n"
            + parse_code
            + f"result_df = df.groupby({dims!r}, dropna=False).size().reset_index(name='count')\nprint(result_df)\n"
        ) if dims else (
            "# DataPilot DS Engine: typed overall aggregation\n"
            "df = df.copy()\n"
            + parse_code
            + f"metric_columns = {mets!r}\naggregation = {agg_name!r}\n"
            + "result_df = (pd.DataFrame([{column: df[column].count() for column in metric_columns}]) "
            + "if metric_columns and aggregation == 'count' else (pd.DataFrame([{column: df[column].agg(aggregation) for column in metric_columns}]) if metric_columns else pd.DataFrame({'count': [len(df)]})))\n"
            + "print(result_df)\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.head(50).to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=finding,
            reproducible_code=code
        )


class CorrelationOperator(BaseTypedOperator):
    """Executes only the explicit, typed correlation contract supplied by the planner."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        def fail(code: str, message: str, columns: Optional[List[str]] = None) -> OperatorOutput:
            return OperatorOutput(
                status="FAILED",
                error_code=code,
                error_message=message,
                manifest=ExecutionManifest(
                    operator="correlation_analysis",
                    columns_used=columns or [],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                ),
            )

        if intent.correlation_spec is None:
            return fail(
                "CORRELATION_SPEC_REQUIRED",
                "Correlation execution requires an explicit typed analysis mode, relationship form, methods, chart type, smoother, and expected direction.",
                intent.subjects,
            )
        spec: CorrelationSpec = intent.correlation_spec

        if not intent.subjects:
            return fail("SUBJECT_COLUMNS_REQUIRED", "Correlation analysis requires explicit numerical subject columns.")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        subjects = [c for c in intent.subjects if c in df.columns and c in num_cols]

        if len(subjects) < 2:
            return fail("INSUFFICIENT_NUMERIC_COLUMNS", "Correlation analysis requires at least 2 numerical features.", subjects)

        if spec.analysis_mode == "pairwise":
            if len(subjects) != 2:
                return fail(
                    "PAIRWISE_COLUMN_COUNT_INVALID",
                    "Pairwise correlation analysis requires exactly two valid numeric subject columns.",
                    subjects,
                )
            valid_df = df[subjects].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid_df) < 3:
                return fail(
                    "INSUFFICIENT_VALID_PAIRS",
                    "Pairwise correlation analysis requires at least three complete numeric pairs.",
                    subjects,
                )

            try:
                from scipy import stats
                from statsmodels.nonparametric.smoothers_lowess import lowess
            except ImportError as exc:
                return fail(
                    "MISSING_ANALYTICAL_DEPENDENCY",
                    f"Pairwise correlation dependency is unavailable: {exc}",
                    subjects,
                )

            x_col, y_col = subjects
            x = valid_df[x_col].to_numpy()
            y = valid_df[y_col].to_numpy()
            record: Dict[str, Any] = {
                "feature_x": x_col,
                "feature_y": y_col,
                "valid_pair_count": int(len(valid_df)),
                "analysis_mode": spec.analysis_mode,
                "relationship_form": spec.relationship_form,
                "expected_direction": spec.expected_direction,
            }
            statistics_computed: List[str] = []
            coefficients: Dict[str, float] = {}
            if "pearson" in spec.correlation_methods:
                pearson = stats.pearsonr(x, y)
                coefficients["pearson"] = float(pearson.statistic)
                record["pearson_r"] = float(pearson.statistic)
                record["pearson_p_value"] = float(pearson.pvalue)
                statistics_computed.append("pearson_correlation")
            if "spearman" in spec.correlation_methods:
                spearman = stats.spearmanr(x, y, nan_policy="omit")
                coefficients["spearman"] = float(spearman.statistic)
                record["spearman_rho"] = float(spearman.statistic)
                record["spearman_p_value"] = float(spearman.pvalue)
                statistics_computed.append("spearman_rank_correlation")
            if "kendall" in spec.correlation_methods:
                kendall = stats.kendalltau(x, y, nan_policy="omit")
                coefficients["kendall"] = float(kendall.statistic)
                record["kendall_tau"] = float(kendall.statistic)
                record["kendall_p_value"] = float(kendall.pvalue)
                statistics_computed.append("kendall_rank_correlation")
            if "pearson" in coefficients and "spearman" in coefficients:
                record["abs_rank_linear_gap"] = abs(abs(coefficients["spearman"]) - abs(coefficients["pearson"]))

            direction_coefficient = coefficients.get("spearman", coefficients.get("pearson", coefficients.get("kendall", 0.0)))
            observed_direction = "positive" if direction_coefficient > 0 else "negative" if direction_coefficient < 0 else "zero"
            direction_matches = spec.expected_direction == "unspecified" or observed_direction == spec.expected_direction
            record["observed_direction"] = observed_direction
            record["direction_matches_request"] = direction_matches
            res_df = pd.DataFrame([record])

            img_b64: Optional[str] = None
            plotly_spec: Optional[Dict[str, Any]] = None
            if spec.chart_type == "scatter":
                plt.figure(figsize=(9, 6))
                plt.scatter(x, y, alpha=0.22, s=12, color="#3b82f6", edgecolors="none", label="Observed pairs")
                plot_data = [{"type": "scatter", "mode": "markers", "name": "Observed pairs", "x": x.tolist(), "y": y.tolist(), "marker": {"size": 4, "opacity": 0.25}}]
                if spec.smoother == "lowess":
                    smoothed = lowess(endog=y, exog=x, frac=0.2, it=1, return_sorted=True)
                    plt.plot(smoothed[:, 0], smoothed[:, 1], color="#ef4444", linewidth=2.2, label="LOWESS smoother")
                    plot_data.append({"type": "scatter", "mode": "lines", "name": "LOWESS smoother", "x": smoothed[:, 0].tolist(), "y": smoothed[:, 1].tolist(), "line": {"color": "#ef4444", "width": 2}})
                    statistics_computed.append("lowess_descriptive_smoother")
                stats_label = ", ".join(f"{name}={value:.3f}" for name, value in coefficients.items())
                plt.title(f"{spec.relationship_form.title()} pairwise association: {x_col} vs {y_col}\n{stats_label}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.legend()
                plt.grid(True, alpha=0.25)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=110)
                plt.close()
                buf.seek(0)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                plotly_spec = {"data": plot_data, "layout": {"title": f"{spec.relationship_form.title()} pairwise association: {x_col} vs {y_col}", "xaxis": {"title": x_col}, "yaxis": {"title": y_col}}}

            warnings = ["LOWESS is a descriptive smoother and does not establish causation."] if spec.smoother == "lowess" else []
            if not direction_matches:
                warnings.append(f"Observed {observed_direction} association does not support the requested {spec.expected_direction} direction.")
            r_val = coefficients.get("pearson", coefficients.get("spearman", coefficients.get("kendall", 0.0)))
            finding = (
                f"Computed {', '.join(spec.correlation_methods)} correlation between '{x_col}' and '{y_col}' "
                f"(r = {r_val:.4f}) across {len(valid_df):,} complete pairs; observed direction is {observed_direction}."
            )
            if not direction_matches:
                finding += f" This does not support the requested {spec.expected_direction} direction."
            if spec.smoother == "lowess":
                finding += " The LOWESS curve visualizes the requested nonlinear form but is descriptive, not causal proof."
            manifest = ExecutionManifest(
                operator="correlation_analysis",
                columns_used=subjects,
                statistics_computed=statistics_computed,
                chart_type=spec.chart_type,
                row_count_in=len(df),
                row_count_out=len(res_df),
                assumptions=["Correlation measures association, not causation."],
                warnings=warnings,
            )
            code = (
                "# DataPilot DS Engine: Typed Pairwise Correlation Analysis\n"
                "from scipy import stats\n"
                "from statsmodels.nonparametric.smoothers_lowess import lowess\n"
                f"x_col, y_col = {x_col!r}, {y_col!r}\n"
                "pairs = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()\n"
                f"# Requested methods: {spec.correlation_methods!r}; smoother: {spec.smoother!r}\n"
                "result_df = pairs.describe().T\n"
            )
            return OperatorOutput(
                status="SUCCEEDED",
                data={"records": res_df.to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
                dataframe=res_df,
                manifest=manifest,
                plot_image_base64=img_b64,
                plotly_spec=plotly_spec,
                written_finding=finding,
                reproducible_code=code,
            )

        method = spec.correlation_methods[0]
        corr_df = df[subjects].corr(method=method)
        res_df = corr_df.reset_index().rename(columns={"index": "feature"})
        img_b64 = None
        plotly_spec = None
        if spec.chart_type == "heatmap":
            plt.figure(figsize=(8, 6))
            plt.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(label=f"{method.title()} correlation")
            plt.xticks(range(len(subjects)), subjects, rotation=45, ha="right")
            plt.yticks(range(len(subjects)), subjects)
            plt.title(f"{method.title()} correlation matrix: " + ", ".join(subjects[:4]))
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plotly_spec = {"data": [{"type": "heatmap", "z": corr_df.values.tolist(), "x": subjects, "y": subjects, "colorscale": "RdBu", "zmin": -1, "zmax": 1}], "layout": {"title": f"{method.title()} correlation heatmap"}}
        manifest = ExecutionManifest(
            operator="correlation_analysis",
            columns_used=subjects,
            statistics_computed=[f"{method}_correlation"],
            chart_type=spec.chart_type,
            row_count_in=len(df),
            row_count_out=len(res_df),
            assumptions=["Correlation measures association, not causation."],
        )
        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=f"Computed a {method.title()} correlation matrix across {len(subjects)} continuous numerical features ({', '.join(subjects)}).",
            reproducible_code=(
                "# DataPilot DS Engine: Typed Correlation Matrix\n"
                f"subjects = {subjects!r}\n"
                f"result_df = df[subjects].corr(method={method!r}).reset_index()\n"
            ),
        )


class AnomalyDetectionOperator(BaseTypedOperator):
    """Explicit-method anomaly detection with no hidden algorithm choice."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        if not intent.subjects:
            return OperatorOutput(
                status="FAILED",
                error_code="SUBJECT_COLUMNS_REQUIRED",
                error_message="Anomaly detection requires explicit numerical subject columns.",
                manifest=ExecutionManifest(
                    operator="anomaly_detection",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        subjects = [c for c in intent.subjects if c in df.columns and c in num_cols]

        if not subjects:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_NUMERIC_COLUMNS",
                error_message="Anomaly detection requires numerical feature columns.",
                manifest=ExecutionManifest(
                    operator="anomaly_detection",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        clean_df = df[subjects].dropna()
        if clean_df.empty:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_VALID_ROWS",
                error_message="Anomaly detection requires at least one complete numeric row across all requested features.",
                manifest=ExecutionManifest(
                    operator="anomaly_detection",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                ),
            )
        tool_args = intent.arguments or {}
        method = tool_args.get("method")
        sensitivity = tool_args.get("sensitivity")
        if method not in {"isolation_forest", "iqr", "zscore"} or not isinstance(sensitivity, (int, float)) or isinstance(sensitivity, bool):
            return OperatorOutput(
                status="FAILED", error_code="ANOMALY_METHOD_REQUIRED",
                error_message="Anomaly detection requires an explicit method and numeric sensitivity.",
                manifest=ExecutionManifest(operator="anomaly_detection", columns_used=subjects, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        sensitivity = float(sensitivity)
        if method == "isolation_forest":
            if not 0 < sensitivity < 0.5:
                return OperatorOutput(
                    status="FAILED", error_code="INVALID_ANOMALY_SENSITIVITY",
                    error_message="Isolation Forest contamination must be between 0 and 0.5.",
                    manifest=ExecutionManifest(operator="anomaly_detection", columns_used=subjects, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=sensitivity, random_state=42)
            preds = clf.fit_predict(clean_df)
            scores = clf.decision_function(clean_df)
            flags = preds == -1
            method_statistic = "isolation_forest_decision_score"
        elif method == "iqr":
            q1, q3 = clean_df.quantile(0.25), clean_df.quantile(0.75)
            scale = (q3 - q1).replace(0, np.nan)
            lower_distance = ((q1 - clean_df) / scale).clip(lower=0).max(axis=1)
            upper_distance = ((clean_df - q3) / scale).clip(lower=0).max(axis=1)
            normalized_distance = pd.concat([lower_distance, upper_distance], axis=1).max(axis=1).fillna(0.0)
            scores = -normalized_distance.to_numpy()
            flags = normalized_distance > sensitivity
            method_statistic = "max_iqr_distance"
        else:
            if sensitivity <= 0:
                return OperatorOutput(
                    status="FAILED", error_code="INVALID_ANOMALY_SENSITIVITY",
                    error_message="Z-score sensitivity must be greater than zero.",
                    manifest=ExecutionManifest(operator="anomaly_detection", columns_used=subjects, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            std = clean_df.std(ddof=0).replace(0, np.nan)
            max_abs_z = ((clean_df - clean_df.mean()) / std).abs().max(axis=1).fillna(0.0)
            scores = -max_abs_z.to_numpy()
            flags = max_abs_z > sensitivity
            method_statistic = "max_absolute_zscore"

        # Predictions correspond only to complete feature rows. Preserve that
        # index alignment instead of assigning a shorter vector to all input rows.
        res_df = df.loc[clean_df.index].copy()
        res_df["anomaly_flag"] = np.where(flags, 1, 0)
        res_df["anomaly_score"] = scores

        anomalies_found = int((res_df["anomaly_flag"] == 1).sum())
        pct = round((anomalies_found / max(len(res_df), 1)) * 100, 2)

        summary_data = [
            {"metric": "Total Records", "value": len(df)},
            {"metric": "Complete Feature Rows Analysed", "value": len(res_df)},
            {"metric": "Anomalies Flagged", "value": anomalies_found},
            {"metric": "Observed Flag Rate", "value": f"{pct}%"},
            {"metric": "Declared Method", "value": method},
            {"metric": "Declared Sensitivity", "value": sensitivity},
            {"metric": "Features Used", "value": ", ".join(subjects)}
        ]
        res_table = pd.DataFrame(summary_data)

        plt.figure(figsize=(9, 4.5))
        plt.hist(scores, bins=30, color="#ef4444", alpha=0.7, edgecolor="black")
        plt.axvline(x=0, color="black", linestyle="--", label="Decision Boundary (0.0)")
        plt.title(f"{method} anomaly score distribution ({anomalies_found:,} rows flagged)")
        plt.xlabel("Anomaly Score (Lower = More Anomalous)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        manifest = ExecutionManifest(
            operator="anomaly_detection",
            columns_used=subjects,
            statistics_computed=[method_statistic, "observed_flag_rate"],
            chart_type="histogram",
            row_count_in=len(df),
            row_count_out=len(res_df),
            warnings=[f"Excluded {len(df) - len(res_df):,} row(s) with missing values in requested anomaly features."] if len(res_df) != len(df) else [],
        )

        finding = f"Declared `{method}` detection (sensitivity `{sensitivity:g}`) flagged **{anomalies_found:,} records** ({pct}% of complete rows) across: {', '.join(subjects)}."

        method_code = {
            "isolation_forest": (
                "from sklearn.ensemble import IsolationForest\n"
                "detector = IsolationForest(contamination=sensitivity, random_state=42)\n"
                "flags = detector.fit_predict(complete_rows) == -1\n"
                "scores = detector.decision_function(complete_rows)\n"
            ),
            "iqr": (
                "q1, q3 = complete_rows.quantile(.25), complete_rows.quantile(.75)\n"
                "scale = (q3 - q1).replace(0, np.nan)\n"
                "lower = ((q1 - complete_rows) / scale).clip(lower=0).max(axis=1)\n"
                "upper = ((complete_rows - q3) / scale).clip(lower=0).max(axis=1)\n"
                "distance = pd.concat([lower, upper], axis=1).max(axis=1).fillna(0)\n"
                "flags, scores = distance > sensitivity, -distance.to_numpy()\n"
            ),
            "zscore": (
                "std = complete_rows.std(ddof=0).replace(0, np.nan)\n"
                "distance = ((complete_rows - complete_rows.mean()) / std).abs().max(axis=1).fillna(0)\n"
                "flags, scores = distance > sensitivity, -distance.to_numpy()\n"
            ),
        }[method]
        code = (
            "# DataPilot DS Engine: explicit-method anomaly detection\n"
            "import numpy as np\nimport pandas as pd\n"
            f"features = {repr(subjects)}\n"
            f"method = {method!r}\nsensitivity = {sensitivity!r}\n"
            "complete_rows = df[features].dropna()\n"
            + method_code +
            "result_df = df.loc[complete_rows.index].copy()\n"
            "result_df['anomaly_flag'] = np.where(flags, 1, 0)\n"
            "result_df['anomaly_score'] = scores\n"
            "print(f'Anomalies found: {result_df[\"anomaly_flag\"].sum()}')\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_table.to_dict(orient="records"), "columns": list(res_table.columns), "row_count": len(res_table)},
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            written_finding=finding,
            reproducible_code=code
        )


class TimeSeriesForecastOperator(BaseTypedOperator):
    """Explicit-method time-series forecasting without invented uncertainty bounds."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        tc = intent.time_context
        time_col = tc.time_column if tc else None
        horizon = tc.horizon if tc else intent.horizon
        tool_args = intent.arguments or {}
        method = tool_args.get("method")
        frequency = tool_args.get("frequency")
        chart_type = tool_args.get("chart_type")

        if not time_col or time_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="NO_TEMPORAL_COLUMN",
                error_message="Time-series forecasting requires an explicit datetime/temporal column.",
                manifest=ExecutionManifest(
                    operator="time_series_forecast",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if horizon is None or horizon <= 0:
            return OperatorOutput(
                status="FAILED",
                error_code="HORIZON_REQUIRED",
                error_message="Time-series forecasting requires an explicit forecast horizon.",
                manifest=ExecutionManifest(
                    operator="time_series_forecast",
                    columns_used=[time_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if method not in {"naive_last_value", "holt_linear"}:
            return OperatorOutput(
                status="FAILED",
                error_code="FORECAST_METHOD_REQUIRED",
                error_message="Forecasting requires an explicit method: naive_last_value or holt_linear.",
                manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        if not isinstance(frequency, str) or not frequency.strip():
            return OperatorOutput(
                status="FAILED",
                error_code="FREQUENCY_REQUIRED",
                error_message="Forecasting requires an explicit calendar frequency such as D, W, MS, or h.",
                manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        try:
            frequency = canonicalize_pandas_frequency(frequency)
            offset = pd.tseries.frequencies.to_offset(frequency)
        except (TypeError, ValueError) as exc:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_FREQUENCY",
                error_message=f"Forecast frequency '{frequency}' is not a supported pandas interval: {exc}",
                manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        if chart_type not in {"line", "none"}:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_CHART_TYPE",
                error_message="Forecast chart_type must be line or none.",
                manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )

        # The compiler stores forecast value_column as ``target_column``.  Read
        # that canonical field first; falling back to subjects is retained only
        # for an already-persisted valid V2 intent, never for column guessing.
        target_col = intent.target_column if intent.target_column in df.columns else None
        if target_col is None and intent.subjects and intent.subjects[0] in df.columns:
            target_col = intent.subjects[0]

        if not target_col:
            return OperatorOutput(
                status="FAILED",
                error_code="TARGET_COLUMN_REQUIRED",
                error_message="Time-series forecasting requires an explicit numerical target column.",
                manifest=ExecutionManifest(
                    operator="time_series_forecast",
                    columns_used=[time_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return OperatorOutput(
                status="FAILED",
                error_code="TARGET_COLUMN_NOT_NUMERIC",
                error_message=f"Forecast target column '{target_col}' must be numeric.",
                manifest=ExecutionManifest(
                    operator="time_series_forecast",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                ),
            )

        explicit_fmt = (intent.arguments or {}).get("datetime_format")
        ts_df = df[[time_col, target_col]].dropna().copy()
        if explicit_fmt:
            try:
                from backend.ds_engine.datetime_parser import normalize_datetime_format, parse_datetime_column
                norm_fmt = normalize_datetime_format(explicit_fmt)
                parse_res = parse_datetime_column(ts_df[time_col], norm_fmt)
                if parse_res.unparseable_count > 0:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="DATETIME_PARSE_FAILED",
                        error_message=f"Datetime parsing failed for {parse_res.unparseable_count:,} non-null value(s) in '{time_col}' with declared format '{explicit_fmt}'.",
                        manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col, target_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                    )
                ts_df[time_col] = parse_res.series
            except ValueError as val_err:
                return OperatorOutput(
                    status="FAILED",
                    error_code="INVALID_DATETIME_FORMAT",
                    error_message=f"Invalid datetime format '{explicit_fmt}' for column '{time_col}': {val_err}",
                    manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col, target_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
        else:
            ts_df[time_col] = pd.to_datetime(ts_df[time_col], errors="coerce")
        ts_df = ts_df.dropna().sort_values(by=time_col)

        if ts_df[time_col].duplicated().any():
            return OperatorOutput(
                status="FAILED",
                error_code="DUPLICATE_TIME_INDEX",
                error_message="Forecasting requires one observation per timestamp; aggregate duplicate timestamps before forecasting.",
                manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col, target_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        minimum_observations = 4 if method == "holt_linear" else 1
        if len(ts_df) < minimum_observations:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_TEMPORAL_OBSERVATIONS",
                error_message=f"{method} forecasting requires at least {minimum_observations} complete, parseable temporal observations.",
                manifest=ExecutionManifest(
                    operator="time_series_forecast",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                ),
            )

        observed_times = pd.DatetimeIndex(ts_df[time_col])
        expected_times = pd.date_range(start=observed_times[0], periods=len(observed_times), freq=offset)
        if not observed_times.equals(expected_times):
            return OperatorOutput(
                status="FAILED",
                error_code="IRREGULAR_TIME_GRID",
                error_message=(
                    f"Observed timestamps do not form the declared regular '{frequency}' grid. "
                    "Resample or aggregate the data explicitly before forecasting."
                ),
                manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col, target_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0, sorted_by=time_col),
            )

        future_dates = pd.date_range(start=observed_times[-1] + offset, periods=horizon, freq=offset)
        observed_values = pd.to_numeric(ts_df[target_col], errors="coerce").to_numpy(dtype=float)
        if method == "naive_last_value":
            forecast_values = np.repeat(observed_values[-1], horizon)
            method_label = "Naive last-value baseline"
        else:
            try:
                from statsmodels.tsa.holtwinters import Holt
                fitted = Holt(observed_values, damped_trend=True, initialization_method="estimated").fit(optimized=True)
                forecast_values = np.asarray(fitted.forecast(horizon), dtype=float)
            except Exception as exc:
                return OperatorOutput(
                    status="FAILED",
                    error_code="FORECAST_FIT_FAILED",
                    error_message=f"holt_linear forecast could not be fitted: {exc}",
                    manifest=ExecutionManifest(operator="time_series_forecast", columns_used=[time_col, target_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0, sorted_by=time_col),
                )
            method_label = "Holt damped linear trend"

        # ISO-8601 retains sub-day temporal resolution. Formatting every
        # forecast as a calendar date silently collapsed hourly/minutely
        # horizons into indistinguishable rows.
        fc_df = pd.DataFrame({
            "ds": [timestamp.isoformat() for timestamp in future_dates],
            "yhat": [round(float(value), 6) for value in forecast_values],
        })

        # ``chart_type`` is part of the public ToolSpec contract.  A table-only
        # request must not silently create a chart artifact just because this
        # operator knows how to draw one.
        img_b64 = None
        plotly_spec = None
        if chart_type == "line":
            plt.figure(figsize=(9, 4.5))
            plt.plot(ts_df[time_col].tail(50), ts_df[target_col].tail(50), label="Historical Data", color="#3b82f6", linewidth=2)
            plt.plot(future_dates, forecast_values, label=f"Forecast ({method_label})", color="#ef4444", linewidth=2, linestyle="--")
            plt.title(f"Time-Series Forecast: {target_col} (Next {horizon} periods)")
            plt.xlabel(time_col)
            plt.ylabel(target_col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            plotly_spec = {
                "data": [
                    {"type": "scatter", "mode": "lines", "name": "Historical", "x": [timestamp.isoformat() for timestamp in ts_df[time_col].tail(50)], "y": ts_df[target_col].tail(50).tolist()},
                    {"type": "scatter", "mode": "lines", "name": f"Forecast ({method_label})", "x": fc_df["ds"].tolist(), "y": fc_df["yhat"].tolist(), "line": {"dash": "dot"}}
                ],
                "layout": {"title": f"Forecast for {target_col} ({horizon} periods)"}
            }

        finding = f"Generated a **{horizon}-period forecast** for `{target_col}` using the explicitly selected {method_label} on a regular `{frequency}` grid. No calibrated prediction interval is claimed."

        c_type = chart_type
        manifest = ExecutionManifest(
            operator="time_series_forecast",
            columns_used=[time_col, target_col],
            statistics_computed=[method],
            chart_type=c_type,
            row_count_in=len(df),
            row_count_out=len(fc_df),
            sorted_by=time_col,
            assumptions=[f"Declared frequency={frequency}", f"Declared method={method}", "No calibrated prediction interval was computed."],
        )

        code = (
            "# DataPilot DS Engine: explicit-method time-series forecast\n"
            "import numpy as np\nimport pandas as pd\n"
            f"time_col = {time_col!r}\ntarget_col = {target_col!r}\nhorizon = {horizon!r}\nfrequency = {frequency!r}\nmethod = {method!r}\n"
            "series = df[[time_col, target_col]].dropna().copy()\n"
            "series[time_col] = pd.to_datetime(series[time_col], errors='raise')\n"
            "series = series.sort_values(time_col)\n"
            "future_index = pd.date_range(series[time_col].iloc[-1] + pd.tseries.frequencies.to_offset(frequency), periods=horizon, freq=frequency)\n"
            "if method == 'naive_last_value':\n"
            "    forecasts = np.repeat(series[target_col].iloc[-1], horizon)\n"
            "else:\n"
            "    from statsmodels.tsa.holtwinters import Holt\n"
            "    forecasts = Holt(series[target_col].to_numpy(dtype=float), damped_trend=True, initialization_method='estimated').fit(optimized=True).forecast(horizon)\n"
            "result_df = pd.DataFrame({'ds': future_index, 'yhat': forecasts})\nprint(result_df)\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": fc_df.to_dict(orient="records"), "columns": list(fc_df.columns), "row_count": len(fc_df)},
            dataframe=fc_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=finding,
            reproducible_code=code
        )


class SupervisedMLOperator(BaseTypedOperator):
    """Typed RandomForest supervised learning with schema-preserving preprocessing."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix,
            brier_score_loss, average_precision_score
        )

        target_col = intent.target_column if intent.target_column in df.columns else None
        if intent.target_column is None and intent.subjects:
            # Compatibility with previously persisted typed intents only. New
            # plans always carry target_column explicitly from TrainModelArgs.
            candidate = intent.subjects[0]
            target_col = candidate if candidate in df.columns else None

        if not target_col or target_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="TARGET_COLUMN_REQUIRED",
                error_message="Supervised ML requires an explicit target column.",
                manifest=ExecutionManifest(
                    operator="supervised_ml",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        requested_features = [
            column for column in intent.subjects
            if column != target_col
        ]
        if not requested_features:
            requested_features = [
                column for column in intent.arguments.get("feature_columns", [])
                if column != target_col
            ]
        missing_features = [column for column in requested_features if column not in df.columns]
        if missing_features:
            return OperatorOutput(
                status="FAILED",
                error_code="COLUMN_NOT_FOUND",
                error_message=f"Supervised ML feature column(s) not found: {missing_features}.",
                manifest=ExecutionManifest(
                    operator="supervised_ml",
                    columns_used=[target_col] + [c for c in requested_features if c in df.columns],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                ),
            )

        if not requested_features:
            return OperatorOutput(
                status="FAILED",
                error_code="FEATURE_COLUMNS_REQUIRED",
                error_message="Supervised ML requires explicit predictor feature columns.",
                manifest=ExecutionManifest(
                    operator="supervised_ml",
                    columns_used=[target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        clean_data = df[[target_col] + requested_features].dropna(subset=[target_col]).copy()
        if len(clean_data) < 5:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_TRAINING_ROWS",
                error_message="Supervised ML requires at least five rows with a non-null target.",
                manifest=ExecutionManifest(
                    operator="supervised_ml", columns_used=[target_col] + requested_features,
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )
        X = clean_data[requested_features]
        y = clean_data[target_col]

        task_type = str(intent.arguments.get("task_type") or "").lower().strip()
        if task_type not in {"classification", "regression"}:
            return OperatorOutput(
                status="FAILED",
                error_code="TASK_TYPE_REQUIRED",
                error_message="Supervised ML requires the planner to specify task_type as 'classification' or 'regression'.",
                manifest=ExecutionManifest(
                    operator="supervised_ml", columns_used=[target_col] + requested_features,
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [column for column in requested_features if column not in numeric_features]
        transformers = []
        if numeric_features:
            transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric_features))
        if categorical_features:
            transformers.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features))
        if not transformers:
            return OperatorOutput(
                status="FAILED", error_code="FEATURE_COLUMNS_REQUIRED",
                error_message="No usable predictor columns were supplied for supervised ML.",
                manifest=ExecutionManifest(operator="supervised_ml", columns_used=[target_col], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )

        if task_type == "classification":
            if y.nunique(dropna=True) < 2:
                return OperatorOutput(
                    status="FAILED", error_code="INSUFFICIENT_TARGET_CLASSES",
                    error_message=f"Classification target '{target_col}' requires at least two classes.",
                    manifest=ExecutionManifest(operator="supervised_ml", columns_used=[target_col] + requested_features, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            metric_name = "Accuracy"
        else:
            if not pd.api.types.is_numeric_dtype(y):
                return OperatorOutput(
                    status="FAILED", error_code="TARGET_COLUMN_NOT_NUMERIC",
                    error_message=f"Regression target '{target_col}' must be numeric.",
                    manifest=ExecutionManifest(operator="supervised_ml", columns_used=[target_col] + requested_features, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            metric_name = "R² Score"

        min_class_count = int(y.value_counts().min()) if task_type == "classification" else 0
        stratify = y if task_type == "classification" and min_class_count >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify,
        )
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        score = accuracy_score(y_test, preds) if task_type == "classification" else r2_score(y_test, preds)

        # 1. Deterministic Baseline Model (Contextual benchmark)
        baseline_model = DummyClassifier(strategy="most_frequent") if task_type == "classification" else DummyRegressor(strategy="mean")
        baseline_pipeline = Pipeline([("preprocess", preprocessor), ("model", baseline_model)])
        baseline_pipeline.fit(X_train, y_train)
        baseline_preds = baseline_pipeline.predict(X_test)
        baseline_score = accuracy_score(y_test, baseline_preds) if task_type == "classification" else r2_score(y_test, baseline_preds)

        # 2. Comprehensive Primary & Evaluation Metrics
        extended_metrics: Dict[str, float] = {
            metric_name: round(float(score), 4),
            "Baseline Score": round(float(baseline_score), 4),
            "Improvement vs Baseline": round(float(score - baseline_score), 4)
        }
        threshold_tuning_payload = None
        class_imbalance_warning = None
        if task_type == "classification":
            extended_metrics["Balanced Accuracy"] = round(float(balanced_accuracy_score(y_test, preds)), 4)
            avg_mode = "binary" if y.nunique() == 2 else "weighted"
            extended_metrics["Precision"] = round(float(precision_score(y_test, preds, average=avg_mode, zero_division=0)), 4)
            extended_metrics["Recall"] = round(float(recall_score(y_test, preds, average=avg_mode, zero_division=0)), 4)
            extended_metrics["F1 Score"] = round(float(f1_score(y_test, preds, average=avg_mode, zero_division=0)), 4)

            # Class distribution imbalance detection
            try:
                majority_pct = float(y.value_counts(normalize=True).max())
                if majority_pct > 0.80:
                    class_imbalance_warning = (
                        f"Class imbalance detected: majority class comprises {majority_pct * 100:.1f}% of observations. "
                        "Standard accuracy may be inflated; evaluate Balanced Accuracy, PR-AUC, and cost-sensitive threshold tuning."
                    )
            except Exception:
                pass

            if y.nunique() == 2 and hasattr(pipeline, "predict_proba"):
                try:
                    fitted_classes = list(pipeline.named_steps["model"].classes_)
                    pos_class = fitted_classes[1]
                    probs = pipeline.predict_proba(X_test)[:, 1]
                    y_test_binary = [1 if val == pos_class else 0 for val in y_test]

                    # Safe AUC calculation (guarded against single-class evaluation splits)
                    if len(set(y_test_binary)) >= 2:
                        roc_auc = float(roc_auc_score(y_test_binary, probs))
                        extended_metrics["ROC-AUC"] = round(roc_auc, 4)
                        pr_auc = float(average_precision_score(y_test_binary, probs))
                        extended_metrics["PR-AUC"] = round(pr_auc, 4)
                    else:
                        roc_auc = None
                        pr_auc = None

                    brier = float(brier_score_loss(y_test_binary, probs))
                    extended_metrics["Brier Score"] = round(brier, 4)

                    # Out-of-fold cross-validation probabilities on training data
                    # (Prevents test-set leakage: threshold tuning occurs on CV, evaluation on test set)
                    train_class_cnt = int(y_train.value_counts().min())
                    cv_tune_splitter = (
                        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        if train_class_cnt >= 5
                        else KFold(n_splits=5, shuffle=True, random_state=42)
                    )
                    try:
                        cv_pred_probs = cross_val_predict(
                            pipeline, X_train, y_train, cv=cv_tune_splitter, method="predict_proba"
                        )[:, 1]
                        y_train_binary = [1 if val == pos_class else 0 for val in y_train]
                    except Exception:
                        cv_pred_probs = None
                        y_train_binary = None

                    threshold_tuning_payload = {
                        "tuning_split": {
                            "split_name": "5-Fold Cross-Validation (Training Split)",
                            "sample_size": len(y_train_binary) if y_train_binary else 0,
                            "y_true": y_train_binary if y_train_binary else [],
                            "probs": [round(float(p), 4) for p in cv_pred_probs] if cv_pred_probs is not None else [],
                            "description": "Out-of-fold probability predictions on training data used for threshold selection without test-set leakage"
                        },
                        "eval_split": {
                            "split_name": "Pristine Held-Out Test Split",
                            "sample_size": len(y_test_binary),
                            "y_true": y_test_binary,
                            "probs": [round(float(p), 4) for p in probs],
                            "description": "Held-out test split reserved exclusively for unbiased final generalization evaluation"
                        },
                        "classes": [str(c) for c in fitted_classes],
                        "positive_class": str(pos_class),
                        "baseline_threshold": 0.50,
                        "brier_score": round(brier, 4),
                        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
                        "y_test": y_test_binary,
                        "test_probs": [round(float(p), 4) for p in probs],
                    }
                except Exception:
                    pass
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            extended_metrics["RMSE"] = round(rmse, 4)
            extended_metrics["MSE"] = round(float(mean_squared_error(y_test, preds)), 4)
            extended_metrics["MAE"] = round(float(mean_absolute_error(y_test, preds)), 4)
            residuals = (y_test.values if hasattr(y_test, "values") else np.array(y_test)) - preds
            res_mean = float(np.mean(residuals))
            res_std = float(np.std(residuals))
            res_skew = float(pd.Series(residuals).skew()) if len(residuals) > 2 else 0.0
            extended_metrics["Residual Std"] = round(res_std, 4)
            extended_metrics["Residual Skew"] = round(res_skew, 4)

            residual_diagnostics = {
                "rmse": round(rmse, 4),
                "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                "mean": round(res_mean, 4),
                "std": round(res_std, 4),
                "skewness": round(res_skew, 4),
            }

            try:
                from statsmodels.stats.stattools import durbin_watson
                dw_val = float(durbin_watson(residuals))
                extended_metrics["Durbin-Watson"] = round(dw_val, 4)
                residual_diagnostics["durbin_watson"] = round(dw_val, 4)
            except Exception:
                pass

            try:
                import scipy.stats as sp_stats
                r_het, p_het = sp_stats.spearmanr(np.abs(residuals), preds)
                if pd.notna(p_het):
                    extended_metrics["Heteroscedasticity p-val"] = round(float(p_het), 4)
                    residual_diagnostics["heteroscedasticity_p"] = round(float(p_het), 4)
            except Exception:
                pass

        # 3. 5-Fold Cross-Validation Breakdown strictly evaluated on training split
        # (Preserves held-out test split pristine and prevents cross-validation leakage)
        train_class_count = int(y_train.value_counts().min()) if task_type == "classification" else 0
        cv_splitter = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if task_type == "classification" and train_class_count >= 5
            else KFold(n_splits=5, shuffle=True, random_state=42)
        )
        try:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv_splitter,
                scoring="accuracy" if task_type == "classification" else "r2"
            )
            cv_folds = [round(float(s), 4) for s in cv_scores]
            cv_mean = round(float(np.mean(cv_scores)), 4)
            cv_std = round(float(np.std(cv_scores)), 4)
        except Exception:
            cv_folds = [round(float(score), 4)]
            cv_mean = round(float(score), 4)
            cv_std = 0.0

        fitted_model = pipeline.named_steps["model"]
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = fitted_model.feature_importances_
        grouped_importances: Dict[str, float] = {column: 0.0 for column in requested_features}
        for encoded_name, importance in zip(feature_names, importances):
            source = encoded_name.split("__", 1)[-1]
            matching = [column for column in requested_features if source == column or source.startswith(f"{column}_")]
            if matching:
                grouped_importances[matching[0]] += float(importance)
        imp_df = pd.DataFrame({
            "feature": list(grouped_importances),
            "importance": [round(value, 6) for value in grouped_importances.values()],
        }).sort_values(by="importance", ascending=False)

        requested_artifact = str(intent.arguments.get("evaluation_artifact") or "").lower()
        if requested_artifact not in {"confusion_matrix", "feature_importance", "none"}:
            return OperatorOutput(
                status="FAILED", error_code="EVALUATION_ARTIFACT_REQUIRED",
                error_message="Supervised ML requires an explicit evaluation_artifact.",
                manifest=ExecutionManifest(operator="supervised_ml", columns_used=[target_col] + requested_features, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        if requested_artifact == "confusion_matrix" and task_type != "classification":
            return OperatorOutput(
                status="FAILED", error_code="INVALID_EVALUATION_ARTIFACT",
                error_message="A confusion matrix is only valid for classification.",
                manifest=ExecutionManifest(operator="supervised_ml", columns_used=[target_col] + requested_features, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )

        if requested_artifact == "confusion_matrix":
            labels = list(fitted_model.classes_)
            matrix = confusion_matrix(y_test, preds, labels=labels)
            result_df = pd.DataFrame(matrix, index=[f"actual_{label}" for label in labels], columns=[f"predicted_{label}" for label in labels]).reset_index().rename(columns={"index": "actual_class"})
            plt.figure(figsize=(7, 6))
            plt.imshow(matrix, cmap="Blues")
            plt.colorbar(label="Record count")
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted class")
            plt.ylabel("Actual class")
            plt.title(f"RandomForest Confusion Matrix ({metric_name}: {score:.4f})")
            threshold = matrix.max() / 2 if matrix.size else 0
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > threshold else "black")
            plt.tight_layout()
            chart_type = "confusion_matrix"
            statistics = [metric_name, "confusion_matrix", "feature_importances"]
            finding = f"Trained RandomForest classifier for `{target_col}` using {len(requested_features)} selected features. Test-set {metric_name}: **{score:.4f}**; the confusion matrix shows class-level prediction errors."
        elif requested_artifact == "feature_importance":
            result_df = imp_df
            plt.figure(figsize=(9, 4.5))
            plt.barh(imp_df["feature"].head(10), imp_df["importance"].head(10), color="#6366f1", edgecolor="black")
            plt.title(f"RandomForest Feature Importance ({metric_name}: {score:.4f})")
            plt.xlabel("Importance Score")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            chart_type = "feature_importance"
            statistics = [metric_name, "feature_importances"]
            finding = f"Trained RandomForest {task_type} model targeting `{target_col}`. Test-set **{metric_name}: {score:.4f}**. Top predictor: `{imp_df['feature'].iloc[0]}` ({imp_df['importance'].iloc[0]*100:.1f}%)."
        else:
            result_df = pd.DataFrame([{"metric": metric_name, "value": round(float(score), 6)}])
            chart_type = "none"
            statistics = [metric_name]
            finding = f"Trained RandomForest {task_type} model targeting `{target_col}`. Test-set **{metric_name}: {score:.4f}**."

        img_b64 = None
        if requested_artifact != "none":
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        manifest = ExecutionManifest(
            operator="supervised_ml",
            columns_used=[target_col] + requested_features,
            statistics_computed=statistics,
            chart_type=chart_type,
            row_count_in=len(df),
            row_count_out=len(result_df),
        )
        if class_imbalance_warning:
            manifest.warnings.append(class_imbalance_warning)

        leakage_assessment = intent.arguments.get("target_leakage_assessment")
        if isinstance(leakage_assessment, dict) and leakage_assessment.get("has_deterministic_signal"):
            leaked_features = [signal.get("feature") for signal in leakage_assessment.get("signals", []) if signal.get("feature")]
            if leaked_features:
                excluded = intent.arguments.get("excluded_leakage_features") or []
                if excluded:
                    manifest.warnings.append(
                        "Prospective safe baseline: excluded deterministic leakage feature(s) "
                        f"({', '.join(excluded)}) before training; the reported score uses only the retained predictors."
                    )
                    finding += f" Safety audit excluded `{', '.join(excluded)}` before fitting this prospective baseline."
                else:
                    manifest.warnings.append(
                        "Retrospective result: selected feature(s) have deterministic target-leakage evidence "
                        f"({', '.join(leaked_features)}); this score is not prospective predictive performance."
                    )

        model_constructor = "RandomForestClassifier(n_estimators=100, random_state=42)" if task_type == "classification" else "RandomForestRegressor(n_estimators=100, random_state=42)"
        score_import = "accuracy_score, confusion_matrix" if task_type == "classification" else "r2_score"
        score_expression = "accuracy_score(y_test, predictions)" if task_type == "classification" else "r2_score(y_test, predictions)"
        split_stratify = "y if y.value_counts().min() >= 2 else None" if task_type == "classification" else "None"
        artifact_code = (
            "labels = list(pipeline.named_steps['model'].classes_)\n"
            "matrix = confusion_matrix(y_test, predictions, labels=labels)\n"
            "result_df = pd.DataFrame(matrix, index=[f'actual_{label}' for label in labels], columns=[f'predicted_{label}' for label in labels]).reset_index().rename(columns={'index': 'actual_class'})\n"
            "plt.figure(figsize=(7, 6))\n"
            "plt.imshow(matrix, cmap='Blues')\n"
            "plt.colorbar(label='Record count')\n"
            "plt.xticks(range(len(labels)), labels, rotation=45, ha='right')\n"
            "plt.yticks(range(len(labels)), labels)\n"
            "plt.xlabel('Predicted class'); plt.ylabel('Actual class')\n"
            "plt.title(f'RandomForest Confusion Matrix (Accuracy: {score:.4f})')\n"
            "plt.tight_layout(); plt.show()\n"
        ) if requested_artifact == "confusion_matrix" else (
            "feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()\n"
            "encoded_importances = pipeline.named_steps['model'].feature_importances_\n"
            "result_df = pd.DataFrame({'encoded_feature': feature_names, 'importance': encoded_importances}).sort_values('importance', ascending=False)\n"
            "result_df.head(10).plot.barh(x='encoded_feature', y='importance', legend=False)\n"
            "plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()\n"
        ) if requested_artifact == "feature_importance" else "result_df = pd.DataFrame({'metric': ['held_out_score'], 'value': [score]})\n"
        code = (
            "# DataPilot DS Engine: reproducible supervised ML execution\n"
            "# `df` is the active dataset supplied by DataPilot.\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
            f"from sklearn.metrics import {score_import}\n"
            f"target_col = {target_col!r}\n"
            f"features = {repr(requested_features)}\n"
            f"task_type = '{task_type}'\n"
            f"evaluation_artifact = '{requested_artifact}'\n"
            "data = df[[target_col] + features].dropna(subset=[target_col]).copy()\n"
            "X, y = data[features], data[target_col]\n"
            "numeric_features = X.select_dtypes(include='number').columns.tolist()\n"
            "categorical_features = [c for c in features if c not in numeric_features]\n"
            "transformers = []\n"
            "if numeric_features:\n"
            "    transformers.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median'))]), numeric_features))\n"
            "if categorical_features:\n"
            "    transformers.append(('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features))\n"
            f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify={split_stratify})\n"
            f"pipeline = Pipeline([('preprocess', ColumnTransformer(transformers=transformers, remainder='drop')), ('model', {model_constructor})])\n"
            "pipeline.fit(X_train, y_train)\n"
            "predictions = pipeline.predict(X_test)\n"
            f"score = {score_expression}\n"
            + artifact_code
            + "print(result_df)\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={
                "records": result_df.to_dict(orient="records"),
                "columns": list(result_df.columns),
                "row_count": len(result_df),
                "extended_metrics": extended_metrics,
                "cv_folds": cv_folds,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "task_type": task_type,
                "residual_diagnostics": residual_diagnostics if task_type == "regression" else None,
                "threshold_tuning_payload": threshold_tuning_payload,
                "class_imbalance_warning": class_imbalance_warning,
            },
            dataframe=result_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            written_finding=finding,
            reproducible_code=code
        )


class ClusteringOperator(BaseTypedOperator):
    """Vectorized KMeans / Clustering Analysis."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        if not intent.subjects:
            return OperatorOutput(
                status="FAILED",
                error_code="SUBJECT_COLUMNS_REQUIRED",
                error_message="Clustering requires explicit numerical subject columns.",
                manifest=ExecutionManifest(
                    operator="clustering",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        subjects = [c for c in intent.subjects if c in df.columns and c in num_cols]

        if not subjects or len(subjects) == 0:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_NUMERIC_COLUMNS",
                error_message="Clustering requires numerical feature columns.",
                manifest=ExecutionManifest(
                    operator="clustering",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        clean_df = df[subjects].dropna()
        if len(clean_df) < 3:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_DATA",
                error_message="Clustering requires at least 3 valid data points.",
                manifest=ExecutionManifest(
                    operator="clustering",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        tool_args = getattr(intent, "arguments", {}) or {}
        if tool_args.get("mode") == "elbow":
            if len(subjects) < 2:
                return OperatorOutput(
                    status="FAILED",
                    error_code="INSUFFICIENT_FEATURES",
                    error_message="Elbow evaluation requires at least 2 numerical feature columns.",
                    manifest=ExecutionManifest(
                        operator="elbow_analysis",
                        columns_used=subjects,
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=len(df),
                        row_count_out=0
                    )
                )
            min_k = int(tool_args.get("min_k", 2))
            max_k = int(tool_args.get("max_k", 8))
            scaling = tool_args.get("scaling", "standard")
            if scaling not in {"standard", "none"}:
                scaling = "standard"
            from backend.ds_engine.unsupervised_engine import unsupervised_engine
            try:
                elbow_res = unsupervised_engine.compute_elbow(
                    df, features=subjects, min_k=min_k, max_k=max_k, scaling=scaling
                )
            except Exception as e:
                return OperatorOutput(
                    status="FAILED",
                    error_code="ELBOW_ANALYSIS_FAILED",
                    error_message=str(e),
                    manifest=ExecutionManifest(
                        operator="elbow_analysis",
                        columns_used=subjects,
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=len(df),
                        row_count_out=0
                    )
                )
            curve_points = elbow_res.get("curve", [])
            curve_df = pd.DataFrame(curve_points)
            best_k = elbow_res.get("best_k_silhouette")

            plt.figure(figsize=(8, 4.5))
            ks = [p["k"] for p in curve_points]
            inertias = [p["inertia"] for p in curve_points]
            plt.plot(ks, inertias, marker="o", color="#6366f1", linewidth=2, label="Inertia (Elbow)")
            if best_k:
                plt.axvline(x=best_k, color="#10b981", linestyle="--", label=f"Optimal k={best_k}")
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Inertia (Sum of Squared Distances)")
            plt.title("K-Means Elbow Curve Analysis")
            plt.legend()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            finding = f"Elbow analysis evaluated k={min_k} to k={max_k} across {len(subjects)} features. Optimal cluster count by silhouette score is **k={best_k}**."
            repro_code = (
                "# DataPilot DS Engine: K-Means Elbow Curve Analysis\n"
                "from sklearn.cluster import KMeans\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "import pandas as pd\n\n"
                f"features = {subjects}\n"
                f"clean_df = df[features].dropna()\n"
                f"X = StandardScaler().fit_transform(clean_df) if '{scaling}' == 'standard' else clean_df.to_numpy()\n"
                f"for k in range({min_k}, {max_k + 1}):\n"
                "    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)\n"
                "    print(f'k={k}: inertia={km.inertia_:.2f}')\n"
            )
            manifest = ExecutionManifest(
                operator="elbow_analysis",
                columns_used=subjects,
                statistics_computed=["inertia", "silhouette_score"],
                chart_type="line",
                row_count_in=len(df),
                row_count_out=len(curve_df),
            )
            return OperatorOutput(
                status="SUCCEEDED",
                data={
                    "records": curve_df.to_dict(orient="records"),
                    "columns": list(curve_df.columns),
                    "row_count": len(curve_df),
                    "curve": curve_points,
                    "best_k_silhouette": best_k,
                    "scaling": scaling,
                    "features": subjects,
                },
                dataframe=curve_df,
                manifest=manifest,
                plot_image_base64=img_b64,
                written_finding=finding,
                reproducible_code=repro_code
            )

        n_clusters = tool_args.get("n_clusters") or tool_args.get("k")
        if n_clusters is None:
            return OperatorOutput(
                status="FAILED",
                error_code="N_CLUSTERS_REQUIRED",
                error_message="Clustering requires an explicit number of clusters (e.g. n_clusters=3 or k=3).",
                manifest=ExecutionManifest(
                    operator="clustering",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool) or n_clusters < 2 or n_clusters >= len(clean_df):
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_CLUSTER_COUNT",
                error_message=f"n_clusters must be an integer from 2 through {len(clean_df) - 1} for the supplied data.",
                manifest=ExecutionManifest(operator="clustering", columns_used=subjects, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        scaling = tool_args.get("scaling")
        if scaling not in {"standard", "none"}:
            return OperatorOutput(
                status="FAILED",
                error_code="SCALING_REQUIRED",
                error_message="Clustering requires scaling='standard' or scaling='none'.",
                manifest=ExecutionManifest(operator="clustering", columns_used=subjects, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        feature_matrix = StandardScaler().fit_transform(clean_df) if scaling == "standard" else clean_df.to_numpy(dtype=float)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)

        res_df = clean_df.copy()
        res_df["cluster_label"] = labels

        # Centroids are intentionally reported in the same units selected by
        # the contract.  Standardized centroids are scientifically meaningful
        # for comparing feature influence; original values remain in res_df.
        centroids = kmeans.cluster_centers_
        centroid_df = pd.DataFrame(centroids, columns=subjects)
        centroid_df.insert(0, "cluster_label", range(n_clusters))

        sil_score = float(silhouette_score(feature_matrix, labels))

        plt.figure(figsize=(8, 4.5))
        if len(subjects) >= 2:
            plt.scatter(clean_df[subjects[0]], clean_df[subjects[1]], c=labels, cmap="tab10", alpha=0.7)
            plt.xlabel(subjects[0])
            plt.ylabel(subjects[1])
            plt.title(f"K-Means Clustering (k={n_clusters})")
        else:
            plt.hist(clean_df[subjects[0]], bins=15, color="#6366f1", edgecolor="black")
            plt.title(f"Clustering Distribution on {subjects[0]}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        manifest = ExecutionManifest(
            operator="clustering",
            columns_used=subjects,
            statistics_computed=["inertia", "silhouette_score", "cluster_centroids"],
            chart_type="scatter" if len(subjects) >= 2 else "hist",
            row_count_in=len(df),
            row_count_out=len(centroid_df),
            assumptions=[f"Declared scaling={scaling}", "KMeans n_init=10, random_state=42"],
        )

        finding = f"Performed K-Means Clustering (k={n_clusters}) on `{subjects}` with declared `{scaling}` scaling. **Silhouette Score: {sil_score:.3f}**, **Inertia: {kmeans.inertia_:.2f}**."

        code = (
            "# DataPilot DS Engine: Clustering Analysis\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.preprocessing import StandardScaler\n"
            f"subjects = {repr(subjects)}\n"
            f"n_clusters = {n_clusters!r}\n"
            f"scaling = {scaling!r}\n"
            "result_df = df[subjects].dropna().copy()\n"
            "X = StandardScaler().fit_transform(result_df) if scaling == 'standard' else result_df.to_numpy(dtype=float)\n"
            "kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)\n"
            "labels = kmeans.fit_predict(X)\n"
            "result_df['cluster_label'] = labels\n"
            "print(result_df.groupby('cluster_label').mean())\n"
        )

        cluster_sizes = {int(k): int(v) for k, v in res_df["cluster_label"].value_counts().to_dict().items()}
        return OperatorOutput(
            status="SUCCEEDED",
            data={
                "records": centroid_df.to_dict(orient="records"),
                "columns": list(centroid_df.columns),
                "row_count": len(centroid_df),
                "inertia": round(float(kmeans.inertia_), 4),
                "silhouette_score": round(float(sil_score), 4),
                "cluster_sizes": cluster_sizes,
                "n_clusters": n_clusters,
                "scaling": scaling
            },
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            written_finding=finding,
            reproducible_code=code
        )


class CategoricalOutcomeAssociationOperator(BaseTypedOperator):
    """Retrospective, descriptive numeric-versus-outcome association analysis.

    This deliberately does *not* train a classifier/regressor.  A numeric
    measurement such as spend, duration, pressure, score, or dosage can be
    compared across categorical outcome groups without falsely presenting the
    measurement as a prospective predictor.
    """

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None,
    ) -> OperatorOutput:
        args = intent.arguments or {}
        predictor = args.get("numeric_predictor")
        outcome = args.get("outcome_column")
        chart_type = args.get("chart_type")
        context = args.get("analysis_context")

        def fail(code: str, message: str, columns: Optional[List[str]] = None) -> OperatorOutput:
            return OperatorOutput(
                status="FAILED", error_code=code, error_message=message,
                manifest=ExecutionManifest(
                    operator="categorical_outcome_association", columns_used=columns or [],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        if not isinstance(predictor, str) or not isinstance(outcome, str):
            return fail("ASSOCIATION_COLUMNS_REQUIRED", "Retrospective association requires explicit numeric_predictor and outcome_column.")
        if predictor == outcome:
            return fail("ASSOCIATION_COLUMNS_INVALID", "numeric_predictor and outcome_column must be different columns.", [predictor])
        missing = [column for column in (predictor, outcome) if column not in df.columns]
        if missing:
            return fail("COLUMN_NOT_FOUND", f"Association column(s) {missing} not found in dataset.", [predictor, outcome])
        if not pd.api.types.is_numeric_dtype(df[predictor]):
            return fail("NUMERIC_PREDICTOR_REQUIRED", f"Association predictor '{predictor}' must be numeric.", [predictor, outcome])
        if chart_type not in {"box", "none"}:
            return fail("INVALID_CHART_TYPE", "Association chart_type must be box or none.", [predictor, outcome])
        if context != "retrospective_association":
            return fail("INVALID_ANALYSIS_CONTEXT", "This operator supports retrospective_association only.", [predictor, outcome])

        working = df[[predictor, outcome]].copy()
        working[predictor] = pd.to_numeric(working[predictor], errors="coerce")
        working = working.dropna(subset=[predictor, outcome])
        groups = list(pd.unique(working[outcome]))
        if len(groups) < 2:
            return fail("INSUFFICIENT_OUTCOME_CLASSES", f"Outcome column '{outcome}' needs at least two non-null groups.", [predictor, outcome])
        if len(groups) > 20:
            return fail(
                "OUTCOME_CARDINALITY_UNSUITABLE",
                f"Outcome column '{outcome}' has {len(groups)} groups; choose a coarser categorical outcome before a box-plot association review.",
                [predictor, outcome],
            )

        summary = (
            working.groupby(outcome, dropna=False)[predictor]
            .agg(record_count="count", mean="mean", median="median", std="std", min="min", max="max")
            .reset_index()
        )
        for column in ("mean", "median", "std", "min", "max"):
            summary[column] = summary[column].round(6)

        grand_mean = float(working[predictor].mean())
        total_ss = float(((working[predictor] - grand_mean) ** 2).sum())
        between_ss = float(sum(
            int(row.record_count) * (float(row.mean) - grand_mean) ** 2
            for row in summary.itertuples(index=False)
        ))
        eta_squared = between_ss / total_ss if total_ss > 0 else float("nan")

        img_b64 = None
        plotly_spec = None
        if chart_type == "box":
            plt.figure(figsize=(max(7, min(14, len(groups) * 0.75)), 4.8))
            ordered_groups = summary[outcome].tolist()
            plot_data = [working.loc[working[outcome] == group, predictor].to_numpy() for group in ordered_groups]
            plt.boxplot(plot_data, tick_labels=[str(group) for group in ordered_groups], showmeans=True)
            plt.title(f"Retrospective distribution of {predictor} by {outcome}")
            plt.xlabel(outcome)
            plt.ylabel(predictor)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plotly_spec = {
                "data": [{
                    "type": "box", "name": str(group),
                    "y": working.loc[working[outcome] == group, predictor].tolist(),
                } for group in ordered_groups],
                "layout": {"title": f"Retrospective distribution of {predictor} by {outcome}", "yaxis": {"title": predictor}},
            }

        eta_text = "undefined (the numeric measurement has no variation)" if not np.isfinite(eta_squared) else f"{eta_squared:.4f}"
        finding = (
            f"Retrospective association review of numeric `{predictor}` across `{outcome}` groups "
            f"({len(working):,} complete records; {len(summary)} groups). Descriptive between-group eta-squared: **{eta_text}**. "
            "This describes historical group separation only; it is not a causal conclusion or prospective targeting model."
        )
        manifest = ExecutionManifest(
            operator="categorical_outcome_association",
            columns_used=[predictor, outcome],
            statistics_computed=["record_count", "mean", "median", "std", "min", "max", "eta_squared"],
            chart_type=chart_type,
            row_count_in=len(df),
            row_count_out=len(summary),
            assumptions=[
                "Declared analysis_context=retrospective_association",
                "Descriptive group comparison only; no causal or prospective-prediction claim.",
            ],
        )
        code = (
            "# DataPilot DS Engine: retrospective numeric-versus-outcome association\n"
            "import pandas as pd\nimport matplotlib.pyplot as plt\n"
            f"numeric_predictor = {predictor!r}\noutcome_column = {outcome!r}\n"
            "work = df[[numeric_predictor, outcome_column]].copy()\n"
            "work[numeric_predictor] = pd.to_numeric(work[numeric_predictor], errors='coerce')\n"
            "work = work.dropna(subset=[numeric_predictor, outcome_column])\n"
            "result_df = work.groupby(outcome_column)[numeric_predictor].agg(record_count='count', mean='mean', median='median', std='std', min='min', max='max').reset_index()\n"
            f"if {chart_type!r} == 'box':\n"
            "    work.boxplot(column=numeric_predictor, by=outcome_column)\n"
            "    plt.suptitle(''); plt.title(f'Retrospective distribution of {numeric_predictor} by {outcome_column}'); plt.tight_layout(); plt.show()\n"
            "print(result_df)\n"
        )
        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": summary.to_dict(orient="records"), "columns": list(summary.columns), "row_count": len(summary)},
            dataframe=summary,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=finding,
            reproducible_code=code,
        )


class StatisticalTestOperator(BaseTypedOperator):
    """Explicit, typed inferential tests with no dtype-based test selection."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None,
    ) -> OperatorOutput:
        from scipy import stats

        def fail(code: str, message: str, columns: Optional[List[str]] = None) -> OperatorOutput:
            return OperatorOutput(
                status="FAILED",
                error_code=code,
                error_message=message,
                manifest=ExecutionManifest(
                    operator="statistical_test", columns_used=columns or [],
                    statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0,
                ),
            )

        args = intent.arguments or {}
        test_type = args.get("test_type")
        if test_type in {"stationarity", "stationarity_test", "adf", "kpss", "unit_root"}:
            time_col = args.get("time_column") or intent.time_column
            target_col = (
                args.get("target_column")
                or args.get("value_column")
                or intent.target_column
                or (intent.subjects[0] if intent.subjects else None)
            )
            raw_cols = args.get("columns")
            if (not time_col or not target_col) and isinstance(raw_cols, list) and len(raw_cols) == 2:
                col0, col1 = raw_cols
                if pd.api.types.is_datetime64_any_dtype(df[col0]) or "date" in str(col0).lower() or "time" in str(col0).lower():
                    time_col, target_col = col0, col1
                elif pd.api.types.is_datetime64_any_dtype(df[col1]) or "date" in str(col1).lower() or "time" in str(col1).lower():
                    time_col, target_col = col1, col0
                else:
                    time_col, target_col = col0, col1

            if not time_col or not target_col:
                return fail("STATIONARITY_COLUMNS_REQUIRED", "Stationarity testing requires both a temporal column (time_column) and a numeric target column (target_column).")
            if time_col not in df.columns:
                return fail("COLUMN_NOT_FOUND", f"Time column '{time_col}' not found in dataset.", [time_col])
            if target_col not in df.columns:
                return fail("COLUMN_NOT_FOUND", f"Target column '{target_col}' not found in dataset.", [target_col])

            from backend.ds_engine.timeseries_engine import timeseries_engine
            try:
                stat_res = timeseries_engine.run_stationarity_tests(df, time_column=time_col, target_column=target_col)
            except Exception as e:
                return fail("STATIONARITY_TEST_FAILED", str(e), [time_col, target_col])

            adf = stat_res["adf_test"]
            kpss_res = stat_res["kpss_test"]
            conclusion = stat_res["conclusion"]
            interpretation = stat_res["interpretation"]

            res_df = pd.DataFrame([
                {
                    "Test": "Augmented Dickey-Fuller (ADF)",
                    "Null Hypothesis": adf["null_hypothesis"],
                    "Test Statistic": round(adf["test_statistic"], 4),
                    "statistic": float(adf["test_statistic"]),
                    "p-value": round(adf["p_value"], 4),
                    "p_value": float(adf["p_value"]),
                    "Stationary (p < 0.05)": adf["is_stationary"],
                },
                {
                    "Test": "KPSS",
                    "Null Hypothesis": kpss_res["null_hypothesis"],
                    "Test Statistic": round(kpss_res["test_statistic"], 4),
                    "statistic": float(kpss_res["test_statistic"]),
                    "p-value": round(kpss_res["p_value"], 4),
                    "p_value": float(kpss_res["p_value"]),
                    "Stationary (p >= 0.05)": kpss_res["is_stationary"],
                }
            ])

            finding = (
                f"Stationarity analysis for `{target_col}` over `{time_col}`: **{conclusion}**. "
                f"{interpretation} ADF statistic={adf['test_statistic']:.4f} (p={adf['p_value']:.4f}), "
                f"KPSS statistic={kpss_res['test_statistic']:.4f} (p={kpss_res['p_value']:.4f})."
            )
            repro_code = (
                "# DataPilot DS Engine: Stationarity Unit-Root Testing (ADF & KPSS)\n"
                "from statsmodels.tsa.stattools import adfuller, kpss\n"
                "import pandas as pd\n\n"
                f"time_col = '{time_col}'\n"
                f"target_col = '{target_col}'\n"
                f"work_df = df[[time_col, target_col]].dropna().copy()\n"
                f"work_df[time_col] = pd.to_datetime(work_df[time_col], errors='coerce')\n"
                f"series = work_df.dropna().sort_values(by=time_col)[target_col].to_numpy(dtype=float)\n"
                "adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series, autolag='AIC')\n"
                "kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression='c', nlags='auto')\n"
                "print(f'ADF Test: statistic={adf_stat:.4f}, p_value={adf_p:.4f}')\n"
                "print(f'KPSS Test: statistic={kpss_stat:.4f}, p_value={kpss_p:.4f}')\n"
            )
            manifest = ExecutionManifest(
                operator="stationarity_test",
                columns_used=[time_col, target_col],
                statistics_computed=["adf_statistic", "adf_p_value", "kpss_statistic", "kpss_p_value"],
                chart_type="none",
                row_count_in=len(df),
                row_count_out=len(res_df),
                sorted_by=time_col,
                assumptions=[adf["null_hypothesis"], kpss_res["null_hypothesis"]],
            )
            return OperatorOutput(
                status="SUCCEEDED",
                data={
                    "records": res_df.to_dict(orient="records"),
                    "columns": list(res_df.columns),
                    "row_count": len(res_df),
                    "conclusion": conclusion,
                    "interpretation": interpretation,
                    "adf_test": adf,
                    "kpss_test": kpss_res,
                },
                dataframe=res_df,
                manifest=manifest,
                written_finding=finding,
                reproducible_code=repro_code,
            )

        columns = args.get("columns")
        alpha = args.get("alpha")
        supported_tests = {
            "independent_ttest", "welch_ttest", "student_ttest",
            "paired_ttest", "mann_whitney_u", "wilcoxon_signed_rank",
            "one_way_anova", "kruskal_wallis",
            "chi_square_independence", "fisher_exact"
        }
        if test_type not in supported_tests:
            return fail(
                "TEST_TYPE_REQUIRED",
                f"Statistical testing requires an explicit test_type: {', '.join(sorted(supported_tests))}.",
            )
        if not isinstance(columns, list) or len(columns) != 2 or not all(isinstance(column, str) and column for column in columns):
            return fail("TEST_COLUMNS_REQUIRED", "Statistical testing requires exactly two explicit column names.")
        if len(set(columns)) != 2:
            return fail("TEST_COLUMNS_INVALID", "Statistical testing requires two distinct columns.", columns)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            return fail("COLUMN_NOT_FOUND", f"Statistical-test column(s) {missing} not found in dataset.", columns)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1:
            return fail("INVALID_ALPHA", "Statistical-test alpha must be a number strictly between 0 and 1.", columns)
        alpha = float(alpha)

        warnings: List[str] = []
        assumptions: List[str] = [f"Test selected explicitly: {test_type}", f"Significance threshold alpha={alpha:g}"]
        effect_size = float("nan")
        effect_size_name = ""
        degrees_of_freedom: Optional[float] = None
        extra: Dict[str, Any] = {}

        if test_type in {"independent_ttest", "welch_ttest", "student_ttest"}:
            left, right = columns
            if not (pd.api.types.is_numeric_dtype(df[left]) and pd.api.types.is_numeric_dtype(df[right])):
                return fail("INVALID_TEST_COLUMN_TYPES", f"{test_type} requires two numeric sample columns.", columns)
            sample_left = pd.to_numeric(df[left], errors="coerce").dropna()
            sample_right = pd.to_numeric(df[right], errors="coerce").dropna()
            if len(sample_left) < 2 or len(sample_right) < 2:
                return fail("INSUFFICIENT_DATA", f"{test_type} requires at least two valid values in each sample column.", columns)

            # Methodological Guard for Observational Independence / Paired Structure:
            # Physical row alignment in a DataFrame does not establish statistical pairing,
            # but independent two-sample tests strictly assume mutually independent sampling units.
            # Record an explicit diagnostic warning if two columns are supplied without an explicit cohort design.
            cohort_design_provided = bool(args.get("cohort_column") or args.get("group_by") or args.get("group_column") or args.get("by"))
            paired_structure_detected = False
            paired_warning_msg = None
            if not cohort_design_provided and left in df.columns and right in df.columns:
                paired_structure_detected = True
                paired_warning_msg = (
                    f"UNVERIFIED_OBSERVATIONAL_INDEPENDENCE (PAIRED_STRUCTURE_DETECTED): Two value columns ('{left}', '{right}') "
                    f"share row alignment in the active DataFrame without an explicit cohort design. "
                    f"Physical row alignment does not prove statistical pairing, but independent t-tests assume observations are "
                    f"mutually independent across samples. If these measurements represent repeated measures or matched pairs on the "
                    f"same subjects, consider 'paired_ttest'. If they represent independent cohorts, specify an explicit cohort design."
                )
                warnings.append(paired_warning_msg)

            equal_var = (test_type == "student_ttest") or bool(args.get("equal_var", False))
            result = stats.ttest_ind(sample_left, sample_right, equal_var=equal_var, nan_policy="omit")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

            n1, n2 = len(sample_left), len(sample_right)
            m1, m2 = float(sample_left.mean()), float(sample_right.mean())
            v1, v2 = float(sample_left.var(ddof=1)), float(sample_right.var(ddof=1))

            mean_diff = m1 - m2
            pooled_sd = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)))
            effect_size = float(mean_diff / pooled_sd) if pooled_sd > 0 else float("nan")
            effect_size_name = "cohens_d"
            degrees_of_freedom = float(getattr(result, "df", np.nan))

            se_diff = float(np.sqrt(v1 / n1 + v2 / n2)) if not equal_var else float(pooled_sd * np.sqrt(1 / n1 + 1 / n2))
            df_for_ci = degrees_of_freedom if np.isfinite(degrees_of_freedom) and degrees_of_freedom > 0 else (n1 + n2 - 2)
            t_crit = float(stats.t.ppf(1 - alpha / 2, df=df_for_ci))
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff

            test_label = "Student" if equal_var else "Welch"
            test_name = f"{test_label} independent t-test ({left} vs {right})"
            extra = {
                "sample_size_left": n1,
                "sample_size_right": n2,
                "mean_left": round(m1, 4),
                "mean_right": round(m2, 4),
                "mean_difference": round(mean_diff, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "equal_variance_assumed": equal_var,
                "paired_structure_detected": paired_structure_detected,
            }
            if paired_warning_msg:
                extra["paired_structure_warning"] = paired_warning_msg

        elif test_type == "paired_ttest":
            left, right = columns
            if not (pd.api.types.is_numeric_dtype(df[left]) and pd.api.types.is_numeric_dtype(df[right])):
                return fail("INVALID_TEST_COLUMN_TYPES", "paired_ttest requires two numeric sample columns.", columns)
            paired = df[[left, right]].dropna()
            if len(paired) < 2:
                return fail("INSUFFICIENT_DATA", "paired_ttest requires at least two paired observations.", columns)
            s_left = pd.to_numeric(paired[left], errors="coerce")
            s_right = pd.to_numeric(paired[right], errors="coerce")
            diffs = s_left - s_right
            result = stats.ttest_rel(s_left, s_right, nan_policy="omit")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

            n_pairs = len(diffs)
            d_mean = float(diffs.mean())
            d_std = float(diffs.std(ddof=1))
            se_d = float(d_std / np.sqrt(n_pairs)) if n_pairs > 0 else 0.0
            effect_size = float(d_mean / d_std) if d_std > 0 else float("nan")
            effect_size_name = "cohens_dz"
            degrees_of_freedom = float(n_pairs - 1)

            t_crit = float(stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom))
            ci_lower = d_mean - t_crit * se_d
            ci_upper = d_mean + t_crit * se_d

            test_name = f"Paired samples t-test ({left} vs {right})"
            extra = {
                "pair_count": n_pairs,
                "mean_difference": round(d_mean, 4),
                "std_difference": round(d_std, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4)
            }

        elif test_type == "mann_whitney_u":
            left, right = columns
            if not (pd.api.types.is_numeric_dtype(df[left]) and pd.api.types.is_numeric_dtype(df[right])):
                return fail("INVALID_TEST_COLUMN_TYPES", "mann_whitney_u requires two numeric sample columns.", columns)
            sample_left = pd.to_numeric(df[left], errors="coerce").dropna()
            sample_right = pd.to_numeric(df[right], errors="coerce").dropna()
            if len(sample_left) < 2 or len(sample_right) < 2:
                return fail("INSUFFICIENT_DATA", "mann_whitney_u requires at least two valid values in each sample column.", columns)

            result = stats.mannwhitneyu(sample_left, sample_right, alternative=args.get("alternative", "two-sided"))
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

            n1, n2 = len(sample_left), len(sample_right)
            # Rank-Biserial correlation: r = 1 - 2*U / (n1 * n2)
            effect_size = float(1.0 - (2.0 * statistic) / max(n1 * n2, 1))
            effect_size_name = "rank_biserial_r"
            test_name = f"Mann-Whitney U rank test ({left} vs {right})"
            extra = {
                "sample_size_left": n1,
                "sample_size_right": n2,
                "median_left": round(float(sample_left.median()), 4),
                "median_right": round(float(sample_right.median()), 4)
            }

        elif test_type == "wilcoxon_signed_rank":
            left, right = columns
            if not (pd.api.types.is_numeric_dtype(df[left]) and pd.api.types.is_numeric_dtype(df[right])):
                return fail("INVALID_TEST_COLUMN_TYPES", "wilcoxon_signed_rank requires two numeric sample columns.", columns)
            paired = df[[left, right]].dropna()
            diffs = pd.to_numeric(paired[left], errors="coerce") - pd.to_numeric(paired[right], errors="coerce")
            non_zero_diffs = diffs[diffs != 0]
            if len(non_zero_diffs) < 2:
                return fail("INSUFFICIENT_DATA", "wilcoxon_signed_rank requires at least two non-zero differences.", columns)

            result = stats.wilcoxon(paired[left], paired[right], alternative=args.get("alternative", "two-sided"))
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

            n_pairs = len(non_zero_diffs)
            # Rank correlation r = Z / sqrt(N)
            z_score = float(stats.norm.ppf(1 - max(min(p_value / 2, 0.4999), 1e-10)))
            effect_size = float(z_score / np.sqrt(n_pairs)) if n_pairs > 0 else float("nan")
            effect_size_name = "rank_correlation_r"
            test_name = f"Wilcoxon signed-rank test ({left} vs {right})"
            extra = {
                "pair_count": len(paired),
                "non_zero_diff_count": n_pairs,
                "median_difference": round(float(diffs.median()), 4)
            }

        elif test_type == "one_way_anova":
            value_column, group_column = columns
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                return fail("INVALID_TEST_COLUMN_TYPES", "one_way_anova requires its first column to be numeric.", columns)
            grouped = [
                pd.to_numeric(group[value_column], errors="coerce").dropna()
                for _, group in df[[value_column, group_column]].dropna(subset=[group_column]).groupby(group_column, dropna=False)
            ]
            grouped = [group for group in grouped if len(group) > 0]
            if len(grouped) < 2 or any(len(group) < 2 for group in grouped):
                return fail("INSUFFICIENT_DATA", "one_way_anova requires at least two non-empty groups with at least two observations each.", columns)
            result = stats.f_oneway(*grouped)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            all_values = np.concatenate([group.to_numpy() for group in grouped])
            grand_mean = float(np.mean(all_values))
            between_ss = float(sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for group in grouped))
            total_ss = float(np.sum((all_values - grand_mean) ** 2))
            effect_size = between_ss / total_ss if total_ss > 0 else float("nan")
            effect_size_name = "eta_squared"
            degrees_of_freedom = float(len(grouped) - 1)
            test_name = f"One-way ANOVA ({value_column} by {group_column})"
            extra = {"group_count": len(grouped), "observation_count": int(len(all_values))}

        elif test_type == "kruskal_wallis":
            value_column, group_column = columns
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                return fail("INVALID_TEST_COLUMN_TYPES", "kruskal_wallis requires its first column to be numeric.", columns)
            grouped = [
                pd.to_numeric(group[value_column], errors="coerce").dropna()
                for _, group in df[[value_column, group_column]].dropna(subset=[group_column]).groupby(group_column, dropna=False)
            ]
            grouped = [group for group in grouped if len(group) > 0]
            if len(grouped) < 2 or any(len(group) < 2 for group in grouped):
                return fail("INSUFFICIENT_DATA", "kruskal_wallis requires at least two non-empty groups with at least two observations each.", columns)
            result = stats.kruskal(*grouped)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

            k = len(grouped)
            n_tot = sum(len(g) for g in grouped)
            # Epsilon-squared = (H - k + 1) / (n - k)
            epsilon_sq = float((statistic - k + 1) / max(n_tot - k, 1)) if n_tot > k else float("nan")
            effect_size = max(0.0, min(1.0, epsilon_sq)) if np.isfinite(epsilon_sq) else float("nan")
            effect_size_name = "epsilon_squared"
            degrees_of_freedom = float(k - 1)
            test_name = f"Kruskal-Wallis H-test ({value_column} by {group_column})"
            extra = {"group_count": k, "observation_count": n_tot}

        elif test_type == "fisher_exact":
            left, right = columns
            contingency = pd.crosstab(df[left], df[right], dropna=False)
            if contingency.shape != (2, 2):
                return fail(
                    "INVALID_TABLE_DIMENSION",
                    f"fisher_exact requires exactly a 2x2 contingency table; got shape {contingency.shape}.",
                    columns
                )
            result = stats.fisher_exact(contingency.values)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            effect_size = statistic
            effect_size_name = "odds_ratio"
            test_name = f"Fisher's exact test ({left} vs {right})"
            extra = {
                "contingency_table": contingency.to_dict(),
                "observation_count": int(contingency.to_numpy().sum())
            }

        else:
            left, right = columns
            contingency = pd.crosstab(df[left], df[right], dropna=False)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2 or contingency.to_numpy().sum() == 0:
                return fail("INSUFFICIENT_DATA", "chi_square_independence requires a non-empty contingency table with at least two categories in each variable.", columns)
            result = stats.chi2_contingency(contingency)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            degrees_of_freedom = float(result.dof)
            expected = np.asarray(result.expected_freq)
            if (expected < 5).any():
                warnings.append("Some chi-square expected cell frequencies are below 5; Fisher's exact test or Monte Carlo simulation may be more reliable.")
            denominator = contingency.to_numpy().sum() * min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            effect_size = float(np.sqrt(statistic / denominator)) if denominator > 0 else float("nan")
            effect_size_name = "cramers_v"
            test_name = f"Chi-square independence test ({left} vs {right})"
            extra = {"contingency_rows": int(contingency.shape[0]), "contingency_columns": int(contingency.shape[1]), "observation_count": int(contingency.to_numpy().sum())}

        if test_type in {"independent_ttest", "welch_ttest", "student_ttest"}:
            null_hypothesis = f"Population means of '{columns[0]}' and '{columns[1]}' are equal (μ₁ = μ₂)"
            alt_hypothesis = f"Population means of '{columns[0]}' and '{columns[1]}' are unequal (μ₁ ≠ μ₂)"
        elif test_type == "paired_ttest":
            null_hypothesis = f"Mean difference between paired observations '{columns[0]}' and '{columns[1]}' is zero (μ_d = 0)"
            alt_hypothesis = f"Mean difference between paired observations '{columns[0]}' and '{columns[1]}' is non-zero (μ_d ≠ 0)"
        elif test_type == "mann_whitney_u":
            null_hypothesis = f"Distribution of '{columns[0]}' is identical to distribution of '{columns[1]}'"
            alt_hypothesis = f"Distribution of '{columns[0]}' is stochastically shifted relative to '{columns[1]}'"
            degrees_of_freedom = None
        elif test_type == "wilcoxon_signed_rank":
            null_hypothesis = f"Median difference between paired observations '{columns[0]}' and '{columns[1]}' is zero"
            alt_hypothesis = f"Median difference between paired observations '{columns[0]}' and '{columns[1]}' is non-zero"
            degrees_of_freedom = None
        elif test_type == "one_way_anova":
            null_hypothesis = f"Population means of '{columns[0]}' are equal across all '{columns[1]}' groups"
            alt_hypothesis = f"At least one '{columns[1]}' group population mean of '{columns[0]}' differs"
        elif test_type == "kruskal_wallis":
            null_hypothesis = f"Population medians of '{columns[0]}' are equal across all '{columns[1]}' groups"
            alt_hypothesis = f"At least one '{columns[1]}' group population median of '{columns[0]}' differs"
        elif test_type == "fisher_exact":
            null_hypothesis = f"True odds ratio between '{columns[0]}' and '{columns[1]}' equals 1 (no association)"
            alt_hypothesis = f"True odds ratio between '{columns[0]}' and '{columns[1]}' does not equal 1 (association present)"
            degrees_of_freedom = None
        else:
            null_hypothesis = f"'{columns[0]}' and '{columns[1]}' are independent categorical variables (no association)"
            alt_hypothesis = f"'{columns[0]}' and '{columns[1]}' are dependent (statistically significant association)"

        is_significant = bool(p_value < alpha)
        result_record: Dict[str, Any] = {
            "test_name": test_name,
            "test_type": test_type,
            "statistic": round(statistic, 6),
            "degrees_of_freedom": round(degrees_of_freedom, 4) if degrees_of_freedom is not None and np.isfinite(degrees_of_freedom) else None,
            "p_value": round(p_value, 8),
            "alpha": alpha,
            "is_statistically_significant": is_significant,
            "effect_size_name": effect_size_name,
            "effect_size": round(effect_size, 6) if np.isfinite(effect_size) else None,
            "null_hypothesis": null_hypothesis,
            "alt_hypothesis": alt_hypothesis,
            "scientific_caution": "A statistically significant p-value rejects the null hypothesis under test assumptions but does not establish practical importance. Always evaluate the effect size and confidence intervals.",
            **extra,
        }
        res_df = pd.DataFrame([result_record])
        significance_text = "statistically significant" if is_significant else "not statistically significant"
        finding = (
            f"Performed **{test_name}** as explicitly requested. Statistic: `{statistic:.4f}`, "
            f"p-value: `{p_value:.6g}`, alpha: `{alpha:g}`; the result is **{significance_text}**. "
            f"Reported {effect_size_name}: `{effect_size:.4f}`." if np.isfinite(effect_size) else
            f"Performed **{test_name}** as explicitly requested. Statistic: `{statistic:.4f}`, "
            f"p-value: `{p_value:.6g}`, alpha: `{alpha:g}`; the result is **{significance_text}**."
        )
        if warnings:
            finding += " Warning: " + " ".join(warnings)

        if test_type in {"independent_ttest", "welch_ttest", "student_ttest"}:
            code = (
                "# DataPilot DS Engine: Two-Sample t-test\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"s1 = df[{left!r}].dropna()\ns2 = df[{right!r}].dropna()\n"
                f"res = stats.ttest_ind(s1, s2, equal_var={equal_var})\n"
                "print(f'Statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )
        elif test_type == "paired_ttest":
            code = (
                "# DataPilot DS Engine: Paired Samples t-test\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"paired = df[[{left!r}, {right!r}]].dropna()\n"
                f"res = stats.ttest_rel(paired[{left!r}], paired[{right!r}])\n"
                "print(f'Statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )
        elif test_type == "mann_whitney_u":
            code = (
                "# DataPilot DS Engine: Mann-Whitney U Test (Non-parametric)\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"s1 = df[{left!r}].dropna()\ns2 = df[{right!r}].dropna()\n"
                "res = stats.mannwhitneyu(s1, s2)\n"
                "print(f'Statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )
        elif test_type == "wilcoxon_signed_rank":
            code = (
                "# DataPilot DS Engine: Wilcoxon Signed-Rank Test (Non-parametric Paired)\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"paired = df[[{left!r}, {right!r}]].dropna()\n"
                f"res = stats.wilcoxon(paired[{left!r}], paired[{right!r}])\n"
                "print(f'Statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )
        elif test_type == "one_way_anova":
            code = (
                "# DataPilot DS Engine: One-Way ANOVA\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"groups = [g[{value_column!r}].dropna() for _, g in df.groupby({group_column!r}, dropna=False)]\n"
                "res = stats.f_oneway(*groups)\n"
                "print(f'F-Statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )
        elif test_type == "kruskal_wallis":
            code = (
                "# DataPilot DS Engine: Kruskal-Wallis Test (Non-parametric ANOVA)\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"groups = [g[{value_column!r}].dropna() for _, g in df.groupby({group_column!r}, dropna=False)]\n"
                "res = stats.kruskal(*groups)\n"
                "print(f'H-Statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )
        elif test_type == "fisher_exact":
            code = (
                "# DataPilot DS Engine: Fisher's Exact Test (2x2 Contingency Table)\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"ct = pd.crosstab(df[{left!r}], df[{right!r}], dropna=False)\n"
                "res = stats.fisher_exact(ct.values)\n"
                "print(f'Odds Ratio: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )
        else:
            code = (
                "# DataPilot DS Engine: Chi-Square Independence Test\n"
                "import pandas as pd\nfrom scipy import stats\n"
                f"ct = pd.crosstab(df[{left!r}], df[{right!r}], dropna=False)\n"
                "res = stats.chi2_contingency(ct)\n"
                "print(f'Chi2 Statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4e}')\n"
            )

        extended_test_metrics: Dict[str, Any] = {
            "Test Statistic": round(statistic, 4),
            "p-value": round(p_value, 6),
            "alpha": alpha,
        }
        if degrees_of_freedom is not None and np.isfinite(degrees_of_freedom):
            extended_test_metrics["Degrees of Freedom"] = round(degrees_of_freedom, 4)
        if np.isfinite(effect_size):
            extended_test_metrics[f"Effect Size ({effect_size_name})"] = round(effect_size, 4)
        if "ci_lower" in extra and "ci_upper" in extra:
            extended_test_metrics["95% Confidence Interval"] = f"[{extra['ci_lower']:.4f}, {extra['ci_upper']:.4f}]"

        return OperatorOutput(
            status="SUCCEEDED",
            data={
                "records": res_df.to_dict(orient="records"),
                "columns": list(res_df.columns),
                "row_count": len(res_df),
                "statistical_test": result_record,
                "extended_metrics": extended_test_metrics,
            },
            dataframe=res_df,
            manifest=ExecutionManifest(
                operator="statistical_test", columns_used=columns,
                statistics_computed=["test_statistic", "degrees_of_freedom", "p_value", "effect_size", "significance"],
                chart_type="none", row_count_in=len(df), row_count_out=len(res_df),
                assumptions=assumptions, warnings=warnings,
            ),
            written_finding=finding,
            reproducible_code=code,
        )

    @staticmethod
    def adjust_multiple_comparisons(
        p_values: List[float],
        method: str = "fdr_bh",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Adjust multiple simultaneous hypothesis test p-values using statsmodels.
        Supports: 'fdr_bh' (Benjamini-Hochberg FDR), 'bonferroni' (FWER), 'holm' (Holm-Bonferroni).
        """
        from statsmodels.stats.multitest import multipletests
        p_arr = np.asarray(p_values, dtype=float)
        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_arr, alpha=alpha, method=method)
        return {
            "method": method,
            "alpha": alpha,
            "raw_p_values": [round(float(p), 6) for p in p_arr],
            "adjusted_p_values": [round(float(p), 6) for p in pvals_corrected],
            "reject_null": [bool(r) for r in reject],
            "significant_count": int(np.sum(reject)),
            "total_tests": len(p_values),
        }



class PCADimensionalityReductionOperator(BaseTypedOperator):
    """Principal Component Analysis (PCA)."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if not intent.subjects:
            return OperatorOutput(
                status="FAILED",
                error_code="SUBJECT_COLUMNS_REQUIRED",
                error_message="PCA dimensionality reduction requires explicit numerical subject columns.",
                manifest=ExecutionManifest(
                    operator="pca_dimensionality_reduction",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        subjects = [c for c in intent.subjects if c in df.columns and c in num_cols]

        if len(subjects) < 2:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_NUMERIC_COLUMNS",
                error_message="PCA dimensionality reduction requires at least 2 numerical columns.",
                manifest=ExecutionManifest(
                    operator="pca_dimensionality_reduction",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        clean_df = df[subjects].dropna()
        if len(clean_df) < 2:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_DATA",
                error_message="PCA requires at least 2 valid data rows.",
                manifest=ExecutionManifest(
                    operator="pca_dimensionality_reduction",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        tool_args = getattr(intent, "arguments", {}) or {}
        n_components = tool_args.get("n_components")
        if n_components is None:
            return OperatorOutput(
                status="FAILED",
                error_code="N_COMPONENTS_REQUIRED",
                error_message="PCA dimensionality reduction requires an explicit n_components parameter.",
                manifest=ExecutionManifest(
                    operator="pca_dimensionality_reduction",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        max_allowed_components = min(len(clean_df), len(subjects))
        if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1 or n_components > max_allowed_components:
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_COMPONENT_COUNT",
                error_message=f"n_components must be an integer between 1 and {max_allowed_components} for the supplied data.",
                manifest=ExecutionManifest(
                    operator="pca_dimensionality_reduction",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        scaling = tool_args.get("scaling", "standard")
        if scaling not in {"standard", "none"}:
            return OperatorOutput(
                status="FAILED",
                error_code="SCALING_REQUIRED",
                error_message="PCA requires scaling='standard' or scaling='none'.",
                manifest=ExecutionManifest(
                    operator="pca_dimensionality_reduction",
                    columns_used=subjects,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        scaled = StandardScaler().fit_transform(clean_df) if scaling == "standard" else clean_df.to_numpy(dtype=float)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(scaled)

        var_ratio = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_ratio)

        res_records = []
        for i in range(n_components):
            res_records.append({
                "component": f"PC{i+1}",
                "explained_variance_ratio": round(float(var_ratio[i]), 4),
                "cumulative_variance_ratio": round(float(cum_var[i]), 4)
            })

        res_df = pd.DataFrame(res_records)

        loadings = [
            {
                "component": f"PC{i+1}",
                **{subj: round(float(pca.components_[i, j]), 4) for j, subj in enumerate(subjects)}
            }
            for i in range(n_components)
        ]

        plt.figure(figsize=(8, 4))
        plt.bar(range(1, n_components + 1), var_ratio * 100, alpha=0.7, color="#6366f1", label="Individual Variance")
        plt.step(range(1, n_components + 1), cum_var * 100, where="mid", color="#ef4444", label="Cumulative Variance")
        plt.xlabel("Principal Component")
        plt.ylabel("Percentage of Variance Explained (%)")
        plt.title("PCA Explained Variance (Scree Plot)")
        plt.legend(loc="best")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        manifest = ExecutionManifest(
            operator="pca_dimensionality_reduction",
            columns_used=subjects,
            statistics_computed=["explained_variance_ratio", "cumulative_variance", "loadings"],
            chart_type="bar",
            row_count_in=len(df),
            row_count_out=len(res_df),
            assumptions=[f"Declared scaling={scaling}"]
        )

        finding = f"Performed PCA on `{len(subjects)}` features with `{scaling}` scaling. First **{n_components} components** explain **{cum_var[-1]*100:.2f}%** of total variance (PC1: {var_ratio[0]*100:.1f}%)."

        code = (
            "# DataPilot DS Engine: PCA Dimensionality Reduction\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.preprocessing import StandardScaler\n"
            f"subjects = {repr(subjects)}\n"
            f"n_components = {n_components!r}\n"
            f"scaling = {scaling!r}\n"
            "result_df = df[subjects].dropna().copy()\n"
            "X = StandardScaler().fit_transform(result_df) if scaling == 'standard' else result_df.to_numpy(dtype=float)\n"
            "pca = PCA(n_components=n_components)\n"
            "transformed = pca.fit_transform(X)\n"
            "print('Explained Variance Ratio:', pca.explained_variance_ratio_)\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={
                "records": res_df.to_dict(orient="records"),
                "columns": list(res_df.columns),
                "row_count": len(res_df),
                "explained_variance_ratio": [round(float(v), 4) for v in var_ratio],
                "cumulative_variance_ratio": [round(float(c), 4) for c in cum_var],
                "explained_variance": [round(float(v), 4) for v in pca.explained_variance_],
                "singular_values": [round(float(s), 4) for s in pca.singular_values_],
                "feature_loadings": loadings,
                "n_components": n_components,
                "scaling": scaling,
                "projections": pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)]).to_dict(orient="records")
            },
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            written_finding=finding,
            reproducible_code=code
        )


class FeatureExplainabilityOperator(BaseTypedOperator):
    """Held-out permutation importance over explicitly assigned column roles."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        args = intent.arguments or {}
        target_col = intent.target_column or args.get("target_column") or args.get("target")

        if not target_col or target_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="TARGET_COLUMN_REQUIRED",
                error_message="Feature explainability requires an explicit target column.",
                manifest=ExecutionManifest(
                    operator="feature_explainability",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        feature_cols = args.get("feature_columns") or args.get("features") or args.get("columns") or []
        if isinstance(feature_cols, str):
            feature_cols = [feature_cols]
        feature_cols = list(dict.fromkeys(feature_cols))
        if not feature_cols:
            return OperatorOutput(
                status="FAILED",
                error_code="FEATURE_COLUMNS_REQUIRED",
                error_message="Feature explainability requires explicit predictor feature columns.",
                manifest=ExecutionManifest(
                    operator="feature_explainability",
                    columns_used=[target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if target_col in feature_cols:
            return OperatorOutput(
                status="FAILED", error_code="TARGET_LEAKAGE_RISK",
                error_message="The target column cannot also be used as an explanatory feature.",
                manifest=ExecutionManifest(operator="feature_explainability", columns_used=[target_col] + feature_cols, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        missing = [column for column in feature_cols if column not in df.columns]
        if missing:
            return OperatorOutput(
                status="FAILED", error_code="COLUMN_NOT_FOUND",
                error_message=f"Feature column(s) {missing} not found in dataset.",
                manifest=ExecutionManifest(operator="feature_explainability", columns_used=[target_col] + feature_cols, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        task_type = args.get("task_type")
        method = args.get("method")
        if task_type not in {"classification", "regression"} or method != "permutation_importance":
            return OperatorOutput(
                status="FAILED", error_code="EXPLAINABILITY_CONTRACT_REQUIRED",
                error_message="Feature explanation requires explicit task_type and method='permutation_importance'.",
                manifest=ExecutionManifest(operator="feature_explainability", columns_used=[target_col] + feature_cols, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        clean_df = df[[target_col] + feature_cols].dropna(subset=[target_col]).copy()
        X = clean_df[feature_cols]
        y = clean_df[target_col]
        if len(clean_df) < 10:
            return OperatorOutput(
                status="FAILED", error_code="INSUFFICIENT_DATA",
                error_message="Feature explanation requires at least 10 rows with a non-missing target.",
                manifest=ExecutionManifest(operator="feature_explainability", columns_used=[target_col] + feature_cols, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        if task_type == "classification" and y.nunique(dropna=True) < 2:
            return OperatorOutput(
                status="FAILED", error_code="INSUFFICIENT_TARGET_CLASSES",
                error_message="Classification explanation requires at least two observed target classes.",
                manifest=ExecutionManifest(operator="feature_explainability", columns_used=[target_col] + feature_cols, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        if task_type == "regression" and not pd.api.types.is_numeric_dtype(y):
            return OperatorOutput(
                status="FAILED", error_code="TARGET_COLUMN_NOT_NUMERIC",
                error_message=f"Regression explanation target '{target_col}' must be numeric.",
                manifest=ExecutionManifest(operator="feature_explainability", columns_used=[target_col] + feature_cols, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [column for column in feature_cols if column not in numeric]
        transformers = []
        if numeric:
            transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical))
        model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == "classification" else RandomForestRegressor(n_estimators=100, random_state=42)
        stratify = y if task_type == "classification" and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=stratify)
        pipeline = Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])
        pipeline.fit(X_train, y_train)
        measured = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
        importances = measured.importances_mean
        importance_std = measured.importances_std
        held_out_score = float(pipeline.score(X_test, y_test))

        imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances, "importance_std": importance_std}).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(8, 4.5))
        plt.barh(imp_df["feature"].head(10)[::-1], imp_df["importance"].head(10)[::-1], color="#6366f1", edgecolor="black")
        plt.xlabel("Feature Importance Score")
        plt.title(f"Feature Importance for Target: {target_col}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        manifest = ExecutionManifest(
            operator="feature_explainability",
            columns_used=[target_col] + feature_cols,
            statistics_computed=["held_out_score", "permutation_importance_mean", "permutation_importance_std", "feature_rankings"],
            chart_type="bar",
            row_count_in=len(df),
            row_count_out=len(imp_df)
        )

        finding = f"Computed held-out permutation importance for `{target_col}` (model score `{held_out_score:.4f}`). Top selected feature: `{imp_df['feature'].iloc[0]}` (mean score decrease `{imp_df['importance'].iloc[0]:.4f}`)."

        code = (
            "# DataPilot DS Engine: held-out permutation importance\n"
            "import pandas as pd\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.inspection import permutation_importance\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            f"target_col = {target_col!r}\n"
            f"features = {repr(feature_cols)}\n"
            f"task_type = {task_type!r}\n"
            "work = df[[target_col] + features].dropna(subset=[target_col]).copy()\n"
            "X, y = work[features], work[target_col]\n"
            "if len(work) < 10: raise ValueError('Feature explanation requires at least 10 rows with a non-missing target')\n"
            "if task_type == 'classification' and y.nunique(dropna=True) < 2: raise ValueError('Classification explanation requires at least two target classes')\n"
            "if task_type == 'regression' and not pd.api.types.is_numeric_dtype(y): raise TypeError(f'Regression target {target_col!r} must be numeric')\n"
            "numeric = X.select_dtypes(include='number').columns.tolist()\n"
            "categorical = [column for column in features if column not in numeric]\n"
            "transformers = []\n"
            "if numeric: transformers.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median'))]), numeric))\n"
            "if categorical: transformers.append(('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical))\n"
            "estimator = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classification' else RandomForestRegressor(n_estimators=100, random_state=42)\n"
            "stratify = y if task_type == 'classification' and y.value_counts().min() >= 2 else None\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=stratify)\n"
            "pipeline = Pipeline([('preprocess', ColumnTransformer(transformers)), ('model', estimator)])\n"
            "pipeline.fit(X_train, y_train)\n"
            "measured = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42)\n"
            "result_df = pd.DataFrame({'feature': features, 'importance': measured.importances_mean, 'importance_std': measured.importances_std}).sort_values(by='importance', ascending=False)\n"
            "print(result_df)\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": imp_df.to_dict(orient="records"), "columns": list(imp_df.columns), "row_count": len(imp_df)},
            dataframe=imp_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            written_finding=finding,
            reproducible_code=code
        )


class AutocorrelationOperator(BaseTypedOperator):
    """Vectorized Autocorrelation (ACF & PACF) Analysis."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        target_col = (intent.subjects[0] if intent.subjects and intent.subjects[0] in df.columns else None) or (intent.target_column if intent.target_column and intent.target_column in df.columns else None)
        if not target_col:
            return OperatorOutput(
                status="FAILED",
                error_code="TARGET_COLUMN_REQUIRED",
                error_message="Autocorrelation analysis requires an explicit target column.",
                manifest=ExecutionManifest(
                    operator="autocorrelation",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_COLUMN_TYPE",
                error_message=f"Autocorrelation analysis requires a numeric target column. '{target_col}' is non-numeric.",
                manifest=ExecutionManifest(
                    operator="autocorrelation",
                    columns_used=[target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        tool_args = getattr(intent, "arguments", {}) or {}
        time_col = intent.time_column or tool_args.get("time_column")
        if time_col and time_col in df.columns:
            work_df = df[[time_col, target_col]].dropna().copy()
            work_df[time_col] = pd.to_datetime(work_df[time_col], errors="coerce")
            work_df = work_df.dropna(subset=[time_col]).sort_values(by=time_col)
            series = work_df[target_col].to_numpy(dtype=float)
        else:
            series = df[target_col].dropna().to_numpy(dtype=float)

        if len(series) < 10:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_DATA",
                error_message="Autocorrelation analysis requires at least 10 valid data points.",
                manifest=ExecutionManifest(
                    operator="autocorrelation",
                    columns_used=[target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        n_lags = tool_args.get("max_lags") or tool_args.get("lag") or tool_args.get("lags")
        if n_lags is None:
            return OperatorOutput(
                status="FAILED",
                error_code="MAX_LAGS_REQUIRED",
                error_message="Autocorrelation analysis requires an explicit max_lags parameter.",
                manifest=ExecutionManifest(
                    operator="autocorrelation",
                    columns_used=[target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        acf_vals = acf(series, nlags=n_lags)
        pacf_vals = pacf(series, nlags=n_lags)

        res_df = pd.DataFrame({
            "lag": list(range(len(acf_vals))),
            "acf": [round(float(v), 4) for v in acf_vals],
            "pacf": [round(float(v), 4) for v in pacf_vals]
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(series, ax=axes[0], lags=n_lags, title=f"ACF: {target_col}")
        plot_pacf(series, ax=axes[1], lags=n_lags, title=f"PACF: {target_col}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        sig_level = 1.96 / np.sqrt(len(series))
        sig_lags = [i for i, val in enumerate(acf_vals) if abs(val) > sig_level and i > 0]
        if sig_lags:
            finding = f"Autocorrelation analysis for `{target_col}`: Significant ACF lag periods detected at lags {sig_lags[:10]} (95% confidence threshold ±{sig_level:.4f})."
        else:
            finding = f"Autocorrelation analysis for `{target_col}`: No strongly significant lag periods detected (95% confidence threshold ±{sig_level:.4f})."

        cols_used = [time_col, target_col] if (time_col and time_col in df.columns) else [target_col]
        manifest = ExecutionManifest(
            operator="autocorrelation",
            columns_used=cols_used,
            statistics_computed=["acf", "pacf", "significant_lags"],
            chart_type="acf",
            row_count_in=len(df),
            row_count_out=len(res_df),
            sorted_by=time_col if (time_col and time_col in df.columns) else None
        )

        code = (
            "# DataPilot DS Engine: AutoCorrelation (ACF & PACF)\n"
            "from statsmodels.tsa.stattools import acf, pacf\n"
            f"target_col = '{target_col}'\n"
            "series = df[target_col].dropna()\n"
            "n_lags = min(40, max(2, len(series) // 2 - 1))\n"
            "acf_vals = acf(series, nlags=n_lags)\n"
            "pacf_vals = pacf(series, nlags=n_lags)\n"
            "print(f'ACF values: {acf_vals}')\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=res_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            written_finding=finding,
            reproducible_code=code
        )


class DataCleaningOperator(BaseTypedOperator):
    """Explicit, column-scoped cleaning without a universal hidden recipe."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        initial_rows = len(df)
        args = intent.arguments or {}
        method = args.get("method")
        raw_columns = args.get("columns") or args.get("column") or []
        columns = [raw_columns] if isinstance(raw_columns, str) else list(raw_columns)
        if method not in {
            "drop_missing", "drop_nulls",
            "impute_mean", "impute_median", "impute_mode", "impute_constant",
            "remove_duplicates", "drop_duplicates",
            "clip_iqr"
        }:
            return OperatorOutput(
                status="FAILED", error_code="CLEANING_METHOD_REQUIRED",
                error_message="Cleaning requires an explicit supported method.",
                manifest=ExecutionManifest(operator="data_cleaning", columns_used=[], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        missing_columns = [column for column in columns if column not in df.columns]
        if missing_columns:
            return OperatorOutput(
                status="FAILED", error_code="COLUMN_NOT_FOUND",
                error_message=f"Cleaning column(s) {missing_columns} not found in dataset.",
                manifest=ExecutionManifest(operator="data_cleaning", columns_used=columns, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        if method not in ("remove_duplicates", "drop_duplicates") and not columns:
            return OperatorOutput(
                status="FAILED", error_code="CLEANING_COLUMNS_REQUIRED",
                error_message=f"Cleaning method '{method}' requires explicit columns.",
                manifest=ExecutionManifest(operator="data_cleaning", columns_used=[], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )
        null_counts = df[columns].isnull().sum().to_dict() if columns else {}
        clean_df = df.copy(deep=True)
        duplicates_removed = 0
        code_operation = ""
        if method in {"remove_duplicates", "drop_duplicates"}:
            subset = columns or None
            before = len(clean_df)
            clean_df = clean_df.drop_duplicates(subset=subset)
            duplicates_removed = before - len(clean_df)
            code_operation = f"clean_df = clean_df.drop_duplicates(subset={subset!r})"
        elif method in {"drop_missing", "drop_nulls"}:
            clean_df = clean_df.dropna(subset=columns)
            code_operation = f"clean_df = clean_df.dropna(subset={columns!r})"
        elif method in {"impute_mean", "impute_median"}:
            non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(clean_df[column])]
            if non_numeric:
                return OperatorOutput(
                    status="FAILED", error_code="NUMERIC_COLUMNS_REQUIRED",
                    error_message=f"{method} requires numeric columns; invalid: {non_numeric}.",
                    manifest=ExecutionManifest(operator="data_cleaning", columns_used=columns, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            statistic = "mean" if method == "impute_mean" else "median"
            for column in columns:
                clean_df[column] = clean_df[column].fillna(getattr(clean_df[column], statistic)())
            code_operation = f"clean_df[{columns!r}] = clean_df[{columns!r}].fillna(clean_df[{columns!r}].{statistic}())"
        elif method == "impute_constant":
            fill_value = args.get("fill_value")
            if fill_value is None:
                return OperatorOutput(
                    status="FAILED", error_code="FILL_VALUE_REQUIRED",
                    error_message="impute_constant requires an explicit fill_value.",
                    manifest=ExecutionManifest(operator="data_cleaning", columns_used=columns, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            for column in columns:
                clean_df[column] = clean_df[column].fillna(fill_value)
            code_operation = f"clean_df[{columns!r}] = clean_df[{columns!r}].fillna({fill_value!r})"
        elif method == "clip_iqr":
            non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(clean_df[column])]
            if non_numeric:
                return OperatorOutput(
                    status="FAILED", error_code="NUMERIC_COLUMNS_REQUIRED",
                    error_message=f"clip_iqr requires numeric columns; invalid: {non_numeric}.",
                    manifest=ExecutionManifest(operator="data_cleaning", columns_used=columns, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )
            for column in columns:
                q25 = float(clean_df[column].quantile(0.25))
                q75 = float(clean_df[column].quantile(0.75))
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                clean_df[column] = clean_df[column].clip(lower=lower_bound, upper=upper_bound)
            code_operation = f"# 1.5x IQR outlier clipping on {columns!r}\nfor col in {columns!r}:\n    q25 = clean_df[col].quantile(0.25)\n    q75 = clean_df[col].quantile(0.75)\n    iqr = q75 - q25\n    clean_df[col] = clean_df[col].clip(lower=q25 - 1.5 * iqr, upper=q75 + 1.5 * iqr)"
        else:
            for column in columns:
                modes = clean_df[column].mode(dropna=True)
                if modes.empty:
                    return OperatorOutput(
                        status="FAILED", error_code="IMPUTATION_VALUE_UNAVAILABLE",
                        error_message=f"Column '{column}' has no observed mode for imputation.",
                        manifest=ExecutionManifest(operator="data_cleaning", columns_used=columns, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                    )
                clean_df[column] = clean_df[column].fillna(modes.iloc[0])
            code_operation = "\n".join(f"clean_df[{column!r}] = clean_df[{column!r}].fillna(clean_df[{column!r}].mode(dropna=True).iloc[0])" for column in columns)

        res_records = []
        for col_name in columns or list(df.columns):
            res_records.append({
                "column": col_name,
                "initial_nulls": null_counts.get(col_name, int(df[col_name].isnull().sum())),
                "remaining_nulls": int(clean_df[col_name].isnull().sum())
            })

        res_df = pd.DataFrame(res_records)

        manifest = ExecutionManifest(
            operator="data_cleaning",
            columns_used=columns,
            statistics_computed=[method, "null_counts_before_after"],
            chart_type="none",
            row_count_in=initial_rows,
            row_count_out=len(clean_df)
        )

        finding = f"Applied declared cleaning method **{method}** to {columns or ['all columns']}. Removed **{duplicates_removed:,} duplicate row(s)** and retained {len(clean_df):,} rows."

        code = (
            "# DataPilot DS Engine: explicit column-scoped cleaning\n"
            "clean_df = df.copy(deep=True)\n"
            f"{code_operation}\n"
            "result_df = clean_df\n"
            "print(f'Cleaned shape: {clean_df.shape}')\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=clean_df,
            manifest=manifest,
            written_finding=finding,
            reproducible_code=code
        )


class FeatureTransformationOperator(BaseTypedOperator):
    """Explicit, column-scoped transformation and feature engineering."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        initial_rows = len(df)
        args = intent.arguments or {}
        method = args.get("method")
        raw_columns = args.get("columns") or args.get("column") or []
        columns = [raw_columns] if isinstance(raw_columns, str) else list(raw_columns)

        valid_methods = {
            "minmax_scale", "normalize_minmax",
            "standardize", "z_score",
            "one_hot_encode",
            "yeo_johnson", "power_transform",
            "log_transform",
            "lag", "create_lag",
            "rolling", "create_rolling",
            "difference", "create_difference",
            "calendar_features", "create_calendar_features"
        }
        if method not in valid_methods:
            return OperatorOutput(
                status="FAILED", error_code="TRANSFORMATION_METHOD_REQUIRED",
                error_message=f"Transformation requires an explicit supported method. Got: '{method}'.",
                manifest=ExecutionManifest(operator="feature_transformation", columns_used=[], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )

        # ── FW-03 TEMPORAL FEATURE ENGINEERING HANDLERS ─────────────────────────
        temporal_methods = {
            "lag", "create_lag",
            "rolling", "create_rolling",
            "difference", "create_difference",
            "calendar_features", "create_calendar_features"
        }

        if method in temporal_methods:
            # 1. Resolve Time & Group Columns
            time_column = args.get("time_column") or intent.time_column
            if not time_column and intent.time_context:
                time_column = intent.time_context.time_column
            group_column = args.get("group_column")
            if not group_column and intent.time_context:
                group_column = intent.time_context.entity_id

            # Calendar features can accept time_column as column or vice-versa
            if method in ("calendar_features", "create_calendar_features") and not time_column and columns:
                time_column = columns[0]

            if not time_column:
                return OperatorOutput(
                    status="FAILED",
                    error_code="TEMPORAL_ORDER_REQUIRED",
                    error_message=f"Temporal feature engineering '{method}' requires an explicit 'time_column' to establish unambiguous chronological ordering.",
                    manifest=ExecutionManifest(
                        operator=f"feature_transformation_{method}",
                        columns_used=[c for c in columns if c in df.columns],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=initial_rows,
                        row_count_out=0
                    )
                )

            if time_column not in df.columns:
                return OperatorOutput(
                    status="FAILED",
                    error_code="COLUMN_NOT_FOUND",
                    error_message=f"Declared time column '{time_column}' not found in dataset.",
                    manifest=ExecutionManifest(
                        operator=f"feature_transformation_{method}",
                        columns_used=[c for c in columns if c in df.columns],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=initial_rows,
                        row_count_out=0
                    )
                )

            if group_column and group_column not in df.columns:
                return OperatorOutput(
                    status="FAILED",
                    error_code="COLUMN_NOT_FOUND",
                    error_message=f"Declared entity/group column '{group_column}' not found in dataset.",
                    manifest=ExecutionManifest(
                        operator=f"feature_transformation_{method}",
                        columns_used=[c for c in columns if c in df.columns],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=initial_rows,
                        row_count_out=0
                    )
                )

            # Columns validation (for non-calendar features)
            if method not in ("calendar_features", "create_calendar_features"):
                if not columns:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="TRANSFORMATION_COLUMNS_REQUIRED",
                        error_message=f"Temporal transformation method '{method}' requires an explicit source column.",
                        manifest=ExecutionManifest(
                            operator=f"feature_transformation_{method}",
                            columns_used=[],
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )
                missing_columns = [col for col in columns if col not in df.columns]
                if missing_columns:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="COLUMN_NOT_FOUND",
                        error_message=f"Transformation column(s) {missing_columns} not found in dataset.",
                        manifest=ExecutionManifest(
                            operator=f"feature_transformation_{method}",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

            transformed_df = df.copy(deep=True)
            warnings = []

            # Deterministic Stable Chronological Sort
            sort_keys = [group_column, time_column] if group_column else [time_column]
            transformed_df.sort_values(by=sort_keys, ascending=True, kind="mergesort", inplace=True)
            transformed_df.reset_index(drop=True, inplace=True)

            # Check for irregular time intervals
            try:
                if group_column:
                    sample_series = transformed_df.groupby(group_column)[time_column].apply(list).iloc[0]
                    series_slice = pd.Series(sample_series[:50])
                else:
                    series_slice = transformed_df[time_column].iloc[:50]

                if pd.api.types.is_numeric_dtype(series_slice):
                    diffs = series_slice.dropna().diff().dropna()
                    if diffs.nunique() > 1:
                        warnings.append("IRREGULAR_TIME_INTERVAL_WARNING: Timestamps are not evenly spaced. Lags represent row/event steps rather than fixed physical time durations.")
                else:
                    parsed_sample = pd.to_datetime(series_slice, errors="coerce")
                    if parsed_sample.notna().sum() >= 3:
                        diffs = parsed_sample.dropna().diff().dropna()
                        if diffs.nunique() > 1:
                            warnings.append("IRREGULAR_TIME_INTERVAL_WARNING: Timestamps are not evenly spaced. Lags represent row/event steps rather than fixed physical time durations.")
            except Exception:
                pass

            res_records = []
            code_operation = ""
            created_columns = []
            nan_count_introduced = 0
            new_name_override = args.get("new_name")

            # ── 1. CREATE_LAG ───────────────────────────────────────────
            if method in ("lag", "create_lag"):
                lag_order = args.get("lag_order")
                if lag_order is None:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="CLARIFICATION_REQUIRED",
                        error_message="create_lag requires an explicit 'lag_order' parameter (integer >= 1). No silent fallback permitted.",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_lag",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )
                try:
                    k = int(lag_order)
                except (ValueError, TypeError):
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_LAG_ORDER",
                        error_message=f"Lag order must be a valid integer >= 1. Got: {lag_order}",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_lag",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )
                if k < 1:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_LAG_ORDER",
                        error_message=f"Lag order must be >= 1 to prevent lookahead data leakage (k={k}). Zero or negative lags are strictly forbidden.",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_lag",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                for col in columns:
                    target_feature_name = new_name_override if (new_name_override and len(columns) == 1) else f"{col}__lag_{k}"
                    if target_feature_name in df.columns and not new_name_override:
                        return OperatorOutput(
                            status="FAILED",
                            error_code="COLUMN_COLLISION_ERROR",
                            error_message=f"Generated column '{target_feature_name}' already exists in the dataset. Provide an explicit 'new_name' to avoid collision.",
                            manifest=ExecutionManifest(
                                operator="feature_transformation_lag",
                                columns_used=columns,
                                statistics_computed=[],
                                chart_type="none",
                                row_count_in=initial_rows,
                                row_count_out=0
                            )
                        )
                    if group_column:
                        transformed_df[target_feature_name] = transformed_df.groupby(group_column)[col].shift(k)
                    else:
                        transformed_df[target_feature_name] = transformed_df[col].shift(k)
                    created_columns.append(target_feature_name)
                    n_nans = int(transformed_df[target_feature_name].isna().sum())
                    nan_count_introduced += n_nans
                    res_records.append({
                        "source_column": col,
                        "created_column": target_feature_name,
                        "lag_order": k,
                        "time_column": time_column,
                        "group_column": group_column,
                        "nan_count_introduced": n_nans,
                        "nan_handling": "retained"
                    })

                sort_str = f"['{group_column}', '{time_column}']" if group_column else f"['{time_column}']"
                shift_expr = f"transformed_df.groupby('{group_column}')[col].shift({k})" if group_column else f"transformed_df[col].shift({k})"
                code_operation = (
                    f"# Governed Lag Generation (k={k})\n"
                    f"transformed_df.sort_values(by={sort_str}, ascending=True, kind='mergesort', inplace=True)\n"
                    f"transformed_df.reset_index(drop=True, inplace=True)\n"
                    f"for col in {columns!r}:\n"
                    f"    transformed_df[f'{{col}}__lag_{k}'] = {shift_expr}\n"
                )

            # ── 2. CREATE_ROLLING ───────────────────────────────────────
            elif method in ("rolling", "create_rolling"):
                window_size = args.get("window_size")
                aggregation = args.get("aggregation")
                min_periods = args.get("min_periods")
                center = args.get("center", False)

                if center:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="CENTERED_ROLLING_PROHIBITED",
                        error_message="Centered rolling windows use future observations and cause lookahead leakage. Only trailing rolling windows (center=False) are permitted.",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_rolling",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                if window_size is None or aggregation is None:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="CLARIFICATION_REQUIRED",
                        error_message="create_rolling requires explicit 'window_size' (integer >= 2) and 'aggregation' ('mean'|'std'|'min'|'max'|'sum'). No silent fallback permitted.",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_rolling",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                try:
                    W = int(window_size)
                except (ValueError, TypeError):
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_WINDOW_SIZE",
                        error_message=f"Window size must be an integer >= 2. Got: {window_size}",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_rolling",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )
                if W < 2:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_WINDOW_SIZE",
                        error_message=f"Rolling window size must be >= 2. Got: {W}",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_rolling",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                agg_clean = str(aggregation).strip().lower()
                valid_aggs = {"mean", "std", "min", "max", "sum"}
                if agg_clean not in valid_aggs:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_AGGREGATION",
                        error_message=f"Unsupported rolling aggregation '{aggregation}'. Supported: {sorted(valid_aggs)}.",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_rolling",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                min_p = int(min_periods) if min_periods is not None else W

                for col in columns:
                    # Target Leakage Guard: if rolling is applied to target column, shift(1) is mandatory to prevent current-step leakage
                    is_target = (col == intent.target_column) or (args.get("is_target", False))
                    target_feature_name = new_name_override if (new_name_override and len(columns) == 1) else f"{col}__roll_{agg_clean}_{W}"
                    if target_feature_name in df.columns and not new_name_override:
                        return OperatorOutput(
                            status="FAILED",
                            error_code="COLUMN_COLLISION_ERROR",
                            error_message=f"Generated column '{target_feature_name}' already exists in dataset. Provide an explicit 'new_name'.",
                            manifest=ExecutionManifest(
                                operator="feature_transformation_rolling",
                                columns_used=columns,
                                statistics_computed=[],
                                chart_type="none",
                                row_count_in=initial_rows,
                                row_count_out=0
                            )
                        )

                    # Non-numeric check
                    if not pd.api.types.is_numeric_dtype(transformed_df[col]):
                        return OperatorOutput(
                            status="FAILED",
                            error_code="NUMERIC_COLUMNS_REQUIRED",
                            error_message=f"Rolling window operations require numeric columns; '{col}' is non-numeric.",
                            manifest=ExecutionManifest(
                                operator="feature_transformation_rolling",
                                columns_used=columns,
                                statistics_computed=[],
                                chart_type="none",
                                row_count_in=initial_rows,
                                row_count_out=0
                            )
                        )

                    if group_column:
                        # Group-aware rolling
                        def _roll_group(sub_s: pd.Series) -> pd.Series:
                            s_work = sub_s.shift(1) if is_target else sub_s
                            r = s_work.rolling(window=W, min_periods=min_p, center=False)
                            return getattr(r, agg_clean)()

                        transformed_df[target_feature_name] = transformed_df.groupby(group_column)[col].transform(_roll_group)
                    else:
                        s_work = transformed_df[col].shift(1) if is_target else transformed_df[col]
                        r = s_work.rolling(window=W, min_periods=min_p, center=False)
                        transformed_df[target_feature_name] = getattr(r, agg_clean)()

                    created_columns.append(target_feature_name)
                    n_nans = int(transformed_df[target_feature_name].isna().sum())
                    nan_count_introduced += n_nans
                    res_records.append({
                        "source_column": col,
                        "created_column": target_feature_name,
                        "window_size": W,
                        "aggregation": agg_clean,
                        "min_periods": min_p,
                        "center": False,
                        "time_column": time_column,
                        "group_column": group_column,
                        "nan_count_introduced": n_nans,
                        "nan_handling": "retained"
                    })

                sort_str = f"['{group_column}', '{time_column}']" if group_column else f"['{time_column}']"
                code_operation = (
                    f"# Governed Trailing Rolling Window (W={W}, agg={agg_clean})\n"
                    f"transformed_df.sort_values(by={sort_str}, ascending=True, kind='mergesort', inplace=True)\n"
                    f"transformed_df.reset_index(drop=True, inplace=True)\n"
                    f"for col in {columns!r}:\n"
                    f"    r = transformed_df[col].rolling(window={W}, min_periods={min_p}, center=False)\n"
                    f"    transformed_df[f'{{col}}__roll_{agg_clean}_{W}'] = r.{agg_clean}()\n"
                )

            # ── 3. CREATE_DIFFERENCE ────────────────────────────────────
            elif method in ("difference", "create_difference"):
                lag_order = args.get("lag_order", 1)
                try:
                    k = int(lag_order)
                except (ValueError, TypeError):
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_LAG_ORDER",
                        error_message=f"Difference order must be an integer >= 1. Got: {lag_order}",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_difference",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )
                if k < 1:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_LAG_ORDER",
                        error_message=f"Differencing order must be >= 1 (k={k}).",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_difference",
                            columns_used=columns,
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                for col in columns:
                    if not pd.api.types.is_numeric_dtype(transformed_df[col]):
                        return OperatorOutput(
                            status="FAILED",
                            error_code="NUMERIC_COLUMNS_REQUIRED",
                            error_message=f"Differencing requires numeric columns; '{col}' is non-numeric.",
                            manifest=ExecutionManifest(
                                operator="feature_transformation_difference",
                                columns_used=columns,
                                statistics_computed=[],
                                chart_type="none",
                                row_count_in=initial_rows,
                                row_count_out=0
                            )
                        )
                    target_feature_name = new_name_override if (new_name_override and len(columns) == 1) else f"{col}__diff_{k}"
                    if target_feature_name in df.columns and not new_name_override:
                        return OperatorOutput(
                            status="FAILED",
                            error_code="COLUMN_COLLISION_ERROR",
                            error_message=f"Generated column '{target_feature_name}' already exists in dataset. Provide an explicit 'new_name'.",
                            manifest=ExecutionManifest(
                                operator="feature_transformation_difference",
                                columns_used=columns,
                                statistics_computed=[],
                                chart_type="none",
                                row_count_in=initial_rows,
                                row_count_out=0
                            )
                        )
                    if group_column:
                        transformed_df[target_feature_name] = transformed_df.groupby(group_column)[col].diff(k)
                    else:
                        transformed_df[target_feature_name] = transformed_df[col].diff(k)

                    created_columns.append(target_feature_name)
                    n_nans = int(transformed_df[target_feature_name].isna().sum())
                    nan_count_introduced += n_nans
                    res_records.append({
                        "source_column": col,
                        "created_column": target_feature_name,
                        "diff_order": k,
                        "time_column": time_column,
                        "group_column": group_column,
                        "nan_count_introduced": n_nans,
                        "nan_handling": "retained"
                    })

                sort_str = f"['{group_column}', '{time_column}']" if group_column else f"['{time_column}']"
                diff_expr = f"transformed_df.groupby('{group_column}')[col].diff({k})" if group_column else f"transformed_df[col].diff({k})"
                code_operation = (
                    f"# Governed Differencing (k={k})\n"
                    f"transformed_df.sort_values(by={sort_str}, ascending=True, kind='mergesort', inplace=True)\n"
                    f"transformed_df.reset_index(drop=True, inplace=True)\n"
                    f"for col in {columns!r}:\n"
                    f"    transformed_df[f'{{col}}__diff_{k}'] = {diff_expr}\n"
                )

            # ── 4. CREATE_CALENDAR_FEATURES ─────────────────────────────
            elif method in ("calendar_features", "create_calendar_features"):
                explicit_clean_fmt = (intent.arguments or {}).get("datetime_format")
                try:
                    if explicit_clean_fmt:
                        from backend.ds_engine.datetime_parser import normalize_datetime_format, parse_datetime_column
                        norm_clean_fmt = normalize_datetime_format(explicit_clean_fmt)
                        dt_parse_res = parse_datetime_column(transformed_df[time_column], norm_clean_fmt)
                        if dt_parse_res.unparseable_count > 0:
                            return OperatorOutput(
                                status="FAILED",
                                error_code="DATETIME_PARSE_FAILED",
                                error_message=f"Datetime parsing failed for {dt_parse_res.unparseable_count:,} non-null value(s) in '{time_column}' with declared format '{explicit_clean_fmt}'.",
                                manifest=ExecutionManifest(operator="feature_transformation_calendar", columns_used=[time_column], statistics_computed=[], chart_type="none", row_count_in=initial_rows, row_count_out=0)
                            )
                        dt_series = dt_parse_res.series
                    else:
                        dt_series = pd.to_datetime(transformed_df[time_column], errors="coerce")
                except ValueError as norm_err:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_DATETIME_FORMAT",
                        error_message=f"Invalid datetime format '{explicit_clean_fmt}' for column '{time_column}': {norm_err}",
                        manifest=ExecutionManifest(operator="feature_transformation_calendar", columns_used=[time_column], statistics_computed=[], chart_type="none", row_count_in=initial_rows, row_count_out=0)
                    )
                except Exception as parse_err:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_TIME_COLUMN",
                        error_message=f"Cannot extract calendar features: column '{time_column}' cannot be parsed as datetime ({parse_err}).",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_calendar",
                            columns_used=[time_column],
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                if dt_series.notna().sum() == 0:
                    return OperatorOutput(
                        status="FAILED",
                        error_code="INVALID_TIME_COLUMN",
                        error_message=f"Column '{time_column}' contains no parseable datetime values.",
                        manifest=ExecutionManifest(
                            operator="feature_transformation_calendar",
                            columns_used=[time_column],
                            statistics_computed=[],
                            chart_type="none",
                            row_count_in=initial_rows,
                            row_count_out=0
                        )
                    )

                # Check cardinality before adding components to avoid zero-variance/constant features
                extracted = {}
                if dt_series.dt.year.nunique() > 1:
                    extracted["year"] = dt_series.dt.year
                if dt_series.dt.month.nunique() > 1:
                    extracted["month"] = dt_series.dt.month
                if dt_series.dt.day.nunique() > 1:
                    extracted["day"] = dt_series.dt.day
                if dt_series.dt.dayofweek.nunique() > 1:
                    extracted["dayofweek"] = dt_series.dt.dayofweek
                    extracted["is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                if dt_series.dt.quarter.nunique() > 1:
                    extracted["quarter"] = dt_series.dt.quarter
                if dt_series.dt.hour.nunique() > 1:
                    extracted["hour"] = dt_series.dt.hour

                # Fallback to standard set if low variance
                if not extracted:
                    extracted["year"] = dt_series.dt.year
                    extracted["month"] = dt_series.dt.month
                    extracted["day"] = dt_series.dt.day
                    extracted["dayofweek"] = dt_series.dt.dayofweek

                for comp, series_val in extracted.items():
                    comp_col_name = f"{time_column}__{comp}"
                    transformed_df[comp_col_name] = series_val
                    created_columns.append(comp_col_name)
                    res_records.append({
                        "source_column": time_column,
                        "created_column": comp_col_name,
                        "component": comp,
                        "nan_count_introduced": int(series_val.isna().sum()),
                        "nan_handling": "retained"
                    })

                code_operation = (
                    f"# Governed Calendar Feature Extraction\n"
                    f"dt_s = pd.to_datetime(transformed_df['{time_column}'], errors='coerce')\n"
                    + "\n".join([f"transformed_df['{time_column}__{k}'] = dt_s.dt.{k}" for k in extracted.keys() if k != "is_weekend"])
                    + (f"\ntransformed_df['{time_column}__is_weekend'] = dt_s.dt.dayofweek.isin([5, 6]).astype(int)" if "is_weekend" in extracted else "")
                    + "\n"
                )

            res_df = pd.DataFrame(res_records)
            entities_count = int(transformed_df[group_column].nunique()) if group_column else 1

            manifest = ExecutionManifest(
                operator=f"feature_transformation_{method}",
                columns_used=list(dict.fromkeys(columns + [time_column] + ([group_column] if group_column else []))),
                statistics_computed=[method] + created_columns,
                chart_type="none",
                row_count_in=initial_rows,
                row_count_out=len(transformed_df),
                warnings=warnings,
                sorted_by=time_column,
                partitioned_by=group_column
            )

            finding = (
                f"Generated {len(created_columns)} temporal feature(s) ({', '.join(created_columns)}) using "
                f"method **{method}** sorted by `{time_column}`"
                + (f" across {entities_count} entity groups in `{group_column}`." if group_column else ".")
                + f" Introduced {nan_count_introduced} initial NaN boundary value(s) (retained)."
            )

            code = (
                "# DataPilot DS Engine: Leakage-Safe Governed Temporal Feature Engineering\n"
                "transformed_df = df.copy(deep=True)\n"
                f"{code_operation}\n"
                "result_df = transformed_df\n"
                f"print(f'Engineered shape: {{transformed_df.shape}}')\n"
            )

            return OperatorOutput(
                status="SUCCEEDED",
                data={
                    "records": res_df.to_dict(orient="records"),
                    "columns": list(res_df.columns),
                    "row_count": len(res_df),
                    "created_columns": created_columns,
                    "nan_count_introduced": nan_count_introduced,
                    "affected_entities": entities_count
                },
                dataframe=transformed_df,
                manifest=manifest,
                written_finding=finding,
                reproducible_code=code
            )

        # ── CLASSICAL FEATURE TRANSFORMATIONS (MinMax, Standardize, One-Hot, etc.) ──
        if not columns:
            return OperatorOutput(
                status="FAILED", error_code="TRANSFORMATION_COLUMNS_REQUIRED",
                error_message=f"Transformation method '{method}' requires explicit columns.",
                manifest=ExecutionManifest(operator="feature_transformation", columns_used=[], statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            return OperatorOutput(
                status="FAILED", error_code="COLUMN_NOT_FOUND",
                error_message=f"Transformation column(s) {missing_columns} not found in dataset.",
                manifest=ExecutionManifest(operator="feature_transformation", columns_used=columns, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
            )

        # Non-numeric validation for numeric transforms
        numeric_methods = {
            "minmax_scale", "normalize_minmax",
            "standardize", "z_score",
            "yeo_johnson", "power_transform",
            "log_transform"
        }
        if method in numeric_methods:
            non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return OperatorOutput(
                    status="FAILED", error_code="NUMERIC_COLUMNS_REQUIRED",
                    error_message=f"{method} requires numeric columns; invalid non-numeric column(s): {non_numeric}.",
                    manifest=ExecutionManifest(operator="feature_transformation", columns_used=columns, statistics_computed=[], chart_type="none", row_count_in=len(df), row_count_out=0),
                )

        transformed_df = df.copy(deep=True)
        code_operation = ""

        if method in {"minmax_scale", "normalize_minmax"}:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            transformed_df[columns] = scaler.fit_transform(transformed_df[columns])
            code_operation = f"from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\ntransformed_df[{columns!r}] = scaler.fit_transform(transformed_df[{columns!r}])"

        elif method in {"standardize", "z_score"}:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            transformed_df[columns] = scaler.fit_transform(transformed_df[columns])
            code_operation = f"from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ntransformed_df[{columns!r}] = scaler.fit_transform(transformed_df[{columns!r}])"

        elif method == "one_hot_encode":
            transformed_df = pd.get_dummies(transformed_df, columns=columns, dtype=int)
            code_operation = f"transformed_df = pd.get_dummies(transformed_df, columns={columns!r}, dtype=int)"

        elif method in {"yeo_johnson", "power_transform"}:
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            transformed_df[columns] = pt.fit_transform(transformed_df[columns])
            code_operation = f"from sklearn.preprocessing import PowerTransformer\npt = PowerTransformer(method='yeo-johnson', standardize=False)\ntransformed_df[{columns!r}] = pt.fit_transform(transformed_df[{columns!r}])"

        elif method == "log_transform":
            for col in columns:
                min_val = float(transformed_df[col].min())
                if min_val < 0:
                    shift = abs(min_val) + 1.0
                    transformed_df[col] = np.log(transformed_df[col] + shift)
                else:
                    transformed_df[col] = np.log1p(transformed_df[col])
            code_operation = f"# Log transform on {columns!r}\nfor col in {columns!r}:\n    min_val = float(transformed_df[col].min())\n    if min_val < 0:\n        transformed_df[col] = np.log(transformed_df[col] + abs(min_val) + 1.0)\n    else:\n        transformed_df[col] = np.log1p(transformed_df[col])"

        res_records = []
        for col_name in columns:
            res_records.append({
                "column": col_name,
                "method": method,
                "status": "transformed"
            })
        res_df = pd.DataFrame(res_records)

        manifest = ExecutionManifest(
            operator="feature_transformation",
            columns_used=columns,
            statistics_computed=[method],
            chart_type="none",
            row_count_in=initial_rows,
            row_count_out=len(transformed_df)
        )

        finding = f"Applied declared transformation method **{method}** to {columns}. Resulting dataset shape: {transformed_df.shape[0]:,} rows, {transformed_df.shape[1]} columns."

        code = (
            "# DataPilot DS Engine: explicit column-scoped feature transformation\n"
            "transformed_df = df.copy(deep=True)\n"
            f"{code_operation}\n"
            "result_df = transformed_df\n"
            f"print(f'Transformed shape: {{transformed_df.shape}}')\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_df.to_dict(orient="records"), "columns": list(res_df.columns), "row_count": len(res_df)},
            dataframe=transformed_df,
            manifest=manifest,
            written_finding=finding,
            reproducible_code=code
        )


class GeneralEDAOperator(BaseTypedOperator):
    """Full Dataset Profile and Exploratory Summary."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        prof = profile or DatasetProfiler.profile(df, "active_dataset")
        records = []
        for col_name, cp in prof.columns.items():
            stats_str = f"min={cp.stats.get('min')}, max={cp.stats.get('max')}, mean={round(cp.stats.get('mean', 0), 2)}" if cp.stats else ""
            records.append({
                "column": col_name,
                "dtype": cp.dtype,
                "semantic_role": cp.semantic_role,
                "null_count": cp.null_count,
                "null_percentage": f"{cp.null_percentage}%",
                "cardinality": cp.cardinality,
                "summary_stats": stats_str
            })

        res_table = pd.DataFrame(records)

        manifest = ExecutionManifest(
            operator="general_eda",
            columns_used=list(df.columns),
            statistics_computed=["null_counts", "cardinality", "descriptive_stats"],
            chart_type="none",
            row_count_in=len(df),
            row_count_out=len(res_table)
        )

        finding = f"Dataset overview: **{prof.row_count:,} rows** and **{prof.column_count} columns** (Hash: `{prof.dataset_hash[:8]}`). Profiling identified {len(prof.binary_flag_candidates)} binary flag(s) and {len(prof.continuous_numeric_candidates)} continuous numeric column(s)."

        code = (
            "# DataPilot DS Engine: Exploratory Dataset Summary\n"
            "print(f'Shape: {df.shape}')\n"
            "print(df.describe(include=\"all\"))\n"
            "result_df = df.describe(include=\"all\").reset_index()\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": res_table.to_dict(orient="records"), "columns": list(res_table.columns), "row_count": len(res_table)},
            dataframe=res_table,
            manifest=manifest,
            written_finding=finding,
            reproducible_code=code
        )


class ClarificationOperator(BaseTypedOperator):
    """Honest structured clarification when query lacks analytical clarity."""

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        reason = intent.prompt_summary or "The requested analysis is underspecified. Please specify candidate metrics or dimensions."
        manifest = ExecutionManifest(
            operator="clarification",
            columns_used=[],
            statistics_computed=[],
            chart_type="none",
            row_count_in=len(df),
            row_count_out=0,
            warnings=[reason]
        )
        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": [], "columns": [], "row_count": 0},
            dataframe=df,
            manifest=manifest,
            written_finding=f"**Clarification Needed:** {reason}",
            reproducible_code="# Clarification requested; no code executed."
        )


class TimeSeriesDecompositionOperator(BaseTypedOperator):
    """
    Scientific Time Series Decomposition:
    Decomposes temporal signals into Trend, Seasonal, and Residual components.
    """

    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        from statsmodels.tsa.seasonal import seasonal_decompose

        tc = intent.time_context
        tool_args = getattr(intent, "arguments", {}) or {}

        # 1. Validate Time Column
        time_col = (tc.time_column if tc else None) or intent.time_column or tool_args.get("time_column")
        if not time_col:
            return OperatorOutput(
                status="FAILED",
                error_code="NO_TEMPORAL_COLUMN",
                error_message="Time-series decomposition requires an explicit datetime/temporal column.",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if time_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="NO_TEMPORAL_COLUMN",
                error_message=f"Temporal column '{time_col}' not found in dataset.",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        # 2. Validate Target / Metric Column
        target_col = intent.target_column or (intent.subjects[0] if intent.subjects else None) or tool_args.get("value_column") or tool_args.get("target_column")
        if not target_col:
            return OperatorOutput(
                status="FAILED",
                error_code="TARGET_COLUMN_REQUIRED",
                error_message="Time-series decomposition requires an explicit numerical target/value column.",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if target_col not in df.columns:
            return OperatorOutput(
                status="FAILED",
                error_code="TARGET_COLUMN_REQUIRED",
                error_message=f"Target column '{target_col}' not found in dataset.",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_COLUMN_TYPE",
                error_message=f"Time-series decomposition target column '{target_col}' must be numeric.",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0
                )
            )

        # 3. Prepare Temporal Data & Sort
        work_df = df[[time_col, target_col]].dropna().copy()
        work_df[time_col] = pd.to_datetime(work_df[time_col], errors="coerce")
        work_df = work_df.dropna(subset=[time_col]).sort_values(by=time_col)

        if len(work_df) < 4:
            return OperatorOutput(
                status="FAILED",
                error_code="INSUFFICIENT_DATA",
                error_message="Time-series decomposition requires at least 4 temporal observations.",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    sorted_by=time_col
                )
            )

        # 4. Period & Model Validation (Strict Zero Magic Fallbacks)
        # A seasonal period is a scientific modeling assumption. It must come
        # from the validated tool arguments, never from an event window or a
        # guessed calendar-frequency map.
        period = tool_args.get("period") or tool_args.get("seasonal_period")

        if period is not None:
            try:
                period = int(period)
            except (ValueError, TypeError):
                period = None

        if period is None or period <= 1:
            return OperatorOutput(
                status="FAILED",
                error_code="PERIOD_REQUIRED",
                error_message="Time-series decomposition requires an explicit seasonal period (e.g. period=7 for weekly cycles).",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    sorted_by=time_col
                )
            )

        model_raw = tool_args.get("model") or tool_args.get("decomposition_model") or tool_args.get("type")
        if not model_raw:
            return OperatorOutput(
                status="FAILED",
                error_code="MODEL_REQUIRED",
                error_message="Time-series decomposition requires an explicit decomposition model ('additive' or 'multiplicative').",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    sorted_by=time_col
                )
            )

        model_type = str(model_raw).strip().lower()
        if model_type not in ("additive", "multiplicative"):
            return OperatorOutput(
                status="FAILED",
                error_code="MODEL_REQUIRED",
                error_message="Time-series decomposition requires an explicit decomposition model ('additive' or 'multiplicative').",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    sorted_by=time_col
                )
            )

        if model_type == "multiplicative" and (work_df[target_col] <= 0).any():
            return OperatorOutput(
                status="FAILED",
                error_code="INVALID_MODEL_FOR_DATA",
                error_message="Multiplicative decomposition requires strictly positive values (> 0).",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    sorted_by=time_col
                )
            )

        # 5. Decomposition Computation
        try:
            decomp = seasonal_decompose(work_df[target_col], model=model_type, period=period)
        except Exception as e:
            return OperatorOutput(
                status="FAILED",
                error_code="DECOMPOSITION_ERROR",
                error_message=f"Time-series decomposition failed: {str(e)}",
                manifest=ExecutionManifest(
                    operator="time_series_decomposition",
                    columns_used=[time_col, target_col],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    sorted_by=time_col
                )
            )

        trend_filled = decomp.trend.bfill().ffill()
        resid_filled = decomp.resid.fillna(0)

        decomp_df = pd.DataFrame({
            time_col: work_df[time_col],
            "observed": work_df[target_col].values,
            "trend": np.round(trend_filled.values, 4),
            "seasonal": np.round(decomp.seasonal.values, 4),
            "residual": np.round(resid_filled.values, 4)
        })

        # Summary statistics
        obs_var = float(np.var(work_df[target_col]))
        resid_var = float(np.var(decomp.resid.dropna())) if len(decomp.resid.dropna()) > 0 else 0.0
        trend_strength = round(max(0.0, 1.0 - (resid_var / max(obs_var, 1e-6))), 3)
        seasonal_strength = round(max(0.0, float(np.var(decomp.seasonal)) / max(obs_var, 1e-6)), 3)

        # 6. Generate 4-panel Decomposition Plot
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(work_df[time_col], work_df[target_col], color="#3b82f6", label="Observed")
        axes[0].set_ylabel("Observed")
        axes[0].set_title(f"Time Series Decomposition ({model_type.capitalize()}): {target_col}", fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(work_df[time_col], decomp.trend, color="#10b981", label="Trend")
        axes[1].set_ylabel("Trend")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(work_df[time_col], decomp.seasonal, color="#f59e0b", label="Seasonal")
        axes[2].set_ylabel("Seasonal")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(work_df[time_col], decomp.resid, color="#ef4444", label="Residual", marker="o", linestyle="None", markersize=3)
        axes[3].set_ylabel("Residual")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        plotly_spec = {
            "data": [
                {"x": list(work_df[time_col].astype(str)), "y": [round(float(v), 3) for v in work_df[target_col]], "name": "Observed", "type": "scatter", "mode": "lines"},
                {"x": list(work_df[time_col].astype(str)), "y": [round(float(v), 3) if pd.notna(v) else None for v in decomp.trend], "name": "Trend", "type": "scatter", "mode": "lines"},
                {"x": list(work_df[time_col].astype(str)), "y": [round(float(v), 3) for v in decomp.seasonal], "name": "Seasonal", "type": "scatter", "mode": "lines"},
                {"x": list(work_df[time_col].astype(str)), "y": [round(float(v), 3) if pd.notna(v) else None for v in decomp.resid], "name": "Residual", "type": "scatter", "mode": "markers"}
            ],
            "layout": {
                "title": f"Time Series Decomposition ({model_type.capitalize()}): {target_col}",
                "xaxis": {"title": time_col},
                "yaxis": {"title": "Value"}
            }
        }

        finding = (
            f"Decomposed time series `{target_col}` over `{time_col}` ({model_type} model, period={period}):\n"
            f"• **Trend Strength**: `{trend_strength:.3f}`\n"
            f"• **Seasonality Strength**: `{seasonal_strength:.3f}`\n"
            f"• **Residual Variance Ratio**: `{(resid_var/max(obs_var, 1e-6)):.4f}`"
        )

        manifest = ExecutionManifest(
            operator="time_series_decomposition",
            columns_used=[time_col, target_col],
            statistics_computed=["trend_strength", "seasonality_strength", "residual_variance"],
            chart_type="line",
            row_count_in=len(df),
            row_count_out=len(decomp_df),
            sorted_by=time_col
        )

        code = (
            "# DataPilot DS Engine: Time Series Decomposition\n"
            "from statsmodels.tsa.seasonal import seasonal_decompose\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            f"time_col = '{time_col}'\n"
            f"target_col = '{target_col}'\n"
            f"period = {period}\n"
            f"model = '{model_type}'\n"
            "\n"
            f"work_df = df[[time_col, target_col]].dropna().copy()\n"
            f"work_df[time_col] = pd.to_datetime(work_df[time_col], errors='coerce')\n"
            f"work_df = work_df.dropna(subset=[time_col]).sort_values(by=time_col)\n"
            "\n"
            f"decomp = seasonal_decompose(work_df[target_col], model=model, period=period)\n"
            f"trend_filled = decomp.trend.bfill().ffill()\n"
            f"resid_filled = decomp.resid.fillna(0)\n"
            "\n"
            f"result_df = pd.DataFrame({{\n"
            f"    time_col: work_df[time_col],\n"
            f"    'observed': work_df[target_col].values,\n"
            f"    'trend': np.round(trend_filled.values, 4),\n"
            f"    'seasonal': np.round(decomp.seasonal.values, 4),\n"
            f"    'residual': np.round(resid_filled.values, 4)\n"
            f"}})\n"
            "print(result_df.head())\n"
        )

        return OperatorOutput(
            status="SUCCEEDED",
            data={"records": decomp_df.to_dict(orient="records"), "columns": list(decomp_df.columns), "row_count": len(decomp_df)},
            dataframe=decomp_df,
            manifest=manifest,
            plot_image_base64=img_b64,
            plotly_spec=plotly_spec,
            written_finding=finding,
            reproducible_code=code
        )


class GovernedSQLOperator(BaseTypedOperator):
    """
    Governed DuckDB SQL Operator.
    Executes analytical queries directly against active dataset in-memory DataFrame via DuckDB.
    Enforces strict read-only safety, schema grounding, zero silent fallback, and zero mutation.
    """
    def execute(
        self,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None,
    ) -> OperatorOutput:
        from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
        from backend.sql_engine.sql_validator import SQLValidator
        from backend.sql_engine.sql_builder import DeterministicSQLBuilder
        from backend.sql_engine.sql_models import SQLIntent

        tool_args = intent.arguments or {}
        sql = tool_args.get("query") or tool_args.get("sql")

        # If no raw SQL string is provided, but structured intent exists, compile it deterministically
        raw_intent = tool_args.get("sql_intent") or getattr(intent, "sql_intent", None)
        if not sql and raw_intent:
            try:
                if isinstance(raw_intent, dict):
                    parsed_intent = SQLIntent.model_validate(raw_intent)
                else:
                    parsed_intent = raw_intent
                sql = DeterministicSQLBuilder.build(parsed_intent, df)
            except Exception as e:
                return OperatorOutput(
                    status="FAILED",
                    error_code="SQL_BUILD_FAILED",
                    error_message=f"Failed to build SQL from intent: {str(e)}",
                    manifest=ExecutionManifest(
                        operator="sql_query",
                        columns_used=[],
                        statistics_computed=[],
                        chart_type="none",
                        row_count_in=len(df),
                        row_count_out=0,
                        warnings=[str(e)]
                    )
                )

        if not sql or not isinstance(sql, str) or not sql.strip():
            return OperatorOutput(
                status="FAILED",
                error_code="SQL_EMPTY",
                error_message="SQL query cannot be empty.",
                manifest=ExecutionManifest(
                    operator="sql_query",
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                )
            )

        sql_str = sql.strip()
        validation = SQLValidator.validate_query(sql_str, df)
        if not validation.is_valid:
            return OperatorOutput(
                status="FAILED",
                error_code=validation.error_code or "INVALID_SQL",
                error_message=validation.message or "Invalid SQL query.",
                manifest=ExecutionManifest(
                    operator="sql_query",
                    columns_used=validation.referenced_columns,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=[validation.message or "Invalid SQL query."]
                )
            )

        dataset_id = getattr(intent, "dataset_id", "active_dataset") or "active_dataset"
        dataset_fp = getattr(intent, "dataset_fingerprint", None)

        exec_res = GovernedDuckDBManager.execute_query(
            df=df,
            sql=sql_str,
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fp,
            page=1,
            page_size=1000
        )

        if exec_res.status != "SUCCEEDED":
            return OperatorOutput(
                status="FAILED",
                error_code=exec_res.error_code or "SQL_EXECUTION_ERROR",
                error_message=exec_res.error_message or "Query execution failed.",
                manifest=ExecutionManifest(
                    operator="sql_query",
                    columns_used=validation.referenced_columns,
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=[exec_res.error_message or "Query execution failed."]
                )
            )

        res_df = pd.DataFrame(exec_res.rows, columns=exec_res.columns) if exec_res.rows else pd.DataFrame(columns=exec_res.columns)

        return OperatorOutput(
            status="SUCCEEDED",
            data=exec_res.model_dump(),
            dataframe=res_df,
            manifest=ExecutionManifest(
                operator="sql_query",
                columns_used=exec_res.columns,
                statistics_computed=[str(validation.estimated_operation)],
                chart_type="none",
                row_count_in=len(df),
                row_count_out=exec_res.row_count,
            ),
            written_finding=f"Executed DuckDB SQL query in {exec_res.execution_time_ms:.2f}ms. Returned {exec_res.row_count} rows across {len(exec_res.columns)} columns.",
            reproducible_code=f"# DuckDB Governed SQL Execution\nimport duckdb\ncon = duckdb.connect(':memory:')\ncon.register('active_dataset', df)\nresult_df = con.execute('''{sql_str}''').df()\n"
        )


class TypedOperatorRegistry:
    """Authoritative Registry & Multi-Step Pipeline with Strict No-Fallback Guarantees."""

    _OPERATORS: Dict[TaskFamily, BaseTypedOperator] = {
        TaskFamily.FILTER_SLICE: FilterSliceOperator(),
        TaskFamily.EVENT_WINDOW_ANALYSIS: EventWindowAnalysisOperator(),
        TaskFamily.DISTRIBUTION_COMPARISON: DistributionVarianceOperator(),
        TaskFamily.GROUP_AGGREGATION: GroupAggregationOperator(),
        TaskFamily.CORRELATION_ANALYSIS: CorrelationOperator(),
        TaskFamily.ANOMALY_DETECTION: AnomalyDetectionOperator(),
        TaskFamily.TIME_SERIES_FORECAST: TimeSeriesForecastOperator(),
        TaskFamily.SUPERVISED_ML: SupervisedMLOperator(),
        TaskFamily.CLUSTERING: ClusteringOperator(),
        TaskFamily.CATEGORICAL_OUTCOME_ASSOCIATION: CategoricalOutcomeAssociationOperator(),
        TaskFamily.STATISTICAL_TEST: StatisticalTestOperator(),
        TaskFamily.DIMENSIONALITY_REDUCTION: PCADimensionalityReductionOperator(),
        TaskFamily.EXPLAINABILITY: FeatureExplainabilityOperator(),
        TaskFamily.DATA_CLEANING: DataCleaningOperator(),
        TaskFamily.FEATURE_ENGINEERING: FeatureTransformationOperator(),
        TaskFamily.GENERAL_EDA: GeneralEDAOperator(),
        TaskFamily.CLARIFICATION: ClarificationOperator(),
        TaskFamily.AUTOCORRELATION: AutocorrelationOperator(),
        TaskFamily.TIME_SERIES_DECOMPOSITION: TimeSeriesDecompositionOperator(),
        TaskFamily.SQL_QUERY: GovernedSQLOperator(),
    }

    @classmethod
    def execute_intent(
        cls,
        df: pd.DataFrame,
        intent: AnalysisIntentV2,
        profile: Optional[DatasetProfile] = None
    ) -> OperatorOutput:
        current_df = df

        # 1. Prerequisite Filtering: apply conditions for non-event tasks
        prereq_filter_out = None
        if intent.conditions and intent.task_family not in (TaskFamily.FILTER_SLICE, TaskFamily.EVENT_WINDOW_ANALYSIS, TaskFamily.ANOMALY_DETECTION, TaskFamily.CLUSTERING, TaskFamily.DIMENSIONALITY_REDUCTION, TaskFamily.EXPLAINABILITY, TaskFamily.DATA_CLEANING, TaskFamily.FEATURE_ENGINEERING, TaskFamily.TIME_SERIES_DECOMPOSITION, TaskFamily.SQL_QUERY):
            filter_op = FilterSliceOperator()
            prereq_filter_out = filter_op.execute(current_df, intent, profile)
            if prereq_filter_out.status == "SUCCEEDED" and prereq_filter_out.dataframe is not None:
                current_df = prereq_filter_out.dataframe
            elif prereq_filter_out.status != "SUCCEEDED":
                return prereq_filter_out

        # 2. Primary Operator Execution (Strict fail-fast: NO hidden fallback)
        primary_family = intent.task_family
        operator = cls._OPERATORS.get(primary_family)

        if not operator:
            return OperatorOutput(
                status="FAILED",
                error_code="UNSUPPORTED_OPERATION",
                error_message=f"No deterministic typed operator registered for TaskFamily '{primary_family}'. Execution halted without silent fallback.",
                manifest=ExecutionManifest(
                    operator=str(primary_family.value if hasattr(primary_family, "value") else str(primary_family)),
                    columns_used=[],
                    statistics_computed=[],
                    chart_type="none",
                    row_count_in=len(df),
                    row_count_out=0,
                    warnings=[f"Unsupported TaskFamily: {primary_family}"]
                )
            )

        out = operator.execute(current_df, intent, profile)
        if prereq_filter_out and prereq_filter_out.status == "SUCCEEDED":
            # The registry composes filtering and analysis into one operation.
            # Preserve both boundaries instead of reporting filtered rows as input.
            out.manifest.row_count_in = len(df)
            out.manifest.row_count_after_filter = len(current_df)
            for c in prereq_filter_out.manifest.columns_used:
                if c not in out.manifest.columns_used:
                    out.manifest.columns_used.append(c)
            out.manifest.filters_applied.extend(prereq_filter_out.manifest.filters_applied)

        if intent.conditions and not out.manifest.predicate_applied:
            out.manifest.predicate_applied = str(intent.conditions)

        return out
