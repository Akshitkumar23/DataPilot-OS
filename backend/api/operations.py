"""
DataPilot OS — Governed Operations API Router.
Handles POST /api/operations/preview and POST /api/operations/apply.
Enforces the strict governed pipeline:
UI / Client Intent
  ↓
Strict Schema / Column Grounding
  ↓
Deterministic Typed Operator (DataCleaningOperator / FeatureTransformationOperator)
  ↓
Semantic Result Validation
  ↓
Dataset Registry Update (Apply only, never on Preview or on Failure)
"""
import time
import uuid
import logging
import math
from typing import Optional, List, Any, Dict
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from backend.core.request_context import request_context_scope
from backend.core.dataset_registry import dataset_registry
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.api.datasets import compute_profile_summary
from backend.persistence.provenance_store import provenance_store
from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import CanonicalQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/operations", tags=["operations"])


def serialize_json_safe_value(val: Any) -> Any:
    """
    Sanitize raw DataFrame values into JSON-safe representations.
    Converts NaN, +Infinity, -Infinity, pd.NA, and NaT into None (JSON null).
    Leaves valid numbers, strings, and booleans strictly unmodified.
    Does NOT mutate any underlying DataFrame.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return val


def serialize_json_safe_list(values: List[Any]) -> List[Any]:
    """Serialize a list of values to JSON-safe representations without mutating source data."""
    return [serialize_json_safe_value(v) for v in values]


class GovernedOperationRequest(BaseModel):
    dataset_id: Optional[str] = None
    operation: str
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    fill_value: Optional[Any] = None
    new_name: Optional[str] = None
    n_bins: Optional[int] = None
    operator: Optional[str] = None
    filter_value: Optional[Any] = None
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    lag_order: Optional[int] = None
    window_size: Optional[int] = None
    aggregation: Optional[str] = None
    min_periods: Optional[int] = None
    is_target: Optional[bool] = None

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            k: v for k, v in {
                "column": self.column,
                "columns": self.columns,
                "fill_value": self.fill_value,
                "new_name": self.new_name,
                "n_bins": self.n_bins,
                "operator": self.operator,
                "filter_value": self.filter_value,
                "time_column": self.time_column,
                "group_column": self.group_column,
                "lag_order": self.lag_order,
                "window_size": self.window_size,
                "aggregation": self.aggregation,
                "min_periods": self.min_periods,
                "is_target": self.is_target,
            }.items() if v is not None
        }


def map_operation_to_tool_call(req: GovernedOperationRequest) -> Dict[str, Any]:
    """Translate UI operation IDs to canonical ToolSpec and arguments."""
    op = (req.operation or "").strip().lower()
    raw_cols = req.columns or ([req.column] if req.column else [])
    cols = [c for c in raw_cols if c]

    clean_map = {
        "fill_mean": ("clean_data", "impute_mean"),
        "impute_mean": ("clean_data", "impute_mean"),
        "fill_median": ("clean_data", "impute_median"),
        "impute_median": ("clean_data", "impute_median"),
        "fill_mode": ("clean_data", "impute_mode"),
        "impute_mode": ("clean_data", "impute_mode"),
        "fill_constant": ("clean_data", "impute_constant"),
        "impute_constant": ("clean_data", "impute_constant"),
        "drop_nulls": ("clean_data", "drop_nulls"),
        "drop_missing": ("clean_data", "drop_nulls"),
        "drop_duplicates": ("clean_data", "drop_duplicates"),
        "remove_duplicates": ("clean_data", "drop_duplicates"),
        "clip_iqr": ("clean_data", "clip_iqr"),
    }

    transform_map = {
        "normalize_minmax": ("transform_data", "minmax_scale"),
        "minmax_scale": ("transform_data", "minmax_scale"),
        "standardize": ("transform_data", "standardize"),
        "z_score": ("transform_data", "standardize"),
        "one_hot_encode": ("transform_data", "one_hot_encode"),
        "yeo_johnson": ("transform_data", "yeo_johnson"),
        "power_transform": ("transform_data", "yeo_johnson"),
        "log_transform": ("transform_data", "log_transform"),
        "create_lag": ("transform_data", "lag"),
        "lag": ("transform_data", "lag"),
        "create_rolling": ("transform_data", "rolling"),
        "rolling": ("transform_data", "rolling"),
        "create_difference": ("transform_data", "difference"),
        "difference": ("transform_data", "difference"),
        "create_calendar_features": ("transform_data", "calendar_features"),
        "calendar_features": ("transform_data", "calendar_features"),
    }

    if op in clean_map:
        tool, method = clean_map[op]
        args: Dict[str, Any] = {"method": method}
        if cols:
            args["columns"] = cols
        if method == "impute_constant":
            args["fill_value"] = req.fill_value
        return {"tool": tool, "arguments": args}

    elif op in transform_map:
        tool, method = transform_map[op]
        args: Dict[str, Any] = {"method": method, "columns": cols}
        if req.time_column:
            args["time_column"] = req.time_column
        if req.group_column:
            args["group_column"] = req.group_column
        if req.lag_order is not None:
            args["lag_order"] = req.lag_order
        if req.window_size is not None:
            args["window_size"] = req.window_size
        if req.aggregation is not None:
            args["aggregation"] = req.aggregation
        if req.min_periods is not None:
            args["min_periods"] = req.min_periods
        if req.new_name is not None:
            args["new_name"] = req.new_name
        if req.is_target is not None:
            args["is_target"] = req.is_target
        return {"tool": tool, "arguments": args}

    raise ValueError(f"Unsupported operation: '{req.operation}'")


def _execute_governed_pipeline(
    req: GovernedOperationRequest,
    is_preview: bool = True
) -> Dict[str, Any]:
    if dataset_registry is None:
        raise HTTPException(status_code=503, detail="Dataset registry is unavailable.")

    # 1. Resolve dataset
    entry = dataset_registry.get(req.dataset_id) if req.dataset_id else dataset_registry.get_active()
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found or no active dataset. Connect and activate a dataset first."
        )

    df = dataset_registry.get_dataframe(entry.dataset_id)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{entry.dataset_id}' could not be loaded into memory."
        )

    t_start = time.time()
    cols = req.columns or ([req.column] if req.column else [])

    # 2. Map operation to canonical tool call
    try:
        tool_call = map_operation_to_tool_call(req)
    except ValueError as val_err:
        if not is_preview:
            provenance_store.record_event(
                dataset_id=entry.dataset_id,
                dataset_name=entry.name,
                operation=req.operation,
                capability="DATA_CLEAN",
                affected_columns=cols,
                parameters=req.parameters,
                df_before=df,
                df_after=None,
                status="FAILED",
                source={"type": "STRUCTURED_UI"},
                error={"error_code": "UNSUPPORTED_OPERATION", "message": str(val_err)},
                execution_time_ms=(time.time() - t_start) * 1000.0,
            )
        raise HTTPException(
            status_code=400,
            detail={"error_code": "UNSUPPORTED_OPERATION", "message": str(val_err)}
        )

    # 3. Execute through unified governed pipeline
    df_work = df.copy(deep=True)
    req_id = f"op_{uuid.uuid4().hex[:8]}"
    output_frames = []

    profile = DatasetProfiler.profile(df_work, dataset_name=entry.name)

    with request_context_scope(
        request_id=req_id,
        dataset_id=entry.dataset_id,
        dataset_hash=profile.dataset_hash,
        model_version="governed_kernel",
    ) as req_ctx:
        response: CanonicalQueryResponse = QueryOrchestrator._process_intent(
            query_raw=f"governed_{req.operation}",
            df=df_work,
            intent_dict=tool_call,
            profile=profile,
            conversation_history=None,
            context={"dataset_id": entry.dataset_id, "dataset_name": entry.name},
            request_id=req_id,
            t_start=t_start,
            output_frames=output_frames,
        )

    status_upper = (response.status or "").upper().strip()

    if status_upper == "NEEDS_CLARIFICATION":
        if not is_preview:
            clarif_msg = response.error.message if response.error else "Operation requires clarification"
            provenance_store.record_event(
                dataset_id=entry.dataset_id,
                dataset_name=entry.name,
                operation=req.operation,
                capability=tool_call.get("tool", req.operation),
                affected_columns=cols,
                parameters=req.parameters,
                df_before=df,
                df_after=None,
                status="NEEDS_CLARIFICATION",
                source={"type": "STRUCTURED_UI"},
                error={"error_code": "NEEDS_CLARIFICATION", "message": clarif_msg},
                execution_time_ms=(time.time() - t_start) * 1000.0,
            )
        raise HTTPException(status_code=422, detail=response.model_dump())

    if status_upper != "SUCCEEDED":
        err_code = "OPERATION_FAILED"
        err_msg = "Operation execution failed."
        if response.error:
            if isinstance(response.error, dict):
                err_code = response.error.get("code", err_code)
                err_msg = response.error.get("message", err_msg)
            elif hasattr(response.error, "code"):
                err_code = getattr(response.error, "code", err_code)
                err_msg = getattr(response.error, "message", err_msg)
        if not is_preview:
            provenance_store.record_event(
                dataset_id=entry.dataset_id,
                dataset_name=entry.name,
                operation=req.operation,
                capability=tool_call.get("tool", req.operation),
                affected_columns=cols,
                parameters=req.parameters,
                df_before=df,
                df_after=None,
                status="FAILED",
                source={"type": "STRUCTURED_UI"},
                error={"error_code": err_code, "message": err_msg},
                execution_time_ms=(time.time() - t_start) * 1000.0,
            )
        raise HTTPException(
            status_code=422,
            detail={
                "error": True,
                "error_code": err_code,
                "message": err_msg,
                "response": response.model_dump()
            }
        )

    out_df = output_frames[-1] if output_frames else df_work

    # 4. Preview vs Apply
    target_col = cols[0] if cols else None

    finding = ""
    code = ""
    if response.result:
        finding = response.result.get("response") or response.result.get("answer") or ""
        code = response.result.get("code") or ""

    if is_preview:
        # Preview: NEVER mutate dataset registry!
        # Detect if a newly engineered feature column was added to out_df
        new_cols = [c for c in out_df.columns if c not in df.columns]
        after_target_col = new_cols[-1] if new_cols else target_col

        if target_col and target_col in df.columns:
            before_vals = serialize_json_safe_list(df[target_col].head(10).tolist())
            before_stats = {
                "nulls": int(df[target_col].isna().sum()),
                "rows": int(len(df))
            }
        else:
            before_vals = [f"Row {i+1}" for i in range(min(10, len(df)))]
            before_stats = {
                "nulls": int(df.isna().sum().sum()),
                "rows": int(len(df))
            }

        if after_target_col and after_target_col in out_df.columns:
            after_vals = serialize_json_safe_list(out_df[after_target_col].head(10).tolist())
            after_stats = {
                "nulls": int(out_df[after_target_col].isna().sum()),
                "rows": int(len(out_df)),
                "created_column": after_target_col if after_target_col != target_col else None
            }
        else:
            after_vals = [f"Row {i+1}" for i in range(min(10, len(out_df)))]
            after_stats = {
                "nulls": int(out_df.isna().sum().sum()),
                "rows": int(len(out_df))
            }

        return {
            "status": "SUCCEEDED",
            "operation": req.operation,
            "dataset_id": entry.dataset_id,
            "preview": {
                "before": before_vals,
                "after": after_vals,
                "before_stats": before_stats,
                "after_stats": after_stats,
            },
            "row_count_before": int(len(df)),
            "row_count_after": int(len(out_df)),
            "column_count_before": int(len(df.columns)),
            "column_count_after": int(len(out_df.columns)),
            "columns_before": list(df.columns),
            "columns_after": list(out_df.columns),
            "finding": finding,
            "code": code,
        }
    else:
        # Apply: Mutate active DataFrame in dataset registry!
        dataset_registry.update_dataframe(entry.dataset_id, out_df)
        new_profile_summary = compute_profile_summary(out_df, entry.dataset_id, entry.name)

        # Record immutable provenance event dynamically derived from DataFrame states
        elapsed_ms = (time.time() - t_start) * 1000.0
        prov_event = provenance_store.record_event(
            dataset_id=entry.dataset_id,
            dataset_name=entry.name,
            operation=req.operation,
            capability=tool_call.get("tool", req.operation),
            affected_columns=cols,
            parameters=req.parameters,
            df_before=df,
            df_after=out_df,
            status="SUCCESS",
            source={"type": "STRUCTURED_UI"},
            finding=finding,
            code=code,
            execution_time_ms=elapsed_ms,
        )

        return {
            "status": "SUCCEEDED",
            "operation": req.operation,
            "dataset_id": entry.dataset_id,
            "row_count": int(len(out_df)),
            "row_count_before": int(len(df)),
            "row_count_after": int(len(out_df)),
            "column_count": int(len(out_df.columns)),
            "columns": list(out_df.columns),
            "profile_summary": new_profile_summary,
            "finding": finding,
            "code": code,
            "provenance_event": prov_event.model_dump(),
        }


@router.post("/preview")
def preview_operation(request: GovernedOperationRequest):
    """Calculates and displays before/after effects without mutating active dataset state."""
    return _execute_governed_pipeline(request, is_preview=True)


@router.post("/apply")
def apply_operation(request: GovernedOperationRequest):
    """Executes governed operation, updates active DataFrame, and recalculates real data profile."""
    return _execute_governed_pipeline(request, is_preview=False)
