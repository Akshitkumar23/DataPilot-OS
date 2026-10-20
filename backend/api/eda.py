"""
DataPilot OS — Governed EDA & Statistics API Router.
Production-grade endpoints for Descriptive Statistics, Missingness Analysis,
Distribution Histograms, Correlation Matrices, and Skewness with full Provenance integration.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.dataset_engine.eda_engine import eda_engine, sanitize_obj
from backend.persistence.provenance_store import provenance_store

try:
    from backend.core.dataset_registry import dataset_registry
except ImportError:
    dataset_registry = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eda", tags=["eda"])


class EDARequest(BaseModel):
    """Structured request for governed exploratory data analysis."""
    dataset_id: Optional[str] = None
    operation: Optional[str] = Field(default=None, description="descriptive | missingness | distribution | correlation | skewness")
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    method: Optional[str] = "pearson"
    n_bins: Optional[int] = 10

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            k: v for k, v in {
                "operation": self.operation,
                "column": self.column,
                "columns": self.columns,
                "method": self.method,
                "n_bins": self.n_bins,
            }.items() if v is not None
        }


def _execute_eda_operation(req: EDARequest) -> Dict[str, Any]:
    """Unified governed execution pipeline for EDA operations."""
    if dataset_registry is None:
        raise HTTPException(status_code=503, detail="Dataset registry is unavailable.")

    # 1. Resolve dataset
    entry = dataset_registry.get(req.dataset_id) if req.dataset_id else dataset_registry.get_active()
    if not entry:
        raise HTTPException(
            status_code=404,
            detail={"error": True, "error_code": "DATASET_NOT_FOUND", "message": "No active dataset connected."}
        )

    df = dataset_registry.get_dataframe(entry.dataset_id)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail={"error": True, "error_code": "DATASET_NOT_LOADED", "message": f"Dataset '{entry.dataset_id}' could not be loaded into memory."}
        )

    t_start = time.time()
    op_clean = (req.operation or "").strip().lower()
    cols = req.columns or ([req.column] if req.column else [])

    # 2. Execute calculation with strict schema validation
    try:
        if op_clean in ("descriptive", "descriptive_stats", "summary_stats", "describe"):
            canonical_op = "DESCRIPTIVE_STATS"
            result_data = eda_engine.compute_descriptive_stats(df, columns=cols if cols else None)
            affected_cols = result_data.get("columns", cols)

        elif op_clean in ("missingness", "missingness_analysis", "null_check", "missing"):
            canonical_op = "MISSINGNESS_ANALYSIS"
            result_data = eda_engine.compute_missingness(df)
            affected_cols = [c["column"] for c in result_data.get("columns", [])]

        elif op_clean in ("distribution", "distribution_analysis", "hist", "histogram"):
            canonical_op = "DISTRIBUTION_ANALYSIS"
            target_col = req.column or (req.columns[0] if req.columns else None)
            result_data = eda_engine.compute_distribution(df, column=target_col, n_bins=req.n_bins or 10)
            affected_cols = [target_col] if target_col else []

        elif op_clean in ("correlation", "correlation_analysis", "corr", "correlation_matrix"):
            canonical_op = "CORRELATION_ANALYSIS"
            result_data = eda_engine.compute_correlation(df, columns=cols if cols else None, method=req.method or "pearson")
            affected_cols = result_data.get("columns", cols)

        elif op_clean in ("skewness", "skewness_analysis", "skew", "normality"):
            canonical_op = "SKEWNESS_ANALYSIS"
            result_data = eda_engine.compute_skewness(df, columns=cols if cols else None)
            affected_cols = result_data.get("columns", cols)

        else:
            raise ValueError(f"UNSUPPORTED_OPERATION: Unknown EDA operation '{req.operation}'. Supported: descriptive, missingness, distribution, correlation, skewness.")

    except ValueError as val_err:
        err_msg = str(val_err)
        if ":" in err_msg:
            err_code, err_text = [p.strip() for p in err_msg.split(":", 1)]
        else:
            err_code = "INVALID_REQUEST"
            err_text = err_msg

        elapsed_ms = (time.time() - t_start) * 1000.0

        # Record failed provenance event
        provenance_store.record_event(
            dataset_id=entry.dataset_id,
            dataset_name=entry.name,
            operation=req.operation,
            capability="EDA",
            affected_columns=cols,
            parameters=req.parameters,
            df_before=df,
            df_after=None,
            status="FAILED",
            source={"type": "STRUCTURED_UI"},
            error={"error_code": err_code, "message": err_text},
            execution_time_ms=elapsed_ms,
        )

        raise HTTPException(
            status_code=422,
            detail={"error": True, "error_code": err_code, "message": err_text}
        )

    # 3. Successful calculation: record provenance event
    elapsed_ms = (time.time() - t_start) * 1000.0
    summary_metrics = {
        "operation": canonical_op,
        "row_count": len(df),
        "columns_analyzed": len(affected_cols),
        "feature_count": len(affected_cols),
    }

    event = provenance_store.record_event(
        dataset_id=entry.dataset_id,
        dataset_name=entry.name,
        operation=canonical_op,
        capability="EDA",
        affected_columns=affected_cols,
        parameters=req.parameters,
        df_before=df,
        df_after=df,  # Non-mutating read-only analysis
        status="SUCCESS",
        source={"type": "STRUCTURED_UI"},
        result_summary=summary_metrics,
        finding=f"Successfully executed {canonical_op} across {len(affected_cols)} column(s) on dataset '{entry.name}'.",
        execution_time_ms=elapsed_ms,
    )

    return sanitize_obj({
        "status": "SUCCEEDED",
        "operation": canonical_op,
        "dataset_id": entry.dataset_id,
        "dataset_name": entry.name,
        "result": result_data,
        "provenance_event": event.model_dump() if event else None,
    })


@router.post("/execute")
def execute_eda(req: EDARequest):
    """Execute any governed EDA operation via a unified structured interface."""
    return _execute_eda_operation(req)


@router.post("/descriptive")
def run_descriptive_stats(req: EDARequest):
    """Calculate descriptive statistics for specified or all numerical columns."""
    req.operation = "descriptive"
    return _execute_eda_operation(req)


@router.post("/missingness")
def run_missingness_analysis(req: EDARequest):
    """Calculate column-wise and overall missingness metrics."""
    req.operation = "missingness"
    return _execute_eda_operation(req)


@router.post("/distribution")
def run_distribution_analysis(req: EDARequest):
    """Calculate empirical distribution bins and sample statistics for a target column."""
    req.operation = "distribution"
    return _execute_eda_operation(req)


@router.post("/correlation")
def run_correlation_analysis(req: EDARequest):
    """Calculate real numerical correlation matrix and ranked association pairs."""
    req.operation = "correlation"
    return _execute_eda_operation(req)


@router.post("/skewness")
def run_skewness_analysis(req: EDARequest):
    """Calculate skewness and distribution normality assessment."""
    req.operation = "skewness"
    return _execute_eda_operation(req)
