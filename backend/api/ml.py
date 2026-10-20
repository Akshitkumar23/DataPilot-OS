"""
DataPilot OS — Governed Machine Learning API Router (Phase 4).
Endpoints for Supervised Classification, Regression, AutoML Tournament,
Model Artifact Inference, and Full Provenance Integration.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.dataset_engine.ml_engine import ml_engine, sanitize_obj
from backend.persistence.provenance_store import provenance_store

try:
    from backend.core.dataset_registry import dataset_registry
except ImportError:
    dataset_registry = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["machine_learning"])


class MLTrainRequest(BaseModel):
    """Structured request for governed ML model training."""
    dataset_id: Optional[str] = None
    target: str = Field(description="Target column name for supervised training")
    features: Optional[List[str]] = Field(default=None, description="Predictor feature column names")
    task: Optional[str] = Field(default="auto", description="classification | regression | auto")
    algorithm: Optional[str] = Field(default="auto", description="auto | random_forest | decision_tree | logistic_regression | linear_regression")
    validation_strategy: Optional[str] = Field(default="split", description="split | cv")
    test_size: Optional[float] = Field(default=0.2, ge=0.05, le=0.5)
    cv_folds: Optional[int] = Field(default=5, ge=2, le=20)
    group_column: Optional[str] = Field(default=None, description="Optional entity grouping column to prevent group leakage")
    time_column: Optional[str] = Field(default=None, description="Optional timestamp column for chronological TimeSeriesSplit")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "features": self.features,
            "task": self.task,
            "algorithm": self.algorithm,
            "validation_strategy": self.validation_strategy,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "group_column": self.group_column,
            "time_column": self.time_column,
        }


class MLPredictRequest(BaseModel):
    """Structured request for model inference."""
    dataset_id: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None


class MLCostPolicyRequest(BaseModel):
    """Structured request for cost-sensitive threshold optimization (FW-02)."""
    dataset_id: Optional[str] = None
    target: str = Field(description="Binary target column name")
    positive_class: Any = Field(description="Label identifying the positive class")
    false_positive_cost: float = Field(description="Cost incurred per false positive (C_FP > 0)")
    false_negative_cost: float = Field(description="Cost incurred per false negative (C_FN > 0)")
    true_positive_cost: Optional[float] = Field(default=0.0, description="Cost/benefit per true positive (C_TP <= 0 for benefit)")
    true_negative_cost: Optional[float] = Field(default=0.0, description="Cost/benefit per true negative (C_TN <= 0 for benefit)")
    features: Optional[List[str]] = Field(default=None, description="Predictor feature column names")
    algorithm: Optional[str] = Field(default="auto", description="auto | random_forest | decision_tree | logistic_regression")
    calibration_policy: Optional[str] = Field(default="auto", description="auto | sigmoid | isotonic | none")
    grid_resolution: Optional[int] = Field(default=100, ge=10, le=1000)
    test_size: Optional[float] = Field(default=0.2, ge=0.05, le=0.5)
    cv_folds: Optional[int] = Field(default=5, ge=2, le=20)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "positive_class": str(self.positive_class),
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
            "true_positive_cost": self.true_positive_cost,
            "true_negative_cost": self.true_negative_cost,
            "features": self.features,
            "algorithm": self.algorithm,
            "calibration_policy": self.calibration_policy,
            "grid_resolution": self.grid_resolution,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
        }


@router.post("/train")
def train_model(req: MLTrainRequest) -> Dict[str, Any]:
    """Train a supervised machine learning model or AutoML tournament on active dataset."""
    if dataset_registry is None:
        raise HTTPException(status_code=503, detail="Dataset registry is unavailable.")

    # 1. Resolve Dataset
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
            detail={"error": True, "error_code": "DATASET_NOT_LOADED", "message": f"Dataset '{entry.dataset_id}' could not be loaded."}
        )

    t_start = time.time()
    cols = [req.target] + (req.features or [])

    # 2. Execute Training with Schema & Precondition Validation
    try:
        result_data = ml_engine.train_model(
            df=df,
            target=req.target,
            features=req.features,
            task=req.task,
            algorithm=req.algorithm or "auto",
            validation_strategy=req.validation_strategy or "split",
            test_size=req.test_size or 0.2,
            cv_folds=req.cv_folds or 5,
            dataset_id=entry.dataset_id,
            group_column=req.group_column,
            time_column=req.time_column,
        )

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
            operation="ML_TRAIN",
            capability="ML",
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

    # 3. Record Successful Provenance Event
    elapsed_ms = (time.time() - t_start) * 1000.0
    canonical_op = "AUTOML_TOURNAMENT" if req.algorithm in ("auto", "automl", "compare") else "ML_TRAIN"

    summary_metrics = {
        "operation": canonical_op,
        "task": result_data["task"],
        "algorithm": result_data["algorithm"],
        "algorithm_name": result_data["algorithm_name"],
        "target": result_data["target"],
        "feature_count": len(result_data["features"]),
        "train_samples": result_data["sample_counts"]["train"],
        "test_samples": result_data["sample_counts"]["test"],
        "primary_metric": result_data["metrics"].get("primary_metric"),
        "primary_score": result_data["metrics"].get("primary_score"),
        "model_id": result_data.get("model_id"),
    }

    prov_event = provenance_store.record_event(
        dataset_id=entry.dataset_id,
        dataset_name=entry.name,
        operation=canonical_op,
        capability="ML",
        affected_columns=[req.target] + result_data["features"],
        parameters=req.parameters,
        df_before=df,
        df_after=df,
        result_summary=summary_metrics,
        status="SUCCESS",
        source={"type": "STRUCTURED_UI"},
        finding=f"Trained {result_data['algorithm_name']} for {result_data['task']} on target '{req.target}' with {result_data['metrics'].get('primary_metric')}={result_data['metrics'].get('primary_score')}.",
        execution_time_ms=elapsed_ms,
    )

    return {
        "status": "SUCCEEDED",
        "dataset_id": entry.dataset_id,
        "result": result_data,
        "provenance_event": {
            "event_id": prov_event.event_id,
            "operation": prov_event.operation,
            "capability": prov_event.capability,
            "timestamp": prov_event.timestamp,
            "execution_time_ms": prov_event.execution_time_ms,
            "summary": summary_metrics,
        }
    }


@router.post("/optimize-threshold")
def optimize_threshold(req: MLCostPolicyRequest) -> Dict[str, Any]:
    """Optimize decision threshold under asymmetric business cost matrix (FW-02)."""
    if dataset_registry is None:
        raise HTTPException(status_code=503, detail="Dataset registry is unavailable.")

    # 1. Resolve Dataset
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
            detail={"error": True, "error_code": "DATASET_NOT_LOADED", "message": f"Dataset '{entry.dataset_id}' could not be loaded."}
        )

    t_start = time.time()
    cols = [req.target] + (req.features or [])

    # Check for missing cost parameters
    if req.false_positive_cost is None or req.false_negative_cost is None:
        raise HTTPException(
            status_code=422,
            detail={"error": True, "error_code": "MISSING_COST_POLICY", "message": "False positive cost and false negative cost must be specified."}
        )

    # 2. Execute Threshold Optimization via ML Engine
    try:
        result_data = ml_engine.optimize_classification_threshold(
            df=df,
            target=req.target,
            positive_class=req.positive_class,
            false_positive_cost=req.false_positive_cost,
            false_negative_cost=req.false_negative_cost,
            true_positive_cost=req.true_positive_cost if req.true_positive_cost is not None else 0.0,
            true_negative_cost=req.true_negative_cost if req.true_negative_cost is not None else 0.0,
            features=req.features,
            algorithm=req.algorithm or "auto",
            calibration_policy=req.calibration_policy or "auto",
            grid_resolution=req.grid_resolution or 100,
            test_size=req.test_size if req.test_size is not None else 0.2,
            cv_folds=req.cv_folds if req.cv_folds is not None else 5,
            dataset_id=entry.dataset_id,
        )
    except ValueError as val_err:
        err_msg = str(val_err)
        if ":" in err_msg:
            err_code, err_text = [p.strip() for p in err_msg.split(":", 1)]
        else:
            err_code = "INVALID_REQUEST"
            err_text = err_msg

        elapsed_ms = (time.time() - t_start) * 1000.0

        provenance_store.record_event(
            dataset_id=entry.dataset_id,
            dataset_name=entry.name,
            operation="ML_THRESHOLD_OPTIMIZATION",
            capability="ML",
            affected_columns=cols,
            parameters=req.parameters,
            df_before=df,
            df_after=None,
            status="FAILED",
            source={"type": "COST_POLICY_UI"},
            error={"error_code": err_code, "message": err_text},
            execution_time_ms=elapsed_ms,
        )

        raise HTTPException(
            status_code=422,
            detail={"error": True, "error_code": err_code, "message": err_text}
        )

    elapsed_ms = (time.time() - t_start) * 1000.0

    summary_metrics = {
        "operation": "ML_THRESHOLD_OPTIMIZATION",
        "algorithm": result_data["algorithm"],
        "target": result_data["target"],
        "positive_class": result_data["positive_class"],
        "model_id": result_data.get("model_id"),
        "baseline_threshold": result_data["threshold_selection"]["baseline_threshold"],
        "optimized_threshold": result_data["threshold_selection"]["optimized_threshold"],
        "theoretical_bayes_threshold": result_data["cost_policy"]["theoretical_bayes_threshold"],
        "baseline_cost": result_data["business_decision_quality"]["baseline_cost"],
        "optimized_cost": result_data["business_decision_quality"]["optimized_cost"],
        "percentage_cost_reduction": result_data["business_decision_quality"]["percentage_cost_reduction"],
        "calibration_applied": result_data["calibration"]["applied"],
        "calibration_method": result_data["calibration"]["method"],
        "brier_score": result_data["calibration"]["brier_score"],
    }

    prov_event = provenance_store.record_event(
        dataset_id=entry.dataset_id,
        dataset_name=entry.name,
        operation="ML_THRESHOLD_OPTIMIZATION",
        capability="ML",
        affected_columns=[req.target] + (req.features or []),
        parameters=req.parameters,
        df_before=df,
        df_after=df,
        result_summary=summary_metrics,
        status="SUCCESS",
        source={"type": "COST_POLICY_UI"},
        finding=(
            f"Optimized decision threshold for '{req.target}' ({result_data['algorithm']}) from 0.50 to "
            f"{result_data['threshold_selection']['optimized_threshold']}, achieving "
            f"{result_data['business_decision_quality']['percentage_cost_reduction']}% cost reduction."
        ),
        execution_time_ms=elapsed_ms,
    )

    return {
        "status": "SUCCEEDED",
        "dataset_id": entry.dataset_id,
        "result": result_data,
        "provenance_event": {
            "event_id": prov_event.event_id,
            "operation": prov_event.operation,
            "capability": prov_event.capability,
            "timestamp": prov_event.timestamp,
            "execution_time_ms": prov_event.execution_time_ms,
            "summary": summary_metrics,
        }
    }


@router.post("/models/{model_id}/predict")
def predict_model(model_id: str, req: MLPredictRequest) -> Dict[str, Any]:
    """Run inference against a saved model artifact."""
    if not req.rows:
        if req.dataset_id:
            if dataset_registry is None:
                raise HTTPException(status_code=503, detail="Dataset registry is unavailable.")
            df = dataset_registry.get_dataframe(req.dataset_id)
            if df is None:
                raise HTTPException(status_code=404, detail="Requested inference dataset not found.")
            rows_data = df
        else:
            raise HTTPException(status_code=422, detail={"error": True, "error_code": "NO_PREDICTION_DATA", "message": "No rows provided for inference."})
    else:
        rows_data = req.rows

    t_start = time.time()
    try:
        prediction_result = ml_engine.predict_with_model(model_id, rows_data)
    except ValueError as val_err:
        err_msg = str(val_err)
        err_code = err_msg.split(":", 1)[0].strip() if ":" in err_msg else "PREDICTION_FAILED"
        err_text = err_msg.split(":", 1)[1].strip() if ":" in err_msg else err_msg
        raise HTTPException(
            status_code=422,
            detail={"error": True, "error_code": err_code, "message": err_text}
        )

    elapsed_ms = (time.time() - t_start) * 1000.0

    return {
        "status": "SUCCEEDED",
        "result": prediction_result,
        "execution_time_ms": elapsed_ms,
    }


@router.get("/models")
def list_models(dataset_id: Optional[str] = None) -> Dict[str, Any]:
    """List persisted model artifacts for a dataset or workspace."""
    models = ml_engine.list_models(dataset_id=dataset_id)
    return {
        "status": "SUCCEEDED",
        "models": models,
        "total": len(models),
    }


@router.get("/algorithms")
def list_algorithms(task: Optional[str] = "classification") -> Dict[str, Any]:
    """List supported machine learning algorithms for a task."""
    task_clean = (task or "classification").strip().lower()
    algorithms = ml_engine.get_supported_algorithms(task_clean)
    return {
        "status": "SUCCEEDED",
        "task": task_clean,
        "algorithms": algorithms,
    }
