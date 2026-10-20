"""
DataPilot OS — Phase 6: Governed Time-Series & Forecasting API Router.
Provides structured endpoints for time-series validation, stationarity diagnostics (ADF, KPSS),
autocorrelation (ACF, PACF), seasonal decomposition, and forecasting (Naive, AutoReg, ARIMA, SARIMAX).
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.dataset_registry import dataset_registry
from backend.ds_engine.timeseries_engine import timeseries_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])


class ValidateRequest(BaseModel):
    dataset_id: str = Field(..., description="Active dataset identifier")
    time_column: str = Field(..., description="Explicit temporal/datetime column")
    target_column: str = Field(..., description="Explicit numerical metric/target column")
    datetime_format: Optional[str] = Field(None, description="Optional explicit datetime format (e.g. DD/MM/YYYY or %d/%m/%Y)")


class StationarityRequest(BaseModel):
    dataset_id: str = Field(..., description="Active dataset identifier")
    time_column: str = Field(..., description="Explicit temporal/datetime column")
    target_column: str = Field(..., description="Explicit numerical metric/target column")
    datetime_format: Optional[str] = Field(None, description="Optional explicit datetime format (e.g. DD/MM/YYYY or %d/%m/%Y)")


class AcfPacfRequest(BaseModel):
    dataset_id: str = Field(..., description="Active dataset identifier")
    time_column: str = Field(..., description="Explicit temporal/datetime column")
    target_column: str = Field(..., description="Explicit numerical metric/target column")
    max_lags: Optional[int] = Field(None, ge=2, le=100, description="Optional maximum lag count")
    datetime_format: Optional[str] = Field(None, description="Optional explicit datetime format (e.g. DD/MM/YYYY or %d/%m/%Y)")


class DecomposeRequest(BaseModel):
    dataset_id: str = Field(..., description="Active dataset identifier")
    time_column: str = Field(..., description="Explicit temporal/datetime column")
    target_column: str = Field(..., description="Explicit numerical metric/target column")
    period: int = Field(..., ge=2, description="Seasonal cycle period (e.g. 7 for daily/weekly, 12 for monthly/annual)")
    model: Optional[str] = Field("additive", description="Decomposition model ('additive' or 'multiplicative')")
    datetime_format: Optional[str] = Field(None, description="Optional explicit datetime format (e.g. DD/MM/YYYY or %d/%m/%Y)")


class ForecastRequest(BaseModel):
    dataset_id: str = Field(..., description="Active dataset identifier")
    time_column: str = Field(..., description="Explicit temporal/datetime column")
    target_column: str = Field(..., description="Explicit numerical metric/target column")
    model_type: Optional[str] = Field("arima", description="Forecasting model: 'naive', 'autoreg', 'arima', 'sarimax'")
    horizon: Optional[int] = Field(7, ge=1, le=365, description="Future forecast horizon steps")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model parameters (e.g. p, d, q, P, D, Q, s, lags)")
    val_size: Optional[int] = Field(None, ge=1, description="Optional chronological validation holdout window size")
    datetime_format: Optional[str] = Field(None, description="Optional explicit datetime format (e.g. DD/MM/YYYY or %d/%m/%Y)")


def _get_dataset(dataset_id: str):
    """Retrieve and validate dataframe from dataset registry."""
    if not dataset_id:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "DATASET_ID_REQUIRED", "error_message": "dataset_id cannot be empty."}
        )

    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "DATASET_NOT_FOUND", "error_message": f"Dataset '{dataset_id}' not found."}
        )

    df = dataset_registry.get_dataframe(dataset_id)
    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "DATASET_EMPTY", "error_message": f"Dataset '{dataset_id}' is empty or unavailable."}
        )

    return df, entry.name or dataset_id


@router.post("/validate")
def validate_temporal_dataset(req: ValidateRequest):
    """
    Validates temporal index, target column, ordering, duplicate timestamps, and frequency grid.
    """
    df, _ = _get_dataset(req.dataset_id)
    try:
        return timeseries_engine.validate_time_series(
            df=df,
            time_column=req.time_column,
            target_column=req.target_column,
            datetime_format=req.datetime_format
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail={"status": "FAILED", "error_message": str(ve)})


@router.post("/stationarity")
def test_stationarity(req: StationarityRequest):
    """
    Executes Augmented Dickey-Fuller (ADF) and KPSS stationarity tests.
    """
    df, _ = _get_dataset(req.dataset_id)
    try:
        return timeseries_engine.run_stationarity_tests(
            df=df,
            time_column=req.time_column,
            target_column=req.target_column,
            datetime_format=req.datetime_format
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail={"status": "FAILED", "error_message": str(ve)})


@router.post("/acf-pacf")
def test_autocorrelation(req: AcfPacfRequest):
    """
    Computes ACF and PACF lags with 95% confidence intervals and diagnostic visualization.
    """
    df, _ = _get_dataset(req.dataset_id)
    try:
        return timeseries_engine.run_acf_pacf(
            df=df,
            time_column=req.time_column,
            target_column=req.target_column,
            max_lags=req.max_lags,
            datetime_format=req.datetime_format
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail={"status": "FAILED", "error_message": str(ve)})


@router.post("/decompose")
def decompose_series(req: DecomposeRequest):
    """
    Decomposes time series into Trend, Seasonal, and Residual components.
    """
    df, _ = _get_dataset(req.dataset_id)
    try:
        return timeseries_engine.run_decomposition(
            df=df,
            time_column=req.time_column,
            target_column=req.target_column,
            period=req.period,
            model=req.model or "additive",
            datetime_format=req.datetime_format
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail={"status": "FAILED", "error_message": str(ve)})


@router.post("/forecast")
def generate_forecast(req: ForecastRequest):
    """
    Fits forecasting model (naive, autoreg, arima, sarimax), performs chronological holdout
    evaluation (MAE, MSE, RMSE, MAPE), generates future projections with 95% CI, and logs provenance.
    """
    df, dataset_name = _get_dataset(req.dataset_id)
    try:
        return timeseries_engine.run_forecast(
            df=df,
            time_column=req.time_column,
            target_column=req.target_column,
            model_type=req.model_type or "arima",
            horizon=req.horizon or 7,
            parameters=req.parameters or {},
            val_size=req.val_size,
            dataset_id=req.dataset_id,
            dataset_name=dataset_name,
            datetime_format=req.datetime_format
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail={"status": "FAILED", "error_message": str(ve)})
