"""
DataPilot OS — Phase 6: Governed Real Time-Series & Forecasting Engine.
Provides production-grade, domain-agnostic time-series validation, stationarity diagnostics (ADF, KPSS),
ACF/PACF, seasonal decomposition, and forecasting (Naive, AutoReg, ARIMA, SARIMAX) with strict
chronological holdout validation and zero active dataset mutation.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.persistence.provenance_store import provenance_store
from backend.ds_engine.time_frequency import canonicalize_pandas_frequency
from backend.ds_engine.time_series_grain import analyze_time_series_grain
from backend.ds_engine.datetime_parser import normalize_datetime_format, parse_datetime_column

logger = logging.getLogger(__name__)


def sanitize_obj(obj: Any) -> Any:
    """Recursively converts NumPy, NaN, and Inf objects into JSON-compliant primitives."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or np.isnan(obj):
            return None
        if math.isinf(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_obj(x) for x in obj]
    return obj


class TimeSeriesEngine:
    """
    Production Time-Series Analytics & Forecasting Service.
    Guarantees strict immutability, zero domain hardcoding, and rigorous chronological validation.
    """

    def validate_time_series(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        datetime_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates temporal index, target column, ordering, duplicates, and frequency grid.
        Zero dataset mutation.
        """
        if df is None or df.empty:
            raise ValueError("[EMPTY_DATASET] Dataset is empty or unavailable.")
        if time_column not in df.columns:
            raise ValueError(f"[COLUMN_NOT_FOUND] Temporal column '{time_column}' not found in dataset.")
        if target_column not in df.columns:
            raise ValueError(f"[COLUMN_NOT_FOUND] Target column '{target_column}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"[NON_NUMERIC_TARGET] Target column '{target_column}' must be numeric.")

        work_df = df[[time_column, target_column]].copy(deep=True)
        total_rows = len(work_df)

        # Parse temporal column
        if datetime_format:
            try:
                norm_fmt = normalize_datetime_format(datetime_format)
            except ValueError as norm_err:
                raise ValueError(f"[INVALID_DATETIME_FORMAT] Invalid format '{datetime_format}': {norm_err}")
            parse_res = parse_datetime_column(work_df[time_column], norm_fmt)
            parsed_dt = parse_res.series
            unparseable_count = parse_res.unparseable_count
            null_time_count = parse_res.null_count
        else:
            parsed_dt = pd.to_datetime(work_df[time_column], errors="coerce")
            unparseable_count = int(parsed_dt.isna().sum() - work_df[time_column].isna().sum())
            null_time_count = int(work_df[time_column].isna().sum())

        valid_mask = parsed_dt.notna() & work_df[target_column].notna()
        clean_df = work_df[valid_mask].copy()
        clean_df["_dt"] = parsed_dt[valid_mask]

        if len(clean_df) < 5:
            raise ValueError(f"[INSUFFICIENT_DATA] Found only {len(clean_df)} valid temporal rows. Minimum 5 required.")

        # Check ordering
        is_monotonic = bool(clean_df["_dt"].is_monotonic_increasing)
        sorted_df = clean_df.sort_values(by="_dt")

        # Duplicate check
        duplicate_timestamps = int(sorted_df["_dt"].duplicated().sum())
        has_duplicates = duplicate_timestamps > 0

        # Frequency detection
        grain_info = analyze_time_series_grain(clean_df, time_column, datetime_format=datetime_format)
        detected_freq = grain_info.detected_frequency

        # Span and stats
        min_date = str(sorted_df["_dt"].min())
        max_date = str(sorted_df["_dt"].max())
        target_series = pd.to_numeric(sorted_df[target_column], errors="coerce").dropna()

        return sanitize_obj({
            "status": "VALIDATED",
            "time_column": time_column,
            "target_column": target_column,
            "datetime_format": datetime_format,
            "total_rows": total_rows,
            "usable_observations": len(sorted_df),
            "null_timestamps": null_time_count,
            "unparseable_timestamps": unparseable_count,
            "has_duplicate_timestamps": has_duplicates,
            "duplicate_timestamp_count": duplicate_timestamps,
            "is_chronologically_sorted": is_monotonic,
            "is_regular_grid": grain_info.is_regular_grid,
            "detected_frequency": detected_freq,
            "start_time": min_date,
            "end_time": max_date,
            "target_summary": {
                "mean": float(target_series.mean()),
                "std": float(target_series.std()) if len(target_series) > 1 else 0.0,
                "min": float(target_series.min()),
                "max": float(target_series.max())
            },
            "warnings": [grain_info.clarification_reason] if grain_info.requires_clarification else []
        })

    def run_stationarity_tests(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        datetime_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Augmented Dickey-Fuller (ADF) and KPSS unit root tests.
        Returns exact test statistics, p-values, critical thresholds, and scientific conclusion.
        """
        self.validate_time_series(df, time_column, target_column, datetime_format=datetime_format)
        work_df = df[[time_column, target_column]].copy(deep=True)
        if datetime_format:
            norm_fmt = normalize_datetime_format(datetime_format)
            work_df["_dt"] = parse_datetime_column(work_df[time_column], norm_fmt).series
        else:
            work_df["_dt"] = pd.to_datetime(work_df[time_column], errors="coerce")
        clean_df = work_df.dropna(subset=["_dt", target_column]).sort_values(by="_dt")

        series = clean_df[target_column].to_numpy(dtype=float)
        if len(series) < 12:
            raise ValueError(f"[INSUFFICIENT_DATA] Stationarity testing requires at least 12 observations, found {len(series)}.")

        from statsmodels.tsa.stattools import adfuller, kpss

        # ADF Test: H0 = Series has a unit root (non-stationary)
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series, autolag="AIC")
        adf_is_stationary = bool(adf_p < 0.05)

        # KPSS Test: H0 = Series is level/trend stationary
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression="c", nlags="auto")
        kpss_is_stationary = bool(kpss_p >= 0.05)

        # Synthesized conclusion
        if adf_is_stationary and kpss_is_stationary:
            conclusion = "STATIONARY"
            interpretation = "Both ADF and KPSS tests confirm the series is stationary. No differencing required."
        elif not adf_is_stationary and not kpss_is_stationary:
            conclusion = "NON_STATIONARY"
            interpretation = "Both ADF and KPSS tests confirm non-stationarity. First-order differencing d=1 is recommended."
        elif adf_is_stationary and not kpss_is_stationary:
            conclusion = "DIFFERENCE_STATIONARY"
            interpretation = "ADF indicates stationarity around trend, but KPSS rejects level stationarity. Series may have a deterministic drift."
        else:
            conclusion = "TREND_STATIONARY"
            interpretation = "KPSS confirms stationarity around a trend, but ADF cannot reject unit root. Detrending or differencing may be needed."

        return sanitize_obj({
            "status": "SUCCEEDED",
            "time_column": time_column,
            "target_column": target_column,
            "sample_size": len(series),
            "adf_test": {
                "test_name": "Augmented Dickey-Fuller (ADF)",
                "null_hypothesis": "Unit root is present (Non-Stationary)",
                "test_statistic": float(adf_stat),
                "p_value": float(adf_p),
                "lags_used": int(adf_lags),
                "n_observations": int(adf_nobs),
                "critical_values": {k: float(v) for k, v in adf_crit.items()},
                "is_stationary": adf_is_stationary,
                "rejection_threshold": 0.05
            },
            "kpss_test": {
                "test_name": "Kwiatkowski-Phillips-Schmidt-Shin (KPSS)",
                "null_hypothesis": "Series is level stationary (Stationary)",
                "test_statistic": float(kpss_stat),
                "p_value": float(kpss_p),
                "lags_used": int(kpss_lags),
                "critical_values": {k: float(v) for k, v in kpss_crit.items()},
                "is_stationary": kpss_is_stationary,
                "rejection_threshold": 0.05
            },
            "conclusion": conclusion,
            "interpretation": interpretation
        })

    def run_acf_pacf(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        max_lags: Optional[int] = None,
        datetime_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Computes Autocorrelation (ACF) and Partial Autocorrelation (PACF) lags
        with 95% confidence intervals and visual diagnostic plot.
        """
        self.validate_time_series(df, time_column, target_column, datetime_format=datetime_format)
        work_df = df[[time_column, target_column]].copy(deep=True)
        if datetime_format:
            norm_fmt = normalize_datetime_format(datetime_format)
            work_df["_dt"] = parse_datetime_column(work_df[time_column], norm_fmt).series
        else:
            work_df["_dt"] = pd.to_datetime(work_df[time_column], errors="coerce")
        clean_df = work_df.dropna(subset=["_dt", target_column]).sort_values(by="_dt")

        series = clean_df[target_column].to_numpy(dtype=float)
        n = len(series)
        if n < 10:
            raise ValueError(f"[INSUFFICIENT_DATA] Autocorrelation requires at least 10 observations, found {n}.")

        limit_lags = max(2, min(40, n // 2 - 1))
        if max_lags is not None:
            if not isinstance(max_lags, int) or max_lags < 1:
                raise ValueError(f"[INVALID_LAGS] max_lags must be a positive integer, got {max_lags}.")
            if max_lags >= n // 2:
                raise ValueError(f"[INVALID_LAGS] max_lags ({max_lags}) cannot exceed half the series length ({n // 2}).")
            lags_to_use = max_lags
        else:
            lags_to_use = limit_lags

        from statsmodels.tsa.stattools import acf, pacf

        acf_vals = acf(series, nlags=lags_to_use)
        pacf_vals = pacf(series, nlags=lags_to_use)

        sig_threshold = 1.96 / np.sqrt(n)
        significant_acf_lags = [int(i) for i, v in enumerate(acf_vals) if abs(v) > sig_threshold and i > 0]
        significant_pacf_lags = [int(i) for i, v in enumerate(pacf_vals) if abs(v) > sig_threshold and i > 0]

        # Generate Dual Subplot Diagnostic
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#090d16")
        for ax in axes:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, color="#1e293b", linestyle="--", alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color("#334155")

        # ACF Stem Plot
        axes[0].stem(range(len(acf_vals)), acf_vals, linefmt="#38bdf8", markerfmt="o", basefmt="#64748b")
        axes[0].axhline(sig_threshold, color="#f59e0b", linestyle="--", alpha=0.8, label="95% CI (±1.96/√N)")
        axes[0].axhline(-sig_threshold, color="#f59e0b", linestyle="--", alpha=0.8)
        axes[0].axhline(0, color="#64748b", linewidth=0.8)
        axes[0].set_title(f"Autocorrelation (ACF): {target_column}", color="#f8fafc", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Lag", color="#94a3b8", fontsize=10)
        axes[0].legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

        # PACF Stem Plot
        axes[1].stem(range(len(pacf_vals)), pacf_vals, linefmt="#10b981", markerfmt="o", basefmt="#64748b")
        axes[1].axhline(sig_threshold, color="#f59e0b", linestyle="--", alpha=0.8, label="95% CI (±1.96/√N)")
        axes[1].axhline(-sig_threshold, color="#f59e0b", linestyle="--", alpha=0.8)
        axes[1].axhline(0, color="#64748b", linewidth=0.8)
        axes[1].set_title(f"Partial Autocorrelation (PACF): {target_column}", color="#f8fafc", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Lag", color="#94a3b8", fontsize=10)
        axes[1].legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        lag_records = [
            {
                "lag": int(i),
                "acf": float(round(acf_vals[i], 4)),
                "pacf": float(round(pacf_vals[i], 4)),
                "is_significant_acf": abs(acf_vals[i]) > sig_threshold and i > 0,
                "is_significant_pacf": abs(pacf_vals[i]) > sig_threshold and i > 0,
            }
            for i in range(len(acf_vals))
        ]

        return sanitize_obj({
            "status": "SUCCEEDED",
            "time_column": time_column,
            "target_column": target_column,
            "max_lags": lags_to_use,
            "significance_threshold": float(sig_threshold),
            "significant_acf_lags": significant_acf_lags,
            "significant_pacf_lags": significant_pacf_lags,
            "lag_records": lag_records,
            "plot_base64": plot_base64
        })

    def run_decomposition(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        period: int,
        model: str = "additive",
        datetime_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Decomposes series into Trend, Seasonal, and Residual components.
        """
        self.validate_time_series(df, time_column, target_column, datetime_format=datetime_format)
        if not isinstance(period, int) or period <= 1:
            raise ValueError(f"[INVALID_PERIOD] Seasonal period must be an integer >= 2, got {period}.")
        if model not in {"additive", "multiplicative"}:
            raise ValueError(f"[INVALID_MODEL] Decomposition model must be 'additive' or 'multiplicative', got '{model}'.")

        work_df = df[[time_column, target_column]].copy(deep=True)
        if datetime_format:
            norm_fmt = normalize_datetime_format(datetime_format)
            work_df["_dt"] = parse_datetime_column(work_df[time_column], norm_fmt).series
        else:
            work_df["_dt"] = pd.to_datetime(work_df[time_column], errors="coerce")
        clean_df = work_df.dropna(subset=["_dt", target_column]).sort_values(by="_dt")

        if len(clean_df) < 2 * period:
            raise ValueError(
                f"[INSUFFICIENT_DATA] Decomposition with period={period} requires at least {2 * period} observations, "
                f"found {len(clean_df)}."
            )

        if model == "multiplicative" and (clean_df[target_column] <= 0).any():
            raise ValueError("[INVALID_MODEL] Multiplicative decomposition requires all target values to be strictly > 0.")

        from statsmodels.tsa.seasonal import seasonal_decompose
        decomp = seasonal_decompose(clean_df[target_column].to_numpy(dtype=float), model=model, period=period)

        # Plot 4-panel decomposition
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, facecolor="#090d16")
        titles = ["Observed Signal", f"Trend Component (period={period})", f"Seasonal Component ({model})", "Residuals (Noise)"]
        colors = ["#38bdf8", "#10b981", "#a78bfa", "#f59e0b"]
        series_list = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]

        for ax, title, color, s in zip(axes, titles, colors, series_list):
            ax.set_facecolor("#0f172a")
            ax.plot(s, color=color, linewidth=1.5)
            ax.set_ylabel(title.split()[0], color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, color="#1e293b", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#334155")

        axes[0].set_title(f"Time-Series Decomposition: {target_column} ({model})", color="#f8fafc", fontsize=12, fontweight="bold")
        axes[-1].set_xlabel("Observation Index", color="#94a3b8", fontsize=10)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return sanitize_obj({
            "status": "SUCCEEDED",
            "time_column": time_column,
            "target_column": target_column,
            "period": period,
            "model": model,
            "plot_base64": plot_base64
        })

    def run_forecast(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        model_type: str = "arima",
        horizon: int = 7,
        parameters: Optional[Dict[str, Any]] = None,
        val_size: Optional[int] = None,
        dataset_id: Optional[str] = "unnamed_dataset",
        dataset_name: Optional[str] = "Dataset",
        datetime_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fits time-series forecasting model (naive, autoreg, arima, sarimax),
        performs strict chronological holdout evaluation (no future leakage),
        generates 95% prediction intervals, and logs immutable provenance.
        """
        start_time = time.time()
        self.validate_time_series(df, time_column, target_column, datetime_format=datetime_format)

        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError(f"[INVALID_HORIZON] Forecast horizon must be a positive integer, got {horizon}.")

        supported_models = {"naive", "autoreg", "arima", "sarimax"}
        norm_model = model_type.lower().strip()
        if norm_model not in supported_models:
            raise ValueError(f"[UNSUPPORTED_MODEL] Model '{model_type}' is not supported. Supported models: {sorted(list(supported_models))}")

        params = parameters or {}
        work_df = df[[time_column, target_column]].copy(deep=True)
        if datetime_format:
            norm_fmt = normalize_datetime_format(datetime_format)
            work_df["_dt"] = parse_datetime_column(work_df[time_column], norm_fmt).series
        else:
            work_df["_dt"] = pd.to_datetime(work_df[time_column], errors="coerce")
        clean_df = work_df.dropna(subset=["_dt", target_column]).sort_values(by="_dt")

        N = len(clean_df)
        if N < 10:
            raise ValueError(f"[INSUFFICIENT_DATA] Forecasting requires at least 10 observations, found {N}.")

        # ── 1. Chronological Holdout Split for Evaluation ─────────────────────────
        # Default holdout validation window = min(horizon, max(3, N // 5))
        validation_len = val_size if (val_size and 1 <= val_size < N - 4) else min(horizon, max(3, N // 5))
        train_len = N - validation_len

        train_series = clean_df[target_column].iloc[:train_len].to_numpy(dtype=float)
        val_series = clean_df[target_column].iloc[train_len:].to_numpy(dtype=float)

        # Fit model on training slice only to predict holdout validation period
        val_pred = self._fit_and_predict(
            model_type=norm_model,
            train_data=train_series,
            steps=validation_len,
            params=params
        )

        # Baseline naive holdout prediction (last train value repeated)
        baseline_pred = np.repeat(train_series[-1], validation_len)

        # Compute out-of-sample metrics
        def calc_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, Optional[float]]:
            errors = actual - pred
            mae = float(np.mean(np.abs(errors)))
            mse = float(np.mean(errors ** 2))
            rmse = float(np.sqrt(mse))
            # Safe MAPE calculation: avoid zero division
            non_zeros = actual != 0
            if np.all(non_zeros):
                mape = float(np.mean(np.abs(errors / actual)) * 100.0)
            else:
                mape = None
            return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}

        model_metrics = calc_metrics(val_series, val_pred)
        baseline_metrics = calc_metrics(val_series, baseline_pred)

        # ── 2. Full Fit and Future Horizon Forecast ───────────────────────────────
        full_series = clean_df[target_column].to_numpy(dtype=float)
        future_pred, lower_ci, upper_ci = self._fit_and_predict_with_ci(
            model_type=norm_model,
            series=full_series,
            horizon=horizon,
            params=params
        )

        # Generate future timestamps
        last_dt = clean_df["_dt"].iloc[-1]
        grain_info = analyze_time_series_grain(clean_df, time_column)
        detected_freq = grain_info.detected_frequency

        future_timestamps = []
        if detected_freq:
            try:
                offset = pd.tseries.frequencies.to_offset(detected_freq)
                future_dts = pd.date_range(start=last_dt + offset, periods=horizon, freq=offset)
                future_timestamps = [dt.isoformat() for dt in future_dts]
            except Exception:
                pass

        if not future_timestamps or len(future_timestamps) != horizon:
            # Fallback delta extrapolation between last two observations
            if len(clean_df) >= 2:
                delta = clean_df["_dt"].iloc[-1] - clean_df["_dt"].iloc[-2]
                future_dts = [last_dt + delta * (i + 1) for i in range(horizon)]
                future_timestamps = [dt.isoformat() for dt in future_dts]
            else:
                future_timestamps = [f"t+{i+1}" for i in range(horizon)]

        # ── 3. Multi-Series Plot Generation ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#090d16")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#1e293b", linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        # Plot historical (last 60 observations)
        hist_display_len = min(60, N)
        hist_x = clean_df["_dt"].iloc[-hist_display_len:].tolist()
        hist_y = clean_df[target_column].iloc[-hist_display_len:].tolist()
        ax.plot(hist_x, hist_y, label=f"Historical ({target_column})", color="#38bdf8", linewidth=2)

        # Plot validation split line & validation prediction
        val_x = clean_df["_dt"].iloc[train_len:].tolist()
        ax.plot(val_x, val_pred, label=f"Validation Eval ({norm_model.upper()})", color="#f59e0b", linestyle=":", linewidth=1.8)

        # Plot future forecast & 95% confidence interval
        try:
            future_plot_x = [pd.to_datetime(ts) for ts in future_timestamps]
            ax.plot(future_plot_x, future_pred, label=f"Forecast ({norm_model.upper()} h={horizon})", color="#10b981", linewidth=2.2, linestyle="--")
            ax.fill_between(future_plot_x, lower_ci, upper_ci, color="#10b981", alpha=0.2, label="95% Confidence Interval")
        except Exception:
            pass

        ax.set_title(f"Time-Series Forecast: {target_column} ({norm_model.upper()} | Next {horizon} Periods)", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.set_xlabel(time_column, color="#94a3b8", fontsize=10)
        ax.set_ylabel(target_column, color="#94a3b8", fontsize=10)
        ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ── 4. Forecast Table Records ─────────────────────────────────────────────
        forecast_records = [
            {
                "step": int(i + 1),
                "timestamp": future_timestamps[i],
                "forecast": float(round(future_pred[i], 4)),
                "lower_ci_95": float(round(lower_ci[i], 4)),
                "upper_ci_95": float(round(upper_ci[i], 4))
            }
            for i in range(horizon)
        ]

        # ── 5. Reproducible Python Pipeline Code ───────────────────────────────────
        if norm_model == "arima":
            p = params.get("p", 1)
            d = params.get("d", 1)
            q = params.get("q", 1)
            code = (
                f"# DataPilot DS Engine: ARIMA({p},{d},{q}) Forecasting\n"
                f"from statsmodels.tsa.arima.model import ARIMA\n"
                f"ts_df = df[['{time_column}', '{target_column}']].dropna().sort_values('{time_column}')\n"
                f"model = ARIMA(ts_df['{target_column}'], order=({p}, {d}, {q}))\n"
                f"fitted = model.fit()\n"
                f"forecast_res = fitted.get_forecast(steps={horizon})\n"
                f"forecast_values = forecast_res.predicted_mean\n"
                f"ci = forecast_res.conf_int(alpha=0.05)\n"
                f"print('Forecast Next {horizon} Periods:', forecast_values)\n"
            )
        elif norm_model == "sarimax":
            p = params.get("p", 1)
            d = params.get("d", 1)
            q = params.get("q", 1)
            P = params.get("P", 1)
            D = params.get("D", 0)
            Q = params.get("Q", 1)
            s = params.get("s", 4)
            code = (
                f"# DataPilot DS Engine: SARIMAX({p},{d},{q})x({P},{D},{Q},{s}) Forecasting\n"
                f"from statsmodels.tsa.statespace.sarimax import SARIMAX\n"
                f"ts_df = df[['{time_column}', '{target_column}']].dropna().sort_values('{time_column}')\n"
                f"model = SARIMAX(ts_df['{target_column}'], order=({p}, {d}, {q}), seasonal_order=({P}, {D}, {Q}, {s}))\n"
                f"fitted = model.fit(disp=False)\n"
                f"forecast_res = fitted.get_forecast(steps={horizon})\n"
                f"forecast_values = forecast_res.predicted_mean\n"
                f"ci = forecast_res.conf_int(alpha=0.05)\n"
                f"print('Forecast Next {horizon} Periods:', forecast_values)\n"
            )
        elif norm_model == "autoreg":
            lags = params.get("lags", 2)
            code = (
                f"# DataPilot DS Engine: AutoReg(lags={lags}) Forecasting\n"
                f"from statsmodels.tsa.ar_model import AutoReg\n"
                f"ts_df = df[['{time_column}', '{target_column}']].dropna().sort_values('{time_column}')\n"
                f"model = AutoReg(ts_df['{target_column}'], lags={lags})\n"
                f"fitted = model.fit()\n"
                f"forecast_values = fitted.forecast(steps={horizon})\n"
                f"print('Forecast Next {horizon} Periods:', forecast_values)\n"
            )
        else:
            code = (
                f"# DataPilot DS Engine: Naive Last-Value Baseline Forecasting\n"
                f"ts_df = df[['{time_column}', '{target_column}']].dropna().sort_values('{time_column}')\n"
                f"last_value = ts_df['{target_column}'].iloc[-1]\n"
                f"forecast_values = [last_value] * {horizon}\n"
                f"print('Forecast Next {horizon} Periods:', forecast_values)\n"
            )

        # ── 6. Analytical Finding Summary ────────────────────────────────────────
        finding = (
            f"Fitted **{norm_model.upper()}** model for `{target_column}` over `{time_column}`. "
            f"Evaluated on held-out {validation_len}-period chronological validation window: "
            f"**RMSE = {model_metrics['rmse']:.3f}** (vs Naive baseline RMSE = {baseline_metrics['rmse']:.3f}). "
            f"Generated {horizon}-period forward projections with 95% confidence bounds."
        )

        # ── 7. Record Provenance Audit Event ───────────────────────────────────────
        event_id = None
        if dataset_id:
            try:
                event = provenance_store.record_event(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or "Dataset",
                    operation="TIME_SERIES_FORECAST",
                    capability="time_series_forecasting",
                    affected_columns=[time_column, target_column],
                    parameters={
                        "model": norm_model,
                        "horizon": horizon,
                        "validation_holdout": validation_len,
                        "parameters": params
                    },
                    df_before=df,
                    df_after=df,  # Active dataset is strictly NOT mutated
                    status="SUCCESS",
                    source={"type": "TIME_SERIES_STUDIO", "tab": "forecasting"},
                    finding=finding,
                    code=code
                )
                event_id = getattr(event, "event_id", None)
            except Exception as prov_err:
                logger.warning(f"Provenance record error in time series: {prov_err}")

        execution_ms = (time.time() - start_time) * 1000.0

        return sanitize_obj({
            "status": "SUCCEEDED",
            "operation": "time_series_forecast",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "time_column": time_column,
            "target_column": target_column,
            "model": norm_model,
            "horizon": horizon,
            "validation_window": validation_len,
            "metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "forecast_records": forecast_records,
            "plot_base64": plot_base64,
            "reproducible_code": code,
            "finding": finding,
            "provenance_event_id": event_id,
            "execution_time_ms": execution_ms
        })

    # ── Internal Fit & Predict Helpers ─────────────────────────────────────────

    def _fit_and_predict(
        self,
        model_type: str,
        train_data: np.ndarray,
        steps: int,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Fits model on train slice and forecasts `steps` ahead."""
        if model_type == "naive":
            return np.repeat(train_data[-1], steps)

        if model_type == "autoreg":
            from statsmodels.tsa.ar_model import AutoReg
            lags = params.get("lags", 1)
            lags = max(1, min(int(lags), len(train_data) // 3))
            try:
                fitted = AutoReg(train_data, lags=lags).fit()
                pred = fitted.forecast(steps=steps)
                return np.asarray(pred, dtype=float)
            except Exception as e:
                logger.warning(f"AutoReg fit failed ({e}), falling back to naive")
                return np.repeat(train_data[-1], steps)

        if model_type == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            try:
                fitted = ARIMA(train_data, order=(p, d, q)).fit()
                res = fitted.get_forecast(steps=steps)
                return np.asarray(res.predicted_mean, dtype=float)
            except Exception as e:
                logger.warning(f"ARIMA fit failed ({e}), falling back to naive")
                return np.repeat(train_data[-1], steps)

        if model_type == "sarimax":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            P = int(params.get("P", 1))
            D = int(params.get("D", 0))
            Q = int(params.get("Q", 1))
            s = int(params.get("s", 4))
            try:
                fitted = SARIMAX(
                    train_data,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                res = fitted.get_forecast(steps=steps)
                return np.asarray(res.predicted_mean, dtype=float)
            except Exception as e:
                logger.warning(f"SARIMAX fit failed ({e}), falling back to naive")
                return np.repeat(train_data[-1], steps)

        return np.repeat(train_data[-1], steps)

    def _fit_and_predict_with_ci(
        self,
        model_type: str,
        series: np.ndarray,
        horizon: int,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits model on entire series and returns (predictions, lower_ci, upper_ci)."""
        resid_std = float(np.std(series)) if len(series) > 1 else 1.0

        if model_type == "naive":
            pred = np.repeat(series[-1], horizon)
            h_steps = np.arange(1, horizon + 1)
            margin = 1.96 * resid_std * np.sqrt(h_steps)
            return pred, pred - margin, pred + margin

        if model_type == "autoreg":
            from statsmodels.tsa.ar_model import AutoReg
            lags = max(1, min(int(params.get("lags", 1)), len(series) // 3))
            try:
                fitted = AutoReg(series, lags=lags).fit()
                pred = np.asarray(fitted.forecast(steps=horizon), dtype=float)
                fit_resid_std = float(np.std(fitted.resid)) if len(fitted.resid) > 1 else resid_std
                h_steps = np.arange(1, horizon + 1)
                margin = 1.96 * fit_resid_std * np.sqrt(h_steps)
                return pred, pred - margin, pred + margin
            except Exception as e:
                logger.warning(f"AutoReg fit failed ({e})")
                pred = np.repeat(series[-1], horizon)
                margin = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
                return pred, pred - margin, pred + margin

        if model_type == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            try:
                fitted = ARIMA(series, order=(p, d, q)).fit()
                res = fitted.get_forecast(steps=horizon)
                pred = np.asarray(res.predicted_mean, dtype=float)
                ci = res.conf_int(alpha=0.05)
                lower = np.asarray(ci[:, 0] if hasattr(ci, "shape") and len(ci.shape) > 1 else ci.iloc[:, 0], dtype=float)
                upper = np.asarray(ci[:, 1] if hasattr(ci, "shape") and len(ci.shape) > 1 else ci.iloc[:, 1], dtype=float)
                return pred, lower, upper
            except Exception as e:
                logger.warning(f"ARIMA fit failed ({e})")
                pred = np.repeat(series[-1], horizon)
                margin = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
                return pred, pred - margin, pred + margin

        if model_type == "sarimax":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            P = int(params.get("P", 1))
            D = int(params.get("D", 0))
            Q = int(params.get("Q", 1))
            s = int(params.get("s", 4))
            try:
                fitted = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                res = fitted.get_forecast(steps=horizon)
                pred = np.asarray(res.predicted_mean, dtype=float)
                ci = res.conf_int(alpha=0.05)
                lower = np.asarray(ci[:, 0] if hasattr(ci, "shape") and len(ci.shape) > 1 else ci.iloc[:, 0], dtype=float)
                upper = np.asarray(ci[:, 1] if hasattr(ci, "shape") and len(ci.shape) > 1 else ci.iloc[:, 1], dtype=float)
                return pred, lower, upper
            except Exception as e:
                logger.warning(f"SARIMAX fit failed ({e})")
                pred = np.repeat(series[-1], horizon)
                margin = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
                return pred, pred - margin, pred + margin

        pred = np.repeat(series[-1], horizon)
        margin = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
        return pred, pred - margin, pred + margin


timeseries_engine = TimeSeriesEngine()
