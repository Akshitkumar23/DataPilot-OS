"""
DataPilot OS — Time Series Grain & Temporal Grid Feasibility.
Generic detection of duplicate timestamps, irregular temporal grids,
and missing explicit resampling decisions.
Zero hardcoded domain heuristics.
"""

from __future__ import annotations

import logging
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TimeSeriesGrainDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_column: str
    row_count: int
    unique_timestamps: int
    has_duplicates: bool
    duplicate_count: int
    is_strictly_monotonic: bool
    is_regular_grid: bool
    detected_frequency: Optional[str] = None
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None
    suggested_grains: List[str] = Field(default_factory=list)


def analyze_time_series_grain(
    df: pd.DataFrame,
    time_column: str,
    datetime_format: Optional[str] = None,
) -> TimeSeriesGrainDiagnostics:
    """
    Analyze the temporal grain and regularity of time_column in df.
    Emits clarification requirement when:
    1. Duplicate timestamps exist (ambiguous row aggregation).
    2. Irregular time grid exists (requires explicit resampling cadence).
    """
    if df is None or time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' does not exist in dataset.")

    series = df[time_column].dropna()
    row_count = len(series)
    if row_count == 0:
        return TimeSeriesGrainDiagnostics(
            time_column=time_column,
            row_count=0,
            unique_timestamps=0,
            has_duplicates=False,
            duplicate_count=0,
            is_strictly_monotonic=False,
            is_regular_grid=False,
            requires_clarification=True,
            clarification_reason=f"Time column '{time_column}' has no non-null values.",
        )

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            if datetime_format:
                from backend.ds_engine.datetime_parser import normalize_datetime_format, parse_datetime_column
                norm_fmt = normalize_datetime_format(datetime_format)
                parsed_res = parse_datetime_column(series, norm_fmt)
                valid_dt = parsed_res.series.dropna()
            else:
                parsed_series = pd.to_datetime(series, errors="coerce")
                valid_dt = parsed_series.dropna()
        except Exception:
            valid_dt = pd.Series([], dtype="datetime64[ns]")
    else:
        valid_dt = series

    unique_ts = int(valid_dt.nunique())
    duplicate_count = len(valid_dt) - unique_ts
    has_duplicates = duplicate_count > 0

    if has_duplicates:
        return TimeSeriesGrainDiagnostics(
            time_column=time_column,
            row_count=row_count,
            unique_timestamps=unique_ts,
            has_duplicates=True,
            duplicate_count=duplicate_count,
            is_strictly_monotonic=False,
            is_regular_grid=False,
            requires_clarification=True,
            clarification_reason=(
                f"Time column '{time_column}' contains {duplicate_count} duplicate timestamp(s). "
                "Time-series operations require a unique temporal index. "
                "Please specify an aggregation grain (e.g. daily 'D', hourly 'h') to consolidate duplicates."
            ),
            suggested_grains=["D", "W", "ME", "h"],
        )

    # Sort to inspect ordering and delta regularity
    sorted_dt = valid_dt.sort_values().drop_duplicates()
    is_monotonic = bool(valid_dt.is_monotonic_increasing)

    detected_freq = None
    if len(sorted_dt) >= 3:
        try:
            detected_freq = pd.infer_freq(pd.DatetimeIndex(sorted_dt))
        except (TypeError, ValueError):
            detected_freq = None

    # Check intervals between successive timestamps
    if len(sorted_dt) >= 3:
        deltas = sorted_dt.diff().dropna()
        unique_deltas = deltas.nunique()
        is_regular = (unique_deltas == 1) or (detected_freq is not None)
    else:
        is_regular = True

    if not is_regular and len(sorted_dt) >= 5:
        return TimeSeriesGrainDiagnostics(
            time_column=time_column,
            row_count=row_count,
            unique_timestamps=unique_ts,
            has_duplicates=False,
            duplicate_count=0,
            is_strictly_monotonic=is_monotonic,
            is_regular_grid=False,
            detected_frequency=detected_freq,
            requires_clarification=True,
            clarification_reason=(
                f"Time column '{time_column}' has an irregular time grid with uneven spacing between timestamps. "
                "Please specify an explicit resampling cadence/grain (e.g. 'D' for daily, 'W' for weekly, 'ME' for monthly) "
                "instead of assuming an invisible frequency."
            ),
            suggested_grains=["D", "W", "ME"],
        )

    return TimeSeriesGrainDiagnostics(
        time_column=time_column,
        row_count=row_count,
        unique_timestamps=unique_ts,
        has_duplicates=False,
        duplicate_count=0,
        is_strictly_monotonic=is_monotonic,
        is_regular_grid=True,
        detected_frequency=detected_freq,
        requires_clarification=False,
    )
