"""
DataPilot OS — Production-Grade Exploratory Data Analysis (EDA) & Statistics Engine.
100% Deterministic, Dataset-Agnostic, Zero-Hardcoding, Factual Calculations on Pandas DataFrames.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def sanitize_float(val: Any) -> Optional[float]:
    """Convert float value to JSON-safe float or None if NaN/Inf."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def sanitize_obj(obj: Any) -> Any:
    """Recursively sanitize data structures for strict JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): sanitize_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_obj(v) for v in obj]
    elif isinstance(obj, float):
        return sanitize_float(obj)
    elif isinstance(obj, (np.floating,)):
        return sanitize_float(float(obj))
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif pd.isna(obj):
        return None
    return obj


class EDAEngine:
    """
    Deterministic EDA calculation engine.
    Calculates descriptive statistics, missingness, distribution histograms,
    correlation matrices, and skewness directly on the active DataFrame.
    """

    @staticmethod
    def compute_descriptive_stats(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate complete descriptive statistics on numerical columns.
        Includes count, mean, median, std, min, max, quartiles (25%, 50%, 75%),
        variance, and IQR.
        """
        if df is None or len(df) == 0:
            raise ValueError("EMPTY_DATASET: The active dataset has no rows.")

        if columns:
            if isinstance(columns, str):
                columns = [columns]
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"COLUMN_NOT_FOUND: Columns not found in dataset: {missing}")
            non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                raise ValueError(f"NON_NUMERIC_COLUMN: Columns are not numeric: {non_numeric}")
            target_cols = [c for c in columns if c in df.columns]
        else:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not target_cols:
            raise ValueError("NO_NUMERIC_COLUMNS: The dataset contains no numeric columns for descriptive statistics.")

        records = []
        for col in target_cols:
            series = df[col]
            valid_s = series.dropna()
            count = int(len(valid_s))
            null_count = int(series.isna().sum())

            if count > 0:
                mean_val = float(valid_s.mean())
                median_val = float(valid_s.median())
                std_val = float(valid_s.std(ddof=1)) if count > 1 else 0.0
                var_val = float(valid_s.var(ddof=1)) if count > 1 else 0.0
                min_val = float(valid_s.min())
                max_val = float(valid_s.max())
                q25_val = float(valid_s.quantile(0.25))
                q50_val = float(valid_s.quantile(0.50))
                q75_val = float(valid_s.quantile(0.75))
                iqr_val = float(q75_val - q25_val)
            else:
                mean_val = median_val = std_val = var_val = None
                min_val = max_val = q25_val = q50_val = q75_val = iqr_val = None

            records.append({
                "column": col,
                "dtype": str(series.dtype),
                "count": count,
                "null_count": null_count,
                "null_percentage": round((null_count / len(df) * 100), 2) if len(df) > 0 else 0.0,
                "mean": sanitize_float(mean_val),
                "median": sanitize_float(median_val),
                "std": sanitize_float(std_val),
                "variance": sanitize_float(var_val),
                "min": sanitize_float(min_val),
                "q25": sanitize_float(q25_val),
                "q50": sanitize_float(q50_val),
                "q75": sanitize_float(q75_val),
                "max": sanitize_float(max_val),
                "iqr": sanitize_float(iqr_val),
            })

        return sanitize_obj({
            "operation": "DESCRIPTIVE_STATS",
            "total_rows": len(df),
            "columns": target_cols,
            "statistics": records
        })

    @staticmethod
    def compute_missingness(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate exact missing value counts and ratios across all dataset columns
        and summary metrics.
        """
        if df is None:
            raise ValueError("EMPTY_DATASET: No active DataFrame available.")

        total_rows = int(len(df))
        total_cols = int(len(df.columns))
        total_cells = total_rows * total_cols
        total_missing = int(df.isna().sum().sum())
        overall_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
        complete_rows = int(len(df.dropna()))
        rows_with_missing = total_rows - complete_rows

        columns_data = []
        for col in df.columns:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0
            non_missing_count = total_rows - missing_count

            if missing_count == 0:
                status = "Complete"
            elif missing_pct < 15.0:
                status = "Low Missingness"
            elif missing_pct < 40.0:
                status = "Moderate Missingness"
            else:
                status = "High Missingness"

            columns_data.append({
                "column": str(col),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "non_missing_count": non_missing_count,
                "status": status,
            })

        return sanitize_obj({
            "operation": "MISSINGNESS_ANALYSIS",
            "summary": {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "total_cells": total_cells,
                "total_missing_cells": total_missing,
                "overall_missing_percentage": overall_pct,
                "complete_rows": complete_rows,
                "rows_with_missing": rows_with_missing,
            },
            "columns": columns_data
        })

    @staticmethod
    def compute_distribution(
        df: pd.DataFrame,
        column: str,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute frequency distribution / histogram bins and empirical bounds
        for a single numerical column.
        """
        if df is None or len(df) == 0:
            raise ValueError("EMPTY_DATASET: The active dataset has no rows.")

        if not column:
            raise ValueError("MISSING_REQUIRED_ARGUMENT: Target column must be specified for distribution analysis.")

        if column not in df.columns:
            raise ValueError(f"COLUMN_NOT_FOUND: Column '{column}' not found in active dataset.")

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"NON_NUMERIC_COLUMN: Column '{column}' is not numeric ({df[column].dtype}). "
                "Distribution analysis requires a numeric variable."
            )

        valid_s = df[column].dropna().astype(float)
        if len(valid_s) == 0:
            raise ValueError(f"INSUFFICIENT_DATA: Column '{column}' contains no non-null values.")

        num_bins = max(2, min(int(n_bins or 10), 50))

        # Handle constant series
        if valid_s.nunique() <= 1:
            val = float(valid_s.iloc[0])
            bins = [{
                "bin_index": 0,
                "bin_start": val,
                "bin_end": val,
                "label": f"{val:.2f}",
                "count": int(len(valid_s)),
                "percentage": 100.0,
            }]
            counts = [len(valid_s)]
            bin_edges = [val, val]
        else:
            counts, bin_edges = np.histogram(valid_s, bins=num_bins)
            bins = []
            for i in range(len(counts)):
                start = float(bin_edges[i])
                end = float(bin_edges[i + 1])
                bins.append({
                    "bin_index": i,
                    "bin_start": sanitize_float(round(start, 4)),
                    "bin_end": sanitize_float(round(end, 4)),
                    "label": f"{start:.2f} - {end:.2f}",
                    "count": int(counts[i]),
                    "percentage": round(float(counts[i] / len(valid_s) * 100), 2),
                })

        mean_val = float(valid_s.mean())
        median_val = float(valid_s.median())
        std_val = float(valid_s.std(ddof=1)) if len(valid_s) > 1 else 0.0
        min_val = float(valid_s.min())
        max_val = float(valid_s.max())
        q25_val = float(valid_s.quantile(0.25))
        q75_val = float(valid_s.quantile(0.75))
        iqr_val = float(q75_val - q25_val)

        lower_bound = q25_val - 1.5 * iqr_val
        upper_bound = q75_val + 1.5 * iqr_val
        outlier_count = int(((valid_s < lower_bound) | (valid_s > upper_bound)).sum())
        zero_count = int((valid_s == 0).sum())
        zero_rate_pct = round(zero_count / len(valid_s) * 100, 2)

        return sanitize_obj({
            "operation": "DISTRIBUTION_ANALYSIS",
            "column": column,
            "total_rows": len(df),
            "sample_size": len(valid_s),
            "null_count": int(df[column].isna().sum()),
            "n_bins": len(bins),
            "bins": bins,
            "summary": {
                "mean": sanitize_float(mean_val),
                "median": sanitize_float(median_val),
                "std": sanitize_float(std_val),
                "min": sanitize_float(min_val),
                "q25": sanitize_float(q25_val),
                "q75": sanitize_float(q75_val),
                "max": sanitize_float(max_val),
                "iqr": sanitize_float(iqr_val),
                "outlier_count": outlier_count,
                "zero_count": zero_count,
                "zero_rate_percentage": zero_rate_pct,
            }
        })

    @staticmethod
    def compute_correlation(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        Compute true correlation matrix and ranked pairs for numerical variables.
        """
        if df is None or len(df) == 0:
            raise ValueError("EMPTY_DATASET: The active dataset has no rows.")

        if columns:
            if isinstance(columns, str):
                columns = [columns]
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"COLUMN_NOT_FOUND: Columns not found in dataset: {missing}")
            non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                raise ValueError(f"NON_NUMERIC_COLUMN: Columns are not numeric: {non_numeric}")
            subjects = [c for c in columns if c in df.columns]
        else:
            subjects = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(subjects) < 2:
            raise ValueError(
                f"INSUFFICIENT_NUMERIC_COLUMNS: Correlation analysis requires at least 2 numerical features; found {len(subjects)}."
            )

        norm_method = method.lower() if method and method.lower() in ("pearson", "spearman", "kendall") else "pearson"
        corr_matrix = df[subjects].corr(method=norm_method)

        matrix = []
        for c1 in subjects:
            row_vals = []
            for c2 in subjects:
                val = corr_matrix.loc[c1, c2]
                row_vals.append(sanitize_float(round(float(val), 4)) if pd.notna(val) else None)
            matrix.append(row_vals)

        pairs = []
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                c1, c2 = subjects[i], subjects[j]
                val = corr_matrix.loc[c1, c2]
                if pd.notna(val):
                    f_val = float(val)
                    abs_val = abs(f_val)
                    if abs_val >= 0.8:
                        strength = "Very Strong"
                    elif abs_val >= 0.6:
                        strength = "Strong"
                    elif abs_val >= 0.4:
                        strength = "Moderate"
                    elif abs_val >= 0.2:
                        strength = "Weak"
                    else:
                        strength = "Negligible"

                    direction = "Positive" if f_val > 0.05 else "Negative" if f_val < -0.05 else "Zero"

                    pairs.append({
                        "feature_a": c1,
                        "feature_b": c2,
                        "correlation": sanitize_float(round(f_val, 4)),
                        "abs_correlation": sanitize_float(round(abs_val, 4)),
                        "direction": direction,
                        "strength": strength,
                    })

        pairs.sort(key=lambda p: (p.get("abs_correlation") or 0.0), reverse=True)

        return sanitize_obj({
            "operation": "CORRELATION_ANALYSIS",
            "method": norm_method,
            "columns": subjects,
            "matrix": matrix,
            "pairs": pairs,
            "feature_count": len(subjects),
            "total_rows": len(df)
        })

    @staticmethod
    def compute_skewness(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute skewness, kurtosis, and distribution normality assessment for numerical columns.
        """
        if df is None or len(df) == 0:
            raise ValueError("EMPTY_DATASET: The active dataset has no rows.")

        if columns:
            if isinstance(columns, str):
                columns = [columns]
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"COLUMN_NOT_FOUND: Columns not found in dataset: {missing}")
            non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                raise ValueError(f"NON_NUMERIC_COLUMN: Columns are not numeric: {non_numeric}")
            subjects = [c for c in columns if c in df.columns]
        else:
            subjects = df.select_dtypes(include=[np.number]).columns.tolist()

        if not subjects:
            raise ValueError("NO_NUMERIC_COLUMNS: The dataset contains no numeric columns for skewness analysis.")

        results = []
        for col in subjects:
            valid_s = df[col].dropna()
            count = int(len(valid_s))

            if count < 3 or valid_s.nunique() <= 1:
                results.append({
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "count": count,
                    "skewness": None,
                    "kurtosis": None,
                    "classification": "Insufficient Data",
                    "direction": "Undefined",
                    "recommendation": "Collect more distinct observations"
                })
                continue

            try:
                skew_val = float(valid_s.skew())
                kurt_val = float(valid_s.kurtosis())
            except Exception:
                skew_val = None
                kurt_val = None

            if skew_val is not None and not math.isnan(skew_val):
                abs_skew = abs(skew_val)
                if abs_skew < 0.5:
                    classification = "Symmetric / Normal"
                    recommendation = "Standard scaling or Gaussian modeling appropriate"
                elif abs_skew <= 1.0:
                    classification = "Moderately Skewed"
                    recommendation = "Robust scaling or Box-Cox/Yeo-Johnson may improve performance"
                else:
                    classification = "Highly Skewed"
                    recommendation = "Log transform or Yeo-Johnson power transform strongly advised"

                if skew_val > 0.05:
                    direction = "Positive (Right-tailed)"
                elif skew_val < -0.05:
                    direction = "Negative (Left-tailed)"
                else:
                    direction = "Symmetric"
            else:
                classification = "Indeterminate"
                direction = "Indeterminate"
                recommendation = "Check for constant or near-constant variance"

            results.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "count": count,
                "skewness": sanitize_float(round(skew_val, 4)) if skew_val is not None else None,
                "kurtosis": sanitize_float(round(kurt_val, 4)) if kurt_val is not None else None,
                "classification": classification,
                "direction": direction,
                "recommendation": recommendation,
            })

        return sanitize_obj({
            "operation": "SKEWNESS_ANALYSIS",
            "columns": subjects,
            "total_rows": len(df),
            "results": results
        })


eda_engine = EDAEngine()
