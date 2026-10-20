"""
DataPilot OS — Strict Operator Input/Output Contracts & Numerical Invariants.
Enforces exact schemas, manifest generation, and mathematical invariants
across all 22 analytical operators.
Zero silent defaults, zero hallucinated approximations.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class OperatorFailureCode(str, Enum):
    COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"
    EMPTY_DATASET = "EMPTY_DATASET"
    INSUFFICIENT_SAMPLE_SIZE = "INSUFFICIENT_SAMPLE_SIZE"
    ZERO_VARIANCE = "ZERO_VARIANCE"
    DEGENERATE_CONTINGENCY_TABLE = "DEGENERATE_CONTINGENCY_TABLE"
    DIVISION_BY_ZERO = "DIVISION_BY_ZERO"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"
    INSUFFICIENT_NUMERIC_COLUMNS = "INSUFFICIENT_NUMERIC_COLUMNS"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    TARGET_LEAKAGE_DETECTED = "TARGET_LEAKAGE_DETECTED"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"


class OperatorManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str
    columns_used: List[str] = Field(default_factory=list)
    row_count_in: int
    row_count_out: int
    parameters: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    execution_time_ms: Optional[float] = None


class NumericalInvariantViolation(ValueError):
    """Raised when an analytical operator violates a fundamental mathematical invariant."""
    pass


class NumericalInvariantValidator:
    """
    Mathematical and statistical invariant verifiers for all analytical results.
    Guarantees that no corrupted or physically impossible figures reach the user.
    """

    @staticmethod
    def validate_correlation_matrix(matrix: pd.DataFrame) -> None:
        """
        Verify correlation matrix invariants:
        1. Square matrix
        2. All entries in [-1.0 - epsilon, 1.0 + epsilon]
        3. Diagonal entries == 1.0
        4. Symmetric matrix: M[i, j] == M[j, i]
        """
        if matrix.empty:
            raise NumericalInvariantViolation("Correlation matrix is empty.")

        n_rows, n_cols = matrix.shape
        if n_rows != n_cols:
            raise NumericalInvariantViolation(
                f"Correlation matrix must be square; got shape ({n_rows}, {n_cols})."
            )

        vals = matrix.to_numpy(dtype=float)

        # Check range [-1.0, 1.0] with floating point tolerance
        if np.any(vals < -1.0001) or np.any(vals > 1.0001):
            bad_vals = vals[(vals < -1.0001) | (vals > 1.0001)]
            raise NumericalInvariantViolation(
                f"Correlation coefficients out of bounds [-1, 1]: {bad_vals.tolist()[:5]}"
            )

        # Check diagonal is 1.0
        diag = np.diag(vals)
        if not np.allclose(diag, 1.0, atol=1e-3, equal_nan=False):
            raise NumericalInvariantViolation(
                f"Correlation diagonal must equal 1.0; got {diag.tolist()}"
            )

        # Check symmetry
        if not np.allclose(vals, vals.T, atol=1e-3, equal_nan=True):
            raise NumericalInvariantViolation(
                "Correlation matrix is asymmetric (M != M^T)."
            )

    @staticmethod
    def validate_rate_or_percentage(
        values: Union[pd.Series, np.ndarray, List[float], float],
        is_percentage: bool = True,
    ) -> None:
        """
        Verify that rates are in [0, 1] or percentages in [0, 100].
        """
        upper_bound = 100.0001 if is_percentage else 1.0001
        arr = np.asarray(values, dtype=float)
        # Filter out NaNs if any
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return

        if np.any(valid < -1e-5):
            raise NumericalInvariantViolation(
                f"Rates/percentages cannot be negative; got minimum {float(valid.min())}."
            )
        if np.any(valid > upper_bound):
            raise NumericalInvariantViolation(
                f"Values exceed upper bound {upper_bound}; got maximum {float(valid.max())}."
            )

    @staticmethod
    def validate_statistical_test_result(
        test_statistic: float,
        p_value: float,
        dof: Optional[float] = None,
    ) -> None:
        """
        Verify inferential test output:
        1. Test statistic must be a finite real number (not NaN or Inf)
        2. p-value must be strictly in [0.0, 1.0]
        3. dof (if present) must be positive
        """
        if not np.isfinite(test_statistic):
            raise NumericalInvariantViolation(
                f"Test statistic must be finite real number; got {test_statistic}."
            )
        if np.isnan(p_value) or p_value < -1e-7 or p_value > 1.000001:
            raise NumericalInvariantViolation(
                f"p-value must be in [0, 1]; got {p_value}."
            )
        if dof is not None and (np.isnan(dof) or dof < 0):
            raise NumericalInvariantViolation(
                f"Degrees of freedom must be non-negative; got {dof}."
            )

    @staticmethod
    def validate_confusion_matrix(
        cm: Union[np.ndarray, List[List[int]]],
        expected_total_samples: Optional[int] = None,
    ) -> None:
        """
        Verify confusion matrix invariants:
        1. Non-negative integer entries
        2. Square matrix (K classes x K classes)
        3. Sum of all entries matches expected evaluation sample count
        """
        arr = np.asarray(cm, dtype=int)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise NumericalInvariantViolation(
                f"Confusion matrix must be 2D square matrix; got shape {arr.shape}."
            )
        if np.any(arr < 0):
            raise NumericalInvariantViolation(
                f"Confusion matrix entries cannot be negative; got min {arr.min()}."
            )
        if expected_total_samples is not None and arr.sum() != expected_total_samples:
            raise NumericalInvariantViolation(
                f"Confusion matrix total entries ({arr.sum()}) does not match expected sample count ({expected_total_samples})."
            )

    @staticmethod
    def validate_pca_variance_ratios(ratios: List[float]) -> None:
        """
        Verify PCA explained variance ratios:
        1. Each ratio in [0, 1]
        2. Monotonically non-increasing: r[i] >= r[i+1]
        3. Cumulative sum <= 1.0001
        """
        if not ratios:
            raise NumericalInvariantViolation("PCA explained variance ratios cannot be empty.")
        arr = np.asarray(ratios, dtype=float)
        if np.any(arr < -1e-6) or np.any(arr > 1.0001):
            raise NumericalInvariantViolation(
                f"PCA explained variance ratios must be in [0, 1]; got {ratios}."
            )
        # Check non-increasing
        diffs = np.diff(arr)
        if np.any(diffs > 1e-6):
            raise NumericalInvariantViolation(
                f"PCA components must be sorted by descending variance; got {ratios}."
            )
        if arr.sum() > 1.0001:
            raise NumericalInvariantViolation(
                f"Cumulative explained variance exceeds 1.0: {arr.sum()}."
            )

    @staticmethod
    def validate_clustering_output(
        labels: Union[np.ndarray, List[int]],
        expected_row_count: int,
        k: Optional[int] = None,
    ) -> None:
        """
        Verify clustering output invariants:
        1. Length of labels equals input row count
        2. Number of unique labels <= k (if k supplied)
        """
        arr = np.asarray(labels, dtype=int)
        if len(arr) != expected_row_count:
            raise NumericalInvariantViolation(
                f"Clustering output label count ({len(arr)}) does not match input row count ({expected_row_count})."
            )
        unique_labels = np.unique(arr[arr >= 0])  # Exclude noise label -1 (e.g. DBSCAN)
        if k is not None and len(unique_labels) > k:
            raise NumericalInvariantViolation(
                f"Clustering returned {len(unique_labels)} clusters, exceeding requested k={k}."
            )
