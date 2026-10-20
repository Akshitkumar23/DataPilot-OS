"""
DataPilot OS — Independent Numerical Oracles.
Provides pure reference implementations of statistical and analytical calculations
used to independently audit operator outputs without trusting operator code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class NumericalOracles:
    """Independent calculation oracles for verification."""

    @staticmethod
    def compute_group_aggregation(
        df: pd.DataFrame,
        group_by: List[str],
        metric: Optional[str] = None,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """Independently compute group aggregation on raw dataframe."""
        if not group_by:
            raise ValueError("group_by cannot be empty.")

        if metric is None or aggregation == "count":
            return df.groupby(group_by, dropna=False).size().reset_index(name="count")

        agg_lower = aggregation.lower()
        if agg_lower == "mean":
            return df.groupby(group_by, dropna=False)[metric].mean().reset_index()
        elif agg_lower == "sum":
            return df.groupby(group_by, dropna=False)[metric].sum().reset_index()
        elif agg_lower == "median":
            return df.groupby(group_by, dropna=False)[metric].median().reset_index()
        elif agg_lower == "min":
            return df.groupby(group_by, dropna=False)[metric].min().reset_index()
        elif agg_lower == "max":
            return df.groupby(group_by, dropna=False)[metric].max().reset_index()
        else:
            raise ValueError(f"Unsupported oracle aggregation: {aggregation}")

    @staticmethod
    def compute_correlation_matrix(
        df: pd.DataFrame,
        columns: List[str],
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Independently compute correlation matrix."""
        subset = df[columns].dropna()
        return subset.corr(method=method)

    @staticmethod
    def compute_independent_ttest(
        sample_a: Union[pd.Series, np.ndarray, List[float]],
        sample_b: Union[pd.Series, np.ndarray, List[float]],
        equal_var: bool = False,
    ) -> Tuple[float, float]:
        """Independently compute Welch's t-test statistic and p-value."""
        s1 = np.asarray(pd.Series(sample_a).dropna(), dtype=float)
        s2 = np.asarray(pd.Series(sample_b).dropna(), dtype=float)
        res = stats.ttest_ind(s1, s2, equal_var=equal_var)
        return float(res.statistic), float(res.pvalue)

    @staticmethod
    def compute_chi_square_independence(
        df: pd.DataFrame,
        col_x: str,
        col_y: str,
    ) -> Tuple[float, float, int]:
        """Independently compute Chi-square test of independence."""
        ct = pd.crosstab(df[col_x], df[col_y], dropna=False)
        res = stats.chi2_contingency(ct)
        return float(res.statistic), float(res.pvalue), int(res.dof)

    @staticmethod
    def compute_one_way_anova(
        df: pd.DataFrame,
        value_column: str,
        group_column: str,
    ) -> Tuple[float, float]:
        """Independently compute One-Way ANOVA."""
        groups = [
            pd.to_numeric(group[value_column], errors="coerce").dropna().to_numpy()
            for _, group in df[[value_column, group_column]].dropna(subset=[group_column]).groupby(group_column)
        ]
        groups = [g for g in groups if len(g) > 0]
        res = stats.f_oneway(*groups)
        return float(res.statistic), float(res.pvalue)
