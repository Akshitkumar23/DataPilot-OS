"""
DataPilot OS — Dataset Statistics Engine.
Generates statistical summaries used for result validation and insight generation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Any, Tuple, Dict
from pydantic import BaseModel

class ColumnStats(BaseModel):
    column: str
    dtype: str
    count: int
    missing: int
    unique: int
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    top_values: List[Any] = []

class DatasetStats(BaseModel):
    row_count: int
    column_count: int
    total_missing: int
    duplicate_rows: int
    memory_usage_mb: float
    column_stats: List[ColumnStats]
    correlation_pairs: List[Dict]

class StatisticsEngine:
    def compute(self, df: pd.DataFrame) -> DatasetStats:
        row_count = len(df)
        column_count = len(df.columns)
        total_missing = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        memory_usage_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        
        col_stats = []
        for col in df.columns:
            col_stats.append(self.compute_column(df[col], str(col)))
            
        correlations = self.get_top_correlations(df)
        
        return DatasetStats(
            row_count=row_count,
            column_count=column_count,
            total_missing=total_missing,
            duplicate_rows=duplicate_rows,
            memory_usage_mb=memory_usage_mb,
            column_stats=col_stats,
            correlation_pairs=correlations
        )

    def compute_column(self, series: pd.Series, col_name: str) -> ColumnStats:
        count = len(series)
        missing = int(series.isna().sum())
        valid_s = series.dropna()
        unique = int(valid_s.nunique())
        dtype_str = str(series.dtype)

        stats = {
            "column": col_name,
            "dtype": dtype_str,
            "count": count,
            "missing": missing,
            "unique": unique,
            "top_values": valid_s.value_counts().head(5).index.tolist() if not valid_s.empty else []
        }
        
        if pd.api.types.is_numeric_dtype(series) and not valid_s.empty:
            stats["min"] = float(valid_s.min())
            stats["max"] = float(valid_s.max())
            stats["mean"] = float(valid_s.mean())
            stats["median"] = float(valid_s.median())
            stats["std"] = float(valid_s.std())
            stats["q25"] = float(valid_s.quantile(0.25))
            stats["q75"] = float(valid_s.quantile(0.75))
            stats["skewness"] = float(valid_s.skew()) if unique > 1 else None
            
        return ColumnStats(**stats)

    def get_top_correlations(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return []
            
        try:
            corr_matrix = numeric_df.corr().abs()
            np.fill_diagonal(corr_matrix.values, np.nan)
            
            pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                     .stack()
                     .sort_values(ascending=False))
                     
            top_pairs = pairs.head(top_n).reset_index()
            top_pairs.columns = ['col1', 'col2', 'correlation']
            
            return top_pairs.to_dict('records')
        except Exception:
            return []

    def validate_result_statistics(self, result_df: pd.DataFrame, expected_shape: Optional[tuple]) -> Tuple[bool, List[str]]:
        issues = []
        if expected_shape:
            if result_df.shape != expected_shape:
                issues.append(f"Expected shape {expected_shape}, got {result_df.shape}")
                
        if len(result_df) == 0:
            issues.append("Result DataFrame is empty")
            
        return len(issues) == 0, issues
