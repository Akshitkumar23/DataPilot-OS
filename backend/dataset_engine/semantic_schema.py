"""
DataPilot OS — Semantic Schema Analyzer.
Infers semantic types from column names, data patterns, and value distributions.
Semantic type inference is critical for dataset-aware code generation.
"""

import pandas as pd
from enum import Enum
from typing import Dict, List

class SemanticType(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    DATETIME = "DATETIME"
    CATEGORICAL = "CATEGORICAL"
    NUMERIC_MEASURE = "NUMERIC_MEASURE"
    MONETARY = "MONETARY"
    PERCENTAGE = "PERCENTAGE"
    BOOLEAN = "BOOLEAN"
    TEXT = "TEXT"
    GEOSPATIAL = "GEOSPATIAL"
    UNKNOWN = "UNKNOWN"

class SemanticTypeInference:
    def infer(self, series: pd.Series, col_name: str) -> SemanticType:
        col_name_lower = str(col_name).lower()
        
        # 1. Check for Identifiers
        if 'id' in col_name_lower or col_name_lower.endswith('_id') or col_name_lower == 'uuid':
            if series.nunique() == len(series) or series.nunique() > len(series) * 0.9:
                return SemanticType.IDENTIFIER

        # 2. Check for Boolean
        if pd.api.types.is_bool_dtype(series) or set(series.dropna().unique()) <= {0, 1} or set(series.dropna().astype(str).str.lower().unique()) <= {'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n'}:
            return SemanticType.BOOLEAN

        # 3. Check for Datetime
        if pd.api.types.is_datetime64_any_dtype(series) or 'date' in col_name_lower or 'time' in col_name_lower:
            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    return SemanticType.DATETIME
                if series.dtype == 'object' and series.str.match(r'^\d{4}-\d{2}-\d{2}').any():
                    return SemanticType.DATETIME
            except Exception:
                pass

        # 4. Geospatial
        if col_name_lower in ['lat', 'latitude', 'lon', 'lng', 'longitude']:
            return SemanticType.GEOSPATIAL

        # 5. Monetary
        if any(kw in col_name_lower for kw in ['price', 'revenue', 'cost', 'amount', 'salary', 'tax', 'fee']):
            if pd.api.types.is_numeric_dtype(series):
                return SemanticType.MONETARY

        # 6. Percentage
        if 'pct' in col_name_lower or 'percent' in col_name_lower or 'rate' in col_name_lower:
            if pd.api.types.is_numeric_dtype(series):
                return SemanticType.PERCENTAGE

        # 7. Numeric Measure
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 20 and len(series) > 100:
                return SemanticType.CATEGORICAL
            return SemanticType.NUMERIC_MEASURE

        # 8. Categorical
        if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
            if series.nunique() < 50 or series.nunique() / len(series) < 0.1:
                return SemanticType.CATEGORICAL
            return SemanticType.TEXT

        return SemanticType.UNKNOWN

    def infer_all(self, df: pd.DataFrame) -> Dict[str, SemanticType]:
        results = {}
        for col in df.columns:
            results[str(col)] = self.infer(df[col], str(col))
        return results

    def get_measure_columns(self, df: pd.DataFrame) -> List[str]:
        inferred = self.infer_all(df)
        return [col for col, sem_type in inferred.items() if sem_type in (SemanticType.NUMERIC_MEASURE, SemanticType.MONETARY, SemanticType.PERCENTAGE)]

    def get_dimension_columns(self, df: pd.DataFrame) -> List[str]:
        inferred = self.infer_all(df)
        return [col for col, sem_type in inferred.items() if sem_type in (SemanticType.CATEGORICAL, SemanticType.TEXT, SemanticType.BOOLEAN, SemanticType.GEOSPATIAL)]

    def get_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        inferred = self.infer_all(df)
        return [col for col, sem_type in inferred.items() if sem_type == SemanticType.DATETIME]

    def get_identifier_columns(self, df: pd.DataFrame) -> List[str]:
        inferred = self.infer_all(df)
        return [col for col, sem_type in inferred.items() if sem_type == SemanticType.IDENTIFIER]
