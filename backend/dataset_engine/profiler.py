"""
DataPilot OS — Dataset Profiler.
Generates a rich DatasetKnowledgeModel from any loaded DataFrame.
This is the foundation for all downstream intelligence: intent, planning, code generation.
"""

import pandas as pd
from typing import List, Optional, Any
from pydantic import BaseModel
from datetime import datetime

from backend.ds_engine.dataset_profiler import DatasetProfiler as CanonicalDatasetProfiler

class ColumnProfile(BaseModel):
    name: str
    physical_type: str
    semantic_type: str
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    sample_values: List[Any]
    is_possible_identifier: bool
    is_possible_key: bool
    is_possible_datetime: bool
    is_possible_measure: bool
    is_possible_dimension: bool

class DatasetKnowledgeModel(BaseModel):
    dataset_id: str
    name: str
    file_path: str
    format: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    identifier_columns: List[str]
    duplicate_count: int
    total_missing_ratio: float
    estimated_domain: str
    domain_confidence: float
    created_at: str
    file_hash: str

class DatasetProfiler:
    def __init__(self):
        # Compatibility facade only. All semantic evidence is produced by the
        # authoritative query-runtime profiler.
        self.semantic_inference = None

    @staticmethod
    def compute_hash(df: pd.DataFrame, dataset_name: str = "dataset") -> str:
        return CanonicalDatasetProfiler.compute_hash(df, dataset_name)

    def profile(self, df: pd.DataFrame, dataset_id: str, name: str, file_path: str, format: str, file_hash: str) -> DatasetKnowledgeModel:
        row_count = len(df)
        column_count = len(df.columns)
        canonical = CanonicalDatasetProfiler.profile(df, dataset_name=name)
        duplicate_count = canonical.duplicate_row_count
        total_missing_ratio = canonical.total_missing_percentage / 100.0

        columns = [self._adapt_column(canonical.columns[str(col)]) for col in df.columns]

        numeric_columns = [col.name for col in columns if col.is_possible_measure]
        categorical_columns = [col.name for col in columns if col.is_possible_dimension]
        datetime_columns = [col.name for col in columns if col.is_possible_datetime]
        identifier_columns = [col.name for col in columns if col.is_possible_identifier]

        return DatasetKnowledgeModel(
            dataset_id=dataset_id,
            name=name,
            file_path=file_path,
            format=format,
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            identifier_columns=identifier_columns,
            duplicate_count=duplicate_count,
            total_missing_ratio=total_missing_ratio,
            estimated_domain="unclassified",
            domain_confidence=0.0,
            created_at=datetime.utcnow().isoformat(),
            file_hash=file_hash
        )

    def _adapt_column(self, profile) -> ColumnProfile:
        semantic_map = {
            "entity_id": "IDENTIFIER",
            "temporal": "DATETIME",
            "binary_flag": "BOOLEAN",
            "continuous_numeric": "NUMERIC_MEASURE",
            "discrete_numeric": "NUMERIC_MEASURE",
            "categorical_nominal": "CATEGORICAL",
            "text": "TEXT",
        }
        semantic_type = semantic_map[profile.semantic_role]
        stats = profile.stats or {}
        is_identifier = profile.semantic_role == "entity_id"
        is_datetime = profile.semantic_role == "temporal"
        is_measure = profile.semantic_role in {"continuous_numeric", "discrete_numeric"}
        is_dimension = profile.semantic_role in {"categorical_nominal", "binary_flag", "text"}
        return ColumnProfile(
            name=profile.name,
            physical_type=profile.dtype,
            semantic_type=semantic_type,
            null_count=profile.null_count,
            null_ratio=profile.null_percentage / 100.0,
            unique_count=profile.cardinality,
            unique_ratio=profile.unique_ratio,
            min_value=stats.get("min"),
            max_value=stats.get("max"),
            mean=stats.get("mean"),
            median=stats.get("median"),
            std=stats.get("std"),
            sample_values=profile.sample_values[:3],
            is_possible_identifier=is_identifier,
            is_possible_key=is_identifier and profile.null_count == 0 and profile.is_unique,
            is_possible_datetime=is_datetime,
            is_possible_measure=is_measure,
            is_possible_dimension=is_dimension
        )

    def to_context_dict(self, model: DatasetKnowledgeModel) -> dict:
        return {
            "name": model.name,
            "row_count": model.row_count,
            "columns": [
                {
                    "name": c.name,
                    "type": c.physical_type,
                    "semantic": c.semantic_type,
                    "samples": c.sample_values
                } for c in model.columns
            ]
        }

dataset_profiler = DatasetProfiler()
