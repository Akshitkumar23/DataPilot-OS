"""
DataPilot OS — Governed Provenance Store (Phase 2B).
Dataset-agnostic, domain-agnostic immutable audit trail for all data mutations.
Calculates all state metrics, column sets, null counts, duplicate counts,
and cryptographic fingerprints dynamically from actual DataFrame execution context.
Zero hardcoding: no domain nouns, column names, or test dataset values.
"""
import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
from backend.ds_engine.dataset_profiler import DatasetProfiler


class DatasetStateSnapshot(BaseModel):
    """Immutable snapshot of dataset state metrics dynamically derived from DataFrame."""
    row_count: int
    column_count: int
    columns: List[str]
    null_count: int
    duplicate_count: int
    fingerprint: str


def canonicalize_capability(cap: str) -> str:
    """Map driver/operator identifiers to canonical capability names."""
    c = (cap or "").strip().upper()
    if "CLEAN" in c:
        return "DATA_CLEAN"
    if "TRANSFORM" in c or "FEATURE" in c:
        return "FEATURE_ENGINEERING"
    if "EDA" in c or "STAT" in c or "CORRELAT" in c or "DISTRIBUT" in c:
        return "EDA"
    if "UNSUPERVISED" in c or "CLUSTER" in c or "PCA" in c:
        return "UNSUPERVISED_LEARNING"
    return c or "DATA_CLEAN"


def canonicalize_operation(op: str) -> str:
    """Map operator method IDs to canonical uppercase operation names."""
    o = (op or "").strip().lower()
    mapping = {
        "drop_nulls": "DROP_NULLS",
        "drop_missing": "DROP_NULLS",
        "drop_duplicates": "DROP_DUPLICATES",
        "remove_duplicates": "DROP_DUPLICATES",
        "fill_mean": "IMPUTE_MEAN",
        "impute_mean": "IMPUTE_MEAN",
        "fill_median": "IMPUTE_MEDIAN",
        "impute_median": "IMPUTE_MEDIAN",
        "fill_mode": "IMPUTE_MODE",
        "impute_mode": "IMPUTE_MODE",
        "fill_constant": "IMPUTE_CONSTANT",
        "impute_constant": "IMPUTE_CONSTANT",
        "clip_iqr": "CLIP_IQR",
        "normalize_minmax": "MIN_MAX_SCALE",
        "minmax_scale": "MIN_MAX_SCALE",
        "standardize": "STANDARDIZE",
        "z_score": "STANDARDIZE",
        "one_hot_encode": "ONE_HOT_ENCODE",
        "yeo_johnson": "YEO_JOHNSON",
        "power_transform": "YEO_JOHNSON",
        "log_transform": "LOG_TRANSFORM",
        "lag": "CREATE_LAG",
        "create_lag": "CREATE_LAG",
        "rolling": "CREATE_ROLLING",
        "create_rolling": "CREATE_ROLLING",
        "difference": "CREATE_DIFFERENCE",
        "create_difference": "CREATE_DIFFERENCE",
        "calendar_features": "CREATE_CALENDAR_FEATURES",
        "create_calendar_features": "CREATE_CALENDAR_FEATURES",
        "descriptive": "DESCRIPTIVE_STATS",
        "descriptive_stats": "DESCRIPTIVE_STATS",
        "summary_stats": "DESCRIPTIVE_STATS",
        "missingness": "MISSINGNESS_ANALYSIS",
        "missingness_analysis": "MISSINGNESS_ANALYSIS",
        "distribution": "DISTRIBUTION_ANALYSIS",
        "distribution_analysis": "DISTRIBUTION_ANALYSIS",
        "correlation": "CORRELATION_ANALYSIS",
        "correlation_matrix": "CORRELATION_ANALYSIS",
        "skewness": "SKEWNESS_ANALYSIS",
        "skewness_check": "SKEWNESS_ANALYSIS",
        "train_model": "ML_TRAIN",
        "ml_train": "ML_TRAIN",
        "model_training": "ML_TRAIN",
        "automl": "AUTOML_TOURNAMENT",
        "automl_tournament": "AUTOML_TOURNAMENT",
        "model_predict": "MODEL_PREDICT",
        "predict": "MODEL_PREDICT",
        "kmeans": "KMEANS_CLUSTERING",
        "kmeans_clustering": "KMEANS_CLUSTERING",
        "pca": "PCA_DIMENSIONALITY_REDUCTION",
        "pca_decomposition": "PCA_DIMENSIONALITY_REDUCTION",
    }
    return mapping.get(o, o.upper())


class ProvenanceEvent(BaseModel):
    """Immutable record of a governed data operation."""
    event_id: str
    timestamp: str
    status: str  # SUCCESS | FAILED | REJECTED | NEEDS_CLARIFICATION
    capability: str  # DATA_CLEAN | FEATURE_ENGINEERING
    operation: str  # DROP_NULLS | MIN_MAX_SCALE etc.
    dataset_id: str
    dataset_name: str
    input_dataset_version: str
    output_dataset_version: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    affected_columns: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    source: Dict[str, Any] = Field(default_factory=dict)
    result_summary: Dict[str, Any] = Field(default_factory=dict)
    before_state: Optional[DatasetStateSnapshot] = None
    after_state: Optional[DatasetStateSnapshot] = None
    error: Optional[Dict[str, Any]] = None
    finding: str = ""
    code: str = ""
    execution_time_ms: float = 0.0
    provenance_version: str = "1.0"
    # Multi-source relational lineage (FW-04)
    parent_dataset_ids: List[str] = Field(default_factory=list)
    source_fingerprints: Dict[str, str] = Field(default_factory=dict)
    join_plan: Dict[str, Any] = Field(default_factory=dict)


def capture_dataset_state_snapshot(df: pd.DataFrame) -> DatasetStateSnapshot:
    """
    Dynamically compute dataset state metrics and cryptographic fingerprint from DataFrame.
    Strictly generic: reads DataFrame properties dynamically. Never assumes column names or counts.
    """
    row_count = int(len(df))
    col_count = int(len(df.columns))
    cols = [str(c) for c in df.columns]
    null_count = int(df.isna().sum().sum())
    try:
        duplicate_count = int(df.duplicated().sum())
    except Exception:
        try:
            duplicate_count = int(df.astype(str).duplicated().sum())
        except Exception:
            duplicate_count = 0
    fingerprint = DatasetProfiler.compute_hash(df)
    return DatasetStateSnapshot(
        row_count=row_count,
        column_count=col_count,
        columns=cols,
        null_count=null_count,
        duplicate_count=duplicate_count,
        fingerprint=fingerprint,
    )


class ProvenanceStore:
    def __init__(self, data_path: str = "backend/data/provenance.json"):
        runtime_data_dir = os.getenv("DATAPILOT_DATA_DIR")
        self.data_path = (
            os.path.join(runtime_data_dir, "provenance.json")
            if runtime_data_dir else os.path.join(os.getcwd(), data_path)
        )
        self._events: List[ProvenanceEvent] = []
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._events = [ProvenanceEvent(**item) for item in data]
            except Exception:
                pass

    def _persist_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in self._events], f, indent=2)
        except Exception:
            pass

    def record_event(
        self,
        dataset_id: str,
        dataset_name: str,
        operation: str,
        capability: str,
        affected_columns: List[str],
        parameters: Dict[str, Any],
        df_before: Optional[pd.DataFrame],
        df_after: Optional[pd.DataFrame] = None,
        status: str = "SUCCESS",
        source: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        finding: str = "",
        code: str = "",
        execution_time_ms: float = 0.0,
        result_summary: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEvent:
        norm_status = status.upper().strip()
        norm_cap = canonicalize_capability(capability)
        norm_op = canonicalize_operation(operation)
        cols = [str(c) for c in (affected_columns or [])]
        params = {k: v for k, v in (parameters or {}).items() if v is not None}
        src = source or {"type": "STRUCTURED_UI"}

        snap_before = capture_dataset_state_snapshot(df_before) if df_before is not None else None
        snap_after = capture_dataset_state_snapshot(df_after) if (df_after is not None and norm_status == "SUCCESS") else None

        input_version = snap_before.fingerprint if snap_before else ""
        output_version = snap_after.fingerprint if snap_after else None

        if result_summary is None:
            if snap_before and snap_after:
                res_sum = {
                    "rows_before": snap_before.row_count,
                    "rows_after": snap_after.row_count,
                    "nulls_before": snap_before.null_count,
                    "nulls_after": snap_after.null_count,
                    "duplicates_before": snap_before.duplicate_count,
                    "duplicates_after": snap_after.duplicate_count,
                }
            elif snap_before:
                res_sum = {
                    "rows_before": snap_before.row_count,
                    "rows_after": snap_before.row_count,
                    "nulls_before": snap_before.null_count,
                    "nulls_after": snap_before.null_count,
                    "duplicates_before": snap_before.duplicate_count,
                    "duplicates_after": snap_before.duplicate_count,
                }
            else:
                res_sum = {}
        else:
            res_sum = dict(result_summary)

        event = ProvenanceEvent(
            event_id=f"prov_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=norm_status,
            capability=norm_cap,
            operation=norm_op,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            input_dataset_version=input_version,
            output_dataset_version=output_version,
            columns=cols,
            affected_columns=cols,
            parameters=params,
            source=src,
            result_summary=res_sum,
            before_state=snap_before,
            after_state=snap_after,
            error=error,
            finding=finding,
            code=code,
            execution_time_ms=execution_time_ms,
        )
        self._events.append(event)
        self._persist_to_disk()
        return event

    def record_multi_source_event(
        self,
        operation: str,
        primary_dataset_id: str,
        primary_dataset_name: str,
        joined_dataset_ids: List[str],
        source_fingerprints: Dict[str, str],
        output_dataset_id: str,
        output_fingerprint: str,
        join_plan: Dict[str, Any],
        metrics: Dict[str, Any],
        sql: str,
        status: str = "SUCCESS",
        execution_time_ms: float = 0.0,
    ) -> ProvenanceEvent:
        """Records a multi-source relational join event with complete parent cryptographic lineage."""
        all_parents = [primary_dataset_id] + list(joined_dataset_ids)
        event = ProvenanceEvent(
            event_id=f"prov_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            capability="RELATIONAL_JOIN",
            operation=operation,
            dataset_id=output_dataset_id,
            dataset_name=primary_dataset_name,
            input_dataset_version=source_fingerprints.get(primary_dataset_id, "unknown"),
            output_dataset_version=output_fingerprint,
            columns=[],
            affected_columns=[],
            parameters=join_plan,
            source={"type": "multi_table_join", "parents": all_parents},
            result_summary=metrics,
            code=sql,
            execution_time_ms=execution_time_ms,
            parent_dataset_ids=all_parents,
            source_fingerprints=source_fingerprints,
            join_plan=join_plan
        )
        self._events.append(event)
        self._persist_to_disk()
        return event

    def get_events(self, dataset_id: Optional[str] = None) -> List[ProvenanceEvent]:
        if dataset_id:
            return [e for e in self._events if e.dataset_id == dataset_id or dataset_id in e.parent_dataset_ids]
        return list(self._events)

    def get_dataset_lineage(self, dataset_id: str) -> List[ProvenanceEvent]:
        return self.get_events(dataset_id=dataset_id)

    def clear(self):
        self._events.clear()
        if os.path.exists(self.data_path):
            try:
                os.remove(self.data_path)
            except Exception:
                pass


provenance_store = ProvenanceStore()
