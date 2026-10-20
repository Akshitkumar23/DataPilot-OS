"""
DataPilot OS — Authoritative Source Execution Planner.
Deterministically decides execution strategy (direct_file_scan vs in_memory_dataframe)
based on source storage, mutation status, disk drift, and engine capabilities.
Never claims zero-copy or pushdown unless verified.
"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd


class ExecutionPlan(BaseModel):
    mode: str = "in_memory_dataframe"  # "direct_file_scan" | "in_memory_dataframe"
    pushdown_supported: bool = False
    reason: str
    source_format: str = "CSV"
    file_path: Optional[str] = None
    sheet_name: Optional[str] = None
    is_mutated: bool = False
    drift_detected: bool = False
    drift_reason: Optional[str] = None


class SourceExecutionPlanner:
    """Authoritative planner deciding between direct source file scan and in-memory execution."""

    @staticmethod
    def plan_sql(
        dataset_entry: Optional[Any],
        df: Optional[pd.DataFrame] = None,
        sql: Optional[str] = None,
        source_file_path: Optional[str] = None,
        fingerprint_before: Optional[str] = None
    ) -> ExecutionPlan:
        """Determines whether a SQL query can be executed via direct DuckDB file scan or requires active DataFrame in RAM."""
        # 1. Resolve source file path
        target_path = source_file_path
        if not target_path and dataset_entry and getattr(dataset_entry, "file_path", None):
            target_path = dataset_entry.file_path

        if not target_path:
            return ExecutionPlan(
                mode="in_memory_dataframe",
                pushdown_supported=False,
                reason="No source file path specified; executing against in-memory DataFrame",
                source_format="RAM"
            )

        target_path = os.path.abspath(target_path)
        ext = os.path.splitext(target_path)[1].lower()
        fmt = ext.lstrip(".").upper() or "UNKNOWN"

        # 2. Check if file exists on disk
        if not os.path.isfile(target_path):
            return ExecutionPlan(
                mode="in_memory_dataframe",
                pushdown_supported=False,
                reason=f"Source file not found on disk at {target_path}; executing against active in-memory DataFrame",
                source_format=fmt,
                file_path=target_path,
                drift_detected=True,
                drift_reason="Source file missing on disk"
            )

        # 3. If dataset entry exists, check mutation and disk drift
        if dataset_entry:
            # Rejection 1: Active state was mutated in-memory (cleaned, transformed, imputed)
            if getattr(dataset_entry, "is_mutated", False):
                return ExecutionPlan(
                    mode="in_memory_dataframe",
                    pushdown_supported=False,
                    reason="Dataset has been modified in memory (cleaning/transformation applied); direct file scan bypassed to keep active RAM state authoritative",
                    source_format=fmt,
                    file_path=target_path,
                    sheet_name=getattr(dataset_entry, "sheet_name", None),
                    is_mutated=True
                )

            # Rejection 2: In-memory fingerprint mismatch with initial load fingerprint
            initial_hash = getattr(dataset_entry, "initial_df_hash", None)
            if initial_hash and fingerprint_before and fingerprint_before != initial_hash:
                return ExecutionPlan(
                    mode="in_memory_dataframe",
                    pushdown_supported=False,
                    reason="In-memory DataFrame fingerprint differs from initial load hash; bypassing direct scan",
                    source_format=fmt,
                    file_path=target_path,
                    is_mutated=True
                )

            # Rejection 3: Active DataFrame row/col shape mismatch
            if df is not None:
                if getattr(dataset_entry, "row_count", None) != len(df) or getattr(dataset_entry, "column_count", None) != len(df.columns):
                    return ExecutionPlan(
                        mode="in_memory_dataframe",
                        pushdown_supported=False,
                        reason="Active DataFrame dimensions mismatch initial catalog metadata; bypassing direct scan",
                        source_format=fmt,
                        file_path=target_path,
                        is_mutated=True
                    )

            # Rejection 4: External file drift on disk
            if hasattr(dataset_entry, "check_drift") and dataset_entry.check_drift():
                return ExecutionPlan(
                    mode="in_memory_dataframe",
                    pushdown_supported=False,
                    reason=f"Source file drifted on disk ({dataset_entry.drift_reason}); bypassing direct scan to maintain session isolation",
                    source_format=fmt,
                    file_path=target_path,
                    drift_detected=True,
                    drift_reason=dataset_entry.drift_reason
                )

        # 4. Check format compatibility for native DuckDB file scans
        # DuckDB supports read_csv_auto and read_parquet natively in-process.
        # Excel (.xlsx, .xls) requires python openpyxl/xlrd and executes via in-memory DataFrame.
        if ext in [".csv", ".tsv"]:
            return ExecutionPlan(
                mode="direct_file_scan",
                pushdown_supported=True,
                reason="Direct DuckDB CSV auto scan with filter/projection pushdown",
                source_format="CSV",
                file_path=target_path,
                is_mutated=False
            )
        elif ext in [".parquet", ".pq"]:
            return ExecutionPlan(
                mode="direct_file_scan",
                pushdown_supported=True,
                reason="Direct DuckDB Parquet scan with columnar projection and predicate pushdown",
                source_format="PARQUET",
                file_path=target_path,
                is_mutated=False
            )
        elif ext in [".xlsx", ".xls"]:
            return ExecutionPlan(
                mode="in_memory_dataframe",
                pushdown_supported=False,
                reason="Excel workbooks execute against loaded in-memory DataFrame (native DuckDB excel extension not bundled in-process)",
                source_format="EXCEL",
                file_path=target_path,
                sheet_name=getattr(dataset_entry, "sheet_name", None) if dataset_entry else None,
                is_mutated=False
            )
        else:
            return ExecutionPlan(
                mode="in_memory_dataframe",
                pushdown_supported=False,
                reason=f"Format '{fmt}' executes against registered in-memory DataFrame",
                source_format=fmt,
                file_path=target_path,
                is_mutated=False
            )

    @staticmethod
    def plan_operation(
        operation_name: str,
        dataset_entry: Optional[Any] = None
    ) -> ExecutionPlan:
        """Determines execution mode for analytical operations (profiling, cleaning, ML, stats, EDA)."""
        op = operation_name.lower()
        fmt = getattr(dataset_entry, "format", "CSV") if dataset_entry else "CSV"
        fpath = getattr(dataset_entry, "file_path", None) if dataset_entry else None
        sheet = getattr(dataset_entry, "sheet_name", None) if dataset_entry else None
        mutated = getattr(dataset_entry, "is_mutated", False) if dataset_entry else False

        # Machine learning, statistical modeling, data cleaning, and EDA require NumPy/Pandas/Scipy
        if op in ["ml", "supervised_ml", "unsupervised_ml", "clustering", "pca", "cleaning", "transform", "eda", "timeseries", "hypothesis_test", "stats"]:
            return ExecutionPlan(
                mode="in_memory_dataframe",
                pushdown_supported=False,
                reason=f"Operation '{operation_name}' executes against in-memory DataFrame via scientific libraries (NumPy/Pandas/Scikit-learn/SciPy)",
                source_format=fmt,
                file_path=fpath,
                sheet_name=sheet,
                is_mutated=mutated
            )

        # Dataset profiling evaluates active DataFrame in RAM
        if op in ["profile", "profiling"]:
            return ExecutionPlan(
                mode="in_memory_dataframe",
                pushdown_supported=False,
                reason="Dataset profiling evaluates live columns and null/uniqueness distributions in RAM",
                source_format=fmt,
                file_path=fpath,
                sheet_name=sheet,
                is_mutated=mutated
            )

        return ExecutionPlan(
            mode="in_memory_dataframe",
            pushdown_supported=False,
            reason=f"Operation '{operation_name}' evaluated on in-memory DataFrame",
            source_format=fmt,
            file_path=fpath,
            sheet_name=sheet,
            is_mutated=mutated
        )


source_execution_planner = SourceExecutionPlanner()
