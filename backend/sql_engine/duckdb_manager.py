"""
DataPilot OS — Phase 7 Governed DuckDB Execution Engine.
Executes read-only SQL queries directly against in-memory Pandas DataFrames with zero copy overhead.
Guarantees:
- Zero DataFrame mutation (fingerprint verified before and after).
- Strict JSON serialization safety (NaN, inf, timestamps, NumPy types handled cleanly).
- Bounded pagination and performance metrics.
- Comprehensive provenance and query history logging.
"""
import math
import os
import time
import uuid
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import duckdb

from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.sql_engine.sql_models import SQLQueryResult
from backend.sql_engine.sql_validator import SQLValidator
from backend.sql_engine.query_history import query_history_manager, SQLQueryHistoryManager
from backend.persistence.provenance_store import provenance_store
from backend.sql_engine.relation_catalog import relation_catalog


def sanitize_value_for_json(val: Any) -> Any:
    """Recursively convert NumPy, Pandas, NaN, and datetime types into JSON-safe Python primitives."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.isoformat()
    if isinstance(val, (list, tuple)):
        return [sanitize_value_for_json(x) for x in val]
    if isinstance(val, dict):
        return {str(k): sanitize_value_for_json(v) for k, v in val.items()}
    return str(val) if not isinstance(val, (str, int, float, bool)) else val


class GovernedDuckDBManager:
    """Authoritative in-process DuckDB manager for DataPilot OS datasets."""

    @classmethod
    def execute_query(
        cls,
        sql: Any,
        df: Any = None,
        dataset_id: str = "active_dataset",
        dataset_name: str = "active_dataset",
        page: int = 1,
        page_size: int = 50,
        record_provenance: bool = True,
        dataset_fingerprint: Optional[str] = None,
        source_file_path: Optional[str] = None
    ) -> SQLQueryResult:
        """
        Executes a governed SQL query on the active DataFrame.
        """
        if isinstance(sql, pd.DataFrame) and (isinstance(df, str) or df is None):
            actual_df = sql
            actual_sql = df
        elif isinstance(df, pd.DataFrame) and isinstance(sql, str):
            actual_df = df
            actual_sql = sql
        else:
            actual_df = df if isinstance(df, pd.DataFrame) else sql
            actual_sql = sql if isinstance(sql, str) else df

        df = actual_df
        sql = actual_sql

        query_id = f"sql_{uuid.uuid4().hex[:12]}"
        fingerprint_before = DatasetProfiler.compute_hash(df)
        start_time = time.perf_counter()

        target_history_path = getattr(SQLQueryHistoryManager, "HISTORY_FILE", None)
        if target_history_path and str(target_history_path) != str(query_history_manager.data_path):
            query_history_manager.data_path = str(target_history_path)
            query_history_manager._load_from_disk()

        # 1. Validate SQL and schema
        validation = SQLValidator.validate(sql, df, dataset_name)
        if validation.status != "VALID":
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0
            # Record failure in query history
            query_history_manager.add_entry(
                query_id=query_id,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint_before,
                sql=sql,
                execution_time_ms=exec_time_ms,
                status="FAILED",
                row_count=0,
                error=validation.message
            )
            # Record failure in provenance if requested
            if record_provenance:
                provenance_store.record_event(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    operation="sql_query",
                    capability="sql",
                    affected_columns=[],
                    parameters={"sql": sql},
                    df_before=df,
                    df_after=None,
                    status="FAILED",
                    error={"code": validation.error_code, "message": validation.message},
                    code=sql,
                    execution_time_ms=exec_time_ms,
                    result_summary={"error": validation.message}
                )

            return SQLQueryResult(
                query_id=query_id,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint_before,
                sql=sql,
                status="FAILED",
                error_code=validation.error_code,
                error_message=validation.message,
                execution_time_ms=exec_time_ms
            )

        # 2. Execute query against DuckDB
        execution_mode = "in_memory_dataframe"
        pushdown_supported = False
        con = None
        try:
            con = duckdb.connect(database=":memory:")
            # Register all catalog tables in the DuckDB session
            catalog_dfs = relation_catalog.get_all_dataframes()
            for tbl_name, tbl_df in catalog_dfs.items():
                con.register(tbl_name, tbl_df)

            # Auto-resolve source_file_path from registry only if None (empty string disables direct scan)
            from backend.core.dataset_registry import dataset_registry
            from backend.core.source_execution_planner import source_execution_planner
            d_entry = dataset_registry.get(dataset_id) if dataset_id else None

            if source_file_path is None and d_entry and getattr(d_entry, "file_path", None):
                source_file_path = d_entry.file_path

            plan = source_execution_planner.plan_sql(
                dataset_entry=d_entry,
                df=df,
                sql=sql,
                source_file_path=source_file_path,
                fingerprint_before=fingerprint_before
            )

            # Determine whether direct file scan is eligible
            direct_scan_succeeded = False
            if plan.mode == "direct_file_scan" and plan.file_path and os.path.isfile(plan.file_path):
                ext = os.path.splitext(plan.file_path)[1].lower()
                try:
                    safe_path = os.path.abspath(plan.file_path).replace("\\", "/").replace("'", "''")
                    scan_func = "read_parquet" if ext in [".parquet", ".pq"] else "read_csv_auto"
                    scan_sql = f"SELECT * FROM {scan_func}('{safe_path}')"

                    for alias in ["active_dataset", "dataset", "df"]:
                        con.execute(f"CREATE OR REPLACE VIEW {alias} AS {scan_sql}")
                    if dataset_name:
                        import re
                        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_name.split(".")[0]).strip("_")
                        if safe_name and not safe_name.isdigit() and safe_name not in catalog_dfs:
                            con.execute(f"CREATE OR REPLACE VIEW {safe_name} AS {scan_sql}")

                    rel = con.query(sql)
                    col_names = [str(c) for c in rel.columns]
                    col_types = {str(col): str(t) for col, t in zip(rel.columns, rel.types)}
                    res_df = rel.df()
                    total_rows = len(res_df)
                    execution_mode = "direct_file_scan"
                    pushdown_supported = True
                    direct_scan_succeeded = True
                except Exception:
                    direct_scan_succeeded = False

            if not direct_scan_succeeded:
                # Register in-memory relation views for active dataset compatibility
                con.register("active_dataset", df)
                con.register("dataset", df)
                con.register("df", df)
                if dataset_name:
                    import re
                    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_name.split(".")[0]).strip("_")
                    if safe_name and not safe_name.isdigit() and safe_name not in catalog_dfs:
                        con.register(safe_name, df)

                # Query relation
                rel = con.query(sql)
                col_names = [str(c) for c in rel.columns]
                col_types = {str(col): str(t) for col, t in zip(rel.columns, rel.types)}

                # Convert result to DataFrame
                res_df = rel.df()
                total_rows = len(res_df)
                execution_mode = "in_memory_dataframe"
                pushdown_supported = False

            # 3. Verify zero mutation invariant on original DataFrame
            fingerprint_after = DatasetProfiler.compute_hash(df)
            if fingerprint_before != fingerprint_after:
                raise RuntimeError(
                    f"Dataset mutation detected! Fingerprint changed from {fingerprint_before} to {fingerprint_after}."
                )

            exec_time_ms = (time.perf_counter() - start_time) * 1000.0

            # 4. Handle pagination
            page_clamped = max(1, page)
            page_size_clamped = max(1, min(page_size, 500))
            total_pages = max(1, (total_rows + page_size_clamped - 1) // page_size_clamped)

            start_idx = (page_clamped - 1) * page_size_clamped
            end_idx = start_idx + page_size_clamped
            page_df = res_df.iloc[start_idx:end_idx]

            # Convert rows to JSON-safe dictionary records
            clean_rows: List[Dict[str, Any]] = []
            for record in page_df.to_dict(orient="records"):
                clean_record = {k: sanitize_value_for_json(v) for k, v in record.items()}
                clean_rows.append(clean_record)

            # 5. Record successful execution in query history
            query_history_manager.add_entry(
                query_id=query_id,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint_before,
                sql=sql,
                execution_time_ms=exec_time_ms,
                status="SUCCEEDED",
                row_count=total_rows,
                error=None
            )

            # 6. Record successful execution in provenance
            if record_provenance:
                provenance_store.record_event(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    operation="sql_query",
                    capability="sql",
                    affected_columns=validation.referenced_columns,
                    parameters={"sql": sql, "operation": validation.estimated_operation},
                    df_before=df,
                    df_after=df,  # unchanged
                    status="SUCCESS",
                    code=sql,
                    finding=f"Executed DuckDB SQL: returned {total_rows} row(s) across {len(col_names)} column(s).",
                    execution_time_ms=exec_time_ms,
                    result_summary={
                        "columns": col_names,
                        "row_count": total_rows,
                        "estimated_operation": validation.estimated_operation,
                        "execution_time_ms": exec_time_ms,
                        "execution_mode": execution_mode,
                        "pushdown_supported": pushdown_supported
                    }
                )

            explain_text = None
            try:
                exp_res = con.execute(f"EXPLAIN {sql}").fetchall()
                if exp_res:
                    explain_text = "\n".join(str(r[1]) if len(r) > 1 else str(r[0]) for r in exp_res)
            except Exception:
                pass

            return SQLQueryResult(
                query_id=query_id,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint_before,
                sql=sql,
                status="SUCCEEDED",
                columns=col_names,
                column_types=col_types,
                row_count=total_rows,
                rows=clean_rows,
                page=page_clamped,
                page_size=page_size_clamped,
                total_pages=total_pages,
                execution_time_ms=round(exec_time_ms, 2),
                explain_plan=explain_text,
                execution_mode=execution_mode,
                pushdown_supported=pushdown_supported
            )

        except Exception as exc:
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = str(exc)
            query_history_manager.add_entry(
                query_id=query_id,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint_before,
                sql=sql,
                execution_time_ms=exec_time_ms,
                status="FAILED",
                row_count=0,
                error=err_msg
            )
            if record_provenance:
                provenance_store.record_event(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    operation="sql_query",
                    capability="sql",
                    affected_columns=[],
                    parameters={"sql": sql},
                    df_before=df,
                    df_after=None,
                    status="FAILED",
                    error={"code": "SQL_EXECUTION_ERROR", "message": err_msg},
                    code=sql,
                    execution_time_ms=exec_time_ms,
                    result_summary={"error": err_msg}
                )

            return SQLQueryResult(
                query_id=query_id,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint_before,
                sql=sql,
                status="FAILED",
                error_code="SQL_EXECUTION_ERROR",
                error_message=err_msg,
                execution_time_ms=round(exec_time_ms, 2)
            )
        finally:
            if con:
                try:
                    con.close()
                except Exception:
                    pass
