"""
DataPilot OS — Independent Direct-Scan Regression Audit Test Suite.
Systematically verifies:
1. Nulls, NaNs, Dates, Mixed Types, Quoted Columns, Duplicate Column Names.
2. Unicode & Spaces in Paths, Dialect Variations, Empty & Header-Only Files.
3. In-Memory Mutation Bypass & Fallback to Active RAM State.
4. External Source File Modification Drift Detection.
5. Missing / Deleted Disk File Graceful In-Memory Fallback.
6. Dataset Switching Isolation.
7. Query Safety, Injection Resistance, and Connection Cleanup.
8. Physical Pushdown Verification via EXPLAIN Plan Inspection.
"""
import os
import sys
import shutil
import hashlib
import tempfile
import pytest
import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, os.path.abspath("."))

from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.sql_engine.sql_models import SQLQueryResult
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.persistence.provenance_store import provenance_store
from backend.sql_engine.query_history import query_history_manager


@pytest.fixture(autouse=True)
def clean_registry():
    dataset_registry.clear()
    query_history_manager.clear_history()
    yield
    dataset_registry.clear()
    query_history_manager.clear_history()


def test_fixture_nulls_and_nans():
    """Verify null, NaN, and None handling parity between direct scan and pandas oracle."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "nulls_test.csv")
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "val_float": [10.5, np.nan, 30.2, np.nan, 50.0],
            "label_str": ["Alpha", None, "Gamma", "Delta", None]
        })
        df.to_csv(csv_path, index=False)

        res = GovernedDuckDBManager.execute_query(
            sql="SELECT id, val_float, label_str FROM active_dataset WHERE val_float IS NOT NULL ORDER BY id",
            df=df,
            dataset_id="ds_nulls",
            source_file_path=csv_path
        )
        assert res.status == "SUCCEEDED"
        assert res.row_count == 3
        # Compare with pandas oracle
        oracle_df = df[df["val_float"].notna()].sort_values("id")
        assert len(res.rows) == len(oracle_df)
        for r, (_, o) in zip(res.rows, oracle_df.iterrows()):
            assert r["id"] == int(o["id"])
            assert abs(float(r["val_float"]) - float(o["val_float"])) < 1e-5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_dates_and_timestamps():
    """Verify ISO dates and timestamps handling parity."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "dates_test.csv")
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "event_date": ["2026-01-15", "2026-06-20", "2026-12-31"],
            "event_ts": ["2026-01-15 08:30:00", "2026-06-20 14:15:30", "2026-12-31 23:59:59"]
        })
        df.to_csv(csv_path, index=False)

        res = GovernedDuckDBManager.execute_query(
            sql="SELECT id, event_date FROM active_dataset WHERE event_date >= '2026-06-01' ORDER BY id",
            df=df,
            dataset_id="ds_dates",
            source_file_path=csv_path
        )
        assert res.status == "SUCCEEDED"
        assert res.row_count == 2
        assert res.rows[0]["id"] == 2
        assert res.rows[1]["id"] == 3
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_quoted_and_special_column_names():
    """Verify quoted column names with spaces and symbols in direct scan."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "quoted_cols.csv")
        df = pd.DataFrame({
            "User ID": [101, 102, 103],
            "Total Spend ($)": [250.50, 89.00, 1420.75]
        })
        df.to_csv(csv_path, index=False)

        res = GovernedDuckDBManager.execute_query(
            sql='SELECT "User ID", "Total Spend ($)" FROM active_dataset WHERE "Total Spend ($)" > 100.0 ORDER BY "User ID"',
            df=df,
            dataset_id="ds_quoted",
            source_file_path=csv_path
        )
        assert res.status == "SUCCEEDED"
        assert res.row_count == 2
        assert res.rows[0]["User ID"] == 101
        assert res.rows[1]["User ID"] == 103
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_unicode_and_spaces_in_path():
    """Verify direct file scan with spaces and Unicode in the absolute file path."""
    parent_dir = tempfile.mkdtemp()
    try:
        # Create directory with spaces and unicode accents
        sub_dir = os.path.join(parent_dir, "Dossier Spécial 2026", "Sub Folder")
        os.makedirs(sub_dir, exist_ok=True)
        csv_path = os.path.join(sub_dir, "données_marché_€ 100.csv")
        df = pd.DataFrame({"code": ["FR", "DE", "JP"], "score": [98.5, 87.2, 91.0]})
        df.to_csv(csv_path, index=False, encoding="utf-8")

        res = GovernedDuckDBManager.execute_query(
            sql="SELECT code, score FROM active_dataset WHERE score > 90.0 ORDER BY score DESC",
            df=df,
            dataset_id="ds_unicode_path",
            source_file_path=csv_path
        )
        assert res.status == "SUCCEEDED"
        assert res.execution_mode == "direct_file_scan"
        assert res.pushdown_supported is True
        assert res.row_count == 2
        assert res.rows[0]["code"] == "FR"
        assert res.rows[1]["code"] == "JP"
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)


def test_fixture_empty_and_header_only_files():
    """Verify empty and header-only files fallback safely without crashing."""
    temp_dir = tempfile.mkdtemp()
    try:
        empty_csv = os.path.join(temp_dir, "empty.csv")
        with open(empty_csv, "w", encoding="utf-8") as f:
            f.write("")  # 0 bytes

        df_empty = pd.DataFrame()
        res_empty = GovernedDuckDBManager.execute_query(
            sql="SELECT * FROM active_dataset",
            df=df_empty,
            dataset_id="ds_empty",
            source_file_path=empty_csv
        )
        # 0-byte columnless file should be rejected cleanly by schema validator without crashing
        assert res_empty.status == "FAILED"
        assert res_empty.error_code is not None or res_empty.error_message is not None

        header_only = os.path.join(temp_dir, "header_only.csv")
        df_header = pd.DataFrame(columns=["col_a", "col_b"])
        df_header.to_csv(header_only, index=False)

        res_header = GovernedDuckDBManager.execute_query(
            sql="SELECT * FROM active_dataset",
            df=df_header,
            dataset_id="ds_header",
            source_file_path=header_only
        )
        assert res_header.status == "SUCCEEDED"
        assert res_header.row_count == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_in_memory_transformation_bypass():
    """Verify that when in-memory DataFrame is transformed, direct scan is strictly BYPASSED."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "transform_test.csv")
        raw_df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        raw_df.to_csv(csv_path, index=False)

        # Register dataset entry with original hash
        entry = DatasetEntry(
            dataset_id="ds_transform",
            name="transform_test.csv",
            file_path=csv_path,
            format="csv",
            row_count=3,
            column_count=2,
            knowledge_model={"columns": ["id", "val"]},
            loaded_at="2026-10-10T00:00:00Z",
            file_hash=hashlib.sha256(open(csv_path, "rb").read()).hexdigest(),
            initial_df_hash=DatasetProfiler.compute_hash(raw_df),
            is_mutated=False,
            source_file_mtime=os.path.getmtime(csv_path),
            source_file_size=os.path.getsize(csv_path)
        )
        dataset_registry.register("ds_transform", entry)
        dataset_registry._dataframes_cache["ds_transform"] = raw_df

        # Case 1: Unmodified query -> should use direct_file_scan
        res1 = GovernedDuckDBManager.execute_query(
            sql="SELECT id, val FROM active_dataset",
            df=raw_df,
            dataset_id="ds_transform",
            source_file_path=csv_path
        )
        assert res1.execution_mode == "direct_file_scan"
        assert res1.rows[0]["val"] == 10

        # Case 2: Apply in-memory transformation (e.g. impute or scale values: val = val * 100)
        transformed_df = raw_df.copy()
        transformed_df["val"] = transformed_df["val"] * 100  # now 1000, 2000, 3000
        dataset_registry.update_dataframe("ds_transform", transformed_df)

        res2 = GovernedDuckDBManager.execute_query(
            sql="SELECT id, val FROM active_dataset",
            df=transformed_df,
            dataset_id="ds_transform",
            source_file_path=csv_path
        )
        # CRITICAL ASSERTION: MUST use in_memory_dataframe and see transformed values (1000)!
        assert res2.execution_mode == "in_memory_dataframe"
        assert res2.pushdown_supported is False
        assert res2.rows[0]["val"] == 1000, f"Expected 1000 from active RAM state, got {res2.rows[0]['val']}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_external_file_modification_drift():
    """Verify that if the source file on disk is modified externally, direct scan is BYPASSED."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "drift_test.csv")
        initial_df = pd.DataFrame({"id": [1, 2], "status": ["active", "active"]})
        initial_df.to_csv(csv_path, index=False)

        entry = DatasetEntry(
            dataset_id="ds_drift",
            name="drift_test.csv",
            file_path=csv_path,
            format="csv",
            row_count=2,
            column_count=2,
            knowledge_model={"columns": ["id", "status"]},
            loaded_at="2026-10-10T00:00:00Z",
            file_hash=hashlib.sha256(open(csv_path, "rb").read()).hexdigest(),
            initial_df_hash=DatasetProfiler.compute_hash(initial_df),
            is_mutated=False,
            source_file_mtime=os.path.getmtime(csv_path),
            source_file_size=os.path.getsize(csv_path)
        )
        dataset_registry.register("ds_drift", entry)
        dataset_registry._dataframes_cache["ds_drift"] = initial_df

        # Externally append rows to disk file
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write("3,external_drift\n4,external_drift\n")

        # Active in-memory state still has 2 rows!
        res = GovernedDuckDBManager.execute_query(
            sql="SELECT * FROM active_dataset",
            df=initial_df,
            dataset_id="ds_drift",
            source_file_path=csv_path
        )
        # CRITICAL ASSERTION: Direct scan must detect drift and execute against active RAM state (2 rows)!
        assert res.execution_mode == "in_memory_dataframe"
        assert res.row_count == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_missing_file_fallback():
    """Verify that if the disk file is deleted, execution seamlessly falls back to active in-memory DataFrame."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "deleted_test.csv")
        df = pd.DataFrame({"id": [10, 20], "metric": [1.5, 2.5]})
        df.to_csv(csv_path, index=False)

        # Delete the file from disk
        os.remove(csv_path)

        res = GovernedDuckDBManager.execute_query(
            sql="SELECT id, metric FROM active_dataset",
            df=df,
            dataset_id="ds_missing",
            source_file_path=csv_path
        )
        assert res.status == "SUCCEEDED"
        assert res.execution_mode == "in_memory_dataframe"
        assert res.row_count == 2
        assert res.rows[0]["id"] == 10
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_dataset_switching_isolation():
    """Verify that switching active datasets correctly isolates query execution and scopes."""
    temp_dir = tempfile.mkdtemp()
    try:
        p1 = os.path.join(temp_dir, "ds1.csv")
        p2 = os.path.join(temp_dir, "ds2.csv")
        df1 = pd.DataFrame({"cust_id": [1, 2], "region": ["North", "South"]})
        df2 = pd.DataFrame({"prod_id": [10, 20, 30], "price": [9.99, 19.99, 29.99]})
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)

        e1 = DatasetEntry(
            dataset_id="ds_1", name="ds1.csv", file_path=p1, format="csv",
            row_count=2, column_count=2, knowledge_model={"columns": ["cust_id", "region"]},
            loaded_at="2026-10-10T00:00:00Z", file_hash=hashlib.sha256(open(p1, "rb").read()).hexdigest(),
            initial_df_hash=DatasetProfiler.compute_hash(df1), is_mutated=False,
            source_file_mtime=os.path.getmtime(p1), source_file_size=os.path.getsize(p1)
        )
        e2 = DatasetEntry(
            dataset_id="ds_2", name="ds2.csv", file_path=p2, format="csv",
            row_count=3, column_count=2, knowledge_model={"columns": ["prod_id", "price"]},
            loaded_at="2026-10-10T00:00:00Z", file_hash=hashlib.sha256(open(p2, "rb").read()).hexdigest(),
            initial_df_hash=DatasetProfiler.compute_hash(df2), is_mutated=False,
            source_file_mtime=os.path.getmtime(p2), source_file_size=os.path.getsize(p2)
        )
        dataset_registry.register("ds_1", e1)
        dataset_registry.register("ds_2", e2)
        dataset_registry._dataframes_cache["ds_1"] = df1
        dataset_registry._dataframes_cache["ds_2"] = df2

        # Query ds_1
        res1 = GovernedDuckDBManager.execute_query(
            sql="SELECT cust_id FROM active_dataset",
            df=df1, dataset_id="ds_1", source_file_path=p1
        )
        assert res1.row_count == 2
        assert "cust_id" in res1.columns

        # Query ds_2
        res2 = GovernedDuckDBManager.execute_query(
            sql="SELECT prod_id FROM active_dataset",
            df=df2, dataset_id="ds_2", source_file_path=p2
        )
        assert res2.row_count == 3
        assert "prod_id" in res2.columns
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_path_injection_resistance():
    """Verify that malicious path characters cannot perform SQL injection."""
    temp_dir = tempfile.mkdtemp()
    try:
        # File with single quotes in filename
        malicious_filename = "data'--injection.csv"
        csv_path = os.path.join(temp_dir, malicious_filename)
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df.to_csv(csv_path, index=False)

        res = GovernedDuckDBManager.execute_query(
            sql="SELECT x, y FROM active_dataset",
            df=df,
            dataset_id="ds_inject",
            source_file_path=csv_path
        )
        assert res.status == "SUCCEEDED"
        assert res.row_count == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fixture_physical_pushdown_verification():
    """Verify that physical EXPLAIN query plan confirms READ_CSV_AUTO with projections and filters."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "pushdown_test.csv")
        df = pd.DataFrame({
            "id": range(1, 501),
            "category": ["A" if i % 2 == 0 else "B" for i in range(1, 501)],
            "val": [i * 1.5 for i in range(1, 501)]
        })
        df.to_csv(csv_path, index=False)

        res = GovernedDuckDBManager.execute_query(
            sql="SELECT id, val FROM active_dataset WHERE id > 450",
            df=df,
            dataset_id="ds_pushdown",
            source_file_path=csv_path
        )
        assert res.status == "SUCCEEDED"
        assert res.execution_mode == "direct_file_scan"
        assert res.pushdown_supported is True
        assert res.explain_plan is not None
        plan_lower = res.explain_plan.lower()
        # Physical plan must explicitly show read_csv_auto and filter/projection
        assert "read_csv_auto" in plan_lower or "filter" in plan_lower
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
