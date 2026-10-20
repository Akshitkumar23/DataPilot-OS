import os
import time
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from backend.dataset_engine.loader import dataset_loader
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.core.source_execution_planner import source_execution_planner
from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.api.datasets import select_excel_sheet, SelectSheetRequest


@pytest.fixture(autouse=True)
def clean_registry():
    dataset_registry.clear()


@pytest.fixture
def temp_workspace(tmp_path):
    """Provides disposable files with various formats, encodings, and delimiters."""
    # 1. Standard CSV
    csv_file = tmp_path / "standard.csv"
    csv_file.write_text("id,val,category\n1,10.5,A\n2,20.0,B\n3,35.2,A\n", encoding="utf-8")

    # 2. Semicolon delimited CSV
    semi_csv = tmp_path / "semicolon.csv"
    semi_csv.write_text("id;metric;status\n100;42.5;Active\n200;99.9;Inactive\n", encoding="utf-8")

    # 3. Tab delimited TSV
    tsv_file = tmp_path / "tab_separated.tsv"
    tsv_file.write_text("code\tscore\tlabel\nX1\t9.8\tPositive\nX2\t4.1\tNegative\n", encoding="utf-8")

    # 4. Multi-sheet Excel workbook
    excel_file = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        pd.DataFrame({"metric": [1, 2, 3], "revenue": [100, 200, 300]}).to_excel(writer, sheet_name="Financials", index=False)
        pd.DataFrame({"emp_id": [10, 20], "dept": ["HR", "Eng"]}).to_excel(writer, sheet_name="Headcount", index=False)
        # Empty sheet with headers only
        pd.DataFrame(columns=["col_a", "col_b"]).to_excel(writer, sheet_name="EmptyHeaders", index=False)

    return {
        "csv": str(csv_file),
        "semi_csv": str(semi_csv),
        "tsv": str(tsv_file),
        "excel": str(excel_file),
        "tmp_dir": tmp_path,
    }


def make_entry(dataset_id: str, name: str, file_path: str, df: pd.DataFrame, format: str = "csv", **kwargs) -> DatasetEntry:
    """Helper to build a valid DatasetEntry for tests."""
    return DatasetEntry(
        dataset_id=dataset_id,
        name=name,
        file_path=file_path,
        format=format,
        row_count=len(df),
        column_count=len(df.columns),
        knowledge_model={"columns": list(df.columns)},
        loaded_at="2026-10-10T00:00:00",
        file_hash="test_hash",
        initial_df_hash=DatasetProfiler.compute_hash(df),
        is_mutated=kwargs.get("is_mutated", False),
        source_file_mtime=os.path.getmtime(file_path) if os.path.exists(file_path) else None,
        source_file_size=os.path.getsize(file_path) if os.path.exists(file_path) else None,
        sheet_name=kwargs.get("sheet_name"),
        available_sheets=kwargs.get("available_sheets") or [],
        is_source_native=kwargs.get("is_source_native", True),
        source_storage=kwargs.get("source_storage", "source_native")
    )


def test_excel_sheet_discovery(temp_workspace):
    """Verify get_excel_sheets discovers all sheets without reading full data."""
    sheets = dataset_loader.get_excel_sheets(temp_workspace["excel"])
    assert sheets == ["Financials", "Headcount", "EmptyHeaders"]


def test_excel_sheet_loading_and_switching(temp_workspace):
    """Verify reading specific sheets from Excel and switching sheets via select_excel_sheet API."""
    excel_path = temp_workspace["excel"]
    df_fin, res_fin = dataset_loader.load(excel_path, sheet_name="Financials")
    assert res_fin.success is True
    assert list(df_fin.columns) == ["metric", "revenue"]
    assert len(df_fin) == 3

    dataset_id = "ds_excel_test"
    entry = make_entry(
        dataset_id=dataset_id,
        name="multi_sheet.xlsx",
        file_path=excel_path,
        df=df_fin,
        format="xlsx",
        sheet_name="Financials",
        available_sheets=["Financials", "Headcount", "EmptyHeaders"]
    )
    dataset_registry.register(dataset_id, entry)
    dataset_registry.register(dataset_id, df_fin)
    dataset_registry.set_active(dataset_id)

    # Call select_excel_sheet API to switch to Headcount
    resp = select_excel_sheet(dataset_id, SelectSheetRequest(sheet_name="Headcount"))
    assert resp["sheet_name"] == "Headcount"
    assert resp["row_count"] == 2

    # Verify registry state is updated
    active_entry = dataset_registry.get(dataset_id)
    assert active_entry.sheet_name == "Headcount"
    switched_df = dataset_registry.get_dataframe(dataset_id)
    assert list(switched_df.columns) == ["emp_id", "dept"]
    assert len(switched_df) == 2


def test_excel_empty_sheet_handling(temp_workspace):
    """Verify loading empty sheet produces empty DataFrame with preserved columns."""
    df_empty, res_empty = dataset_loader.load(temp_workspace["excel"], sheet_name="EmptyHeaders")
    assert res_empty.success is True
    assert len(df_empty) == 0
    assert list(df_empty.columns) == ["col_a", "col_b"]


def test_csv_delimiter_sniffing(temp_workspace):
    """Verify dataset_loader auto-detects semicolon and tab delimiters."""
    df_semi, res_semi = dataset_loader.load(temp_workspace["semi_csv"])
    assert res_semi.success is True
    assert list(df_semi.columns) == ["id", "metric", "status"]
    assert len(df_semi) == 2

    df_tsv, res_tsv = dataset_loader.load(temp_workspace["tsv"])
    assert res_tsv.success is True
    assert list(df_tsv.columns) == ["code", "score", "label"]
    assert len(df_tsv) == 2


def test_source_execution_planner_csv_vs_excel(temp_workspace):
    """Verify planner routes CSV to direct_file_scan and Excel to in_memory_dataframe."""
    # 1. Unmodified CSV -> direct_file_scan
    csv_path = temp_workspace["csv"]
    df_csv, res_csv = dataset_loader.load(csv_path)
    csv_entry = make_entry("ds_csv", "standard.csv", csv_path, df_csv, format="csv")
    dataset_registry.register("ds_csv", csv_entry)
    dataset_registry.register("ds_csv", df_csv)

    plan_csv = source_execution_planner.plan_sql(dataset_entry=csv_entry, df=df_csv, sql="SELECT * FROM dataset")
    assert plan_csv.mode == "direct_file_scan"
    assert plan_csv.pushdown_supported is True

    # 2. Excel -> in_memory_dataframe
    xl_path = temp_workspace["excel"]
    df_xl, res_xl = dataset_loader.load(xl_path, sheet_name="Financials")
    xl_entry = make_entry("ds_xl", "multi_sheet.xlsx", xl_path, df_xl, format="xlsx", sheet_name="Financials")
    dataset_registry.register("ds_xl", xl_entry)
    dataset_registry.register("ds_xl", df_xl)

    plan_xl = source_execution_planner.plan_sql(dataset_entry=xl_entry, df=df_xl, sql="SELECT * FROM dataset")
    assert plan_xl.mode == "in_memory_dataframe"
    assert plan_xl.pushdown_supported is False


def test_in_memory_mutation_prevents_direct_scan(temp_workspace):
    """Verify that any in-memory cleaning or mutation falls back to in_memory_dataframe."""
    csv_path = temp_workspace["csv"]
    df_csv, _ = dataset_loader.load(csv_path)
    dataset_id = "ds_mut"
    entry = make_entry(dataset_id, "standard.csv", csv_path, df_csv, format="csv")
    dataset_registry.register(dataset_id, entry)
    dataset_registry.register(dataset_id, df_csv)

    # Initially direct scan
    initial_plan = source_execution_planner.plan_sql(dataset_entry=entry, df=df_csv, sql="SELECT COUNT(*) FROM dataset")
    assert initial_plan.mode == "direct_file_scan"

    # Simulate cleaning mutation
    mutated_df = df_csv.copy()
    mutated_df["val"] = mutated_df["val"] * 2
    dataset_registry.update_dataframe(dataset_id, mutated_df)

    updated_entry = dataset_registry.get(dataset_id)
    assert updated_entry.is_mutated is True

    # Mutated dataset must NEVER be directly scanned from original file
    post_mutation_plan = source_execution_planner.plan_sql(dataset_entry=updated_entry, df=mutated_df, sql="SELECT COUNT(*) FROM dataset")
    assert post_mutation_plan.mode == "in_memory_dataframe"
    assert post_mutation_plan.pushdown_supported is False
    assert "modified in memory" in post_mutation_plan.reason.lower()


def test_drift_detection_on_external_modification(temp_workspace):
    """Verify that external changes to the file trigger drift detection and prevent stale direct scans."""
    file_path = temp_workspace["csv"]
    df_csv, _ = dataset_loader.load(file_path)
    dataset_id = "ds_drift"
    entry = make_entry(dataset_id, "standard.csv", file_path, df_csv, format="csv")
    dataset_registry.register(dataset_id, entry)
    dataset_registry.register(dataset_id, df_csv)

    # Initial check
    has_drift = entry.check_drift()
    assert has_drift is False
    assert entry.drift_detected is False

    # Wait 50ms and append data to the file on disk
    time.sleep(0.05)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("4,99.9,B\n")

    has_drift_after = entry.check_drift()
    assert has_drift_after is True
    assert entry.drift_detected is True

    # Planner must reject direct scan on drifted file
    drifted_plan = source_execution_planner.plan_sql(dataset_entry=entry, df=df_csv, sql="SELECT * FROM dataset")
    assert drifted_plan.mode == "in_memory_dataframe"
    assert "drifted on disk" in drifted_plan.reason.lower()


def test_duckdb_execution_and_oracle_parity(temp_workspace):
    """Verify DuckDB execution result matches pandas Oracle exactly."""
    file_path = temp_workspace["csv"]
    df_csv, _ = dataset_loader.load(file_path)
    dataset_id = "ds_oracle"

    sql = "SELECT category, SUM(val) as total_val FROM dataset GROUP BY category ORDER BY category"
    res = GovernedDuckDBManager.execute_query(
        sql=sql,
        df=df_csv,
        dataset_id=dataset_id,
        source_file_path=file_path
    )

    assert res.status == "SUCCEEDED"
    assert res.execution_mode == "direct_file_scan"
    assert res.pushdown_supported is True

    # Oracle comparison with Pandas
    oracle_df = df_csv.groupby("category")["val"].sum().reset_index().rename(columns={"val": "total_val"}).sort_values("category")
    duck_rows = res.rows
    assert len(duck_rows) == len(oracle_df)
    for duck_row, (_, oracle_row) in zip(duck_rows, oracle_df.iterrows()):
        assert duck_row["category"] == oracle_row["category"]
        assert pytest.approx(duck_row["total_val"]) == oracle_row["total_val"]


def test_save_result_dataset_metadata():
    """Regression test: verify saved analytical results are registered with is_source_native=False and source_storage=managed_upload."""
    from backend.api.datasets import save_result_as_dataset, SaveAsDatasetRequest

    save_req = SaveAsDatasetRequest(
        name="regression_test_saved_table",
        columns=["dim_a", "metric_b"],
        rows=[{"dim_a": "alpha", "metric_b": 100}, {"dim_a": "beta", "metric_b": 200}],
        parent_dataset_id="ds_parent_123"
    )
    resp = save_result_as_dataset(save_req)
    saved_id = resp["dataset_id"]
    entry = dataset_registry.get(saved_id)
    assert entry is not None
    assert entry.is_source_native is False
    assert entry.source_storage == "managed_upload"
