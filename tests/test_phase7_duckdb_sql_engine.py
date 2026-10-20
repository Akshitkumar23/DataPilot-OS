"""
Phase 7 Test Suite: Governed DuckDB SQL Engine.
Verifies:
1. In-process DuckDB querying directly over active pandas DataFrame
2. Read-only execution contract (blocking DROP, DELETE, UPDATE, ALTER, COPY, INSTALL, LOAD, ATTACH, DETACH)
3. Schema grounding (rejecting unknown columns with structured error)
4. Zero mutation invariant (fingerprint before == fingerprint after)
5. JSON serialization (NaN, inf, timestamps, NumPy/Pandas scalar handling)
6. Complex analytical queries (CTEs, Window functions, Aggregations)
7. Query history logging and pagination
"""
import pytest
import pandas as pd
import numpy as np

from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.sql_engine.sql_validator import SQLValidator
from backend.sql_engine.query_history import SQLQueryHistoryManager


@pytest.fixture
def sample_financial_df():
    """Generates a realistic schema-driven test DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "account_id": [f"ACC_{i:04d}" for i in range(100)],
        "department": np.random.choice(["Finance", "Engineering", "Marketing", "Operations"], size=100),
        "budget": np.random.uniform(10000.0, 95000.0, size=100).round(2),
        "headcount": np.random.randint(1, 25, size=100),
        "is_active": np.random.choice([True, False], size=100),
        "notes": [None if i % 5 == 0 else f"Note {i}" for i in range(100)]
    })


def test_basic_sql_selection(sample_financial_df):
    """Verifies basic SELECT querying directly over in-memory DataFrame."""
    sql = "SELECT account_id, department, budget FROM active_dataset WHERE budget > 50000 ORDER BY budget DESC LIMIT 10"
    
    validation = SQLValidator.validate_query(sql, sample_financial_df)
    assert validation.is_valid is True
    assert "account_id" in validation.referenced_columns
    assert "department" in validation.referenced_columns
    assert "budget" in validation.referenced_columns

    result = GovernedDuckDBManager.execute_query(
        sample_financial_df,
        sql,
        dataset_id="financial_test",
        page=1,
        page_size=10
    )
    assert result.status == "SUCCEEDED"
    assert result.row_count == 10
    assert len(result.rows) == 10
    assert result.columns == ["account_id", "department", "budget"]
    assert result.rows[0]["budget"] >= result.rows[1]["budget"]


def test_complex_analytics_cte_and_window(sample_financial_df):
    """Verifies advanced analytical queries: CTEs, Window functions, aggregations."""
    sql = """
    WITH dept_ranks AS (
        SELECT 
            department, 
            account_id, 
            budget,
            AVG(budget) OVER (PARTITION BY department) AS avg_dept_budget,
            ROW_NUMBER() OVER (PARTITION BY department ORDER BY budget DESC) as rnk
        FROM active_dataset
    )
    SELECT department, account_id, budget, avg_dept_budget
    FROM dept_ranks
    WHERE rnk = 1
    ORDER BY budget DESC
    """
    validation = SQLValidator.validate_query(sql, sample_financial_df)
    assert validation.is_valid is True
    assert validation.estimated_operation in ("CTE", "WINDOW", "AGGREGATION")

    result = GovernedDuckDBManager.execute_query(
        sample_financial_df,
        sql,
        dataset_id="financial_test"
    )
    assert result.status == "SUCCEEDED"
    assert result.row_count == 4  # 4 distinct departments
    assert "avg_dept_budget" in result.columns


def test_schema_grounding_rejects_unknown_columns(sample_financial_df):
    """Verifies strict schema grounding rejects unknown columns without fuzzy guessing."""
    sql = "SELECT department, non_existent_revenue_col FROM active_dataset"
    validation = SQLValidator.validate_query(sql, sample_financial_df)
    
    assert validation.is_valid is False
    assert validation.error_code == "UNKNOWN_COLUMN"
    assert "non_existent_revenue_col" in validation.message
    assert "Available columns:" in validation.message
    # Make sure it lists actual columns
    assert "department" in validation.message
    assert "budget" in validation.message


def test_read_only_contract_blocks_mutations(sample_financial_df):
    """Verifies that mutating SQL commands are rejected."""
    forbidden_queries = [
        "DROP TABLE active_dataset",
        "DELETE FROM active_dataset WHERE headcount < 5",
        "UPDATE active_dataset SET budget = 0",
        "ALTER TABLE active_dataset ADD COLUMN leaked INT",
        "ATTACH 'evil.db' AS evil",
        "INSTALL httpfs",
        "LOAD httpfs",
        "COPY active_dataset TO '/tmp/leak.csv'",
    ]

    for forbidden_sql in forbidden_queries:
        validation = SQLValidator.validate_query(forbidden_sql, sample_financial_df)
        assert validation.is_valid is False, f"Expected validation failure for: {forbidden_sql}"
        assert validation.error_code in ("MUTATION_BLOCKED", "PARSE_ERROR")

        # Execution attempt must also fail
        result = GovernedDuckDBManager.execute_query(
            sample_financial_df,
            forbidden_sql,
            dataset_id="financial_test"
        )
        assert result.status == "FAILED"
        assert result.error_code in ("MUTATION_BLOCKED", "VALIDATION_FAILED")


def test_zero_mutation_dataset_invariant(sample_financial_df):
    """Verifies dataset fingerprint before and after SQL execution remains strictly identical."""
    from backend.ds_engine.dataset_profiler import DatasetProfiler
    
    profile_before = DatasetProfiler.profile(sample_financial_df)
    fp_before = profile_before.fingerprint

    # Execute read-only analytical query
    sql = "SELECT department, COUNT(*), SUM(budget) FROM active_dataset GROUP BY department"
    result = GovernedDuckDBManager.execute_query(
        sample_financial_df,
        sql,
        dataset_id="financial_test",
        dataset_fingerprint=fp_before
    )
    assert result.status == "SUCCEEDED"

    profile_after = DatasetProfiler.profile(sample_financial_df)
    fp_after = profile_after.fingerprint

    assert fp_before == fp_after, "Dataset was mutated by SQL query execution!"


def test_json_safe_serialization():
    """Verifies NaN, infinity, Timestamps, and NumPy scalar types are serialized to JSON without crashing."""
    df_with_edge_cases = pd.DataFrame({
        "id": [1, 2, 3],
        "with_nan": [1.5, np.nan, 3.5],
        "with_inf": [10.0, np.inf, -np.inf],
        "with_date": pd.date_range("2025-01-01", periods=3, freq="D"),
        "with_null_str": ["val1", None, "val3"]
    })

    sql = "SELECT * FROM active_dataset"
    result = GovernedDuckDBManager.execute_query(df_with_edge_cases, sql, dataset_id="edge_cases")
    
    assert result.status == "SUCCEEDED"
    assert len(result.rows) == 3
    # Check that inf and nan are transformed to JSON safe representations (None)
    assert result.rows[1]["with_nan"] is None
    assert result.rows[1]["with_inf"] is None
    assert result.rows[2]["with_inf"] is None
    assert isinstance(result.rows[0]["with_date"], str)


def test_pagination_and_query_history(sample_financial_df, tmp_path, monkeypatch):
    """Verifies pagination slicing and persistent local query history logging."""
    history_file = tmp_path / "test_sql_history.json"
    monkeypatch.setattr(SQLQueryHistoryManager, "HISTORY_FILE", history_file)

    sql = "SELECT account_id, department, budget FROM active_dataset ORDER BY budget ASC"
    
    # Page 1 (20 items)
    p1 = GovernedDuckDBManager.execute_query(sample_financial_df, sql, dataset_id="test_ds", page=1, page_size=20)
    assert p1.page == 1
    assert p1.row_count == 100
    assert len(p1.rows) == 20
    assert p1.total_pages == 5

    # Page 2 (20 items)
    p2 = GovernedDuckDBManager.execute_query(sample_financial_df, sql, dataset_id="test_ds", page=2, page_size=20)
    assert p2.page == 2
    assert len(p2.rows) == 20
    # Page 1 and Page 2 items should not overlap
    p1_ids = {r["account_id"] for r in p1.rows}
    p2_ids = {r["account_id"] for r in p2.rows}
    assert p1_ids.isdisjoint(p2_ids)

    # Check query history recorded both executions
    history = SQLQueryHistoryManager.get_history(limit=10)
    assert len(history) >= 2
    assert history[0]["status"] == "SUCCEEDED"
    assert history[0]["row_count"] == 100
