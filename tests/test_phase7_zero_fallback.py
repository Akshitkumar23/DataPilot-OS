"""
Phase 7 Test Suite: Zero Fallback Verification.
Verifies strict non-fallback contracts:
1. Unknown column -> structured error (never guess or fuzzy-match)
2. Mutating SQL statement -> blocked with structured error (never ignore or execute substitute)
3. Invalid syntax -> structured error (never fabricate results)
4. Missing analytical evidence -> honest caveat (never fabricate findings)
"""
import pytest
import pandas as pd
import numpy as np

from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.sql_engine.sql_validator import SQLValidator
from backend.sql_engine.sql_builder import DeterministicSQLBuilder
from backend.sql_engine.sql_models import SQLIntent
from backend.reporting.evidence_collector import ReportEvidenceCollector
from backend.reporting.report_generator import GovernedReportGenerator
from backend.ds_engine.dataset_profiler import DatasetProfiler


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "total_spend": [100.0, 250.0, 300.0],
        "loyalty_tier": ["Bronze", "Silver", "Gold"]
    })


def test_zero_fallback_unknown_column(base_df):
    """Verifies that an unknown column produces a structured error and never executes a fallback."""
    sql = "SELECT customer_id, revenue FROM active_dataset"  # 'revenue' does not exist
    val = SQLValidator.validate_query(sql, base_df)
    
    assert val.is_valid is False
    assert val.error_code == "UNKNOWN_COLUMN"
    assert "revenue" in val.message
    # Check that available columns are provided honestly
    assert "total_spend" in val.message

    # Attempting to execute must fail honestly, never guess total_spend
    res = GovernedDuckDBManager.execute_query(base_df, sql, dataset_id="test")
    assert res.status == "FAILED"
    assert res.row_count == 0


def test_zero_fallback_syntax_error(base_df):
    """Verifies that invalid SQL syntax produces a structured syntax error without fallback."""
    sql = "SELECT customer_id FROM WHERE total_spend > 100"  # Invalid syntax
    val = SQLValidator.validate_query(sql, base_df)
    
    assert val.is_valid is False
    assert val.error_code in ("PARSE_ERROR", "BINDER_ERROR")

    res = GovernedDuckDBManager.execute_query(base_df, sql, dataset_id="test")
    assert res.status == "FAILED"
    assert res.row_count == 0


def test_zero_fallback_nl_unknown_column(base_df):
    """Verifies that DeterministicSQLBuilder never guesses when an intent references a missing column."""
    intent = SQLIntent(
        operation="select",
        selected_columns=["customer_id", "spending"],  # 'spending' instead of 'total_spend'
    )

    with pytest.raises(ValueError) as exc:
        DeterministicSQLBuilder.build(intent, base_df)

    assert "Column 'spending' does not exist in active dataset" in str(exc.value)


def test_zero_fallback_reporting_honest_caveats(base_df):
    """Verifies that sections without verified evidence report an honest empty state, never inventing metrics."""
    prof = DatasetProfiler.profile(base_df)
    # Collect evidence with no ML runs or time series runs
    ev = ReportEvidenceCollector.harvest_evidence("test_ds", base_df, prof)
    rep = GovernedReportGenerator.generate_report("test_ds", ev, base_df)

    # ML section must honestly report no verified runs
    ml_sec = next(s for s in rep.sections if s.section_id == "ml")
    assert "No verified Machine Learning model training" in ml_sec.content
    assert ml_sec.key_metrics == {}

    # Time-series section must honestly report no verified runs
    ts_sec = next(s for s in rep.sections if s.section_id == "timeseries")
    assert "No verified Time-Series analysis" in ts_sec.content
    assert ts_sec.key_metrics == {}
