"""
Phase 7 Test Suite: Provenance Audit Trail.
Verifies that all Phase 7 actions (DuckDB SQL execution, analytical report generation)
log authoritative, immutable provenance events to provenance_store.
"""
import pytest
import pandas as pd
import numpy as np

from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.reporting.report_generator import GovernedReportGenerator
from backend.reporting.evidence_collector import ReportEvidenceCollector
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.persistence.provenance_store import provenance_store


@pytest.fixture
def audit_df():
    return pd.DataFrame({
        "tx_id": [f"TX_{i:04d}" for i in range(20)],
        "amount": np.random.uniform(10, 500, size=20).round(2),
        "status": np.random.choice(["cleared", "pending"], size=20)
    })


def test_sql_execution_provenance_logging(audit_df):
    """Verifies that SQL execution emits a provenance event with query metadata."""
    initial_event_count = len(provenance_store.get_events())
    
    sql = "SELECT status, COUNT(*), SUM(amount) FROM active_dataset GROUP BY status"
    result = GovernedDuckDBManager.execute_query(
        audit_df,
        sql,
        dataset_id="provenance_test_ds"
    )
    assert result.status == "SUCCEEDED"

    events = provenance_store.get_events()
    assert len(events) > initial_event_count
    latest_event = events[-1]
    assert latest_event.operation.lower() == "sql_query"
    assert latest_event.parameters.get("sql") == sql
    assert latest_event.result_summary.get("row_count") == result.row_count


def test_report_generation_provenance_logging(audit_df):
    """Verifies that generating an analytical report emits a provenance event."""
    initial_event_count = len(provenance_store.get_events())

    prof = DatasetProfiler.profile(audit_df)
    ev = ReportEvidenceCollector.harvest_evidence("audit_ds", audit_df, prof)
    report = GovernedReportGenerator.generate_report(
        dataset_id="audit_ds",
        evidence_package=ev,
        df=audit_df,
        title="Provenance Audit Report"
    )

    events = provenance_store.get_events()
    assert len(events) > initial_event_count
    latest_event = events[-1]
    assert latest_event.operation.lower() == "generate_report"
    assert latest_event.result_summary.get("report_id") == report.report_id
    assert latest_event.result_summary.get("sections_count") == 9
