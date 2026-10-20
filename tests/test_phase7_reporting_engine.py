"""
Phase 7 Test Suite: Analytical Reporting Engine.
Verifies:
1. Evidence collection from active dataset profile and provenance history
2. Stale evidence detection on fingerprint mismatch
3. Production of complete 9-section AnalyticalReport with real key findings
4. Standalone publication-quality HTML export with @media print styling
"""
import pytest
import pandas as pd
import numpy as np

from backend.reporting.evidence_collector import ReportEvidenceCollector
from backend.reporting.report_generator import GovernedReportGenerator
from backend.reporting.report_exporter import ReportExporter
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.persistence.provenance_store import provenance_store


@pytest.fixture
def sales_dataset():
    np.random.seed(101)
    n = 60
    return pd.DataFrame({
        "order_id": [f"ORD_{i:03d}" for i in range(n)],
        "region": np.random.choice(["North", "South", "East", "West"], size=n),
        "sales_amount": np.random.normal(500, 120, size=n).round(2),
        "quantity": np.random.randint(1, 10, size=n),
        "discount": np.random.choice([0.0, 0.05, 0.1, 0.15], size=n)
    })


def test_evidence_collection_from_profile(sales_dataset):
    """Verifies that the collector harvests valid profile metrics into evidence items."""
    profile = DatasetProfiler.profile(sales_dataset)
    evidence_package = ReportEvidenceCollector.harvest_evidence(
        dataset_id="sales_ds",
        df=sales_dataset,
        profile=profile
    )

    assert evidence_package.dataset_id == "sales_ds"
    assert evidence_package.dataset_fingerprint == profile.fingerprint
    assert evidence_package.stale_warning is False
    assert len(evidence_package.evidence_items) >= 1

    # Check overview evidence item
    overview_item = next((item for item in evidence_package.evidence_items if item.capability == "PROFILE"), None)
    assert overview_item is not None
    assert overview_item.metrics.get("row_count") == 60
    assert overview_item.metrics.get("column_count") == 5


def test_stale_evidence_detection(sales_dataset):
    """Verifies that modifying the DataFrame triggers a stale evidence alert."""
    profile = DatasetProfiler.profile(sales_dataset)
    stale_fingerprint = "00000000000000000000000000000000"

    # Pass in a mismatched fingerprint
    evidence_package = ReportEvidenceCollector.harvest_evidence(
        dataset_id="sales_ds",
        df=sales_dataset,
        profile=profile,
        stale_fingerprint_check=stale_fingerprint
    )

    assert evidence_package.stale_warning is True
    assert "mutated" in evidence_package.stale_reason.lower() or "mismatch" in evidence_package.stale_reason.lower()


def test_report_generation_9_sections(sales_dataset):
    """Verifies that GovernedReportGenerator synthesizes all 9 sections without hallucination."""
    profile = DatasetProfiler.profile(sales_dataset)
    evidence_package = ReportEvidenceCollector.harvest_evidence(
        dataset_id="sales_ds",
        df=sales_dataset,
        profile=profile
    )

    report = GovernedReportGenerator.generate_report(
        dataset_id="sales_ds",
        evidence_package=evidence_package,
        df=sales_dataset,
        title="Comprehensive Sales Analytical Audit"
    )

    assert report.report_id.startswith("rep_")
    assert report.dataset_id == "sales_ds"
    assert len(report.sections) == 9
    
    # Check section IDs
    expected_sections = [
        "overview", "governance", "eda", "statistics",
        "ml", "unsupervised", "timeseries", "sql", "limitations"
    ]
    for s_id in expected_sections:
        sec = next((s for s in report.sections if s.section_id == s_id), None)
        assert sec is not None, f"Missing report section: {s_id}"

    # Overview must reflect real row & column count
    overview_sec = next(s for s in report.sections if s.section_id == "overview")
    assert "60 rows" in overview_sec.content
    assert "5 columns" in overview_sec.content

    # Executive summary and findings must be present
    assert len(report.executive_summary) > 30
    assert len(report.key_findings) >= 1


def test_standalone_html_export(sales_dataset):
    """Verifies standalone HTML export produces valid HTML containing print styles and real data."""
    profile = DatasetProfiler.profile(sales_dataset)
    evidence_package = ReportEvidenceCollector.harvest_evidence(
        dataset_id="sales_ds",
        df=sales_dataset,
        profile=profile
    )
    report = GovernedReportGenerator.generate_report(
        dataset_id="sales_ds",
        evidence_package=evidence_package,
        df=sales_dataset
    )

    html = ReportExporter.to_standalone_html(report)
    assert "<!DOCTYPE html>" in html
    assert "@media print" in html
    assert report.title in html
    assert report.dataset_fingerprint in html
    assert "Executive Summary" in html
    assert "DataPilot OS" in html
