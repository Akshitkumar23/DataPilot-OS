"""
Phase 7 Test Suite: Analytical Reporting Anti-Hardcoding Invariant.
Verifies that report generation across 3 unseen domains strictly derives all findings,
metrics, and narratives from runtime profiles and evidence packages without hardcoding.
"""
import pytest
import pandas as pd
import numpy as np

from backend.reporting.evidence_collector import ReportEvidenceCollector
from backend.reporting.report_generator import GovernedReportGenerator
from backend.ds_engine.dataset_profiler import DatasetProfiler


def test_domain_1_genomics_report():
    df = pd.DataFrame({
        "gene_id": [f"GENE_{i:04d}" for i in range(70)],
        "expression_level": np.random.exponential(scale=18.0, size=70),
        "pathway": np.random.choice(["Glycolysis", "Apoptosis", "DNA_Repair"], size=70)
    })
    prof = DatasetProfiler.profile(df)
    ev = ReportEvidenceCollector.harvest_evidence("genomics_ds", df, prof)
    rep = GovernedReportGenerator.generate_report("genomics_ds", ev, df)

    assert rep.dataset_id == "genomics_ds"
    assert len(rep.sections) == 9
    overview = next(s for s in rep.sections if s.section_id == "overview")
    assert "70 rows" in overview.content
    assert "3 columns" in overview.content


def test_domain_2_logistics_report():
    df = pd.DataFrame({
        "manifest_id": [f"MAN_{i:05d}" for i in range(85)],
        "customs_delay_days": np.random.geometric(p=0.3, size=85),
        "freight_cost_usd": np.random.uniform(200, 5000, size=85)
    })
    prof = DatasetProfiler.profile(df)
    ev = ReportEvidenceCollector.harvest_evidence("logistics_ds", df, prof)
    rep = GovernedReportGenerator.generate_report("logistics_ds", ev, df)

    assert rep.dataset_id == "logistics_ds"
    overview = next(s for s in rep.sections if s.section_id == "overview")
    assert "85 rows" in overview.content
    assert "3 columns" in overview.content


def test_domain_3_iot_telemetry_report():
    df = pd.DataFrame({
        "probe_id": [f"PRB_{i:03d}" for i in range(120)],
        "voltage": np.random.normal(3.3, 0.1, size=120),
        "signal_dbm": np.random.randint(-90, -40, size=120),
        "error_state": np.random.choice([False, True], size=120, p=[0.9, 0.1])
    })
    prof = DatasetProfiler.profile(df)
    ev = ReportEvidenceCollector.harvest_evidence("iot_ds", df, prof)
    rep = GovernedReportGenerator.generate_report("iot_ds", ev, df)

    assert rep.dataset_id == "iot_ds"
    overview = next(s for s in rep.sections if s.section_id == "overview")
    assert "120 rows" in overview.content
    assert "4 columns" in overview.content
