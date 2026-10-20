"""Audit DataPilot's live ToolSpec registry against a professional DS lifecycle.

This is an engineering inventory, not a runtime router.  Missing entries stay
visible as backlog instead of being disguised by planner prompts or training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry


CAPABILITIES = [
    ("ingest_profile", "Dataset loading, schema/profile and quality evidence", [], "partial"),
    ("filter_sort", "Typed filtering, slicing, sorting and limits", ["filter_data", "sort_data"], "implemented"),
    ("group_metrics", "Grouped metrics, rates and distribution audits", ["group_data", "aggregate_data", "conditional_rate", "audit_metric_distribution"], "implemented"),
    ("correlation_eda", "EDA and linear/monotonic/nonlinear association", ["eda_data", "correlate_data", "categorical_outcome_association"], "implemented"),
    ("cleaning", "Explicit missing-value and duplicate operations", ["clean_data"], "partial"),
    ("inferential_statistics", "Core hypothesis tests with explicit roles", ["statistical_test"], "partial"),
    ("anomaly_detection", "Explicit unsupervised anomaly definitions", ["detect_anomalies"], "partial"),
    ("clustering_dimension_reduction", "K-means and PCA", ["cluster_data", "pca_analysis"], "partial"),
    ("supervised_ml", "Leakage-aware baseline training and evaluation", ["train_model", "explain_features"], "partial"),
    ("time_series", "Forecast, decomposition and autocorrelation", ["forecast_data", "decompose_time_series", "autocorrelation"], "partial"),
    ("event_windows", "Row/onset-scoped pre-event analysis", ["event_window_analysis"], "implemented"),
    ("join_reshape", "Join/merge, pivot, melt and relational validation", [], "missing"),
    ("derived_features_windows", "Typed expressions, bins, rolling/window and resampling", [], "missing"),
    ("business_analytics", "Cohort, retention, funnel and contribution analysis", [], "missing"),
    ("experimentation_causal", "A/B tests, power, causal estimands and assumptions", [], "missing"),
    ("advanced_model_validation", "Cross-validation, tuning, calibration, imbalance and fairness", [], "missing"),
    ("forecast_validation", "Backtesting, probabilistic intervals and temporal CV", [], "missing"),
    ("model_lifecycle", "Artifact registry, reproducibility, drift and deployment monitoring", [], "missing"),
    ("sql_pushdown", "Typed multi-table SQL planning and database execution", [], "missing"),
    ("governance", "PII controls, access policy, lineage and audit retention", [], "missing"),
]


def build_report() -> dict:
    registry = CapabilityRegistry(auto_bootstrap=True)
    live = {spec.name for spec in registry.list_tool_specs()}
    rows = []
    for capability_id, description, tools, maturity in CAPABILITIES:
        missing_tools = sorted(set(tools) - live)
        effective = "missing" if missing_tools else maturity
        rows.append({
            "capability_id": capability_id,
            "description": description,
            "maturity": effective,
            "required_live_tools": tools,
            "missing_live_tools": missing_tools,
        })
    return {
        "schema_version": "datapilot-professional-capability-matrix-v1",
        "live_tools": sorted(live),
        "live_tool_count": len(live),
        "summary": {status: sum(row["maturity"] == status for row in rows) for status in ("implemented", "partial", "missing")},
        "capabilities": rows,
        "interpretation": "Training may improve selection of live tools; it cannot implement capabilities marked missing.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
