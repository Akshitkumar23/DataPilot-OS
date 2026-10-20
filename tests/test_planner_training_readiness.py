"""Regression coverage for the offline, real-public-data QLoRA gate."""

from pathlib import Path

from scripts.planner_training_readiness import assess
from scripts.validate_planner_corpus import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_real_only_seed_reports_gaps_without_accepting_synthetic_reference_data():
    report = assess(
        load_jsonl(ROOT / "training" / "planner_sft_real_only_seed.jsonl"),
        provenance_path=ROOT / "data" / "real_benchmarks" / "provenance.json",
        min_train=1000,
        min_validation=100,
        min_test=200,
        min_train_per_tool=25,
        min_clarification_train=100,
        min_distinct_train_datasets=4,
        min_distinct_test_datasets=2,
        real_only=True,
        require_disjoint_test_datasets=True,
    )
    assert report["ready"] is False
    assert report["non_observed_dataset_ids"] == []
    assert any("reviewed train examples" in item for item in report["blockers"])
    assert any("overlap" in item for item in report["blockers"])
