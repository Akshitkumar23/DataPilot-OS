import json
from pathlib import Path

from training.build_training_master_status import build_status


ROOT = Path(__file__).resolve().parents[1]


def test_master_status_reports_truthful_training_blockers():
    report = build_status(
        ROOT / "training" / "planner_sft_contract_verified.jsonl",
        ROOT / "training" / "review_queue" / "full-scope-core" / "coverage-and-evidence.json",
        ROOT / "training" / "artifacts" / "numeric-oracle-review.json",
        ROOT / "training" / "artifacts" / "DataPilot-Kaggle-Preflight-V4",
    )
    assert report["report_schema"] == "datapilot-training-master-status-v1"
    assert report["live_registry"]["analytical_tool_count"] == 22
    assert report["full_scope_seed"]["independent_reference_plans"] == 20
    assert report["independent_numeric_evidence"]["unique_plan_status_counts"]["numeric_plan_verified"] == 54
    assert report["prompt_contract"]["aligned"] is False
    assert report["gates"]["candidate_release_eligible"] is False
    assert report["next_required_phase"] == "P1_RUNTIME_TRAINING_CONTRACT_ALIGNMENT"
    assert report["blockers"]


def test_master_status_is_json_serializable():
    report = build_status(
        ROOT / "training" / "planner_sft_contract_verified.jsonl",
        ROOT / "training" / "review_queue" / "full-scope-core" / "coverage-and-evidence.json",
        ROOT / "training" / "artifacts" / "numeric-oracle-review.json",
        ROOT / "training" / "artifacts" / "DataPilot-Kaggle-Preflight-V4",
    )
    assert json.loads(json.dumps(report))["purpose"].startswith("Pre-training")
