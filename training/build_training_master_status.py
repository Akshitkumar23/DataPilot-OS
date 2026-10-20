"""Build one evidence-backed status report before any new planner training.

The report is diagnostic only.  It never edits labels, changes the active
Ollama model, starts training, or promotes a Kaggle artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry
from modules.semantic_planner import SemanticPlanner
from training.audit_corpus_quality import audit
from training.corpus_contract import reference_commands
from training.planner_prompt import build_prompt

DEFAULT_CORPUS = ROOT / "training" / "planner_sft_contract_verified.jsonl"
DEFAULT_FULL_SCOPE = ROOT / "training" / "review_queue" / "full-scope-core" / "coverage-and-evidence.json"
DEFAULT_NUMERIC = ROOT / "training" / "artifacts" / "numeric-oracle-review.json"
DEFAULT_BUNDLE = ROOT / "training" / "artifacts" / "DataPilot-Kaggle-Preflight-V4"
DEFAULT_OUTPUT = ROOT / "training" / "artifacts" / "training-master-status.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _bundle_drift(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        return {"available": False, "path": str(bundle), "reason": "BUNDLE_MANIFEST.json not found"}
    manifest = _json(manifest_path)
    matches, changed, missing, generated_only = [], [], [], []
    for relative, recorded_hash in manifest.get("files", {}).items():
        current = ROOT / relative
        if not current.exists():
            generated_only.append(relative)
        elif not current.is_file():
            missing.append(relative)
        elif _sha256(current) == recorded_hash:
            matches.append(relative)
        else:
            changed.append(relative)
    return {
        "available": True,
        "path": str(bundle),
        "bundle_name": manifest.get("bundle"),
        "created_at_utc": manifest.get("created_at_utc"),
        "manifest_sha256": _sha256(manifest_path),
        "comparable_current_files": len(matches) + len(changed) + len(missing),
        "matching_current_files": len(matches),
        "changed_current_files": changed,
        "missing_current_files": missing,
        "bundle_generated_files_without_root_counterpart": generated_only,
        "current_source_snapshot_matches_bundle": not changed and not missing,
    }


def _local_evaluation_reports() -> list[str]:
    reports = []
    for path in (ROOT / "training").rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("metrics"), dict) and isinstance(value.get("results"), list):
            reports.append(str(path))
    return sorted(reports)


def build_status(corpus: Path, full_scope: Path, numeric: Path, bundle: Path) -> dict[str, Any]:
    records = _jsonl(corpus)
    corpus_quality = audit(records)
    registry = CapabilityRegistry(auto_bootstrap=True)
    specs = registry.list_tool_specs()
    analytical = sorted(
        spec.name for spec in specs
        if spec.category not in {"general_chat", "clarification", "clarify"}
    )
    command_lengths = [len(reference_commands(record)) for record in records]

    full_scope_report = _json(full_scope) if full_scope.exists() else {}
    numeric_report = _json(numeric) if numeric.exists() else {}
    planner_source = inspect.getsource(SemanticPlanner.extract_semantic_intent)
    production_sequence_disabled = "allow_command_sequence=False" in planner_source
    offline_prompt_source = inspect.getsource(build_prompt)
    offline_uses_commands = '"commands"' in offline_prompt_source or "commands\\\"" in offline_prompt_source
    # Source inspection can identify a mismatch, not prove model/template/
    # packaged-runtime parity. Keep this release gate unverified.
    contract_aligned = False

    blockers = list(corpus_quality.get("blockers", []))
    if not contract_aligned:
        blockers.append(
            "End-to-end packaged model/runtime contract parity remains unverified; local renderer tests are not release evidence"
        )
    if corpus_quality["splits"]["train"].get("human_reviewed", 0) == 0:
        blockers.append("No labels are recorded as human-reviewed")
    if not full_scope_report.get("training_ready", False):
        blockers.append("Full-scope seed is a review queue, not an approved split corpus")
    bundle_report = _bundle_drift(bundle)
    if bundle_report.get("available") and not bundle_report.get("current_source_snapshot_matches_bundle"):
        blockers.append("Existing V4 bundle source snapshot differs from the current project")

    local_eval_reports = _local_evaluation_reports()
    return {
        "report_schema": "datapilot-training-master-status-v1",
        "purpose": "Pre-training evidence inventory; this report does not authorize training or deployment.",
        "corpus": {
            "path": str(corpus),
            "sha256": _sha256(corpus),
            "records": len(records),
            "single_command_records": sum(length == 1 for length in command_lengths),
            "multi_command_records": sum(length > 1 for length in command_lengths),
            "quality": corpus_quality,
        },
        "live_registry": {
            "tool_count": len(specs),
            "analytical_tool_count": len(analytical),
            "analytical_tools": analytical,
        },
        "full_scope_seed": {
            "path": str(full_scope),
            "analytical_tools_covered": full_scope_report.get("analytical_tools_covered"),
            "independent_reference_plans": full_scope_report.get("independent_reference_plans"),
            "language_variants": full_scope_report.get("language_variants"),
            "execution_passed": full_scope_report.get("execution_passed"),
            "human_reviewed": full_scope_report.get("human_reviewed"),
            "training_ready": full_scope_report.get("training_ready", False),
            "limits": full_scope_report.get("limits", []),
        },
        "independent_numeric_evidence": {
            "path": str(numeric),
            "record_status_counts": numeric_report.get("record_status_counts", {}),
            "unique_plan_status_counts": numeric_report.get("unique_plan_status_counts", {}),
            "note": "Numeric-plan agreement is not query-semantic or human review.",
        },
        "prompt_contract": {
            "offline_prompt_version": "runtime-contract-sequence-v1",
            "offline_target_envelope": "tool+arguments OR ordered commands[]; no mode field",
            "production_response_schema": "tool+arguments OR ordered commands[]",
            "production_disables_command_sequences": production_sequence_disabled,
            "aligned": contract_aligned,
        },
        "historical_kaggle": {
            "saved_summary_available": (ROOT / "docs" / "KAGGLE_TRAINING_RESEARCH_AUDIT.md").exists(),
            "local_per_case_evaluation_reports": local_eval_reports,
            "raw_per_case_fix4_predictions_found_locally": None,
            "recorded_fix4": {
                "test_cases": 200,
                "base_exact_plan": 0,
                "candidate_exact_plan": 0,
                "candidate_tool_match": 158,
                "candidate_schema_valid": 131,
                "decision": "release rejected",
            },
            "evidence_limit": (
                "Candidate per-case reports exist locally and still require Fix4 identity verification."
                if local_eval_reports else
                "Recorded summary is available; raw per-case Fix4 exports were not found locally."
            ),
        },
        "bundle": bundle_report,
        "gates": {
            "inventory_complete": True,
            "runtime_training_contract_aligned": contract_aligned,
            "corpus_quality_ready": corpus_quality.get("ready_for_new_training", False),
            "full_scope_corpus_ready": full_scope_report.get("training_ready", False),
            "gpu_smoke_passed_for_current_bundle": False,
            "candidate_release_eligible": False,
            "desktop_deployment_verified": False,
        },
        "blockers": blockers,
        "next_required_phase": "P1_RUNTIME_TRAINING_CONTRACT_ALIGNMENT",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--full-scope", type=Path, default=DEFAULT_FULL_SCOPE)
    parser.add_argument("--numeric", type=Path, default=DEFAULT_NUMERIC)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_status(args.corpus, args.full_scope, args.numeric, args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
