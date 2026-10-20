"""Report whether a reviewed DataPilot planner corpus is safe to fine-tune.

This is an offline governance gate.  It never calls an LLM, never changes the
desktop application's active model, and never treats synthetic candidates as
training labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry
from scripts.validate_planner_corpus import DEFAULT_CORPUS, DEFAULT_PROVENANCE, load_jsonl, validate_records
from training.corpus_contract import reference_commands

APPROVED_LABELS = {"seed-reviewed", "human-reviewed"}
CONTRACT_VERIFIED_LABEL = "contract-verified"


def load_data_kinds(provenance_path: Path) -> dict[str, str]:
    """Resolve source kind by dataset id from the versioned source manifest."""
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest_ref = provenance.get("manifest", "data/benchmark_sources.json")
    manifest_path = ROOT / Path(str(manifest_ref).replace("\\", "/"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        item["id"]: item.get("data_kind", "unknown")
        for item in manifest.get("datasets", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def analytical_tool_names(registry: CapabilityRegistry) -> set[str]:
    """Return live registry tools that require demonstrated planner coverage."""
    return {
        spec.name
        for spec in registry.list_tool_specs()
        if spec.category not in {"general_chat", "clarification"}
    }


def assess(
    records: list[dict[str, Any]],
    *,
    provenance_path: Path,
    min_train: int,
    min_validation: int,
    min_test: int,
    min_train_per_tool: int,
    min_clarification_train: int,
    min_distinct_train_datasets: int,
    min_distinct_test_datasets: int,
    real_only: bool,
    require_disjoint_test_datasets: bool,
    accepted_labels: set[str] | None = None,
) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validate_records(records, provenance)
    registry = CapabilityRegistry(auto_bootstrap=True)
    kinds = load_data_kinds(provenance_path)
    accepted_labels = accepted_labels or APPROVED_LABELS
    reviewed = [record for record in records if record.get("label_status") in accepted_labels]
    human_reviewed = [record for record in records if record.get("label_status") == "human-reviewed"]
    unreviewed = [record for record in records if record.get("label_status") not in accepted_labels]
    split_counts = Counter(record.get("split") for record in reviewed)
    train = [record for record in reviewed if record.get("split") == "train"]
    test = [record for record in reviewed if record.get("split") == "test"]
    tool_counts = Counter(
        command['tool']
        for record in train
        for command in reference_commands(record)
    )
    required_tools = analytical_tool_names(registry)
    missing_tools = sorted(tool for tool in required_tools if tool_counts[tool] < min_train_per_tool)
    clarification_train = sum(record.get("expected_decision") == "clarification" for record in train)
    train_datasets = {record.get("dataset_id") for record in train if isinstance(record.get("dataset_id"), str)}
    test_datasets = {record.get("dataset_id") for record in test if isinstance(record.get("dataset_id"), str)}
    non_observed = sorted({
        record.get("dataset_id")
        for record in reviewed
        if kinds.get(record.get("dataset_id"), "unknown") != "observed_public"
    })
    duplicate_queries = sorted({
        query for query, count in Counter(
            str(record.get("query", "")).strip().casefold() for record in reviewed
        ).items() if query and count > 1
    })
    executable_commands = [
        json.dumps(reference_commands(record), sort_keys=True, separators=(",", ":"))
        for record in reviewed
        if record.get("expected_decision") == "tool"
    ]
    unique_command_count = len(set(executable_commands))
    execution_verified_count = sum(
        isinstance(record.get("verification"), dict)
        and record["verification"].get("kind") == "execution"
        for record in reviewed
    )

    blockers: list[str] = []
    if split_counts["train"] < min_train:
        blockers.append(f"reviewed train examples {split_counts['train']} < required {min_train}")
    if split_counts["validation"] < min_validation:
        blockers.append(f"reviewed validation examples {split_counts['validation']} < required {min_validation}")
    if split_counts["test"] < min_test:
        blockers.append(f"reviewed test examples {split_counts['test']} < required {min_test}")
    if clarification_train < min_clarification_train:
        blockers.append(f"reviewed train clarification examples {clarification_train} < required {min_clarification_train}")
    if missing_tools:
        blockers.append("insufficient train coverage for live tools: " + ", ".join(missing_tools))
    if len(train_datasets) < min_distinct_train_datasets:
        blockers.append(f"distinct train datasets {len(train_datasets)} < required {min_distinct_train_datasets}")
    if len(test_datasets) < min_distinct_test_datasets:
        blockers.append(f"distinct test datasets {len(test_datasets)} < required {min_distinct_test_datasets}")
    overlap = sorted(train_datasets & test_datasets)
    if require_disjoint_test_datasets and overlap:
        blockers.append("test datasets overlap training datasets: " + ", ".join(overlap))
    if real_only and non_observed:
        blockers.append("real-only policy excludes dataset(s): " + ", ".join(non_observed))
    if duplicate_queries:
        blockers.append(f"duplicate reviewed queries detected: {len(duplicate_queries)}")

    return {
        "ready": not blockers,
        "accepted_label_statuses": sorted(accepted_labels),
        # "reviewed_*" is retained for compatibility with older callers.  It
        # means accepted by the currently selected gate, not necessarily
        # reviewed by a human (the research gate may accept contract-verified
        # records).  Consumers should display accepted_* instead.
        "accepted_record_count": len(reviewed),
        "human_reviewed_record_count": len(human_reviewed),
        "human_reviewed_split_counts": dict(Counter(record.get("split") for record in human_reviewed)),
        "not_accepted_record_count": len(unreviewed),
        "reviewed_record_count": len(reviewed),
        "unreviewed_record_count": len(unreviewed),
        "reviewed_split_counts": dict(split_counts),
        "train_tool_counts": {tool: tool_counts[tool] for tool in sorted(required_tools)},
        "missing_or_undercovered_tools": missing_tools,
        "train_clarification_count": clarification_train,
        "unique_executable_command_count": unique_command_count,
        "execution_verified_record_count": execution_verified_count,
        "train_datasets": sorted(train_datasets),
        "test_datasets": sorted(test_datasets),
        "test_train_dataset_overlap": overlap,
        "real_only": real_only,
        "non_observed_dataset_ids": non_observed,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--min-train", type=int, default=1000)
    parser.add_argument("--min-validation", type=int, default=100)
    parser.add_argument("--min-test", type=int, default=200)
    parser.add_argument("--min-train-per-tool", type=int, default=25)
    parser.add_argument("--min-clarification-train", type=int, default=100)
    parser.add_argument("--min-distinct-train-datasets", type=int, default=4)
    parser.add_argument("--min-distinct-test-datasets", type=int, default=2)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument(
        "--allow-contract-verified",
        action="store_true",
        help="Allow live-ToolSpec contract-verified examples for an explicitly unpromoted research run; never use this for production approval.",
    )
    parser.add_argument("--allow-test-dataset-overlap", action="store_true")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero when the corpus is not training-ready.")
    args = parser.parse_args()
    report = assess(
        load_jsonl(args.corpus), provenance_path=args.provenance,
        min_train=args.min_train, min_validation=args.min_validation, min_test=args.min_test,
        min_train_per_tool=args.min_train_per_tool,
        min_clarification_train=args.min_clarification_train,
        min_distinct_train_datasets=args.min_distinct_train_datasets,
        min_distinct_test_datasets=args.min_distinct_test_datasets,
        real_only=args.real_only,
        require_disjoint_test_datasets=not args.allow_test_dataset_overlap,
        accepted_labels=APPROVED_LABELS | ({CONTRACT_VERIFIED_LABEL} if args.allow_contract_verified else set()),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] or not args.enforce else 2


if __name__ == "__main__":
    raise SystemExit(main())
