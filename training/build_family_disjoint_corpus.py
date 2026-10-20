"""Build deterministic leakage-resistant splits from approved review output.

Related paraphrases are assigned as one family.  Test datasets are held out
entirely, while validation families are held out within development datasets.
No label content is created or corrected by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.corpus_contract import reference_commands, reference_identity
from training.review_contract import canonical_json, sha256_file


def semantic_family(record: dict[str, Any]) -> str:
    payload = {
        "dataset_id": record["dataset_id"],
        "decision": record["expected_decision"],
        "plan": reference_identity(record),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stable_bucket(value: str, modulus: int = 10000) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) % modulus


def assign_splits(
    records: list[dict[str, Any]], *, test_datasets: set[str], validation_fraction: float = 0.15
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5")
    if any(record.get("label_status") != "human-reviewed" for record in records):
        raise ValueError("Only human-reviewed records may enter the release corpus")
    known_datasets = {record["dataset_id"] for record in records}
    unknown = test_datasets - known_datasets
    if unknown:
        raise ValueError(f"Unknown test dataset id(s): {sorted(unknown)}")
    if not test_datasets or test_datasets == known_datasets:
        raise ValueError("Test datasets must be a non-empty strict subset of reviewed datasets")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[semantic_family(record)].append(record)

    output: list[dict[str, Any]] = []
    family_splits: dict[str, str] = {}
    threshold = int(validation_fraction * 10000)
    for family_id, members in sorted(groups.items()):
        datasets = {member["dataset_id"] for member in members}
        if len(datasets) != 1:
            raise ValueError(f"Family {family_id} unexpectedly spans datasets: {sorted(datasets)}")
        dataset = next(iter(datasets))
        if dataset in test_datasets:
            split = "test"
        else:
            split = "validation" if stable_bucket(family_id) < threshold else "train"
        family_splits[family_id] = split
        for member in members:
            output.append({**member, "split": split, "task_family_id": family_id})

    output.sort(key=lambda record: (record["split"], record["case_id"]))
    tool_counts = {}
    for split in ("train", "validation", "test"):
        tool_counts[split] = dict(Counter(
            command["tool"]
            for record in output if record["split"] == split
            for command in reference_commands(record)
        ))
    split_counts = dict(Counter(record["split"] for record in output))
    missing_test_tools = sorted(set(tool_counts["train"]) - set(tool_counts["test"]))
    blockers = []
    for split in ("train", "validation", "test"):
        if not split_counts.get(split):
            blockers.append(f"{split} split is empty")
    if missing_test_tools:
        blockers.append("held-out test datasets do not cover tools: " + ", ".join(missing_test_tools))
    report = {
        "schema_version": "datapilot-family-disjoint-split-v1",
        "record_count": len(output),
        "family_count": len(groups),
        "test_datasets": sorted(test_datasets),
        "development_datasets": sorted(known_datasets - test_datasets),
        "split_counts": split_counts,
        "tool_counts": tool_counts,
        "missing_test_tools": missing_test_tools,
        "family_overlap": 0,
        "dataset_overlap_with_test": 0,
        "release_split_ready": not blockers,
        "blockers": blockers,
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-corpus", type=Path, required=True)
    parser.add_argument("--test-dataset", action="append", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite split or audit evidence")
    records = [json.loads(line) for line in args.approved_corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    output, report = assign_splits(
        records, test_datasets=set(args.test_dataset), validation_fraction=args.validation_fraction
    )
    report["approved_corpus_sha256"] = sha256_file(args.approved_corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    report["output_corpus_sha256"] = sha256_file(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["release_split_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
