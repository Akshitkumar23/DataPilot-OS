"""Validate planner SFT labels against the live registry and pinned dataset schemas.

This is deliberately an offline curation gate.  It never changes the active
Ollama model, writes a Modelfile, or alters runtime query routing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Running ``python scripts/validate_planner_corpus.py`` makes ``scripts`` the
# import root, unlike pytest.  Add the repository root explicitly so the same
# command works from a checkout, a venv, and CI.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry
from training.corpus_contract import reference_commands


DEFAULT_CORPUS = ROOT / "training" / "planner_sft_seed.jsonl"
DEFAULT_PROVENANCE = ROOT / "data" / "real_benchmarks" / "provenance.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: every record must be an object")
        value["_line_number"] = line_number
        records.append(value)
    if not records:
        raise ValueError(f"{path}: corpus is empty")
    return records


def validate_records(records: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    # A bare registry is intentionally empty in the app until its owner
    # bootstraps it.  The curation gate needs the actual canonical contracts.
    registry = CapabilityRegistry(auto_bootstrap=True)
    datasets = {
        row["id"]: set(row["columns"])
        for row in provenance.get("datasets", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("columns"), list)
    }
    case_ids: set[str] = set()
    errors: list[str] = []

    for record in records:
        where = f"line {record.get('_line_number', '?')}"
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{where}: missing case_id")
            continue
        if case_id in case_ids:
            errors.append(f"{where}: duplicate case_id '{case_id}'")
        case_ids.add(case_id)
        if record.get("schema_version") not in {"planner-sft-v1", "planner-sft-v2"}:
            errors.append(f"{where} ({case_id}): unsupported schema_version")
        if record.get("split") not in {"train", "validation", "test"}:
            errors.append(f"{where} ({case_id}): split must be train, validation, or test")
        dataset_id = record.get("dataset_id")
        if dataset_id not in datasets:
            errors.append(f"{where} ({case_id}): dataset_id '{dataset_id}' is absent from provenance")
            continue
        required_columns = record.get("required_columns")
        if not isinstance(required_columns, list) or not all(isinstance(c, str) for c in required_columns):
            errors.append(f"{where} ({case_id}): required_columns must be a string list")
        else:
            missing = sorted(set(required_columns) - datasets[dataset_id])
            if missing:
                errors.append(f"{where} ({case_id}): unknown required columns: {missing}")
        messages = record.get("messages")
        query = record.get('query')
        if not isinstance(query, str) or not query.strip():
            errors.append(f'{where} ({case_id}): query must be a non-empty string')
        if not isinstance(messages, list) or len(messages) != 2:
            errors.append(f"{where} ({case_id}): exactly one user and one assistant message are required")
        elif [item.get("role") for item in messages if isinstance(item, dict)] != ["user", "assistant"]:
            errors.append(f"{where} ({case_id}): messages must be user then assistant")
        elif not all(isinstance(item.get("content"), str) and item["content"].strip() for item in messages):
            errors.append(f"{where} ({case_id}): each message needs non-empty string content")
        elif messages[0]['content'] != query:
            errors.append(f'{where} ({case_id}): query must exactly match the user message used to author the label')

        outcome = record.get("expected_decision")
        if outcome == "clarification":
            if "expected_command" in record or 'expected_commands' in record:
                errors.append(f"{where} ({case_id}): clarification labels cannot contain expected_command")
            if isinstance(messages, list) and len(messages) == 2 and isinstance(messages[1], dict):
                try:
                    response = json.loads(messages[1]["content"])
                    if (not isinstance(response, dict) or response.get("mode") != "clarify"
                            or not isinstance(response.get("clarification"), str)
                            or not response['clarification'].strip() or response.get('commands')):
                        errors.append(f"{where} ({case_id}): clarification assistant label must contain mode=clarify and text")
                except (KeyError, TypeError, json.JSONDecodeError):
                    errors.append(f"{where} ({case_id}): clarification assistant label is not JSON")
            continue
        if outcome != "tool":
            errors.append(f"{where} ({case_id}): expected_decision must be tool or clarification")
            continue
        try:
            expected_commands = reference_commands(record)
        except ValueError as exc:
            errors.append(f'{where} ({case_id}): {exc}')
            continue
        for command in expected_commands:
            try:
                registry.validate_tool_call(command["tool"], command["arguments"])
            except Exception as exc:  # Present all malformed labels in one pass.
                errors.append(f"{where} ({case_id}): invalid {command['tool']} contract: {exc}")
        if isinstance(messages, list) and len(messages) == 2 and isinstance(messages[1], dict):
            try:
                response = json.loads(messages[1]["content"])
                commands = response.get("commands") if isinstance(response, dict) else None
                if not isinstance(response, dict) or response.get("mode") != "tool" or commands != expected_commands:
                    errors.append(f"{where} ({case_id}): assistant label must exactly match expected_command")
            except (KeyError, TypeError, json.JSONDecodeError):
                errors.append(f"{where} ({case_id}): tool assistant label is not JSON")

    if errors:
        raise ValueError("Planner corpus validation failed:\n- " + "\n- ".join(errors))
    present_splits = {record.get("split") for record in records}
    missing_splits = {"train", "validation", "test"} - present_splits
    if missing_splits:
        raise ValueError("Planner corpus needs non-empty split(s): " + ", ".join(sorted(missing_splits)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()
    records = load_jsonl(args.corpus)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    validate_records(records, provenance)
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    print(f"VALID corpus: {len(records)} records; " + ", ".join(f"{split}={count}" for split, count in sorted(split_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
