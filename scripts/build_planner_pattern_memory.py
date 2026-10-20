"""Compile reviewed planner labels into non-executable local retrieval memory.

The output deliberately redacts source column names and excludes argument values.
It can guide natural-language interpretation, but cannot provide a column,
filter value, time horizon, or executable tool command to the live planner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry
from scripts.validate_planner_corpus import DEFAULT_CORPUS, load_jsonl

DEFAULT_OUTPUT = ROOT / "training" / "planner_pattern_memory.jsonl"
APPROVED_LABELS = {"seed-reviewed", "human-reviewed"}


def redact_columns(query: str, columns: list[str]) -> str:
    result = query
    for column in sorted({column for column in columns if isinstance(column, str)}, key=len, reverse=True):
        result = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(column) + r"(?![A-Za-z0-9_])", "<column>", result, flags=re.IGNORECASE)
    return result


def compile_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = CapabilityRegistry(auto_bootstrap=True)
    memory: list[dict[str, Any]] = []
    for record in records:
        if record.get("label_status") not in APPROVED_LABELS:
            continue
        decision = record.get("expected_decision")
        command = record.get("expected_command") if decision == "tool" else None
        tool = command.get("tool") if isinstance(command, dict) else "clarification_needed"
        spec = registry.get_tool_spec(tool)
        if spec is None:
            raise ValueError(f"{record.get('case_id')}: tool '{tool}' is not registered")
        schema = spec.input_model.model_json_schema()
        fields = sorted(schema.get("properties", {}).keys())
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{record.get('case_id')}: query is required")
        memory.append({
            "schema_version": "planner-pattern-memory-v1",
            "case_id": record["case_id"],
            "review_status": record["label_status"],
            "tool": tool,
            "query_pattern": redact_columns(query, record.get("required_columns") or []),
            "argument_fields": fields,
        })
    if not memory:
        raise ValueError("No reviewed planner labels available for memory compilation")
    return memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    memory = compile_records(load_jsonl(args.corpus))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in memory) + "\n", encoding="utf-8")
    print(f"Wrote {len(memory)} reviewed, redacted planner patterns to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
