"""Execute representative labelled plans against their real source CSVs.

This verifier bypasses the LLM entirely: it feeds the already-labelled ToolSpec
call through the same resolver, V2 intent mapper, profiler, and deterministic
typed operators used by DataPilot.  It therefore distinguishes a merely valid
JSON label from one that produces a real typed result.  It never edits labels
or promotes a model; human review remains required.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ds_engine.command_resolver import CommandResolver
from backend.ds_engine.dataset_profiler import DatasetProfiler
from backend.ds_engine.typed_operators import TypedOperatorRegistry
from backend.ds_engine.result_validator import SemanticResultValidator
from modules.query_orchestrator import _map_intent_to_v2
from scripts.validate_planner_corpus import DEFAULT_PROVENANCE, load_jsonl, validate_records
from training.corpus_contract import reference_commands, reference_family
from training.reference_execution import execute_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "training" / "planner_sft_contract_verified.jsonl")
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--samples-per-tool", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero if a selected executable plan fails.")
    args = parser.parse_args()
    if args.samples_per_tool < 1:
        raise ValueError("--samples-per-tool must be at least 1")
    records = load_jsonl(args.corpus)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    validate_records(records, provenance)
    files = {row["id"]: ROOT / Path(str(row["file"]).replace("\\", "/")) for row in provenance["datasets"]}
    selected: list[dict[str, Any]] = []
    per_tool: Counter[str] = Counter()
    seen_commands: set[str] = set()
    for record in records:
        if record["split"] != args.split or record["expected_decision"] != "tool":
            continue
        call = reference_commands(record)
        tool = reference_family(record)
        signature = json.dumps([record['dataset_id'], call], sort_keys=True)
        if signature in seen_commands or per_tool[tool] >= args.samples_per_tool:
            continue
        selected.append(record)
        per_tool[tool] += 1
        seen_commands.add(signature)

    dataframe_cache: dict[str, pd.DataFrame] = {}
    profile_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for record in selected:
        dataset_id = record["dataset_id"]
        if dataset_id not in dataframe_cache:
            dataframe_cache[dataset_id] = pd.read_csv(files[dataset_id])
            profile_cache[dataset_id] = DatasetProfiler.profile(dataframe_cache[dataset_id])
        df = dataframe_cache[dataset_id].copy(deep=True)
        profile = profile_cache[dataset_id]
        started = perf_counter()
        try:
            outputs = execute_reference(record, df)
            output = outputs[-1]
            manifest = output.manifest.model_dump() if hasattr(output.manifest, "model_dump") else dict(output.manifest)
            results.append({"case_id": record["case_id"], "tool": reference_family(record), "dataset_id": dataset_id, "passed": True, "latency_ms": round((perf_counter() - started) * 1000, 2), "manifest": manifest,
                            'step_manifests': [o.manifest.model_dump(mode='json') for o in outputs]})
        except Exception as exc:
            results.append({"case_id": record["case_id"], "tool": reference_family(record), "dataset_id": dataset_id, "passed": False, "latency_ms": round((perf_counter() - started) * 1000, 2), "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "status": "execution_evidence_only",
        "split": args.split,
        "samples_per_tool": args.samples_per_tool,
        "selected_count": len(results),
        "passed_count": sum(item["passed"] for item in results),
        "failed_count": sum(not item["passed"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("selected_count", "passed_count", "failed_count")}, indent=2))
    return 0 if not args.enforce or (payload['selected_count'] > 0 and payload["failed_count"] == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
