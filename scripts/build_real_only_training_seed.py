"""Create an auditable real-public-only copy of a planner corpus.

The source corpus is never altered.  Records whose documented source is not
``observed_public`` are excluded rather than relabelled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.planner_training_readiness import load_data_kinds
from scripts.validate_planner_corpus import DEFAULT_CORPUS, DEFAULT_PROVENANCE, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "planner_sft_real_only_seed.jsonl")
    args = parser.parse_args()
    kinds = load_data_kinds(args.provenance)
    retained = [
        record for record in load_jsonl(args.corpus)
        if kinds.get(record.get("dataset_id")) == "observed_public"
    ]
    if not retained:
        raise RuntimeError("No observed-public records were found in the source corpus.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in retained) + "\n",
        encoding="utf-8",
    )
    excluded = sorted({record.get("dataset_id") for record in load_jsonl(args.corpus)} - {record.get("dataset_id") for record in retained})
    print(json.dumps({"retained": len(retained), "excluded_dataset_ids": excluded, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
