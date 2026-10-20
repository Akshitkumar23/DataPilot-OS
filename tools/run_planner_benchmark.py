"""Run a reviewed DataPilot planner benchmark against local Ollama models.

This is an offline evaluator. It never calls an endpoint which changes the
active model. Supply a CSV and explicit JSON benchmark cases owned by the user.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# When executed as ``python tools/run_planner_benchmark.py``, Python adds the
# tools directory—not the repository root—to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.planner_benchmark import (
    BenchmarkCase,
    PlannerBenchmarkRunner,
    PromotionPolicy,
    evaluate_promotion,
)
from modules.semantic_planner import SemanticPlanner


def _load_cases(path: Path) -> list[BenchmarkCase]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Cases JSON must be an array of benchmark case objects.")
    return [BenchmarkCase.model_validate(item) for item in raw]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local Ollama planner models safely.")
    parser.add_argument("--dataset", required=True, type=Path, help="Validation CSV path")
    parser.add_argument("--cases", required=True, type=Path, help="Reviewed BenchmarkCase JSON array")
    parser.add_argument("--baseline-model", required=True, help="Current local Ollama planner model")
    parser.add_argument("--candidate-model", required=True, help="Candidate local Ollama planner model")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--max-pass-rate-regression", type=float, default=0.0)
    parser.add_argument("--max-p95-latency-ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path, help="Optional explicit JSON report output path")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    cases = _load_cases(args.cases)
    runner = PlannerBenchmarkRunner(SemanticPlanner.extract_semantic_intent)
    baseline = runner.run(cases, df, model_name=args.baseline_model, timeout_sec=args.timeout_sec)
    candidate = runner.run(cases, df, model_name=args.candidate_model, timeout_sec=args.timeout_sec)
    decision = evaluate_promotion(
        baseline,
        candidate,
        PromotionPolicy(
            min_pass_rate=args.min_pass_rate,
            max_pass_rate_regression=args.max_pass_rate_regression,
            max_p95_latency_ratio=args.max_p95_latency_ratio,
        ),
    )
    payload = {
        "baseline": baseline.model_dump(),
        "candidate": candidate.model_dump(),
        "promotion_decision": decision.model_dump(),
        "note": "This output is advisory only; no active model was changed.",
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if decision.eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
