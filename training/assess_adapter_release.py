"""Compare held-out base/candidate adapter evaluations without deployment.

An eligible result is still advisory: this command never modifies Ollama or
DataPilot configuration.  It simply creates a signed-off artifact a human can
review before considering a separate export/import step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"{path} is not an adapter evaluation report")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--min-exact-plan-rate", type=float, default=0.85)
    parser.add_argument("--max-tool-match-regression", type=float, default=0.0)
    parser.add_argument("--max-schema-validity-regression", type=float, default=0.0)
    parser.add_argument("--min-execution-pass-rate", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--report-only', action='store_true', help='Write rejection as a decision, not a subprocess crash. Does not change eligibility.')
    args = parser.parse_args()
    base, candidate = load(args.baseline), load(args.candidate)
    if base.get("split") != candidate.get("split") or base.get("case_count") != candidate.get("case_count"):
        raise ValueError("Baseline and candidate must evaluate the same split and case count.")
    bm, cm = base["metrics"], candidate["metrics"]
    reasons: list[str] = []
    for label, report in (('baseline', base), ('candidate', candidate)):
        if report.get('status') != 'evaluation_only_unpromoted':
            reasons.append(f'{label} evaluation is incomplete or has an unrecognized status')
        rows = report.get('results', [])
        if not rows or len(rows) != report.get('case_count') or any(not r.get('case_id') or not r.get('prompt_sha256') for r in rows):
            reasons.append(f'{label} has incomplete case evidence')
    for key in ('corpus_sha256', 'prompt_version', 'base_model'):
        if not base.get(key) or base.get(key) != candidate.get(key):
            reasons.append(f'Missing or mismatched evaluation identity: {key}')
    for key in ('runtime_contract', 'execution_grader_version', 'generation_config'):
        if not base.get(key) or base.get(key) != candidate.get(key):
            reasons.append(f'Missing or mismatched evaluation identity: {key}')
    if base.get('smoke_only', True) or candidate.get('smoke_only', True):
        reasons.append('Smoke or legacy evaluation is not release evidence')
    if [r.get('case_id') for r in base.get('results', [])] != [r.get('case_id') for r in candidate.get('results', [])]:
        reasons.append('Evaluation case identities differ')
    if [r.get('prompt_sha256') for r in base.get('results', [])] != [r.get('prompt_sha256') for r in candidate.get('results', [])]:
        reasons.append('Evaluation prompts differ')
    required = set(candidate.get('required_test_families', []))
    if not required or required != set(base.get('required_test_families', [])) or not required <= set(candidate.get('per_tool', {})):
        reasons.append('Evaluation lacks broad capability and clarification coverage')
    if cm.get("passed_rate", 0.0) < args.min_exact_plan_rate:
        reasons.append(f"candidate exact-plan rate {cm.get('passed_rate', 0.0):.1%} is below {args.min_exact_plan_rate:.1%}")
    if cm.get("tool_match_rate", 0.0) < bm.get("tool_match_rate", 0.0) - args.max_tool_match_regression:
        reasons.append("candidate tool-match rate regressed")
    if cm.get("schema_valid_rate", 0.0) < bm.get("schema_valid_rate", 0.0) - args.max_schema_validity_regression:
        reasons.append("candidate schema-validity rate regressed")
    execution = candidate.get('execution_metrics', {})
    if not execution.get('evaluated'):
        reasons.append('Candidate has no independently executed output evidence')
    elif execution.get('pass_rate', 0.0) < args.min_execution_pass_rate:
        reasons.append(f"candidate execution pass rate {execution.get('pass_rate', 0.0):.1%} is below {args.min_execution_pass_rate:.1%}")
    languages = candidate.get('per_language', {})
    expected_languages = {'en', 'hi', 'hinglish'}
    if not expected_languages <= set(languages):
        reasons.append('Candidate evaluation lacks English, Hindi, and Hinglish slices')
    payload = {
        "eligible_for_manual_export_review": not reasons,
        "reasons": reasons,
        "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline), "metrics": bm},
        "candidate": {"path": str(args.candidate), "sha256": sha256(args.candidate), "metrics": cm},
        "invariant": "This decision does not promote or import a model into Ollama; human approval and execution-result evaluation remain mandatory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["eligible_for_manual_export_review"] or args.report_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
