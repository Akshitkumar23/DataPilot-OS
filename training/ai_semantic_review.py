"""Resumable, provenance-preserving semantic triage for corpus review tasks.

AI verdicts are evidence only. This script never writes `human-reviewed` and
never modifies the source corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.intelligence.ollama_service import OllamaRequest, ollama_service

VERDICT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['decision', 'checks', 'rationale', 'human_review_required'],
    'properties': {
        'decision': {'type': 'string', 'enum': ['accept', 'reject', 'uncertain']},
        'checks': {
            'type': 'object', 'additionalProperties': False,
            'required': ['task', 'columns', 'filters', 'aggregation', 'visualization', 'ambiguity'],
            'properties': {name: {'type': 'string', 'enum': ['pass', 'fail', 'not_applicable', 'uncertain']}
                           for name in ('task', 'columns', 'filters', 'aggregation', 'visualization', 'ambiguity')},
        },
        'rationale': {'type': 'string', 'minLength': 12},
        'human_review_required': {'type': 'boolean'},
    },
}

SYSTEM = """You are a skeptical data-science plan reviewer. Judge whether the candidate
tool plan faithfully answers the user request using only the supplied dataset schema.
Do not assume the candidate is correct. Check task type, exact columns, filters and
denominators, aggregation/time semantics, requested visualization, and whether the
request is genuinely underspecified. Mark uncertain when domain/time/leakage meaning
cannot be established from the text and schema. Return only schema-valid JSON."""


def normalize_verdict(verdict):
    checks = verdict.get('checks', {})
    decisive = bool(checks) and all(value in {'pass', 'not_applicable'} for value in checks.values())
    if verdict.get('decision') == 'accept' and (not decisive or verdict.get('human_review_required')):
        return {**verdict, 'decision': 'uncertain', 'human_review_required': True,
                'rationale': 'Inconsistent accept verdict downgraded: ' + verdict.get('rationale', '')}
    if verdict.get('decision') in {'reject', 'uncertain'}:
        verdict['human_review_required'] = True
    return verdict


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, default=ROOT / 'data/real_benchmarks/provenance.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='qwen2.5:3b-instruct')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    report = json.loads(args.tasks.read_text(encoding='utf-8'))
    datasets = {d['id']: d for d in json.loads(args.provenance.read_text(encoding='utf-8'))['datasets']}
    completed = {}
    if args.output.exists():
        for line in args.output.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row['task_id']] = row
    pending = [t for t in report['tasks'] if t['task_id'] not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a', encoding='utf-8') as handle:
        for index, task in enumerate(pending, 1):
            source = datasets[task['dataset_id']]
            payload = {
                'dataset_id': task['dataset_id'], 'columns': source['columns'],
                'queries': [v['query'] for v in task['variants'][:3]],
                'candidate_command': task.get('reference_command'),
                'candidate_commands': task.get('reference_commands'),
                'expected_kind': task['tool'],
            }
            response = ollama_service.structured_request(OllamaRequest(
                model=args.model, system_prompt=SYSTEM,
                prompt=json.dumps(payload, ensure_ascii=False),
                response_schema=VERDICT_SCHEMA, temperature=0.0, top_p=0.1,
                max_tokens=500, timeout_sec=180,
            ))
            if not response.success or not response.parsed_json:
                verdict = {'decision': 'uncertain', 'checks': {name: 'uncertain' for name in
                    ('task', 'columns', 'filters', 'aggregation', 'visualization', 'ambiguity')},
                    'rationale': response.error_message or 'Reviewer returned no structured verdict.',
                    'human_review_required': True}
            else:
                verdict = normalize_verdict(response.parsed_json)
            row = {'task_id': task['task_id'], 'input_sha256': fingerprint(payload),
                   'review_kind': 'ai-semantic-triage', 'model': response.model_used or args.model,
                   'latency_ms': response.latency_ms, **verdict,
                   'human_reviewed': False,
                   'warning': 'AI triage is not human review and cannot promote labels by itself.'}
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            print(f"Reviewed {index}/{len(pending)} {task['task_id']}: {row['decision']}", flush=True)


if __name__ == '__main__':
    main()
