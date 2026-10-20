"""Produce evidence-linked curation tasks without changing or approving labels."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_prompt import fingerprint
from training.corpus_contract import reference_identity, reference_family
from training.review_contract import (
    REVIEW_SCHEMA_VERSION, batch_content_sha256, record_sha256, sha256_file,
)


def prepare(records, registry, evidence=()):
    groups = defaultdict(list)
    evidence_by_case = {r['case_id']: r for r in evidence}
    for record in records:
        command = reference_identity(record)
        # Keep clarification questions separate: equal missing-field labels do not
        # establish that two different questions are semantically equivalent.
        key = fingerprint([record['dataset_id'], command if command else record['query']])
        groups[key].append(record)
    tasks = []
    for family_id, members in sorted(groups.items()):
        splits = sorted({r['split'] for r in members})
        observations = [evidence_by_case[r['case_id']] for r in members if r['case_id'] in evidence_by_case]
        failed = [r for r in observations if not r['passed']]
        reasons = []
        if failed:
            reasons.append('EXECUTION_OR_RESULT_CONTRACT_FAILURE')
        if len(splits) > 1:
            reasons.append('CROSS_SPLIT_REFERENCE_PLAN_OVERLAP')
        if not observations:
            reasons.append('NO_EXECUTION_EVIDENCE')
        if any(r.get('label_status') != 'human-reviewed' for r in members):
            reasons.append('HUMAN_SEMANTIC_REVIEW_REQUIRED')
        first = members[0]
        tasks.append({
            'task_id': family_id, 'dataset_id': first['dataset_id'],
            'tool': reference_family(first),
            'reference_command': first.get('expected_command'),
            'reference_commands': first.get('expected_commands'),
            'splits': splits, 'priority': 0 if failed else 1 if len(splits) > 1 else 2,
            'reasons': reasons,
            'variants': [{'case_id': r['case_id'], 'query': r['query'], 'split': r['split'],
                          'label_status': r.get('label_status')} for r in members],
            'execution_evidence': observations,
            'review': {'status': 'pending', 'reviewer': None, 'rationale': None,
                       'independent_expected_result': None},
            'warning': 'Execution evidence applies only to listed case IDs. No variant is automatically approved.'})
    tasks.sort(key=lambda t: (t['priority'], t['tool'], t['task_id']))
    families = {s.name for s in registry.list_tool_specs()
                if s.category not in {'general_chat', 'clarify', 'clarification'}} | {'clarification'}
    coverage = []
    for family in sorted(families):
        counts = {split: sum(t['tool'] == family and split in t['splits'] for t in tasks)
                  for split in ('train', 'validation', 'test')}
        coverage.append({'tool': family, 'reference_families_by_split': counts,
                         'needs_independent_test_cases': counts['test'] == 0})
    return {'schema_version': 'corpus-review-tasks-v1', 'corpus_sha256': fingerprint(records),
            'record_count': len(records), 'reference_family_count': len(tasks),
            'task_count_by_priority': {str(p): sum(t['priority'] == p for t in tasks) for p in range(3)},
            'coverage': coverage, 'tasks': tasks,
            'invariant': 'No labels, splits or training approval are changed by this report.'}


def markdown(report):
    lines = ['# Corpus review worklist', '',
             f"Records: {report['record_count']}; reference families: {report['reference_family_count']}.", '',
             'Reference families are exact dataset/plan groups, not certified independent business problems.', '',
             '## Priorities', '',
             '- P0: inspect execution/result-contract failures; do not train on these until the cause is resolved.',
             '- P1: redesign overlapping splits; keep related variants together and curate independent held-out cases.',
             '- P2: verify query meaning, evidence, numerical ground truth and clarification behavior.', '',
             'A passing execution does not approve a query or all its paraphrases. Keep actual human review distinct.', '',
             '## Capability coverage', '',
             '| Tool | Train families | Validation families | Test families |',
             '|---|---:|---:|---:|']
    for row in report['coverage']:
        c = row['reference_families_by_split']
        lines.append(f"| {row['tool']} | {c['train']} | {c['validation']} | {c['test']} |")
    lines += ['', '## Execution failures', '']
    failures = [t for t in report['tasks'] if t['priority'] == 0]
    if not failures:
        lines.append('None in the supplied execution sample; unexecuted families remain unverified.')
    for task in failures:
        lines.append(f"- `{task['tool']}` — family `{task['task_id']}`")
        for item in task['execution_evidence']:
            if not item['passed']:
                lines.append(f"  - `{item['case_id']}`: {item.get('error', 'Execution failed')}")
    lines += ['', 'Full query variants, case IDs, reference plans and review fields are in review_tasks.json.',
              'Complete every row in variant_review_template.csv. Review identity is self-declared unless your workflow adds an external signature.',
              'Import only with import_human_reviews.py and the bound review_manifest.json; the importer rejects stale or edited immutable fields.',
              'Do not approve source labels based only on this report or automated execution.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing review workspace')
    from modules.capability_registry import CapabilityRegistry
    records = [json.loads(line) for line in args.corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
    observations = []
    for path in args.evidence:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, list):
            observations.extend(payload)
        elif isinstance(payload, dict) and isinstance(payload.get('results'), list):
            observations.extend(payload['results'])
        else:
            raise ValueError(f'Unsupported evidence payload in {path}: expected a list or an object with results[]')
    report = prepare(records, CapabilityRegistry(auto_bootstrap=True), observations)
    args.output.mkdir(parents=True)
    (args.output / 'review_tasks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'README.md').write_text(markdown(report), encoding='utf-8')
    # Human-editable review sheet. Keeping one row per exact dataset/plan
    # family prevents repeated paraphrases from inflating review counts.
    with (args.output / 'review_template.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['task_id', 'dataset_id', 'tool', 'priority', 'splits', 'reasons',
                  'representative_query', 'review_status', 'reviewer', 'rationale',
                  'independent_expected_result']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in report['tasks']:
            writer.writerow({
                'task_id': task['task_id'], 'dataset_id': task['dataset_id'],
                'tool': task['tool'], 'priority': task['priority'],
                'splits': '|'.join(task['splits']), 'reasons': '|'.join(task['reasons']),
                'representative_query': task['variants'][0]['query'],
                'review_status': 'pending', 'reviewer': '', 'rationale': '',
                'independent_expected_result': '',
            })
    corpus_sha256 = sha256_file(args.corpus)
    batch_id = f"{args.output.name}-{corpus_sha256[:12]}"
    variant_rows = []
    with (args.output / 'variant_review_template.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['review_schema_version', 'review_batch_id', 'corpus_sha256', 'record_sha256',
                  'case_id', 'task_id', 'dataset_id', 'tool', 'split', 'query',
                  'proposed_plan_json', 'review_status', 'reviewer', 'reviewer_role',
                  'rationale', 'independent_expected_result', 'reviewed_at_utc']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in report['tasks']:
            proposed = task.get('reference_commands') or task.get('reference_command') or {'clarification': True}
            for variant in task['variants']:
                source_record = next(r for r in records if r['case_id'] == variant['case_id'])
                row = {
                    'review_schema_version': REVIEW_SCHEMA_VERSION,
                    'review_batch_id': batch_id,
                    'corpus_sha256': corpus_sha256,
                    'record_sha256': record_sha256(source_record),
                    'case_id': variant['case_id'], 'task_id': task['task_id'],
                    'dataset_id': task['dataset_id'], 'tool': task['tool'],
                    'split': variant['split'], 'query': variant['query'],
                    'proposed_plan_json': json.dumps(proposed, ensure_ascii=False, separators=(',', ':')),
                    'review_status': 'pending', 'reviewer': '', 'reviewer_role': '',
                    'rationale': '', 'independent_expected_result': '', 'reviewed_at_utc': '',
                }
                writer.writerow(row)
                variant_rows.append(row)
    manifest = {
        'review_schema_version': REVIEW_SCHEMA_VERSION,
        'review_batch_id': batch_id,
        'source_corpus': str(args.corpus.resolve()),
        'corpus_sha256': corpus_sha256,
        'record_count': len(records),
        'record_hashes': {r['case_id']: record_sha256(r) for r in records},
        'review_sheet_content_sha256': batch_content_sha256(variant_rows),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'identity_assurance': 'Content integrity only; reviewer identity requires authenticated workflow or external signature.',
    }
    (args.output / 'review_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (args.output / 'reviewer_allowlist.example.json').write_text(
        json.dumps(["replace-with-exact-reviewer-name"], indent=2) + "\n", encoding='utf-8'
    )
    print(json.dumps({key: report[key] for key in ('record_count', 'reference_family_count', 'task_count_by_priority')}, indent=2))


if __name__ == '__main__':
    main()
