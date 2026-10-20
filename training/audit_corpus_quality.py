"""Measure corpus diversity independently of nominal record count; no labels are rewritten."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.planner_prompt import fingerprint
from training.corpus_contract import reference_identity, reference_family


def semantic_family(record):
    """Prefer the curation identity; derive the same exact plan family otherwise."""
    return record.get('task_family_id') or fingerprint([record['dataset_id'], reference_identity(record)])


def audit(records, *, min_test_families=20, require_dataset_disjoint=True):
    splits = {}
    signatures = {}
    for split in ('train', 'validation', 'test'):
        rows = [r for r in records if r['split'] == split]
        # Group by dataset and reference task: changing a prefix is not a new task.
        signatures[split] = {semantic_family(r) for r in rows}
        splits[split] = {'records': len(rows), 'unique_dataset_plans': len(signatures[split]),
                         'tools': dict(Counter(reference_family(r) for r in rows)),
                         'human_reviewed': sum(r.get('label_status') == 'human-reviewed' for r in rows)}
    overlap = len(signatures['train'] & signatures['validation'])
    test_overlap = len((signatures['train'] | signatures['validation']) & signatures['test'])
    development_datasets = {r['dataset_id'] for r in records if r['split'] in {'train', 'validation'}}
    test_datasets = {r['dataset_id'] for r in records if r['split'] == 'test'}
    dataset_overlap = sorted(development_datasets & test_datasets)
    missing = sorted(set(splits['train']['tools']) - set(splits['test']['tools']))
    blockers = []
    if overlap:
        blockers.append(f'{overlap} dataset-plan families shared by train and validation')
    if test_overlap:
        blockers.append(f'{test_overlap} test dataset-plan families seen in development splits')
    if require_dataset_disjoint and dataset_overlap:
        blockers.append('Test datasets overlap development datasets: ' + ', '.join(dataset_overlap))
    if missing:
        blockers.append('Test coverage missing: ' + ', '.join(missing))
    if splits['test']['unique_dataset_plans'] < min_test_families:
        blockers.append(f'Test contains fewer than {min_test_families} independent dataset-plan families')
    return {'splits': splits, 'train_validation_family_overlap': overlap,
            'development_test_family_overlap': test_overlap,
            'development_test_dataset_overlap': dataset_overlap,
            'missing_test_families': missing, 'ready_for_new_training': not blockers,
            'blockers': blockers, 'note': 'Contract verification is not human review or result correctness.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--enforce', action='store_true')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
    report = audit(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 2 if args.enforce and not report['ready_for_new_training'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
