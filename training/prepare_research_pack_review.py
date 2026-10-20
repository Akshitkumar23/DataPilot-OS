"""Create a cryptographically bound human-review workspace for a research pack."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.review_contract import REVIEW_SCHEMA_VERSION, batch_content_sha256, record_sha256, sha256_file
from training.validate_research_candidate_pack import validate_pack


DIMENSIONS = [
    'semantic_intent_check', 'grounding_check', 'method_check',
    'expected_behavior_check', 'artifact_check', 'language_naturalness_check',
    'dependency_check', 'safety_governance_check',
]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _normalise(record: dict, kind: str) -> dict:
    case_id = record.get('record_id') or record.get('workflow_id')
    query = record.get('query_template') or record.get('user_request')
    return {
        'schema_version': 'datapilot-research-review-candidate-v1',
        'case_id': case_id,
        'dataset_id': 'dataset-neutral-placeholder-contract',
        'query': query,
        'required_columns': record.get('dataset_schema_placeholders', []),
        'expected_decision': record.get('expected_behavior'),
        'expected_command': {'candidate_contract': record},
        'label_status': 'research-candidate-pending-review',
        'split': 'review',
        'language': record.get('language'),
        'review_lane': kind,
        'human_reviewed': False,
        'source_record': record,
    }


def build(pack_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite review workspace: {output}')
    intake = validate_pack(pack_dir)
    records = []
    for filename, kind in (
        ('hindi_candidates.jsonl', 'hindi_language_and_semantic'),
        ('hinglish_candidates.jsonl', 'hinglish_language_and_semantic'),
        ('multistep_candidates.jsonl', 'workflow_dependency_and_semantic'),
    ):
        records.extend(_normalise(row, kind) for row in _read_jsonl(pack_dir / filename))
    case_ids = [record['case_id'] for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError('REVIEW_WORKSPACE_DUPLICATE_CASE_ID')

    output.mkdir(parents=True)
    corpus_path = output / 'research_candidates.jsonl'
    corpus_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    corpus_hash = sha256_file(corpus_path)
    batch_id = f'{output.name}-{corpus_hash[:12]}'
    rows = []
    fields = [
        'review_schema_version', 'review_batch_id', 'corpus_sha256', 'record_sha256',
        'case_id', 'review_lane', 'language', 'task_family', 'expected_behavior', 'query',
        'candidate_contract_json', *DIMENSIONS, 'review_status', 'reviewer', 'reviewer_role',
        'rationale', 'independent_expected_result', 'reviewed_at_utc',
    ]
    with (output / 'variant_review_template.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            source = record['source_record']
            row = {
                'review_schema_version': REVIEW_SCHEMA_VERSION,
                'review_batch_id': batch_id,
                'corpus_sha256': corpus_hash,
                'record_sha256': record_sha256(record),
                'case_id': record['case_id'],
                'review_lane': record['review_lane'],
                'language': record['language'],
                'task_family': source.get('task_family'),
                'expected_behavior': source.get('expected_behavior'),
                'query': record['query'],
                'candidate_contract_json': json.dumps(source, ensure_ascii=False, separators=(',', ':')),
                **{dimension: '' for dimension in DIMENSIONS},
                'review_status': 'pending', 'reviewer': '', 'reviewer_role': '',
                'rationale': '', 'independent_expected_result': '', 'reviewed_at_utc': '',
            }
            writer.writerow(row)
            rows.append(row)

    manifest = {
        'review_schema_version': REVIEW_SCHEMA_VERSION,
        'review_batch_id': batch_id,
        'source_pack': str(pack_dir.resolve()),
        'source_pack_receipt': intake,
        'source_corpus': str(corpus_path.resolve()),
        'corpus_sha256': corpus_hash,
        'record_count': len(records),
        'record_hashes': {record['case_id']: record_sha256(record) for record in records},
        'review_sheet_content_sha256': batch_content_sha256(rows),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'identity_assurance': 'Content integrity only; reviewer identity requires an authenticated workflow or external signature.',
        'training_authorized': False,
    }
    (output / 'review_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'reviewer_allowlist.example.json').write_text('["replace-with-exact-reviewer-name"]\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# DataPilot research candidate human review\n\n'
        'This workspace contains 630 AI-assisted candidates. It does not contain human approvals.\n\n'
        'Complete every row in `variant_review_template.csv`. Each review dimension must be `pass`, `fail`, or '
        '`not_applicable`. Use `approved` only when no dimension fails; otherwise use `rejected` or `needs_changes`.\n\n'
        'A reviewer must independently check language naturalness, semantic intent, method, artifacts, expected '
        'clarification/unsupported behavior, dependency edges and safety. Fill reviewer, reviewer_role, rationale, '
        'independent_expected_result and a timezone-aware reviewed_at_utc.\n\n'
        'The manifest binds each displayed record to the frozen corpus. Do not edit query or contract columns. '
        'Run `training/import_human_reviews.py` only after completing every row. Human approval still does not make '
        'abstract placeholder candidates executable SFT records; approved archetypes must next be instantiated against '
        'real dataset schemas and verified by execution oracles.\n',
        encoding='utf-8',
    )
    return {'records': len(records), 'corpus_sha256': corpus_hash, 'review_batch_id': batch_id, 'training_authorized': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.pack_dir, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
