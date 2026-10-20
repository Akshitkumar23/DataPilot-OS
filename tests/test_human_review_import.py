import csv
import json
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path

import pytest

from training.import_human_reviews import apply_reviews, load_reviews, verify_review_binding
from training.review_contract import REVIEW_SCHEMA_VERSION, batch_content_sha256, record_sha256, sha256_file


def review(status='approved'):
    return {'case_id': 'c1', 'review_status': status, 'reviewer': 'A. Reviewer',
            'reviewer_role': 'data scientist', 'rationale': 'The query and plan agree.',
            'independent_expected_result': 'One verified grouped table.',
            'reviewed_at_utc': '2026-09-12T12:00:00+00:00'}


def test_only_approved_record_becomes_human_reviewed():
    accepted, rejected = apply_reviews([{'case_id': 'c1', 'label_status': 'contract-verified'}], {'c1': review()})
    assert accepted[0]['label_status'] == 'human-reviewed'
    assert accepted[0]['human_reviewed'] is True
    assert accepted[0]['human_review']['reviewer_role'] == 'data scientist'
    assert rejected == []


def test_rejection_is_preserved_outside_training():
    accepted, rejected = apply_reviews([{'case_id': 'c1'}], {'c1': review('rejected')})
    assert accepted == [] and rejected[0]['human_review']['status'] == 'rejected'
    assert rejected[0]['human_reviewed'] is False


def test_review_dimension_and_binding_evidence_are_preserved():
    row = {
        **review(),
        'semantic_intent_check': 'pass',
        'grounding_check': 'pass',
        'method_check': 'not_applicable',
        'expected_behavior_check': 'pass',
        'artifact_check': 'pass',
        'language_naturalness_check': 'pass',
        'dependency_check': 'not_applicable',
        'safety_governance_check': 'pass',
        'review_schema_version': REVIEW_SCHEMA_VERSION,
        'review_batch_id': 'batch-1',
        'corpus_sha256': 'corpus-hash',
        'record_sha256': 'record-hash',
    }
    accepted, rejected = apply_reviews([{'case_id': 'c1'}], {'c1': row})
    assert rejected == []
    evidence = accepted[0]['human_review']
    assert evidence['dimensions']['semantic_intent_check'] == 'pass'
    assert evidence['dimensions']['method_check'] == 'not_applicable'
    assert evidence['evidence_binding'] == {
        'review_schema_version': REVIEW_SCHEMA_VERSION,
        'review_batch_id': 'batch-1',
        'corpus_sha256': 'corpus-hash',
        'record_sha256': 'record-hash',
    }


def test_review_must_cover_exact_corpus():
    with pytest.raises(ValueError, match='REVIEW_COVERAGE_MISMATCH'):
        apply_reviews([{'case_id': 'c1'}, {'case_id': 'c2'}], {'c1': review()})


def test_future_review_timestamp_is_rejected(tmp_path):
    row = review()
    row['reviewed_at_utc'] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    path = tmp_path / 'reviews.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader(); writer.writerow(row)
    with pytest.raises(ValueError, match='in the future'):
        load_reviews(path)


def test_review_binding_rejects_changed_query(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    records = [{'schema_version': 'planner-sft-v1', 'case_id': 'c1', 'dataset_id': 'd1',
                'query': 'original', 'required_columns': [], 'expected_decision': 'clarification'}]
    corpus.write_text(json.dumps(records[0]) + '\n', encoding='utf-8')
    row = {**review(), 'review_schema_version': REVIEW_SCHEMA_VERSION,
           'review_batch_id': 'batch', 'corpus_sha256': sha256_file(corpus),
           'record_sha256': record_sha256(records[0]), 'query': 'tampered'}
    reviews = {'c1': row}
    manifest = tmp_path / 'review_manifest.json'
    manifest.write_text(json.dumps({
        'review_schema_version': REVIEW_SCHEMA_VERSION, 'review_batch_id': 'batch',
        'corpus_sha256': sha256_file(corpus), 'record_count': 1,
        'record_hashes': {'c1': record_sha256(records[0])},
        'review_sheet_content_sha256': batch_content_sha256([row]),
    }), encoding='utf-8')
    with pytest.raises(ValueError, match='displayed query'):
        verify_review_binding(corpus, records, tmp_path / 'reviews.csv', reviews, manifest)
