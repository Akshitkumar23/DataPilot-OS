import csv
import json

import pytest

from training.import_human_reviews import load_reviews
from training.prepare_research_pack_review import build
from tests.test_research_candidate_pack import _pack


def test_review_workspace_binds_all_candidates(tmp_path):
    pack = tmp_path / 'pack'
    pack.mkdir()
    _pack(pack)
    output = tmp_path / 'review'
    result = build(pack, output)
    rows = list(csv.DictReader((output / 'variant_review_template.csv').open(encoding='utf-8-sig')))
    manifest = json.loads((output / 'review_manifest.json').read_text(encoding='utf-8'))
    assert result['records'] == 510
    assert len(rows) == 510 == manifest['record_count']
    assert result['training_authorized'] is False


def test_approved_research_review_cannot_hide_failed_dimension(tmp_path):
    path = tmp_path / 'reviews.csv'
    row = {
        'case_id': 'c1', 'review_status': 'approved', 'reviewer': 'Reviewer',
        'reviewer_role': 'data scientist', 'rationale': 'Checked the complete candidate contract.',
        'independent_expected_result': 'A structured clarification response.',
        'reviewed_at_utc': '2026-09-13T12:00:00+00:00',
        'semantic_intent_check': 'fail', 'grounding_check': 'pass', 'method_check': 'not_applicable',
        'expected_behavior_check': 'pass', 'artifact_check': 'pass', 'language_naturalness_check': 'pass',
        'dependency_check': 'not_applicable', 'safety_governance_check': 'pass',
    }
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader(); writer.writerow(row)
    with pytest.raises(ValueError, match='approved review cannot contain'):
        load_reviews(path)
