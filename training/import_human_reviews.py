"""Import genuine record-level reviews without inventing approval provenance."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.review_contract import (
    REVIEW_SCHEMA_VERSION, batch_content_sha256, record_sha256, sha256_file,
)

ALLOWED = {'approved', 'rejected', 'needs_changes'}
ALLOWED_ROLES = {'data scientist', 'statistician', 'data analyst', 'ml engineer', 'domain expert'}
ALLOWED_ROLES.add('language reviewer')
MIN_RATIONALE_LENGTH = 20
MIN_EXPECTED_RESULT_LENGTH = 10
RESEARCH_REVIEW_DIMENSIONS = (
    'semantic_intent_check', 'grounding_check', 'method_check',
    'expected_behavior_check', 'artifact_check', 'language_naturalness_check',
    'dependency_check', 'safety_governance_check',
)
ALLOWED_DIMENSION_VERDICTS = {'pass', 'fail', 'not_applicable'}


def load_reviews(path, *, now=None, reviewer_allowlist=None):
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
    by_case = {}
    errors = []
    for line, row in enumerate(rows, 2):
        case_id = row.get('case_id', '').strip()
        status = row.get('review_status', '').strip().lower()
        if not case_id or case_id in by_case:
            errors.append(f'line {line}: missing or duplicate case_id')
            continue
        if status not in ALLOWED:
            errors.append(f'{case_id}: review_status must be approved, rejected, or needs_changes')
        for field in ('reviewer', 'reviewer_role', 'rationale', 'independent_expected_result', 'reviewed_at_utc'):
            if not row.get(field, '').strip():
                errors.append(f'{case_id}: {field} is required')
        if len(row.get('rationale', '').strip()) < MIN_RATIONALE_LENGTH:
            errors.append(f'{case_id}: rationale must contain at least {MIN_RATIONALE_LENGTH} characters')
        if len(row.get('independent_expected_result', '').strip()) < MIN_EXPECTED_RESULT_LENGTH:
            errors.append(f'{case_id}: independent_expected_result must contain at least {MIN_EXPECTED_RESULT_LENGTH} characters')
        role = row.get('reviewer_role', '').strip().casefold()
        if role and role not in ALLOWED_ROLES:
            errors.append(f"{case_id}: reviewer_role must be one of {sorted(ALLOWED_ROLES)}")
        reviewer = row.get('reviewer', '').strip().casefold()
        if reviewer_allowlist is not None and reviewer not in reviewer_allowlist:
            errors.append(f'{case_id}: reviewer is not present in the supplied allowlist')
        if any(field in row for field in RESEARCH_REVIEW_DIMENSIONS):
            verdicts = {field: row.get(field, '').strip().lower() for field in RESEARCH_REVIEW_DIMENSIONS}
            for field, verdict in verdicts.items():
                if verdict not in ALLOWED_DIMENSION_VERDICTS:
                    errors.append(f'{case_id}: {field} must be pass, fail, or not_applicable')
            if status == 'approved' and any(verdict == 'fail' for verdict in verdicts.values()):
                errors.append(f'{case_id}: approved review cannot contain a failed review dimension')
        try:
            timestamp = datetime.fromisoformat(row.get('reviewed_at_utc', '').replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                raise ValueError
            reference_now = now or datetime.now(timezone.utc)
            if timestamp > reference_now + timedelta(minutes=5):
                errors.append(f'{case_id}: reviewed_at_utc is in the future')
        except ValueError:
            errors.append(f'{case_id}: reviewed_at_utc must be timezone-aware ISO-8601')
        by_case[case_id] = row
    if errors:
        raise ValueError('INVALID_HUMAN_REVIEW:\n- ' + '\n- '.join(errors[:50]))
    return by_case


def verify_review_binding(corpus_path, records, review_path, reviews, manifest_path):
    if not manifest_path.exists():
        raise ValueError('REVIEW_MANIFEST_MISSING: a v2 review_manifest.json is required')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    corpus_hash = sha256_file(corpus_path)
    errors = []
    if manifest.get('review_schema_version') != REVIEW_SCHEMA_VERSION:
        errors.append('manifest review_schema_version mismatch')
    if manifest.get('corpus_sha256') != corpus_hash:
        errors.append('source corpus SHA-256 mismatch; this review batch is stale')
    if manifest.get('record_count') != len(records):
        errors.append('manifest record_count mismatch')
    record_hashes = manifest.get('record_hashes', {})
    for record in records:
        case_id = record['case_id']
        expected_hash = record_sha256(record)
        row = reviews.get(case_id, {})
        if record_hashes.get(case_id) != expected_hash:
            errors.append(f'{case_id}: manifest record fingerprint mismatch')
        if row.get('record_sha256', '').strip() != expected_hash:
            errors.append(f'{case_id}: review row record fingerprint mismatch')
        if row.get('corpus_sha256', '').strip() != corpus_hash:
            errors.append(f'{case_id}: review row corpus SHA-256 mismatch')
        if row.get('review_schema_version', '').strip() != REVIEW_SCHEMA_VERSION:
            errors.append(f'{case_id}: review schema version mismatch')
        if row.get('review_batch_id', '').strip() != manifest.get('review_batch_id'):
            errors.append(f'{case_id}: review batch id mismatch')
        if row.get('query', '').strip() != str(record.get('query', '')).strip():
            errors.append(f'{case_id}: displayed query no longer matches corpus')
    if errors:
        raise ValueError('STALE_OR_TAMPERED_REVIEW_BATCH:\n- ' + '\n- '.join(errors[:50]))
    # The digest covers immutable binding columns only, so reviewers can fill
    # verdict fields without invalidating the frozen source identity.
    if manifest.get('review_sheet_content_sha256') != batch_content_sha256(reviews.values()):
        raise ValueError('STALE_OR_TAMPERED_REVIEW_BATCH: immutable review-sheet binding columns changed')


def apply_reviews(records, reviews):
    case_ids = {r['case_id'] for r in records}
    if set(reviews) != case_ids:
        missing, extra = sorted(case_ids - set(reviews)), sorted(set(reviews) - case_ids)
        raise ValueError(f'REVIEW_COVERAGE_MISMATCH: missing={len(missing)} extra={len(extra)}')
    accepted, rejected = [], []
    for record in records:
        review = reviews[record['case_id']]
        dimensions = {
            field: review.get(field, '').strip().lower()
            for field in RESEARCH_REVIEW_DIMENSIONS
            if field in review
        }
        enriched = {**record, 'human_review': {
            'status': review['review_status'].strip().lower(),
            'reviewer': review['reviewer'].strip(), 'reviewer_role': review['reviewer_role'].strip(),
            'rationale': review['rationale'].strip(),
            'independent_expected_result': review['independent_expected_result'].strip(),
            'reviewed_at_utc': review['reviewed_at_utc'].strip(),
            'dimensions': dimensions,
            'evidence_binding': {
                'review_schema_version': review.get('review_schema_version', '').strip(),
                'review_batch_id': review.get('review_batch_id', '').strip(),
                'corpus_sha256': review.get('corpus_sha256', '').strip(),
                'record_sha256': review.get('record_sha256', '').strip(),
            },
        }}
        if enriched['human_review']['status'] == 'approved':
            enriched['label_status'] = 'human-reviewed'
            enriched['human_reviewed'] = True
            accepted.append(enriched)
        else:
            enriched['human_reviewed'] = False
            rejected.append(enriched)
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--approved-output', type=Path, required=True)
    parser.add_argument('--rejected-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Defaults to review_manifest.json beside the review CSV.')
    parser.add_argument('--reviewer-allowlist', type=Path, help='Optional JSON array of permitted reviewer names.')
    args = parser.parse_args()
    for path in (args.approved_output, args.rejected_output):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite review evidence: {path}')
    records = [json.loads(line) for line in args.corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
    allowlist = None
    if args.reviewer_allowlist:
        raw_allowlist = json.loads(args.reviewer_allowlist.read_text(encoding='utf-8'))
        if not isinstance(raw_allowlist, list) or not all(isinstance(v, str) and v.strip() for v in raw_allowlist):
            raise ValueError('Reviewer allowlist must be a JSON array of non-empty names')
        allowlist = {value.strip().casefold() for value in raw_allowlist}
    reviews = load_reviews(args.reviews, reviewer_allowlist=allowlist)
    verify_review_binding(args.corpus, records, args.reviews, reviews, args.manifest or args.reviews.with_name('review_manifest.json'))
    accepted, rejected = apply_reviews(records, reviews)
    for path, rows in ((args.approved_output, accepted), (args.rejected_output, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(json.dumps({'approved': len(accepted), 'rejected_or_needs_changes': len(rejected)}))


if __name__ == '__main__':
    main()
