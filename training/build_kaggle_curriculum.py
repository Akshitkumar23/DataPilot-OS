"""Assemble and independently execute a multilingual numerical research curriculum.

Every new question is AI/programmatically authored. Review of a parent
archetype does not transfer human approval to the bound dataset example.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.build_multilingual_multistep_candidates import _balanced, _render
from training.corpus_contract import reference_commands
from training.planner_prompt import fingerprint
from training.pilot_policy import validate_pilot


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def assemble(source, archetypes):
    records = deepcopy(source)
    for row in records:
        row.update(language='en', human_reviewed=False, language_provenance='programmatically-authored')
    parents = {}
    for row in archetypes:
        if row.get('human_review', {}).get('status') == 'approved':
            parents.setdefault((row['source_record']['task_family'], row['language']), row)
    for row in _balanced(source, 120):
        for language in ('hi', 'hinglish'):
            parent = parents.get(('filter_aggregate', language))
            if parent is None:
                raise ValueError('Missing reviewed filter/aggregate archetype')
            query = _render(row, language)
            variant = deepcopy(row)
            variant.update(case_id=row['case_id'] + '-' + language, query=query, language=language,
                           language_provenance='ai-authored-derived-query', human_reviewed=False,
                           parent_archetype_id=parent['case_id'],
                           parent_archetype_sha256=fingerprint(parent),
                           source_case_id=row['case_id'], review_required=True,
                           label_status='contract-verified')
            variant['messages'][0]['content'] = query
            records.append(variant)
    # Explicit chart curriculum, balanced across existing source/split families.
    buckets = {}
    for row in source:
        calls = reference_commands(row)
        if len(calls) == 1 and calls[0]['tool'] == 'group_data':
            buckets.setdefault((row['dataset_id'], row['split']), []).append(row)
    for key, rows in sorted(buckets.items()):
        for row in sorted(rows, key=lambda r: r['case_id'])[:4]:
            for chart in ('bar', 'line'):
                variant = deepcopy(row)
                variant['expected_commands'][0]['arguments']['chart_type'] = chart
                query = row['query'].replace('Return a table; no chart.', f'Return a table and a {chart} chart.')
                if query == row['query']:
                    raise ValueError('Chart curriculum requires an explicit table-only source request')
                variant.update(case_id=row['case_id'] + '-' + chart, query=query, language='en',
                               language_provenance='programmatically-authored', human_reviewed=False,
                               source_case_id=row['case_id'], review_required=True)
                variant['messages'] = [{'role': 'user', 'content': query}, {'role': 'assistant', 'content':
                    json.dumps({'mode': 'tool', 'commands': variant['expected_commands']})}]
                records.append(variant)
    return records


def build(source_dir, approved, output):
    if output.exists():
        raise FileExistsError(output)
    source = read(source_dir / 'pilot.jsonl')
    validate_pilot(source, source_dir / 'pilot.jsonl', source_dir / 'pilot-policy.json')
    records = assemble(source, read(approved))
    from scripts.validate_planner_corpus import DEFAULT_PROVENANCE, validate_records
    from training.reference_execution import execute_reference
    from training.verify_numeric_oracles import reference_sql, assert_equivalent
    from training.audit_corpus_quality import audit
    import duckdb
    import pandas as pd
    provenance = json.loads(DEFAULT_PROVENANCE.read_text(encoding='utf-8'))
    validate_records(records, provenance)
    quality = audit(records)
    if not quality['ready_for_new_training']:
        raise ValueError(quality['blockers'])
    datasets = {d['id']: d for d in provenance['datasets']}
    frames, hashes, evidence, cache = {}, {}, [], {}
    output.mkdir(parents=True)
    for index, row in enumerate(records):
        did = row['dataset_id']
        if did not in frames:
            spec = datasets[did]
            path = ROOT / spec['file'].replace('\\', '/')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != spec['sha256']:
                raise ValueError('Public source content hash changed: ' + did)
            frames[did], hashes[did] = pd.read_csv(path), digest
        commands = reference_commands(row)
        plan_sha = fingerprint([did, commands])
        if plan_sha not in cache:
            expected = frames[did]
            trace = []
            with duckdb.connect() as con:
                for call in commands:
                    sql, params = reference_sql(call, expected.columns)
                    con.register('source_data', expected)
                    expected = con.execute(sql, params).df()
                    trace.append({'sql': sql, 'parameters': params})
            outputs = execute_reference(row, frames[did])
            assert_equivalent(outputs[-1].dataframe, expected)
            cache[plan_sha] = {'passed': True, 'plan_sha256': plan_sha,
                'source_sha256': hashes[did], 'sql_trace': trace,
                'result_sha256': fingerprint(json.loads(expected.to_json(orient='split', index=False))),
                'execution_manifest': outputs[-1].manifest.model_dump(mode='json')}
        checked = cache[plan_sha]
        row['verification'] = {'kind': 'independent-numeric-plan', 'query_semantics_verified': False,
                              'source_sha256': hashes[did], 'result_sha256': checked['result_sha256'],
                              'plan_sha256': plan_sha}
        evidence.append({'case_id': row['case_id'], **checked})
        if (index + 1) % 50 == 0:
            print(f'Oracle/manifest verified {index + 1}/{len(records)}', flush=True)
    corpus = output / 'pilot.jsonl'
    corpus.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    policy = {'kind': 'scoped-numeric-research-pilot-v1',
              'allowed_tools': ['filter_data', 'aggregate_data', 'group_data'],
              'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
              'source_hashes': hashes, 'record_count': len(records), 'numeric_verified': len(records),
              'human_reviewed': 0, 'deployment_allowed': False, 'plan_binding_required': True,
              'split_counts': dict(Counter(r['split'] for r in records)),
              'language_counts': dict(Counter(r['language'] for r in records)),
              'limitation': 'Research curriculum on public observed data; generated requests require semantic evaluation.'}
    for name, payload in [('pilot-policy.json', policy), ('numeric-evidence.json', evidence), ('quality.json', quality)]:
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    validate_pilot(records, corpus, output / 'pilot-policy.json')
    print(json.dumps(policy, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--approved-archetypes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source_dir, args.approved_archetypes, args.output)
