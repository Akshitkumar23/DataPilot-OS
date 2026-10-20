"""Build a scoped, numerically checked curriculum from observed public data.

Questions are programmatically authored, not observed human requests. No label
is marked human-reviewed. This builder never modifies the existing corpus.
"""
import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import pandas as pd
from training.planner_prompt import fingerprint
from training.corpus_contract import reference_family
from training.verify_numeric_oracles import reference_sql, assert_equivalent
from training.reference_execution import execute_reference
from scripts.validate_planner_corpus import DEFAULT_PROVENANCE, validate_records

TOOLS = ['filter_data', 'aggregate_data', 'group_data']


def make_record(dataset, commands, fields, text, family):
    identity = fingerprint([dataset, commands])
    return {'schema_version': 'planner-sft-v2', 'case_id': 'pilot-' + identity[:24],
            'split': 'train', 'dataset_id': dataset, 'query': text, 'required_columns': sorted(set(fields)),
            'expected_decision': 'tool', 'expected_commands': commands,
            'task_family_id': fingerprint([dataset, family]),
            'messages': [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': json.dumps({'mode': 'tool', 'commands': commands})}],
            'label_status': 'contract-verified', 'query_origin': 'programmatically-authored',
            'review_required': True}


def candidates(dataset, df):
    nums = [c for c in df.select_dtypes(include='number') if df[c].nunique() > 2]
    dims = [c for c in df if 2 <= df[c].nunique() <= 60]
    for metric, dimension in itertools.product(nums, dims):
        if metric == dimension:
            continue
        for agg in ('mean', 'sum', 'min', 'max', 'median', 'count'):
            text = f'Group by `{dimension}` and compute {agg} of `{metric}` for each group. Return a table; no chart.'
            cmd = {'tool': 'group_data', 'arguments': {'group_by': [dimension], 'metric': metric, 'aggregation': agg, 'chart_type': 'none'}}
            yield make_record(dataset, [cmd], [metric, dimension], text, ['group', metric, dimension])
    for filtered, metric in itertools.product(nums, nums):
        thresholds = sorted(set(float(df[filtered].quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)))
        for threshold, agg in itertools.product(thresholds, ('mean', 'sum', 'min', 'max', 'median', 'count')):
            filt = {'column': filtered, 'operator': '>', 'value': threshold}
            text = f'Keep rows where `{filtered}` > {threshold!r}, then compute {agg} of `{metric}` on those rows. Return a table; no chart.'
            commands = [{'tool': 'filter_data', 'arguments': filt},
                        {'tool': 'aggregate_data', 'arguments': {'column': metric, 'aggregation': agg}}]
            yield make_record(dataset, commands, [filtered, metric], text, ['filtered_metric', filtered, metric])
            # Alternate representation is kept in the SAME split family.
            single = [{'tool': 'aggregate_data', 'arguments': {'column': metric, 'aggregation': agg,
                       'filters': [{'column': filtered, 'condition': '>', 'value': threshold}]}}]
            direct = f'Compute {agg} of `{metric}` restricted to rows satisfying `{filtered}` > {threshold!r}. Return one table.'
            yield make_record(dataset, single, [filtered, metric], direct, ['filtered_metric', filtered, metric])
    for col in nums:
        for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            threshold = float(df[col].quantile(q))
            for op in ('>', '<'):
                cmd = {'tool': 'filter_data', 'arguments': {'column': col, 'operator': op, 'value': threshold}}
                yield make_record(dataset, [cmd], [col], f'Return all rows with `{col}` {op} {threshold!r}. Do not truncate.', ['slice', col])


def choose(pool, count):
    # Balanced across dataset and reference family, deterministic, no duplicate plans.
    buckets = {}
    for r in sorted(pool, key=lambda r: r['case_id']):
        buckets.setdefault((r['dataset_id'], reference_family(r)), []).append(r)
    result = []
    while len(result) < count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(result) < count:
                result.append(buckets[key].pop())
    if len(result) < count:
        raise ValueError(f'Insufficient distinct plans: {len(result)} < {count}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a prior pilot')
    provenance = json.loads(DEFAULT_PROVENANCE.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / str(provenance['manifest']).replace('\\', '/')).read_text(encoding='utf-8'))
    observed = {d['id'] for d in manifest['datasets'] if d.get('data_kind') == 'observed_public'}
    # Fixed source-level holdout, shared with the pre-existing benchmark policy.
    test_ids = {'uci_bike_sharing_hourly', 'uci_online_retail'}
    frames, hashes, pools = {}, {}, {'train': [], 'validation': [], 'test': []}
    for spec in provenance['datasets']:
        if spec['id'] not in observed:
            continue
        path = ROOT / str(spec['file']).replace('\\', '/')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != spec['sha256']:
            raise ValueError('Dataset integrity failure: ' + spec['id'])
        did = spec['id']
        frames[did], hashes[did] = pd.read_csv(path), actual
        seen = set()
        for r in candidates(did, frames[did]):
            if r['case_id'] in seen:
                continue
            seen.add(r['case_id'])
            split = 'test' if did in test_ids else 'validation' if int(r['task_family_id'][:8], 16) % 7 == 0 else 'train'
            r['split'] = split
            pools[split].append(r)
    records = sum((choose(pools[s], n) for s, n in [('train', 1050), ('validation', 150), ('test', 240)]), [])
    validate_records(records, provenance)
    args.output.mkdir(parents=True)
    evidence = []
    for index, r in enumerate(records):
        expected = frames[r['dataset_id']]
        trace = []
        with duckdb.connect() as con:
            for cmd in r['expected_commands']:
                sql, params = reference_sql(cmd, expected.columns)
                con.register('source_data', expected)
                expected = con.execute(sql, params).df()
                trace.append({'sql': sql, 'parameters': params})
        outputs = execute_reference(r, frames[r['dataset_id']])
        assert_equivalent(outputs[-1].dataframe, expected)
        result_hash = fingerprint(json.loads(expected.to_json(orient='split', index=False)))
        r['verification'] = {'kind': 'independent-numeric-plan', 'query_semantics_verified': False,
                             'source_sha256': hashes[r['dataset_id']], 'result_sha256': result_hash}
        evidence.append({'case_id': r['case_id'], 'passed': True, 'sql_trace': trace, 'result_sha256': result_hash})
        if (index + 1) % 25 == 0:
            print(f'Independent verification {index + 1}/{len(records)}', flush=True)
    corpus = args.output / 'pilot.jsonl'
    corpus.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n', encoding='utf-8')
    policy = {'kind': 'scoped-numeric-research-pilot-v1', 'allowed_tools': TOOLS,
              'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
              'source_hashes': hashes, 'record_count': len(records), 'numeric_verified': len(evidence),
              'human_reviewed': 0, 'query_origin': 'programmatically-authored',
              'split_counts': dict(Counter(r['split'] for r in records)),
              'deployment_allowed': False, 'limitation': 'Numeric reference-plan verification is not semantic or human review.'}
    (args.output / 'pilot-policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    (args.output / 'numeric-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(policy, indent=2))


if __name__ == '__main__':
    main()
