"""Independent SQL checks of supported reference plans; never human approval.

This is offline corpus tooling, not runtime routing or natural-language parsing.
Unsupported argument combinations are explicitly left unverified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from training.planner_prompt import fingerprint
from training.corpus_contract import reference_commands


def identifier(name, columns):
    if not isinstance(name, str) or name not in columns:
        raise ValueError(f'Unknown column: {name!r}')
    return '"' + name.replace('"', '""') + '"'


def reference_sql(command, columns):
    tool, args = command['tool'], command['arguments']
    allowed = {
        'filter_data': {'column', 'operator', 'value'},
        'group_data': {'group_by', 'metric', 'aggregation', 'chart_type', 'datetime_parses'},
        'aggregate_data': {'column', 'aggregation', 'filters'},
    }
    if tool not in allowed or set(args) - allowed[tool]:
        raise NotImplementedError('Oracle does not cover this tool/argument combination')
    if args.get('datetime_parses'):
        raise NotImplementedError('Datetime parsing requires its own independent oracle')
    params, conditions = [], []
    filters = [args] if tool == 'filter_data' else args.get('filters', [])
    for spec in filters:
        if tool != 'filter_data' and set(spec) - {'column', 'condition', 'operator', 'value'}:
            raise NotImplementedError('Unsupported predicate fields')
        op = spec.get('condition', spec.get('operator'))
        ops = {'==': '=', '=': '=', '!=': '<>', '>': '>', '>=': '>=', '<': '<', '<=': '<='}
        if op not in ops or spec.get('value') is None:
            raise NotImplementedError('Unsupported predicate semantics')
        conditions.append(f'{identifier(spec["column"], columns)} {ops[op]} ?')
        params.append(spec['value'])
    where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
    if tool == 'filter_data':
        return 'SELECT * FROM source_data' + where, params
    grouping = args.get('group_by', [])
    if not isinstance(grouping, list):
        raise NotImplementedError('Grouping must be an explicit list')
    metrics = args.get('metric', args.get('column'))
    metrics = [metrics] if isinstance(metrics, str) else metrics or []
    aggregation = args.get('aggregation')
    funcs = {'mean': 'AVG', 'sum': 'SUM', 'count': 'COUNT', 'min': 'MIN', 'max': 'MAX', 'median': 'MEDIAN'}
    if aggregation not in funcs:
        raise NotImplementedError('Oracle requires an explicit supported aggregation')
    dims = [identifier(c, columns) for c in grouping]
    if not metrics:
        if aggregation != 'count':
            raise NotImplementedError('A measure is required')
        measures = ['COUNT(*) AS count']
    else:
        measures = []
        for col in metrics:
            quoted = identifier(col, columns)
            expression = f'{funcs[aggregation]}({quoted})'
            # Match the documented Pandas sum identity for empty/all-null groups.
            if aggregation == 'sum':
                expression = f'COALESCE({expression}, 0)'
            measures.append(f'{expression} AS {quoted}')
    sql = 'SELECT ' + ', '.join(dims + measures) + ' FROM source_data' + where
    if dims:
        sql += ' GROUP BY ' + ', '.join(dims)
    return sql, params


def assert_equivalent(actual, expected):
    if actual is None or set(actual.columns) != set(expected.columns):
        raise AssertionError('Result column inventory differs from independent SQL')
    columns = list(expected.columns)
    def canonical(df):
        return df[columns].sort_values(columns, na_position='last').reset_index(drop=True)
    pd.testing.assert_frame_equal(canonical(actual), canonical(expected),
                                  check_dtype=False, check_exact=False, rtol=1e-8, atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite oracle evidence')
    import duckdb
    from backend.ds_engine.command_resolver import CommandResolver
    from backend.ds_engine.dataset_profiler import DatasetProfiler
    from backend.ds_engine.typed_operators import TypedOperatorRegistry
    from backend.ds_engine.result_validator import SemanticResultValidator
    from modules.query_orchestrator import _map_intent_to_v2
    from scripts.validate_planner_corpus import DEFAULT_PROVENANCE, validate_records, load_jsonl
    provenance = json.loads(DEFAULT_PROVENANCE.read_text(encoding='utf-8'))
    records = load_jsonl(args.corpus)
    validate_records(records, provenance)
    datasets = {r['id']: r for r in provenance['datasets']}
    cache, frames, profiles, results = {}, {}, {}, []
    for record in records:
        commands = reference_commands(record)
        if not commands:
            results.append({'case_id': record['case_id'], 'status': 'not_covered', 'reason': 'Clarification needs semantic review'})
            continue
        if len(commands) > 1:
            results.append({'case_id': record['case_id'], 'status': 'not_covered', 'reason': 'Sequential SQL oracle not implemented; no automatic approval'})
            continue
        command = commands[0]
        key = fingerprint([record['dataset_id'], command])
        if key not in cache:
            spec = datasets[record['dataset_id']]
            try:
                sql, params = reference_sql(command, spec['columns'])
            except NotImplementedError as exc:
                cache[key] = {'status': 'not_covered', 'reason': str(exc)}
            else:
                try:
                    did = record['dataset_id']
                    if did not in frames:
                        path = ROOT / str(spec['file']).replace('\\', '/')
                        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                        if actual_hash != spec['sha256']:
                            raise ValueError('Source dataset hash mismatch')
                        frames[did] = pd.read_csv(path)
                        profiles[did] = DatasetProfiler.profile(frames[did])
                    df = frames[did].copy(deep=True)
                    with duckdb.connect() as con:
                        con.register('source_data', df)
                        expected = con.execute(sql, params).df()
                    # Empty raw query prevents the numerical checker relying on
                    # the wording of one paraphrase. Semantic review is separate.
                    resolved = CommandResolver.resolve_command(command, df=df, raw_query='')
                    if not resolved.is_valid or not resolved.semantic_intent:
                        raise ValueError(f'Resolver rejected plan: {resolved.error_message}')
                    intent = _map_intent_to_v2(resolved.semantic_intent, '', df=df, validated_tool_args=command['arguments'])
                    output = TypedOperatorRegistry.execute_intent(df, intent, profiles[did])
                    if output.status != 'SUCCEEDED':
                        raise ValueError(f'Execution failed: {output.error_code}')
                    validation = SemanticResultValidator.validate_manifest(intent, output.manifest)
                    if not validation.is_valid:
                        raise ValueError(f'Manifest failure: {validation.issues}')
                    assert_equivalent(output.dataframe, expected)
                    cache[key] = {'status': 'numeric_plan_verified', 'sql': sql, 'parameters': params,
                                  'rows': len(expected), 'source_sha256': spec['sha256'],
                                  'result_sha256': fingerprint(json.loads(expected.to_json(orient='split', index=False))),
                                  'verified_plan_sha256': key}
                except Exception as exc:
                    cache[key] = {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
        results.append({'case_id': record['case_id'], 'dataset_id': record['dataset_id'], **cache[key],
                        'query_semantics_verified': False, 'human_reviewed': False})
    report = {'corpus_sha256': fingerprint(records), 'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'duckdb_version': duckdb.__version__, 'record_status_counts': dict(Counter(r['status'] for r in results)),
              'unique_plan_status_counts': dict(Counter(r['status'] for r in cache.values())), 'results': results,
              'scope': 'Independent numerical plan agreement only. No approval of query meaning, training readiness or human review.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('record_status_counts', 'unique_plan_status_counts')}, indent=2))


if __name__ == '__main__':
    main()
