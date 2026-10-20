"""Re-execute a frozen research corpus against SQL without relabeling its authorship."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.corpus_contract import reference_commands
from training.planner_prompt import fingerprint


def run(corpus, provenance, output):
    import duckdb
    import pandas as pd
    from training.reference_execution import execute_reference
    from training.verify_numeric_oracles import reference_sql, assert_equivalent
    from training.runtime_identity import runtime_identity
    if output.exists():
        raise FileExistsError(output)
    records = [json.loads(line) for line in corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
    datasets = {d['id']: d for d in json.loads(provenance.read_text(encoding='utf-8'))['datasets']}
    frames, cache, results = {}, {}, []
    for i, row in enumerate(records):
        did = row['dataset_id']
        if did not in frames:
            source = ROOT / datasets[did]['file'].replace('\\', '/')
            if hashlib.sha256(source.read_bytes()).hexdigest() != datasets[did]['sha256']:
                raise ValueError('Dataset source hash mismatch: ' + did)
            frames[did] = pd.read_csv(source)
        commands = reference_commands(row)
        key = fingerprint([did, commands])
        if key not in cache:
            try:
                expected = frames[did]
                with duckdb.connect() as con:
                    for call in commands:
                        sql, params = reference_sql(call, expected.columns)
                        con.register('source_data', expected)
                        expected = con.execute(sql, params).df()
                observed = execute_reference(row, frames[did])[-1]
                assert_equivalent(observed.dataframe, expected)
                cache[key] = {'passed': True}
            except Exception as exc:
                cache[key] = {'passed': False, 'error': str(exc)}
        results.append({'case_id': row['case_id'], 'plan_sha256': key, **cache[key]})
        if (i + 1) % 50 == 0:
            print(f'Revalidated {i+1}/{len(records)}; distinct plans={len(cache)}', flush=True)
    report = {'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
              'runtime_identity': runtime_identity(), 'records': len(records),
              'distinct_plans': len(cache), 'passed': sum(r['passed'] for r in results),
              'failures': [r for r in results if not r['passed']],
              'query_semantics_verified': False, 'deployment_allowed': False,
              'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'runtime_identity')}, indent=2))
    if report['failures']:
        raise ValueError('Runtime numerical revalidation failed; inspect receipt')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--provenance', type=Path, default=ROOT/'data/real_benchmarks/provenance.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.corpus, a.provenance, a.output)
