"""Refresh a tokenization receipt after a source-only runtime change.

Safe only when corpus, policy and pinned model revision are byte-identical to a
previous successful tokenizer preflight. It does not claim a new tokenizer run.
"""
import argparse, hashlib, json, sys
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(bundle, prior, output):
    bundle, prior, output = map(Path, (bundle, prior, output))
    report = json.loads(prior.read_text(encoding='utf-8'))
    corpus = bundle / 'training/planner_sft_seed.jsonl'
    policy = bundle / 'training/pilot-policy.json'
    if report.get('corpus_file_sha256','').lower() != sha(corpus).lower():
        raise ValueError('Prior preflight corpus differs; run tokenizer preflight again')
    if report.get('policy_file_sha256','').lower() != sha(policy).lower():
        raise ValueError('Prior preflight policy differs; run tokenizer preflight again')
    if report.get('ready_for_gpu_smoke') is not True or not report.get('tokenizer_checked'):
        raise ValueError('Prior receipt is not a successful tokenizer preflight')
    sys.path.insert(0, str(bundle))
    from training.runtime_identity import runtime_identity
    current_identity = runtime_identity()
    if report.get('refreshed_runtime_identity_only'):
        raise ValueError('A refreshed receipt is not original preflight evidence. Run kaggle_preflight.py again.')
    if report.get('runtime_contract') != current_identity:
        raise ValueError('Runtime changed. Fresh prompt/schema/tokenizer verification is required; fingerprints cannot be rewritten.')
    config = json.loads((bundle / 'RUN_CONFIG.json').read_text(encoding='utf-8'))
    if any(report.get(key) != config.get(key) for key in ('base_model', 'base_revision')):
        raise ValueError('Pinned model identity differs; run tokenizer preflight again')
    if output.exists():
        raise FileExistsError(output)
    report['refreshed_runtime_identity_only'] = False
    report['prior_preflight_sha256'] = sha(prior)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'records': report['records'], 'max_observed_tokens': report['max_observed_tokens'],
                      'ready_for_gpu_smoke': report['ready_for_gpu_smoke'],
                      'runtime_sha256': report['runtime_contract']['sha256']} , indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--bundle',required=True); p.add_argument('--prior',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); main(a.bundle,a.prior,a.output)
