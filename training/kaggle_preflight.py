"""Validate a complete bundle and its actual tokenized runtime prompts before GPU work."""
from __future__ import annotations
import argparse
import json
import math
import hashlib
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(corpus, provenance_path, policy_path, output, model, revision, tokenize=True):
    if output.exists():
        raise FileExistsError(output)
    from scripts.validate_planner_corpus import load_jsonl, validate_records
    from training.pilot_policy import validate_pilot
    from training.audit_corpus_quality import audit
    from training.prepare_runtime_corpus import RuntimeCorpusBuilder
    from modules.capability_registry import CapabilityRegistry
    from training.runtime_identity import runtime_identity
    records = load_jsonl(corpus)
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    validate_records(records, provenance)
    policy = validate_pilot(records, corpus, policy_path)
    quality = audit(records)
    if not quality['ready_for_new_training']:
        raise ValueError(quality['blockers'])
    registry = CapabilityRegistry(auto_bootstrap=True)
    builder = RuntimeCorpusBuilder(provenance, registry)
    tokenizer = None
    if tokenize:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    examples, lengths, prompt_lengths, completion_lengths, failures = [], [], [], [], []
    for index, row in enumerate(records):
        try:
            ex = builder.example(row)
            if tokenizer:
                turns = ex['prompt'] + ex['completion']
                n = len(tokenizer.apply_chat_template(turns, tokenize=True))
                p = len(tokenizer.apply_chat_template(ex['prompt'], tokenize=True, add_generation_prompt=True))
                lengths.append(n)
                prompt_lengths.append(p)
                completion_lengths.append(n-p)
            examples.append({k: ex[k] for k in ('case_id', 'split', 'prompt_sha256', 'source_record_sha256', 'response_schema_sha256', 'reference_in_shortlist')})
        except Exception as exc:
            failures.append({'case_id': row['case_id'], 'reason': str(exc)})
        if (index + 1) % 100 == 0:
            print(f'Runtime prompt/schema checked {index+1}/{len(records)}', flush=True)
    max_new = max(384, math.ceil((max(completion_lengths, default=0)+64)/64)*64)
    required = max(max(lengths, default=0), max(prompt_lengths, default=0)+max_new)
    ceiling = math.ceil(required/256)*256 if tokenizer else None
    if ceiling and ceiling > 16384:
        failures.append({'reason': f'Full input needs {ceiling} tokens; exceeds measured-run limit 16384. No truncation allowed.'})
    report = {'schema_version': 'datapilot-kaggle-preflight-v1', 'records': len(records),
              'schema_valid': len(examples), 'failures': failures, 'quality': quality,
              'language_counts': dict(Counter(r.get('language', 'en') for r in records)),
              'tokenizer_checked': bool(tokenizer), 'max_seq_length': ceiling,
              'max_new_tokens': max_new, 'max_observed_tokens': max(lengths, default=None),
              'base_model': model, 'base_revision': revision,
              'corpus_file_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
              'policy_file_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
              'runtime_contract': runtime_identity(), 'case_evidence': examples,
              'training_scope': policy['allowed_tools'], 'deployment_allowed': False,
              'ready_for_gpu_smoke': not failures and bool(tokenizer)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'case_evidence', 'quality', 'runtime_contract'}}, indent=2))
    if failures:
        raise ValueError(f'PREFLIGHT_FAILED: {len(failures)} cases/issues; see {output}')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=ROOT/'training/planner_sft_seed.jsonl')
    p.add_argument('--provenance', type=Path, default=ROOT/'data/real_benchmarks/provenance.json')
    p.add_argument('--pilot-policy', type=Path, default=ROOT/'training/pilot-policy.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--base-revision', default='main')
    p.add_argument('--skip-tokenizer', action='store_true', help='Local schema audit only; cannot authorize GPU training.')
    a = p.parse_args()
    run(a.corpus, a.provenance, a.pilot_policy, a.output, a.base_model, a.base_revision, not a.skip_tokenizer)
