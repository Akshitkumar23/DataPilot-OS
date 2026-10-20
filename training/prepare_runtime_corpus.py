"""Prepare SFT examples against the actual desktop planner's first-call contract.

This migration rejects unsupported targets and records retrieval/schema gaps;
it never alters a reference plan to make it fit. It does not assign review status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from time import perf_counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.planner_prompt_contract import render_system_prompt, render_user_prompt
from modules.planner_prompt_contract import eligible_tools, render_runtime_system
from training.corpus_contract import reference_commands
from training.planner_prompt import fingerprint


def reference_target(record, registry):
    commands = reference_commands(record)
    if not commands:
        response = json.loads(record['messages'][-1]['content'])
        reason = response.get('clarification')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('MISSING_CLARIFICATION_REFERENCE')
        specs = [s for s in registry.list_tool_specs() if s.category == 'clarification']
        if len(specs) != 1:
            raise ValueError('CLARIFICATION_CONTRACT_UNAVAILABLE')
        return {'tool': specs[0].name, 'arguments': {'reason': reason}}
    if len(commands) > 8:
        raise ValueError('INVALID_PLAN_SEQUENCE: maximum 8 steps')
    return commands[0] if len(commands) == 1 else {'commands': commands}


def prepare_record(record, frame, registry):
    from jsonschema import Draft202012Validator
    # Retrieval sees only the request, exactly as the production first attempt.
    specs = registry.select_planner_tools(record['query'], has_dataset=frame is not None)
    available = eligible_tools(registry, frame is not None)
    schema = registry.get_dynamic_command_json_schema(specs, allow_command_sequence=True)
    messages = [
        {'role': 'system', 'content': render_runtime_system(specs, available, record['query'])},
        {'role': 'user', 'content': render_user_prompt(
            record['query'], frame, record.get('conversation_history'), record.get('context'))},
    ]
    target = reference_target(record, registry)
    calls = target.get('commands', [target])
    if any(call['tool'] not in {s.name for s in specs} for call in calls):
        raise ValueError(f"RETRIEVAL_MISS: reference tools {[c['tool'] for c in calls]!r} absent from "
                         f"production retrieval {[s.name for s in specs]!r}")
    errors = sorted(Draft202012Validator(schema).iter_errors(target), key=lambda e: str(e.path))
    if errors:
        raise ValueError('PLANNER_TARGET_SCHEMA_MISMATCH: ' + errors[0].message)
    for call in calls:
        registry.validate_tool_call(call['tool'], call['arguments'])
    return {
        'case_id': record['case_id'],
        'dataset_id': record['dataset_id'],
        'source_record_sha256': fingerprint(record),
        'prompt_sha256': fingerprint(messages),
        'response_schema_sha256': fingerprint(schema),
        'response_schema': schema,
        'selected_tools': [s.name for s in specs],
        'available_tools': [s.name for s in available],
        'reference_in_shortlist': all(c['tool'] in {s.name for s in specs} for c in calls),
        'prompt': messages,
        'completion': [{'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False)}],
        'split': record.get('split'),
        'label_status': record.get('label_status'),
        'human_reviewed': record.get('human_reviewed', False),
        'limits': ['First-attempt prompt; registry and memory configuration must match deployment.',
                   'Schema acceptance is not semantic or numerical verification.'],
    }


class RuntimeCorpusBuilder:
    """Cache hash-checked public frames and construct answer-independent inputs."""
    def __init__(self, provenance, registry):
        self.datasets = {d['id']: d for d in provenance['datasets']}
        self.registry = registry
        self.frames = {}

    def frame(self, record):
        import pandas as pd
        key = record['dataset_id']
        if key not in self.frames:
            spec = self.datasets[key]
            path = (ROOT / spec['file'].replace('\\', '/')).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('SOURCE_PATH_OUTSIDE_PROJECT')
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError('SOURCE_HASH_MISMATCH')
            self.frames[key] = pd.read_csv(path)
        return self.frames[key]

    def inputs(self, record):
        frame = self.frame(record)
        specs = self.registry.select_planner_tools(record['query'], has_dataset=True)
        return [
            {'role': 'system', 'content': render_runtime_system(specs, eligible_tools(self.registry, True), record['query'])},
            {'role': 'user', 'content': render_user_prompt(record['query'], frame,
                record.get('conversation_history'), record.get('context'))},
        ]

    def example(self, record):
        return prepare_record(record, self.frame(record), self.registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, default=ROOT / 'data/real_benchmarks/provenance.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve migration evidence')
    import pandas as pd
    from modules.capability_registry import CapabilityRegistry
    registry = CapabilityRegistry(auto_bootstrap=True)
    provenance = json.loads(args.provenance.read_text(encoding='utf-8'))
    datasets = {d['id']: d for d in provenance['datasets']}
    rows = [json.loads(line) for line in args.corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
    frames, accepted, rejected = {}, [], []
    for row in rows:
        started = perf_counter()
        try:
            key = row['dataset_id']
            if key not in frames:
                spec = datasets[key]
                path = (ROOT / spec['file'].replace('\\', '/')).resolve()
                if not path.is_relative_to(ROOT):
                    raise ValueError('SOURCE_PATH_OUTSIDE_PROJECT')
                if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                    raise ValueError('SOURCE_HASH_MISMATCH')
                frames[key] = pd.read_csv(path)
            accepted.append(prepare_record(row, frames[key], registry))
        except Exception as exc:
            rejected.append({'case_id': row.get('case_id'), 'error_type': type(exc).__name__, 'reason': str(exc)})
        print(f"Prepared/checked {len(accepted) + len(rejected)}/{len(rows)}: {row.get('case_id')} "
              f"({perf_counter() - started:.2f}s)", flush=True)
    args.output.mkdir(parents=True)
    (args.output / 'prepared.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in accepted), encoding='utf-8')
    report = {'total': len(rows), 'prepared': len(accepted), 'rejected': rejected,
              'training_ready': False, 'reason': 'Migration diagnostic; split/review/scientific gates remain required.',
              'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              'renderer_sha256': hashlib.sha256((ROOT / 'modules/planner_prompt_contract.py').read_bytes()).hexdigest()}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'total': len(rows), 'prepared': len(accepted), 'rejected': len(rejected), 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
