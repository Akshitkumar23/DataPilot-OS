"""Explicit limited-scope research policy; does not authorize deployment."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from training.corpus_contract import reference_commands, reference_family
from training.planner_prompt import fingerprint


def validate_pilot(records, corpus, policy_path):
    policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    if policy.get('kind') != 'scoped-numeric-research-pilot-v1' or policy.get('deployment_allowed') is not False:
        raise ValueError('Invalid research-only pilot policy')
    if hashlib.sha256(Path(corpus).read_bytes()).hexdigest() != policy.get('corpus_sha256'):
        raise ValueError('Pilot corpus hash mismatch')
    allowed = set(policy.get('allowed_tools', []))
    if allowed != {'filter_data', 'aggregate_data', 'group_data'}:
        raise ValueError('Unsupported scope for this independently verified pilot')
    counts = Counter(r['split'] for r in records)
    if len({r['case_id'] for r in records}) != len(records):
        raise ValueError('Duplicate pilot case IDs')
    if policy.get('record_count') != len(records) or policy.get('split_counts') != dict(counts):
        raise ValueError('Pilot inventory/count mismatch')
    if any(counts[s] < n for s, n in [('train', 1000), ('validation', 100), ('test', 200)]):
        raise ValueError('Pilot split counts insufficient')
    families = {}
    for r in records:
        if not r.get('task_family_id'):
            raise ValueError('Missing task-family identity')
        families.setdefault(r['task_family_id'], set()).add(r['split'])
        if any(c['tool'] not in allowed for c in reference_commands(r)):
            raise ValueError('Out-of-scope command in pilot')
        v = r.get('verification', {})
        if v.get('kind') != 'independent-numeric-plan' or not v.get('result_sha256'):
            raise ValueError('Missing independent numeric evidence')
        if v.get('source_sha256') != policy.get('source_hashes', {}).get(r['dataset_id']):
            raise ValueError('Dataset evidence hash mismatch')
    if any(len(s) > 1 for s in families.values()):
        raise ValueError('Task family crosses splits')
    development = {r['dataset_id'] for r in records if r['split'] != 'test'}
    if development & {r['dataset_id'] for r in records if r['split'] == 'test'}:
        raise ValueError('Test dataset overlaps development')
    evidence = json.loads((Path(policy_path).parent / 'numeric-evidence.json').read_text(encoding='utf-8'))
    by_id = {e['case_id']: e for e in evidence}
    if set(by_id) != {r['case_id'] for r in records} or len(by_id) != len(evidence):
        raise ValueError('Evidence case inventory mismatch')
    for r in records:
        e = by_id[r['case_id']]
        if e.get('passed') is not True or e.get('result_sha256') != r['verification']['result_sha256']:
            raise ValueError('Case evidence mismatch')
        if policy.get('plan_binding_required'):
            expected = fingerprint([r['dataset_id'], reference_commands(r)])
            if e.get('plan_sha256') != expected or r['verification'].get('plan_sha256') != expected:
                raise ValueError('Plan evidence hash mismatch')
            if e.get('source_sha256') != r['verification']['source_sha256']:
                raise ValueError('Oracle source evidence mismatch')
    return policy


class ScopedRegistry:
    def __init__(self, registry, names):
        self.registry, self.names = registry, set(names)

    def list_tool_specs(self):
        return [s for s in self.registry.list_tool_specs() if s.name in self.names]

    def validate_tool_call(self, name, arguments):
        if name not in self.names:
            raise ValueError('Tool outside explicit research scope')
        return self.registry.validate_tool_call(name, arguments)
