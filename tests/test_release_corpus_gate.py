from types import SimpleNamespace

import training.release_corpus_gate as gate


class Registry:
    def __init__(self, *args, **kwargs):
        pass

    def list_tool_specs(self):
        return [SimpleNamespace(name='group_data', requires_dataset=True, category='analysis')]


def record(case_id, split, language='en', commands=None):
    commands = commands or [{'tool': 'group_data', 'arguments': {'group_by': ['g'], 'metric': 'm', 'aggregation': 'mean'}}]
    value = {
        'schema_version': 'planner-sft-v2' if len(commands) > 1 else 'planner-sft-v1',
        'case_id': case_id, 'dataset_id': 'test-data' if split == 'test' else 'dev-data',
        'query': case_id, 'required_columns': ['g', 'm'], 'expected_decision': 'tool',
        'label_status': 'human-reviewed', 'split': split, 'language': language,
        'language_provenance': 'human-authored',
    }
    if len(commands) > 1:
        value['expected_commands'] = commands
    else:
        value['expected_command'] = commands[0]
    return value


def test_release_gate_requires_human_language_and_multistep(monkeypatch):
    monkeypatch.setattr(gate, 'CapabilityRegistry', Registry)
    rows = [record('train', 'train'), record('validation', 'validation'), record('test', 'test')]
    report = gate.assess_release_corpus(rows, min_multistep=1, min_hindi=1, min_hinglish=1, min_test_families=1)
    assert not report['ready']
    assert any('multi-step' in item for item in report['blockers'])
    assert any('Hindi' in item for item in report['blockers'])


def test_release_gate_accepts_declared_coverage(monkeypatch):
    monkeypatch.setattr(gate, 'CapabilityRegistry', Registry)
    sequence = [
        {'tool': 'group_data', 'arguments': {'group_by': ['g'], 'metric': 'm', 'aggregation': 'mean'}},
        {'tool': 'group_data', 'arguments': {'group_by': ['g'], 'metric': 'm', 'aggregation': 'max'}},
    ]
    rows = [record('train', 'train', 'hi', sequence), record('validation', 'validation', 'hinglish'),
            record('test', 'test'), record('test-sequence', 'test', commands=sequence)]
    report = gate.assess_release_corpus(rows, min_multistep=1, min_hindi=1, min_hinglish=1, min_test_families=1)
    assert report['ready'], report['blockers']
