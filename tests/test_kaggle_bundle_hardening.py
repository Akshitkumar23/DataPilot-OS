import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from scripts.build_kaggle_training_bundle import copy_source_tree, notebook_payload
from training.audit_corpus_quality import audit
from training.planner_prompt import build_prompt, check_token_budget
from training.evaluate_planner_adapter import compare_record, evaluation_messages
from training.train_planner_adapter import completion_dataset


class Args(BaseModel):
    column: str
    aggregation: str = 'mean'


class Registry:
    def list_tool_specs(self):
        return [SimpleNamespace(name='summarize', description='Summarize a column', input_model=Args)]

    def validate_tool_call(self, tool, arguments):
        if tool != 'summarize':
            raise ValueError('Unknown tool')
        return Args.model_validate(arguments)


def record(split='train'):
    expected = {'tool': 'summarize', 'arguments': {'column': 'measure'}}
    return {'dataset_id': 'observed', 'split': split, 'query': 'Average measure',
            'expected_decision': 'execute', 'expected_command': expected,
            'required_columns': ['measure'], 'label_status': 'contract-verified',
            'messages': [{'role': 'user', 'content': 'Average measure'},
                         {'role': 'assistant', 'content': json.dumps({'mode': 'tool', 'commands': [expected]})}]}


def test_shared_prompt_has_schema_and_contracts_without_gold_leakage():
    r = record()
    provenance = {'datasets': [{'id': 'observed', 'columns': ['measure', 'segment']}]}
    prompt = build_prompt(r, provenance, Registry())
    changed = copy.deepcopy(r)
    changed['expected_command']['arguments']['column'] = 'SECRET_ANSWER'
    changed['required_columns'] = ['SECRET_ANSWER']
    assert build_prompt(changed, provenance, Registry()) == prompt
    assert 'segment' in prompt[0]['content'] and 'summarize' in prompt[0]['content']
    dataset = completion_dataset([prompt + r['messages'][-1:]])
    assert dataset['prompt'][0] == evaluation_messages(r, provenance, Registry())


def test_empty_registry_and_truncation_are_not_silently_accepted():
    with pytest.raises(ValueError, match='REGISTRY_UNAVAILABLE'):
        build_prompt(record(), {'datasets': [{'id': 'observed', 'columns': []}]},
                     SimpleNamespace(list_tool_specs=lambda: []))
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: list(range(20)))
    with pytest.raises(ValueError, match='PROMPT_TOO_LONG'):
        check_token_budget(tokenizer, [], 10)


def test_canonical_defaults_not_spurious_failure():
    r = record()
    command = copy.deepcopy(r['expected_command'])
    command['arguments']['aggregation'] = 'mean'
    score = compare_record(r, {'mode': 'tool', 'commands': [command]}, Registry())
    assert score['passed'] and not score['raw_exact_plan']


def test_literal_string_is_not_proof_of_correct_column_binding():
    command = {'tool': 'summarize', 'arguments': {'column': 'other', 'aggregation': 'measure'}}
    score = compare_record(record(), {'mode': 'tool', 'commands': [command]}, Registry())
    assert not score['columns_grounded'] and not score['passed']


def test_paraphrases_do_not_count_as_independent_plans():
    records = [record(), record('validation'), record('test')]
    report = audit(records)
    assert report['train_validation_family_overlap'] == 1
    assert not report['ready_for_new_training']
    assert report['splits']['train']['human_reviewed'] == 0


def test_notebook_has_valid_cells_and_no_automatic_export():
    nb = notebook_payload('safe-bundle')
    assert len({c['id'] for c in nb['cells']}) == len(nb['cells'])
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            assert 'outputs' not in cell and 'execution_count' not in cell
        else:
            assert 'export_adapter_for_ollama.py' not in ''.join(cell['source'])
            compile(''.join(cell['source']), '<notebook>', 'exec')
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert code.index('audit_corpus_quality.py') < code.index("nvidia-smi")
    assert code.index('audit_corpus_quality.py') < code.index('pip install')


def test_source_copy_excludes_secret_and_non_source_files(tmp_path):
    src = tmp_path / 'source'
    src.mkdir()
    (src / 'app.py').write_text('pass')
    (src / '.env_token').write_text('test-fixture-not-a-real-secret')
    (src / 'credentials.json').write_text('{}')
    copy_source_tree(src, tmp_path / 'out')
    assert {p.name for p in (tmp_path / 'out').iterdir()} == {'app.py'}


def test_numerical_pilot_notebook_is_fail_closed_and_token_measured():
    nb = notebook_payload('verified-curriculum', pilot=True)
    code_cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for source in code_cells:
        compile(source, '<pilot-notebook>', 'exec')
    code = '\n'.join(code_cells)
    assert 'subprocess.run(args, check=True)' in code
    assert code.index('kaggle_preflight.py') < code.index("'--max-steps', '2'") < code.index("'--epochs', '2'")
    assert 'corpus_file_sha256' in code and 'policy_file_sha256' in code
    assert "payload['adapter_files'] == current_files" in code
    assert "report['runtime_contract'] == runtime" in code
    assert 'export_adapter_for_ollama.py' not in code
    assert '--resume' in code and '--report-only' in code
    assert "OLLAMA_PLANNER_MEMORY_ENABLED'] = '0'" in code


def test_live_planner_schema_accepts_canonical_group_metric_and_record_count():
    from jsonschema import Draft202012Validator
    from modules.capability_registry import CapabilityRegistry
    registry = CapabilityRegistry(auto_bootstrap=True)
    spec = next(item for item in registry.list_tool_specs() if item.name == 'group_data')
    schema = registry.get_dynamic_command_json_schema([spec], allow_command_sequence=False)
    validator = Draft202012Validator(schema)
    metric_count = {'tool': 'group_data', 'arguments': {'group_by': ['segment'],
                    'metric': 'revenue', 'aggregation': 'count', 'chart_type': 'none'}}
    record_count = {'tool': 'group_data', 'arguments': {'group_by': ['segment'],
                    'aggregation': 'count', 'chart_type': 'bar'}}
    assert not list(validator.iter_errors(metric_count))
    assert not list(validator.iter_errors(record_count))
    registry.validate_tool_call('group_data', metric_count['arguments'])
    registry.validate_tool_call('group_data', record_count['arguments'])


def test_curriculum_derivatives_cannot_inherit_human_approval(monkeypatch):
    from training import build_kaggle_curriculum as curriculum
    source = {'case_id': 'heldout', 'dataset_id': 'unseen', 'task_family_id': 'family-1',
              'schema_version': 'planner-sft-v2',
              'split': 'test', 'human_reviewed': True, 'query': 'source',
              'messages': [{'role': 'user', 'content': 'source'}],
              'expected_commands': [{'tool': 'filter_data', 'arguments': {}}]}
    parents = [{'case_id': language, 'source_record': {'task_family': 'filter_aggregate'},
                'language': language, 'human_review': {'status': 'approved'}}
               for language in ['hi', 'hinglish']]
    monkeypatch.setattr(curriculum, '_balanced', lambda rows, n: rows)
    monkeypatch.setattr(curriculum, '_render', lambda row, language: language + ' request')
    result = curriculum.assemble([source], parents)
    assert len(result) == 3
    assert all(row['human_reviewed'] is False for row in result)
    assert all(row['split'] == 'test' and row['task_family_id'] == 'family-1' for row in result)
    assert source['human_reviewed'] is True
    assert all(row['review_required'] and row['parent_archetype_sha256'] for row in result[1:])
