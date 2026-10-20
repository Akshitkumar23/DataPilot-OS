import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from modules.capability_registry import CapabilityRegistry
from scripts.validate_planner_corpus import validate_records, DEFAULT_PROVENANCE
from training.corpus_contract import reference_commands
from training.evaluate_planner_adapter import compare_record
from training.reference_execution import execute_reference


def record(split='train'):
    commands = [
        {'tool': 'filter_data', 'arguments': {'column': 'age', 'operator': '>', 'value': 40}},
        {'tool': 'group_data', 'arguments': {'group_by': ['marital'], 'metric': 'balance', 'aggregation': 'mean', 'chart_type': 'bar'}},
    ]
    query = 'Keep customers older than 40, then calculate mean balance by marital status with a bar chart.'
    return {'schema_version': 'planner-sft-v2', 'case_id': split, 'split': split,
            'dataset_id': 'uci_bank_marketing', 'query': query, 'expected_decision': 'tool',
            'expected_commands': commands, 'required_columns': ['age', 'marital', 'balance'],
            'messages': [{'role': 'user', 'content': query},
                         {'role': 'assistant', 'content': json.dumps({'mode': 'tool', 'commands': commands})}]}


def test_validator_accepts_v2_and_preserves_order():
    records = [record(s) for s in ('train', 'validation', 'test')]
    provenance = json.loads(DEFAULT_PROVENANCE.read_text(encoding='utf-8'))
    validate_records(records, provenance)
    records[0]['messages'][1]['content'] = json.dumps({'mode': 'tool', 'commands': list(reversed(records[0]['expected_commands']))})
    with pytest.raises(ValueError, match='exactly match'):
        validate_records(records, provenance)


def test_evaluator_rejects_missing_extra_or_reordered_steps():
    r = record()
    registry = CapabilityRegistry(auto_bootstrap=True)
    commands = r['expected_commands']
    assert compare_record(r, {'mode': 'tool', 'commands': commands}, registry)['passed']
    for wrong in (commands[:1], commands + commands[:1], list(reversed(commands))):
        assert not compare_record(r, {'mode': 'tool', 'commands': wrong}, registry)['passed']


def test_v1_and_mixed_references_are_not_silently_reinterpreted():
    r = record()
    r['schema_version'] = 'planner-sft-v1'
    with pytest.raises(ValueError, match='requires planner-sft-v2'):
        reference_commands(r)
    r = record()
    r['expected_command'] = r['expected_commands'][0]
    with pytest.raises(ValueError, match='not both'):
        reference_commands(r)


def test_real_data_filter_is_carried_into_group_result():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / 'data/real_benchmarks/bank_marketing.csv')
    before = df.copy(deep=True)
    outputs = execute_reference(record(), df)
    expected = df.loc[df['age'] > 40].groupby('marital', dropna=False)['balance'].mean().reset_index()
    pd.testing.assert_frame_equal(outputs[-1].dataframe.sort_values('marital').reset_index(drop=True),
                                  expected.sort_values('marital').reset_index(drop=True), check_dtype=False)
    assert len(outputs) == 2
    assert len(outputs[0].dataframe) == int((df['age'] > 40).sum())
    pd.testing.assert_frame_equal(df, before)
