from types import SimpleNamespace

from training.prepare_corpus_review import prepare


def registry():
    return SimpleNamespace(list_tool_specs=lambda: [
        SimpleNamespace(name='group_data', category='group_aggregate'),
        SimpleNamespace(name='train_model', category='auto_ml')])


def example(case_id, split):
    return {'case_id': case_id, 'dataset_id': 'public', 'split': split,
            'query': 'Average measure by segment', 'label_status': 'contract-verified',
            'expected_command': {'tool': 'group_data', 'arguments': {'metric': 'measure'}}}


def test_review_groups_overlap_without_promoting_variants():
    records = [example('one', 'train'), example('two', 'validation')]
    report = prepare(records, registry(), [{'case_id': 'one', 'passed': True}])
    task = report['tasks'][0]
    assert task['priority'] == 1 and len(task['variants']) == 2
    assert task['review']['status'] == 'pending'
    assert task['execution_evidence'] == [{'case_id': 'one', 'passed': True}]
    assert all(r['label_status'] == 'contract-verified' for r in records)
    assert all(c['needs_independent_test_cases'] for c in report['coverage'])


def test_execution_failures_take_priority():
    report = prepare([example('one', 'train')], registry(),
                     [{'case_id': 'one', 'passed': False, 'error': 'incorrect chart'}])
    assert report['tasks'][0]['priority'] == 0
    assert 'EXECUTION_OR_RESULT_CONTRACT_FAILURE' in report['tasks'][0]['reasons']


def test_different_datasets_are_not_merged():
    first, second = example('one', 'train'), example('two', 'test')
    second['dataset_id'] = 'other'
    report = prepare([first, second], registry())
    assert report['reference_family_count'] == 2
