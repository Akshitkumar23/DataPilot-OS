import pytest

from training.build_family_disjoint_corpus import assign_splits, semantic_family


def record(case_id, dataset, tool='group_data'):
    return {
        'schema_version': 'planner-sft-v1', 'case_id': case_id,
        'dataset_id': dataset, 'query': f'query {case_id}',
        'required_columns': ['value'], 'expected_decision': 'tool',
        'expected_command': {'tool': tool, 'arguments': {'metric': 'value'}},
        'label_status': 'human-reviewed', 'split': 'train',
    }


def test_family_variants_never_cross_splits_and_test_dataset_is_held_out():
    first = record('a', 'development')
    second = {**first, 'case_id': 'b', 'query': 'a paraphrase'}
    test = record('c', 'heldout')
    rows, report = assign_splits([first, second, test], test_datasets={'heldout'}, validation_fraction=0.2)
    assigned = {row['case_id']: row['split'] for row in rows}
    assert assigned['a'] == assigned['b']
    assert assigned['c'] == 'test'
    assert report['dataset_overlap_with_test'] == 0


def test_unreviewed_rows_are_refused():
    row = record('a', 'development')
    row['label_status'] = 'contract-verified'
    with pytest.raises(ValueError, match='human-reviewed'):
        assign_splits([row], test_datasets={'development'})
