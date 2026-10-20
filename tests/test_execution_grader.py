import pandas as pd

from training.execution_grader import grade_execution


def _record():
    return {
        'case_id': 'execution-grade-1', 'schema_version': 'planner-sft-v2',
        'expected_decision': 'tool', 'query': 'Mean amount by segment.',
        'expected_commands': [{'tool': 'group_data', 'arguments': {
            'group_by': ['segment'], 'metric': 'amount', 'aggregation': 'mean',
            'chart_type': 'none', 'datetime_parses': [],
        }}],
    }


def test_execution_grader_compares_candidate_output_to_independent_sql():
    frame = pd.DataFrame({'segment': ['a', 'a', 'b'], 'amount': [2.0, 4.0, 10.0]})
    response = {'commands': _record()['expected_commands']}
    result = grade_execution(_record(), response, frame)
    assert result['execution_verified'] is True


def test_execution_grader_rejects_semantically_wrong_valid_plan():
    frame = pd.DataFrame({'segment': ['a', 'a', 'b'], 'amount': [2.0, 4.0, 10.0]})
    response = {'commands': [{'tool': 'group_data', 'arguments': {
        'group_by': ['segment'], 'metric': 'amount', 'aggregation': 'sum',
        'chart_type': 'none', 'datetime_parses': [],
    }}]}
    result = grade_execution(_record(), response, frame)
    assert result['execution_verified'] is False


def test_execution_grader_marks_unscored_families_explicitly():
    record = _record()
    record['expected_commands'] = [{'tool': 'forecast_data', 'arguments': {}}]
    result = grade_execution(record, {'commands': record['expected_commands']}, pd.DataFrame({'x': [1]}))
    assert result == {'execution_verified': None, 'execution_reason': 'independent_oracle_not_available'}
