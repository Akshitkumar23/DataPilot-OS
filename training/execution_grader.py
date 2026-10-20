"""Execute emitted plans for the SQL-backed curriculum; never count abstention as a pass."""
from copy import deepcopy

from training.corpus_contract import reference_commands


def grade_execution(record, response, frame):
    import duckdb
    from training.reference_execution import execute_reference
    from training.verify_numeric_oracles import reference_sql, assert_equivalent
    commands = response.get('commands', [response] if 'tool' in response else [])
    allowed = {'filter_data', 'group_data', 'aggregate_data'}
    if record.get('expected_decision') != 'tool':
        return {'execution_verified': None, 'execution_reason': 'clarification_requires_semantic_review'}
    if any(c['tool'] not in allowed for c in reference_commands(record)):
        return {'execution_verified': None, 'execution_reason': 'independent_oracle_not_available'}
    if not isinstance(commands, list) or not 1 <= len(commands) <= 8:
        return {'execution_verified': False, 'execution_reason': 'invalid_command_sequence'}
    if any(not isinstance(c, dict) or c.get('tool') not in allowed for c in commands):
        return {'execution_verified': False, 'execution_reason': 'emitted_operation_outside_evaluation_scope'}
    try:
        expected = frame
        with duckdb.connect() as con:
            for command in reference_commands(record):
                sql, params = reference_sql(command, expected.columns)
                con.register('source_data', expected)
                expected = con.execute(sql, params).df()
        candidate = deepcopy(record)
        candidate.pop('expected_command', None)
        candidate.update(schema_version='planner-sft-v2', expected_commands=commands)
        outputs = execute_reference(candidate, frame)
        assert_equivalent(outputs[-1].dataframe, expected)
        requested = reference_commands(record)[-1]['arguments'].get('chart_type', 'none')
        if requested != outputs[-1].manifest.chart_type:
            raise ValueError('Emitted plan did not preserve requested visualization')
        return {'execution_verified': True, 'execution_reason': 'typed_output_matches_independent_sql'}
    except Exception as exc:
        return {'execution_verified': False, 'execution_reason': str(exc)}
