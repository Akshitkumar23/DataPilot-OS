import duckdb
import pandas as pd
import pytest
from training.verify_numeric_oracles import reference_sql, assert_equivalent


def run_reference(command, frame):
    sql, params = reference_sql(command, frame.columns)
    with duckdb.connect() as con:
        con.register('source_data', frame)
        return con.execute(sql, params).df()


def test_group_average_includes_null_group():
    df = pd.DataFrame({'segment': ['x', 'x', None], 'value': [2., 4., 8.]})
    result = run_reference({'tool': 'group_data', 'arguments': {
        'group_by': ['segment'], 'metric': 'value', 'aggregation': 'mean'}}, df)
    assert_equivalent(df.groupby('segment', dropna=False)['value'].mean().reset_index(), result)


def test_filter_values_are_parameters_not_sql():
    value = "x' OR 1=1 --"
    df = pd.DataFrame({'odd"column': [value, 'other']})
    result = run_reference({'tool': 'filter_data', 'arguments': {
        'column': 'odd"column', 'operator': '==', 'value': value}}, df)
    assert len(result) == 1 and result.iloc[0, 0] == value


def test_wrong_value_or_lost_rows_fail():
    with pytest.raises(AssertionError):
        assert_equivalent(pd.DataFrame({'x': [2]}), pd.DataFrame({'x': [3]}))
    with pytest.raises(AssertionError):
        assert_equivalent(pd.DataFrame({'x': [2]}), pd.DataFrame({'x': [2, 2]}))


def test_unsupported_chart_or_preprocessing_is_not_silently_ignored():
    with pytest.raises(NotImplementedError):
        reference_sql({'tool': 'group_data', 'arguments': {
            'aggregation': 'mean', 'metric': 'v', 'datetime_parses': [{'column': 'time'}]}}, ['v', 'time'])
    with pytest.raises(NotImplementedError):
        reference_sql({'tool': 'filter_data', 'arguments': {
            'column': 'v', 'operator': '>', 'value': 2, 'limit': 1}}, ['v'])


def test_empty_sum_matches_zero_and_count_excludes_null_values():
    df = pd.DataFrame({'value': pd.Series([None, None], dtype='float64')})
    for agg, expected in [('sum', 0), ('count', 0)]:
        result = run_reference({'tool': 'aggregate_data', 'arguments': {'column': 'value', 'aggregation': agg}}, df)
        assert result.iloc[0, 0] == expected
