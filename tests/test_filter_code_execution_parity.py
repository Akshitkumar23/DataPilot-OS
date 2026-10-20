from types import SimpleNamespace

import pandas as pd
import pytest

from backend.ds_engine.intent_schema import FilterSpec
from backend.ds_engine.typed_operators import FilterSliceOperator


@pytest.mark.parametrize('operator,value', [
    ('eq', 3), ('neq', 3), ('gt', 3), ('gte', 3), ('lt', 3), ('lte', 3),
    ('in', [1, 3]), ('not_in', [1, 3]), ('is_null', None),
    ('is_not_null', None), ('missing', None), ('gt', '2.5'),
])
def test_numeric_filter_export_matches_execution(operator, value):
    frame = pd.DataFrame({'measurement': [1., 3., 5., None]})
    intent = SimpleNamespace(conditions=[FilterSpec(column='measurement', operator=operator, value=value)], arguments={})
    result = FilterSliceOperator().execute(frame, intent)
    namespace = {'df': frame.copy()}
    exec(result.reproducible_code, namespace)
    pd.testing.assert_frame_equal(namespace['result_df'], result.dataframe)


def test_literal_substring_export_matches_execution():
    frame = pd.DataFrame({'label': ['a.b', 'axb', 'A.B', None]})
    intent = SimpleNamespace(conditions=[FilterSpec(column='label', operator='contains', value='a.b')], arguments={})
    result = FilterSliceOperator().execute(frame, intent)
    namespace = {'df': frame.copy()}
    exec(result.reproducible_code, namespace)
    pd.testing.assert_frame_equal(namespace['result_df'], result.dataframe)
    assert list(result.dataframe.index) == [0, 2]
