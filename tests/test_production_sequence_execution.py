import json
from types import SimpleNamespace

import pandas as pd

from modules.query_orchestrator import QueryOrchestrator


def run(monkeypatch, frame, commands):
    monkeypatch.setattr('modules.query_orchestrator.ollama_service.verify_ready', lambda: SimpleNamespace(ready=True))
    monkeypatch.setattr('modules.query_orchestrator.SemanticPlanner.extract_semantic_intent',
                        lambda **kwargs: {'commands': commands})
    return QueryOrchestrator.process_query('Filter measure above 2, then mean by category.', frame)


def test_production_sequence_uses_full_intermediate_and_preserves_source(monkeypatch):
    frame = pd.DataFrame({'measure': [1, 3, 5, 8], 'category': ['a', 'a', 'a', 'b']})
    before = frame.copy(deep=True)
    response = run(monkeypatch, frame, [
        {'tool': 'filter_data', 'arguments': {'column': 'measure', 'operator': '>', 'value': 2}},
        {'tool': 'group_data', 'arguments': {'group_by': ['category'], 'metric': 'measure', 'aggregation': 'mean', 'chart_type': 'bar'}},
    ])
    assert response.status == 'SUCCEEDED', response
    assert len(response.plan['execution_plan']) == 2
    result = pd.DataFrame(response.result['data']['records'])
    assert result.set_index('category')['measure'].to_dict() == {'a': 4.0, 'b': 8.0}
    assert response.result['plot_image']
    assert len(response.result['steps']) == 2
    namespace = {'df': before.copy(deep=True)}
    exec(response.result['code'], namespace)
    assert namespace['result_df'].set_index('category')['measure'].to_dict() == {'a': 4.0, 'b': 8.0}
    json.dumps(response.model_dump())
    pd.testing.assert_frame_equal(frame, before)


def test_production_sequence_stops_on_intermediate_schema_error(monkeypatch):
    frame = pd.DataFrame({'measure': [3, 5], 'category': ['a', 'a']})
    response = run(monkeypatch, frame, [
        {'tool': 'filter_data', 'arguments': {'column': 'measure', 'operator': '>', 'value': 2}},
        {'tool': 'group_data', 'arguments': {'group_by': ['missing'], 'metric': 'measure', 'aggregation': 'mean', 'chart_type': 'none'}},
        {'tool': 'filter_data', 'arguments': {'column': 'measure', 'operator': '>', 'value': 2}},
    ])
    assert response.status != 'SUCCEEDED'
    assert response.plan['failed_step'] == 2
    assert len(response.plan['execution_plan']) == 2
    assert response.result is None
