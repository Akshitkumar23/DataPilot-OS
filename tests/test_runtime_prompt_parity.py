import json
from copy import deepcopy

import pandas as pd
import pytest

from backend.intelligence.ollama_service import OllamaResponse
from modules.capability_registry import capability_registry
from modules.semantic_planner import SemanticPlanner
from training.prepare_runtime_corpus import prepare_record, reference_target


def test_prepared_prompt_matches_actual_first_ollama_request(monkeypatch):
    frame = pd.DataFrame({'measure': [1, 4, 9]})
    target = {'tool': 'filter_data', 'arguments': {'column': 'measure', 'operator': '>', 'value': 3}}
    record = {'case_id': 'parity', 'dataset_id': 'fixture', 'query': 'Keep measure above 3.',
              'expected_command': target, 'expected_decision': 'tool',
              'conversation_history': [{'role': 'user', 'content': 'Use this dataset.'}],
              'context': {'active_module': 'data'}}
    specs = [capability_registry.get_tool_spec('filter_data')]
    monkeypatch.setattr(capability_registry, 'select_planner_tools', lambda *args, **kwargs: specs)
    monkeypatch.setattr('modules.semantic_planner.planner_learning_memory.render_guidance', lambda *args: '')
    prepared = prepare_record(record, frame, capability_registry)
    captured = []
    def respond(request):
        captured.append(request)
        return OllamaResponse(success=True, parsed_json=target, content=json.dumps(target))
    monkeypatch.setattr('modules.semantic_planner.ollama_service.structured_request', respond)
    result = SemanticPlanner.extract_semantic_intent(record['query'], frame,
                                                    record['conversation_history'], record['context'])
    assert result is not None and len(captured) == 1
    assert prepared['prompt'] == [{'role': 'system', 'content': captured[0].system_prompt},
                                  {'role': 'user', 'content': captured[0].prompt}]
    assert prepared['response_schema'] == captured[0].response_schema
    assert json.loads(prepared['completion'][0]['content']) == target
    changed = deepcopy(record)
    changed['expected_command']['arguments']['value'] = 8
    assert prepare_record(changed, frame, capability_registry)['prompt'] == prepared['prompt']


def test_multistep_reference_is_preserved_without_dropping_steps():
    command = {'tool': 'filter_data', 'arguments': {'column': 'x', 'operator': '>', 'value': 0}}
    record = {'schema_version': 'planner-sft-v2', 'expected_commands': [command, command]}
    before = deepcopy(record)
    assert reference_target(record, capability_registry) == {'commands': [command, command]}
    assert record == before


def test_known_tool_outside_shortlist_is_rejected_without_gold_routing(monkeypatch):
    specs = [capability_registry.get_tool_spec('filter_data')]
    monkeypatch.setattr(capability_registry, 'select_planner_tools', lambda *args, **kwargs: specs)
    record = {'case_id': 'shortlist', 'dataset_id': 'fixture', 'query': 'sort x',
              'expected_command': {'tool': 'sort_data', 'arguments': {'column': 'x', 'order': 'asc'}}}
    with pytest.raises(ValueError, match="RETRIEVAL_MISS"):
        prepare_record(record, pd.DataFrame({'x': [1]}), capability_registry)
