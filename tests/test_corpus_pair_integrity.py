import copy
import json
from pathlib import Path

import pytest

from scripts.validate_planner_corpus import load_jsonl, validate_records, DEFAULT_PROVENANCE

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return load_jsonl(ROOT / 'training/planner_sft_seed.jsonl'), json.loads(DEFAULT_PROVENANCE.read_text(encoding='utf-8'))


def test_query_and_label_user_message_cannot_diverge():
    records, provenance = inputs()
    records[0]['query'] = 'A completely different request'
    with pytest.raises(ValueError, match='query must exactly match'):
        validate_records(records, provenance)


@pytest.mark.parametrize('content', ['[]', 'null', '42'])
def test_nonobject_assistant_json_is_reported_not_crashed(content):
    records, provenance = inputs()
    records[0]['messages'][1]['content'] = content
    with pytest.raises(ValueError, match='assistant label'):
        validate_records(records, provenance)


@pytest.mark.parametrize('response', [
    {'mode': 'clarify', 'clarification': ' '},
    {'mode': 'clarify', 'clarification': 'Which column?', 'commands': [{'tool': 'anything'}]},
    [],
])
def test_clarification_cannot_be_empty_or_execute(response):
    records, provenance = inputs()
    record = next(r for r in records if r['expected_decision'] == 'clarification')
    record['messages'][1]['content'] = json.dumps(response)
    with pytest.raises(ValueError, match='clarification assistant label'):
        validate_records(records, provenance)
