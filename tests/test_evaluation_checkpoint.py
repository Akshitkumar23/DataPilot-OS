import json

import pytest

from training.evaluation_checkpoint import write_checkpoint, require_context_budget


def test_partial_evidence_remains_explicitly_incomplete(tmp_path):
    path = tmp_path / 'evaluation.partial.json'
    write_checkpoint(path, {'base_model': 'fixture'}, [], 2)
    write_checkpoint(path, {'base_model': 'fixture'}, [{'case_id': 'one', 'passed': True}], 2)
    result = json.loads(path.read_text())
    assert result['status'] == 'evaluation_incomplete'
    assert result['case_count'] == 1 and result['expected_case_count'] == 2
    assert result['deployment_allowed'] is False
    assert not path.with_suffix('.json.tmp').exists()


def test_context_budget_includes_output_without_silent_truncation():
    require_context_budget(100, 20, 120)
    with pytest.raises(ValueError, match='PROMPT_TOO_LONG'):
        require_context_budget(101, 20, 120)


def test_bundle_copies_runtime_training_dependencies():
    from pathlib import Path
    source = Path('scripts/build_kaggle_training_bundle.py').read_text(encoding='utf-8')
    assert '"prepare_runtime_corpus.py"' in source
    assert '"evaluation_checkpoint.py"' in source
    assert '"execution_grader.py"' in source
