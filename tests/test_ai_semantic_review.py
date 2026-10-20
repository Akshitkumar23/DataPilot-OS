from training.ai_semantic_review import normalize_verdict


def test_inconsistent_accept_is_downgraded():
    verdict = {'decision': 'accept', 'checks': {'task': 'pass', 'columns': 'uncertain'},
               'rationale': 'Long enough rationale', 'human_review_required': False}
    normalized = normalize_verdict(verdict)
    assert normalized['decision'] == 'uncertain'
    assert normalized['human_review_required'] is True


def test_consistent_accept_is_preserved():
    verdict = {'decision': 'accept', 'checks': {'task': 'pass', 'columns': 'not_applicable'},
               'rationale': 'Long enough rationale', 'human_review_required': False}
    assert normalize_verdict(verdict)['decision'] == 'accept'


def test_rejection_always_requires_human_review():
    verdict = {'decision': 'reject', 'checks': {'task': 'fail'},
               'rationale': 'Long enough rationale', 'human_review_required': False}
    assert normalize_verdict(verdict)['human_review_required'] is True
