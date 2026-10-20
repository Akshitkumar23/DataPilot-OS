from training.build_full_scope_seed import seeds, WORDING
from training.corpus_contract import reference_commands


def test_seeds_cover_all_authored_tools_in_two_languages_without_fake_review():
    rows = seeds()
    assert len(rows) == 40
    assert {r['language'] for r in rows} == {'en', 'hinglish'}
    assert {reference_commands(r)[0]['tool'] for r in rows} == set(WORDING)
    assert len({r['task_family_id'] for r in rows}) == 20
    for r in rows:
        assert r['human_reviewed'] is False and 'split' not in r
        assert r['query'] == r['messages'][0]['content']


def test_previously_implicit_choices_are_explicit():
    assert 'Drop rows' in WORDING['clean_data'][0]
    assert '10-minute' in WORDING['forecast_data'][0]
    assert 'Holt linear' in WORDING['forecast_data'][0]
    assert all(col in WORDING['explain_features'][0] for col in ['age', 'education_num', 'hours_per_week'])
