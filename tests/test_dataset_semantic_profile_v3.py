import pandas as pd

from backend.ds_engine.dataset_profiler import DatasetProfiler


def test_profile_uses_observed_structure_not_column_keywords():
    frame = pd.DataFrame(
        {
            "alpha": list(range(1, 13)),
            "beta": pd.date_range("2025-01-01", periods=12, freq="D").strftime("%Y/%m/%d"),
            "gamma": ["north", "south"] * 6,
            "delta": [10.0, 11.5, 13.0, 9.5, 8.0, 7.5, 12.0, 14.0, 15.0, 10.5, 9.0, 8.5],
        }
    )
    profile = DatasetProfiler.profile(frame, "opaque")

    assert profile.columns["alpha"].semantic_role == "entity_id"
    assert "alpha" in profile.sequence_column_candidates
    assert profile.columns["beta"].semantic_role == "temporal"
    assert profile.columns["beta"].datetime_parse_ratio == 1.0
    assert profile.columns["beta"].inferred_frequency == "D"
    assert "gamma" in profile.categorical_candidates
    assert profile.columns["delta"].semantic_role == "continuous_numeric"


def test_two_level_non_boolean_column_exposes_ambiguity_instead_of_guessing_event_meaning():
    frame = pd.DataFrame({"opaque": ["red", "blue"] * 10, "measure": range(20)})
    profile = DatasetProfiler.profile(frame)

    assert profile.columns["opaque"].semantic_role == "categorical_nominal"
    assert profile.role_ambiguities["opaque"] == ["categorical_nominal", "binary_flag"]
    assert "opaque" in profile.binary_flag_candidates


def test_profile_reports_quality_evidence_and_is_deterministic():
    frame = pd.DataFrame(
        {
            "x": [1.0, 1.0, None, 2.0],
            "y": ["a", "a", "b", "c"],
        }
    )
    first = DatasetProfiler.profile(frame, "quality")
    second = DatasetProfiler.profile(frame, "quality")

    assert first.dataset_hash == second.dataset_hash
    assert first.duplicate_row_count == 1
    assert first.total_missing_percentage == 12.5
    assert first.columns["x"].null_count == 1
    assert first.columns["x"].unique_ratio == 2 / 3


def test_compact_context_surfaces_ambiguity_and_ordering_evidence():
    frame = pd.DataFrame({"a": range(10), "b": ["left", "right"] * 5})
    context = DatasetProfiler.profile(frame).to_compact_prompt_context()

    assert "Ordered Sequence Candidates: a" in context
    assert "b=>categorical_nominal,binary_flag" in context
