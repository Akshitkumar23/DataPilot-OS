"""
DataPilot OS — Phase 3 Dataset Grounding Unit Test Suite.
Tests:
- Exact column existence checks (case-sensitive, case-insensitive, normalized).
- No silent auto-substitution for typos; typo suggestions surface clearly.
- Candidate deduplication and role validation.
- Profile caching keyed by SHA-256 dataset hash.
- Decision record generation and invariants.
"""

import pandas as pd
import pytest

from backend.ds_engine.dataset_profiler import DatasetProfiler, DatasetProfile
from backend.ds_engine.dataset_grounding import DatasetGrounding, GroundingMatch
from backend.ds_engine.analysis_decision import (
    AnalysisDecision,
    DecisionType,
    DecisionRecord,
    make_execute_decision,
    make_clarify_decision,
    make_unsupported_decision,
    make_unsafe_decision,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "User_ID": [101, 102, 103, 104, 105],
        "Monthly_Revenue": [1200.50, 850.00, 2300.75, 410.20, 1950.00],
        "signup_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "Churn_Flag": [0, 1, 0, 0, 1],
        "Customer_Segment": ["Enterprise", "SMB", "Enterprise", "Consumer", "SMB"],
    })


def test_exact_column_matching(sample_df):
    cols = list(sample_df.columns)

    # 1. Exact case-sensitive match
    m1 = DatasetGrounding.find_column_match("Monthly_Revenue", cols)
    assert m1.found
    assert m1.matched_column == "Monthly_Revenue"
    assert m1.is_exact
    assert m1.confidence == 1.0

    # 2. Case-insensitive match
    m2 = DatasetGrounding.find_column_match("monthly_revenue", cols)
    assert m2.found
    assert m2.matched_column == "Monthly_Revenue"
    assert m2.is_case_insensitive
    assert m2.confidence == 0.98

    # 3. Normalized punctuation/space match
    m3 = DatasetGrounding.find_column_match("monthly revenue", cols)
    assert m3.found
    assert m3.matched_column == "Monthly_Revenue"
    assert m3.is_normalized
    assert m3.confidence == 0.92


def test_typo_suggestions_without_auto_substitution(sample_df):
    cols = list(sample_df.columns)

    # User typos column name
    m = DatasetGrounding.find_column_match("Monthly_Reveneu", cols)
    # Must NOT auto-substitute!
    assert not m.found
    assert m.matched_column is None
    # Must provide suggestion
    assert "Monthly_Revenue" in m.suggestions


def test_ambiguous_normalized_match_does_not_guess():
    cols = ["First Name", "first_name"]
    m = DatasetGrounding.find_column_match("firstname", cols)
    # Multiple columns normalize to 'firstname' -> must NOT silently pick one
    assert not m.found
    assert len(m.suggestions) == 2
    assert set(m.suggestions) == {"First Name", "first_name"}


def test_profile_cache_keyed_by_hash(sample_df):
    DatasetProfiler.clear_cache()
    h = DatasetProfiler.compute_hash(sample_df, "test_ds")

    # Initial profile computes and stores in cache
    p1 = DatasetProfiler.profile(sample_df, "test_ds", use_cache=True)
    assert p1.dataset_hash == h
    assert DatasetProfiler.get_cached_profile(h) is not None

    # Second call returns identical cached instance
    p2 = DatasetProfiler.profile(sample_df, "test_ds", use_cache=True)
    assert p1 is p2

    # Clear cache removes it
    DatasetProfiler.clear_cache()
    assert DatasetProfiler.get_cached_profile(h) is None


def test_candidate_lists_are_deduplicated(sample_df):
    profile = DatasetProfiler.profile(sample_df, "dedup_test", use_cache=False)

    for cand_list in [
        profile.temporal_column_candidates,
        profile.target_column_candidates,
        profile.entity_id_candidates,
        profile.binary_flag_candidates,
        profile.continuous_numeric_candidates,
        profile.categorical_candidates,
        profile.sequence_column_candidates,
        profile.datetime_parse_candidates,
    ]:
        assert len(cand_list) == len(set(cand_list)), f"Candidate list contains duplicates: {cand_list}"


def test_column_role_validation(sample_df):
    profile = DatasetProfiler.profile(sample_df, "role_test", use_cache=False)

    # Monthly_Revenue is numeric
    val_num = DatasetGrounding.validate_column_role("Monthly_Revenue", "numeric", profile)
    assert val_num.is_role_valid

    # Monthly_Revenue is NOT temporal
    val_time = DatasetGrounding.validate_column_role("Monthly_Revenue", "temporal", profile)
    assert not val_time.is_role_valid
    assert "incompatible with required role 'temporal'" in val_time.error_message

    # signup_date is temporal
    val_date = DatasetGrounding.validate_column_role("signup_date", "temporal", profile)
    assert val_date.is_role_valid

    # Customer_Segment is categorical
    val_cat = DatasetGrounding.validate_column_role("Customer_Segment", "categorical", profile)
    assert val_cat.is_role_valid


def test_ground_columns_or_clarify(sample_df):
    cols = list(sample_df.columns)

    # All valid columns
    grounded, decision = DatasetGrounding.ground_columns(
        ["monthly_revenue", "User_ID"], cols
    )
    assert decision is None
    assert grounded == ["Monthly_Revenue", "User_ID"]

    # One missing column
    grounded_fail, decision_fail = DatasetGrounding.ground_columns(
        ["monthly_revenue", "nonexistent_col"], cols
    )
    assert grounded_fail == []
    assert decision_fail is not None
    assert decision_fail.requires_clarification
    assert "nonexistent_col" in decision_fail.reason


def test_decision_record_invariants():
    exec_dec = make_execute_decision(grounded_columns=["sales", "region"])
    assert exec_dec.is_executable
    assert not exec_dec.requires_clarification

    clarify_dec = make_clarify_decision(
        reason="Multiple dates found",
        candidate_columns=["date_1", "date_2"]
    )
    assert clarify_dec.requires_clarification
    assert not clarify_dec.is_executable
    assert clarify_dec.candidate_columns == ["date_1", "date_2"]

    unsup_dec = make_unsupported_decision(reason="Quantum optimization not supported")
    assert unsup_dec.is_unsupported
    assert not unsup_dec.is_executable

    unsafe_dec = make_unsafe_decision(reason="Target leakage detected")
    assert unsafe_dec.is_unsafe
    assert not unsafe_dec.is_executable
