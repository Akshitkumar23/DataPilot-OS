"""
Regression test suite for Defect D-02: Paired-Structure Guard in StatisticalTestOperator.
Verifies that independent t-tests on two columns from the same rows emit PAIRED_STRUCTURE_DETECTED
warnings without silently substituting the test, while independent cohort designs and explicit
paired tests execute cleanly without false warnings.
"""

import pytest
import numpy as np
import pandas as pd
from backend.ds_engine.typed_operators import StatisticalTestOperator
from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily


def test_independent_ttest_same_rows_emits_paired_warning():
    """
    Comparing two numeric columns from the same rows using Welch/independent t-test
    must emit PAIRED_STRUCTURE_DETECTED warning without silently changing the test.
    """
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "fresh_spend": np.random.lognormal(mean=9.0, sigma=0.8, size=n),
        "milk_spend": np.random.lognormal(mean=8.5, sigma=0.7, size=n),
    })

    op = StatisticalTestOperator()
    intent = AnalysisIntentV2(
        task_family=TaskFamily.STATISTICAL_TEST,
        subjects=["fresh_spend", "milk_spend"],
        arguments={"test_type": "welch_ttest", "columns": ["fresh_spend", "milk_spend"], "alpha": 0.05},
    )

    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    # 1. Calculation executed
    assert "p_value" in res.dataframe.columns
    assert "statistic" in res.dataframe.columns
    # 2. Emitted PAIRED_STRUCTURE_DETECTED warning in manifest warnings and finding
    assert any("PAIRED_STRUCTURE_DETECTED" in w for w in res.manifest.warnings)
    assert "PAIRED_STRUCTURE_DETECTED" in res.written_finding
    # 3. Test was not silently substituted
    assert res.dataframe["test_type"].iloc[0] == "welch_ttest"


def test_independent_cohort_design_emits_no_paired_warning():
    """
    Independent two-sample t-test with an explicit cohort design (e.g. cohort_column / group_by)
    should NOT trigger the paired structure warning.
    """
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "site_a_yield": np.random.normal(loc=10.0, scale=2.0, size=n),
        "site_b_yield": np.random.normal(loc=12.0, scale=2.0, size=n),
        "facility_type": ["Urban"] * n,
    })

    op = StatisticalTestOperator()
    intent = AnalysisIntentV2(
        task_family=TaskFamily.STATISTICAL_TEST,
        subjects=["site_a_yield", "site_b_yield"],
        arguments={
            "test_type": "welch_ttest",
            "columns": ["site_a_yield", "site_b_yield"],
            "cohort_column": "facility_type",
            "alpha": 0.05,
        },
    )

    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    assert not any("PAIRED_STRUCTURE_DETECTED" in w for w in res.manifest.warnings)
    assert "PAIRED_STRUCTURE_DETECTED" not in res.written_finding


def test_explicit_paired_ttest_executes_without_false_warning():
    """
    When the analyst explicitly selects paired_ttest on before vs after measurements,
    the paired test executes cleanly and does NOT produce a PAIRED_STRUCTURE_DETECTED warning.
    """
    np.random.seed(42)
    n = 60
    before = np.random.normal(loc=120.0, scale=15.0, size=n)
    after = before - np.random.normal(loc=8.0, scale=3.0, size=n)

    df = pd.DataFrame({"bp_before": before, "bp_after": after})

    op = StatisticalTestOperator()
    intent = AnalysisIntentV2(
        task_family=TaskFamily.STATISTICAL_TEST,
        subjects=["bp_before", "bp_after"],
        arguments={"test_type": "paired_ttest", "columns": ["bp_before", "bp_after"], "alpha": 0.05},
    )

    res = op.execute(df, intent)

    assert res.status == "SUCCEEDED"
    assert not any("PAIRED_STRUCTURE_DETECTED" in w for w in res.manifest.warnings)
    assert "PAIRED_STRUCTURE_DETECTED" not in res.written_finding
    assert res.dataframe["p_value"].iloc[0] < 0.05
    assert "paired" in res.dataframe["test_name"].iloc[0].lower()
