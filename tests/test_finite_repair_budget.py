"""
Tests for Finite Self-Healing Repair Budget Manager (Phase 9).
Verifies:
- Max 2 repair iterations.
- Progress detection (diff-check); identical repair fails with LOOP_NO_PROGRESS.
- Budget exhaustion after 2 failed iterations fails with LOOP_BUDGET_EXCEEDED.
- Full telemetry and audit history tracking.
"""

import pytest
from backend.core.error_states import ErrorState
from backend.ds_engine.repair_budget import RepairBudgetManager


def test_repair_budget_allows_max_two_attempts():
    mgr = RepairBudgetManager(max_attempts=2)
    assert mgr.can_attempt_repair()
    assert mgr.attempts_used == 0


def test_identical_repair_triggers_loop_no_progress():
    mgr = RepairBudgetManager(max_attempts=2)
    plan_1 = {"tool": "group_data", "arguments": {"group_by": ["cat"], "aggregation": "mean"}}
    # Propose identical plan
    plan_repaired = {"tool": "group_data", "arguments": {"group_by": ["cat"], "aggregation": "mean"}}

    err = mgr.check_progress_and_record(
        original_plan=plan_1,
        proposed_plan=plan_repaired,
        error_diagnosed="COLUMN_NOT_FOUND",
        adjustment_summary="None",
        success=False,
    )

    assert err is not None
    assert err.code == ErrorState.LOOP_NO_PROGRESS
    assert "identical" in err.message.lower()
    assert len(mgr.history) == 1
    assert mgr.history[0].is_identical


def test_budget_exhaustion_triggers_loop_budget_exceeded():
    mgr = RepairBudgetManager(max_attempts=2)
    plan_initial = {"tool": "group_data", "arguments": {"group_by": ["nonexistent"]}}

    # Attempt 1: Changed, but still invalid
    plan_attempt_1 = {"tool": "group_data", "arguments": {"group_by": ["other_wrong_col"]}}
    err_1 = mgr.check_progress_and_record(
        original_plan=plan_initial,
        proposed_plan=plan_attempt_1,
        error_diagnosed="COLUMN_NOT_FOUND",
        adjustment_summary="Tried other column",
        success=False,
    )
    # Attempt 1 made progress (different hash), so no budget exceeded error yet
    assert err_1 is None
    assert mgr.attempts_used == 1
    assert mgr.can_attempt_repair()

    # Attempt 2: Changed again, but still not successful
    plan_attempt_2 = {"tool": "group_data", "arguments": {"group_by": ["third_wrong_col"]}}
    err_2 = mgr.check_progress_and_record(
        original_plan=plan_attempt_1,
        proposed_plan=plan_attempt_2,
        error_diagnosed="COLUMN_NOT_FOUND",
        adjustment_summary="Tried third column",
        success=False,
    )
    # Now budget (2) is exhausted
    assert err_2 is not None
    assert err_2.code == ErrorState.LOOP_BUDGET_EXCEEDED
    assert "budget exhausted" in err_2.message.lower()
    assert mgr.attempts_used == 2
    assert not mgr.can_attempt_repair()


def test_successful_repair_completes_cleanly():
    mgr = RepairBudgetManager(max_attempts=2)
    plan_initial = {"tool": "group_data", "arguments": {"group_by": ["nonexistent"]}}
    plan_repaired = {"tool": "group_data", "arguments": {"group_by": ["valid_category"]}}

    err = mgr.check_progress_and_record(
        original_plan=plan_initial,
        proposed_plan=plan_repaired,
        error_diagnosed="COLUMN_NOT_FOUND",
        adjustment_summary="Substituted grounded valid_category",
        success=True,
    )

    assert err is None
    assert mgr.attempts_used == 1
    assert len(mgr.history) == 1
    assert mgr.history[0].success
    assert not mgr.history[0].is_identical
