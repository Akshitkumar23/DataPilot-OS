"""Tests for the offline planner benchmark and non-mutating promotion gate."""

import pandas as pd
import pytest

from modules.planner_benchmark import (
    BenchmarkCase,
    PlannerBenchmarkReport,
    PlannerBenchmarkRunner,
    PromotionPolicy,
    evaluate_promotion,
)


def test_runner_scores_grounded_tool_and_preserves_model_argument():
    received_models = []

    def planner(**kwargs):
        received_models.append(kwargs["planner_model"])
        return {
            "tool_name": "event_window_analysis",
            "feature_columns": ["vibration_x_g", "vibration_y_g"],
            "event_predicate": {"column": "machine_failure", "operator": "eq", "value": 1},
        }

    report = PlannerBenchmarkRunner(planner).run(
        [BenchmarkCase(
            case_id="event-window",
            query="Compare the vibration signals before failure",
            expected_tool="event_window_analysis",
            required_columns=["vibration_x_g", "vibration_y_g", "machine_failure"],
        )],
        pd.DataFrame({"vibration_x_g": [1.0], "vibration_y_g": [2.0], "machine_failure": [0]}),
        model_name="candidate:local",
    )

    assert report.pass_rate == 1.0
    assert report.results[0].observed_tool == "event_window_analysis"
    assert received_models == ["candidate:local"]


def test_runner_rejects_wrong_tool_or_missing_grounding():
    def planner(**_kwargs):
        return {"tool_name": "eda_data", "feature_columns": ["vibration_x_g"]}

    report = PlannerBenchmarkRunner(planner).run(
        [BenchmarkCase(
            case_id="wrong-route",
            query="Analyse failures",
            expected_tool="event_window_analysis",
            required_columns=["vibration_x_g", "machine_failure"],
        )],
        pd.DataFrame({"vibration_x_g": [1.0], "machine_failure": [1]}),
        model_name="candidate:local",
    )

    result = report.results[0]
    assert not result.passed
    assert result.observed_tool == "eda_data"
    assert result.missing_required_columns == ["machine_failure"]


def test_runner_rejects_wrong_semantic_role_binding_even_when_columns_exist():
    """Presence of the right columns is insufficient for model promotion."""
    def planner(**_kwargs):
        return {
            "tool_name": "group_data",
            "dimensions": ["Date", "Season"],
            "metrics": ["demand"],
            "aggregation": "mean",
            "chart_type": "line",
        }

    report = PlannerBenchmarkRunner(planner).run(
        [BenchmarkCase(
            case_id="hourly-profile",
            query="Average demand by Hour and Season as a line chart.",
            expected_tool="group_data",
            required_columns=["Hour", "Season", "demand"],
            expected_fields={"dimensions": ["Hour", "Season"], "chart_type": "line"},
        )],
        pd.DataFrame({"Date": ["01/01/2024"], "Hour": [8], "Season": ["Winter"], "demand": [10]}),
        model_name="candidate:local",
    )

    result = report.results[0]
    assert result.passed is False
    assert result.mismatched_expected_fields["dimensions"]["observed"] == ["Date", "Season"]


def test_clarification_case_is_scored_as_a_valid_outcome():
    def planner(**_kwargs):
        return {"is_ambiguous": True, "ambiguity_reason": "An order column is required."}

    report = PlannerBenchmarkRunner(planner).run(
        [BenchmarkCase(
            case_id="missing-lookback",
            query="What happens before an event?",
            expected_outcome="clarification",
        )],
        pd.DataFrame({"sequence": [1]}),
        model_name="candidate:local",
    )

    assert report.results[0].passed
    assert report.results[0].observed_outcome == "clarification"


def test_promotion_gate_rejects_accuracy_or_latency_regression():
    baseline = PlannerBenchmarkReport(
        model_name="baseline",
        case_count=10,
        passed_count=10,
        pass_rate=1.0,
        mean_latency_ms=10.0,
        p95_latency_ms=20.0,
        results=[],
    )
    candidate = PlannerBenchmarkReport(
        model_name="candidate",
        case_count=10,
        passed_count=8,
        pass_rate=0.8,
        mean_latency_ms=20.0,
        p95_latency_ms=50.0,
        results=[],
    )

    decision = evaluate_promotion(
        baseline, candidate, PromotionPolicy(min_pass_rate=0.9, max_p95_latency_ratio=2.0)
    )

    assert not decision.eligible
    assert any("pass rate" in reason.lower() for reason in decision.reasons)
    assert any("latency" in reason.lower() for reason in decision.reasons)


def test_promotion_gate_allows_an_equal_or_better_candidate():
    baseline = PlannerBenchmarkReport(
        model_name="baseline",
        case_count=2,
        passed_count=1,
        pass_rate=0.5,
        mean_latency_ms=30.0,
        p95_latency_ms=40.0,
        results=[],
    )
    candidate = PlannerBenchmarkReport(
        model_name="candidate",
        case_count=2,
        passed_count=2,
        pass_rate=1.0,
        mean_latency_ms=40.0,
        p95_latency_ms=70.0,
        results=[],
    )

    decision = evaluate_promotion(
        baseline, candidate, PromotionPolicy(min_pass_rate=0.9, max_p95_latency_ratio=2.0)
    )

    assert decision.eligible
    assert decision.reasons == []


def test_invalid_case_contract_is_rejected_before_a_run():
    with pytest.raises(ValueError, match="expected_tool"):
        BenchmarkCase(case_id="bad", query="do something")
