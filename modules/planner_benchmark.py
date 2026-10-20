"""Offline, evidence-based evaluation for DataPilot's semantic planner.

This module deliberately does *not* register runtime tools, alter an active
Ollama model, or route user queries.  A benchmark case is a labelled example
used only to decide whether a human may promote a planner/model configuration.
"""

from __future__ import annotations

from math import ceil
from statistics import fmean
from time import perf_counter
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


PlannerCallable = Callable[..., Optional[Dict[str, Any]]]


class BenchmarkCase(BaseModel):
    """One explicitly labelled, non-runtime planner evaluation example."""

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_outcome: Literal["tool", "clarification"] = "tool"
    expected_tool: Optional[str] = None
    required_columns: List[str] = Field(default_factory=list)
    expected_fields: Dict[str, Any] = Field(default_factory=dict)
    """Exact resolved-plan fields that must match for this labelled case.

    Required columns catch omission but cannot distinguish an incorrect role
    binding (for example, using a parsing timestamp as a group dimension while
    omitting the requested time bucket).  This is evaluation-only metadata;
    it never participates in runtime routing or execution.
    """

    @field_validator("required_columns")
    @classmethod
    def unique_nonempty_columns(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("required_columns must not contain duplicates")
        return cleaned

    def model_post_init(self, __context: Any) -> None:
        if self.expected_outcome == "tool" and not self.expected_tool:
            raise ValueError("expected_tool is required when expected_outcome is 'tool'")
        if self.expected_outcome == "clarification" and self.expected_tool:
            raise ValueError("expected_tool must be omitted when expected_outcome is 'clarification'")


class BenchmarkCaseResult(BaseModel):
    case_id: str
    passed: bool
    latency_ms: float
    observed_tool: Optional[str] = None
    observed_outcome: Literal["tool", "clarification", "failure"]
    missing_required_columns: List[str] = Field(default_factory=list)
    mismatched_expected_fields: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    reason: Optional[str] = None


class PlannerBenchmarkReport(BaseModel):
    """Immutable measurement output; it has no side effect on model selection."""

    model_name: str
    case_count: int
    passed_count: int
    pass_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    results: List[BenchmarkCaseResult]


class PromotionPolicy(BaseModel):
    """Human-selected gate thresholds; this policy cannot promote a model itself."""

    min_pass_rate: float = Field(ge=0.0, le=1.0)
    max_pass_rate_regression: float = Field(default=0.0, ge=0.0, le=1.0)
    max_p95_latency_ratio: float = Field(default=2.0, gt=0.0)


class PromotionDecision(BaseModel):
    baseline_model: str
    candidate_model: str
    eligible: bool
    reasons: List[str] = Field(default_factory=list)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ranked = sorted(values)
    index = max(0, min(len(ranked) - 1, ceil(percentile * len(ranked)) - 1))
    return float(ranked[index])


def _columns_from_plan(plan: Dict[str, Any]) -> set[str]:
    """Extract values which are column references from a resolved plan.

    This is a scoring aid only. Runtime schema validation remains authoritative.
    """
    columns: set[str] = set()
    scalar_fields = ("target_column", "time_column", "sequence_column", "event_column")
    list_fields = ("feature_columns", "metrics", "dimensions", "group_by", "columns")
    for field in scalar_fields:
        value = plan.get(field)
        if isinstance(value, str) and value:
            columns.add(value)
    for field in list_fields:
        value = plan.get(field, [])
        if isinstance(value, list):
            columns.update(item for item in value if isinstance(item, str) and item)

    predicate = plan.get("event_predicate")
    if isinstance(predicate, dict) and isinstance(predicate.get("column"), str):
        columns.add(predicate["column"])
    for filter_spec in plan.get("filters", []):
        if isinstance(filter_spec, dict) and isinstance(filter_spec.get("column"), str):
            columns.add(filter_spec["column"])
    return columns


def _observable_plan_fields(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Expose resolved command arguments to offline benchmark labels only."""
    observed = dict(plan)
    execution_plan = plan.get("execution_plan")
    if isinstance(execution_plan, list) and execution_plan:
        first_step = execution_plan[0]
        if isinstance(first_step, dict):
            arguments = first_step.get("arguments")
            if isinstance(arguments, dict):
                observed.update(arguments)
    return observed


class PlannerBenchmarkRunner:
    """Runs injected planner calls without changing the active Ollama model."""

    def __init__(self, planner: PlannerCallable):
        self._planner = planner

    def run(
        self,
        cases: List[BenchmarkCase],
        df: pd.DataFrame,
        *,
        model_name: str,
        timeout_sec: float = 60.0,
    ) -> PlannerBenchmarkReport:
        if not cases:
            raise ValueError("At least one benchmark case is required")
        if df is None:
            raise ValueError("A dataset is required for grounded planner benchmarking")
        if not model_name or not model_name.strip():
            raise ValueError("model_name is required; benchmark reports must be attributable")

        results: List[BenchmarkCaseResult] = []
        for case in cases:
            started = perf_counter()
            try:
                plan = self._planner(
                    query_raw=case.query,
                    df=df,
                    conversation_history=[],
                    context={},
                    timeout_sec=timeout_sec,
                    planner_model=model_name,
                )
                latency_ms = (perf_counter() - started) * 1000
            except Exception as exc:
                results.append(BenchmarkCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    latency_ms=(perf_counter() - started) * 1000,
                    observed_outcome="failure",
                    reason=f"Planner raised {type(exc).__name__}: {exc}",
                ))
                continue

            if not isinstance(plan, dict):
                results.append(BenchmarkCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    latency_ms=latency_ms,
                    observed_outcome="failure",
                    reason="Planner returned no resolved plan.",
                ))
                continue

            is_clarification = bool(plan.get("is_ambiguous"))
            observed_outcome: Literal["tool", "clarification"] = (
                "clarification" if is_clarification else "tool"
            )
            observed_tool = plan.get("tool_name") if isinstance(plan.get("tool_name"), str) else None
            missing_columns = sorted(set(case.required_columns) - _columns_from_plan(plan))
            observed_fields = _observable_plan_fields(plan)
            mismatched_fields = {
                field: {"expected": expected, "observed": observed_fields.get(field)}
                for field, expected in case.expected_fields.items()
                if observed_fields.get(field) != expected
            }
            outcome_matches = observed_outcome == case.expected_outcome
            tool_matches = case.expected_outcome == "clarification" or observed_tool == case.expected_tool
            passed = outcome_matches and tool_matches and not missing_columns and not mismatched_fields

            reasons: List[str] = []
            if not outcome_matches:
                reasons.append(f"Expected {case.expected_outcome}, got {observed_outcome}")
            if not tool_matches:
                reasons.append(f"Expected tool '{case.expected_tool}', got '{observed_tool}'")
            if missing_columns:
                reasons.append("Missing required columns: " + ", ".join(missing_columns))
            if mismatched_fields:
                reasons.append("Mismatched semantic fields: " + ", ".join(sorted(mismatched_fields)))

            results.append(BenchmarkCaseResult(
                case_id=case.case_id,
                passed=passed,
                latency_ms=latency_ms,
                observed_tool=observed_tool,
                observed_outcome=observed_outcome,
                missing_required_columns=missing_columns,
                mismatched_expected_fields=mismatched_fields,
                reason="; ".join(reasons) if reasons else None,
            ))

        latencies = [result.latency_ms for result in results]
        passed_count = sum(result.passed for result in results)
        return PlannerBenchmarkReport(
            model_name=model_name.strip(),
            case_count=len(results),
            passed_count=passed_count,
            pass_rate=passed_count / len(results),
            mean_latency_ms=float(fmean(latencies)),
            p95_latency_ms=_percentile(latencies, 0.95),
            results=results,
        )


def evaluate_promotion(
    baseline: PlannerBenchmarkReport,
    candidate: PlannerBenchmarkReport,
    policy: PromotionPolicy,
) -> PromotionDecision:
    """Return an advisory decision only; never call set_active_model()."""
    reasons: List[str] = []
    if baseline.case_count != candidate.case_count:
        reasons.append("Reports use different case counts and cannot be compared safely.")
    if candidate.pass_rate < policy.min_pass_rate:
        reasons.append(
            f"Candidate pass rate {candidate.pass_rate:.1%} is below policy minimum {policy.min_pass_rate:.1%}."
        )
    required_rate = baseline.pass_rate - policy.max_pass_rate_regression
    if candidate.pass_rate < required_rate:
        reasons.append(
            f"Candidate pass rate {candidate.pass_rate:.1%} regresses below baseline {baseline.pass_rate:.1%}."
        )
    if baseline.p95_latency_ms > 0:
        allowed_p95 = baseline.p95_latency_ms * policy.max_p95_latency_ratio
        if candidate.p95_latency_ms > allowed_p95:
            reasons.append(
                f"Candidate p95 latency {candidate.p95_latency_ms:.1f} ms exceeds allowed {allowed_p95:.1f} ms."
            )

    return PromotionDecision(
        baseline_model=baseline.model_name,
        candidate_model=candidate.model_name,
        eligible=not reasons,
        reasons=reasons,
    )
