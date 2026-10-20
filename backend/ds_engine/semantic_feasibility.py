"""Dataset-grounded feasibility checks for validated ToolSpec commands.

The planner decides the user's intent. This module never re-classifies raw
natural-language text with keywords; it only proves whether the selected tool
has the explicit, structurally valid inputs needed for safe execution.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.ds_engine.dataset_profiler import DatasetProfile
from backend.ds_engine.intent_schema import FilterSpec


class FeasibilityCheckResult(BaseModel):
    is_feasible: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    clarification_prompt: Optional[str] = None
    candidate_columns: List[str] = Field(default_factory=list)
    suggested_adjustments: Dict[str, Any] = Field(default_factory=dict)


def _normalise_column_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    # Retain combining vowel marks: dropping these conflates Hindi names.
    return "".join(c for c in text if c.isalnum() or unicodedata.category(c).startswith("M"))


def _resolve_explicit_column(profile: DatasetProfile, requested: Any) -> Optional[str]:
    """Resolve an explicitly supplied name, allowing only unambiguous formatting aliases."""
    if not isinstance(requested, str) or not requested.strip():
        return None
    if requested in profile.columns:
        return requested
    normalised = _normalise_column_name(requested)
    if not normalised:
        return None
    matches = [name for name in profile.columns if _normalise_column_name(name) == normalised]
    return matches[0] if len(matches) == 1 else None


def _find_sequence_candidates(profile: Optional[DatasetProfile]) -> List[str]:
    """Return only structurally evidenced sequences; names are never inspected."""
    if not profile:
        return []
    candidates: List[str] = []
    for name, column in profile.columns.items():
        numeric = "int" in column.dtype.lower() or "float" in column.dtype.lower()
        if numeric and column.is_unique and (
            column.is_monotonic_increasing or column.is_monotonic_decreasing
        ):
            candidates.append(name)
    return candidates


def _clarification(message: str, candidates: List[str]) -> FeasibilityCheckResult:
    # A column can legitimately have more than one semantic role in the
    # dataset profile (for example binary target and binary flag).  Preserve
    # profile order while presenting each valid candidate only once.
    candidates = list(dict.fromkeys(candidates))
    choice = ", ".join(f"`{column}`" for column in candidates)
    suffix = f" Available candidates: {choice}." if candidates else ""
    return FeasibilityCheckResult(
        is_feasible=False,
        error_code="CLARIFICATION_REQUIRED",
        error_message=message,
        clarification_prompt=message + suffix,
        candidate_columns=candidates,
    )


class SemanticFeasibilityValidator:
    """Validates selected ToolSpecs using only contracts and dataset evidence."""

    @classmethod
    def validate_plan(
        cls,
        primary_tool_name: str,
        tool_args: Dict[str, Any],
        profile: Optional[DatasetProfile],
        query_raw: str = "",  # API compatibility only; deliberately not interpreted.
        df: Optional[Any] = None,
    ) -> FeasibilityCheckResult:
        from modules.capability_registry import capability_registry

        spec = capability_registry.get_tool_spec((primary_tool_name or "").lower().strip())
        if not spec:
            return FeasibilityCheckResult(
                is_feasible=False,
                error_code="UNSUPPORTED_OPERATION",
                error_message=f"Unknown or unsupported semantic tool: '{primary_tool_name}'",
            )

        if spec.requires_dataset and (profile is None or not profile.columns):
            return FeasibilityCheckResult(
                is_feasible=False,
                error_code="NO_DATASET_LOADED",
                error_message="An active dataset is required for this operation.",
            )

        prereq = spec.prerequisites if (spec and profile) else None

        if prereq:
            if prereq.requires_temporal_column:
                resolved = _resolve_explicit_column(profile, tool_args.get("time_column"))
                if not resolved:
                    if profile.temporal_column_candidates:
                        return _clarification(
                            f"Tool '{spec.name}' requires an explicit temporal column.",
                            profile.temporal_column_candidates,
                        )
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="NO_TEMPORAL_COLUMN",
                        error_message=f"Tool '{spec.name}' requires a temporal column, but none is present in this dataset.",
                    )
                if resolved not in profile.temporal_column_candidates:
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="INVALID_TEMPORAL_COLUMN",
                        error_message=f"Column '{resolved}' is not a temporal column for tool '{spec.name}'.",
                    )
                tool_args["time_column"] = resolved

            if prereq.requires_ordering_column:
                time_col = _resolve_explicit_column(profile, tool_args.get("time_column"))
                sequence_col = _resolve_explicit_column(
                    profile,
                    tool_args.get("sequence_column") or tool_args.get("sequence_order") or tool_args.get("order_column"),
                )
                if time_col and sequence_col:
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="AMBIGUOUS_ORDERING_CONTEXT",
                        error_message="Provide either one temporal column or one sequence-order column, not both.",
                    )
                if time_col:
                    if time_col in profile.temporal_column_candidates:
                        tool_args["time_column"] = time_col
                    elif time_col in _find_sequence_candidates(profile):
                        tool_args["time_column"] = time_col
                        tool_args["sequence_column"] = time_col
                    else:
                        return FeasibilityCheckResult(
                            is_feasible=False,
                            error_code="INVALID_ORDERING_COLUMN",
                            error_message=f"Column '{time_col}' is not a temporal or monotonic sequence column.",
                        )
                elif sequence_col:
                    if sequence_col not in _find_sequence_candidates(profile):
                        return FeasibilityCheckResult(
                            is_feasible=False,
                            error_code="INVALID_ORDERING_COLUMN",
                            error_message=f"Column '{sequence_col}' is not a unique monotonic sequence column.",
                        )
                    tool_args["sequence_column"] = sequence_col
                else:
                    candidates = list(profile.temporal_column_candidates) + _find_sequence_candidates(profile)
                    if candidates:
                        return _clarification(
                            f"Tool '{spec.name}' requires an explicit temporal or sequence-order column.",
                            candidates,
                        )
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="NO_ORDERING_COLUMN",
                        error_message=f"Tool '{spec.name}' requires a temporal or structurally ordered sequence column.",
                    )

            if prereq.requires_event_column:
                predicate = tool_args.get("event_predicate")
                predicate_column = predicate.get("column") if isinstance(predicate, dict) else None
                resolved = _resolve_explicit_column(profile, tool_args.get("event_column") or predicate_column)
                if not resolved:
                    return _clarification(
                        f"Tool '{spec.name}' requires an explicit event column.",
                        list(profile.binary_flag_candidates) + list(profile.target_column_candidates),
                    )
                tool_args["event_column"] = resolved

            if prereq.requires_event_predicate:
                predicate = tool_args.get("event_predicate")
                if not isinstance(predicate, dict) or not any(key in predicate for key in ("operator", "condition", "op")):
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="EVENT_PREDICATE_REQUIRED",
                        error_message=f"Tool '{spec.name}' requires an explicit structured event predicate.",
                    )
                predicate_column = _resolve_explicit_column(
                    profile, predicate.get("column") or tool_args.get("event_column")
                )
                if predicate_column != tool_args.get("event_column"):
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="EVENT_PREDICATE_COLUMN_MISMATCH",
                        error_message="The event predicate column must match the explicit event column.",
                    )
                try:
                    parsed = FilterSpec.model_validate({
                        **predicate,
                        "column": predicate_column,
                    })
                except ValueError as exc:
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="EVENT_PREDICATE_REQUIRED",
                        error_message=str(exc),
                    )
                tool_args["event_predicate"] = parsed.model_dump(mode="json")

            if prereq.requires_pre_event_window:
                raw_window = tool_args.get("pre_event_window")
                try:
                    window = int(raw_window) if raw_window is not None else None
                except (TypeError, ValueError):
                    window = None
                if window is None or window < 1:
                    return _clarification(
                        f"Tool '{spec.name}' requires an explicit positive pre-event lookback window (number of ordered records before each event).",
                        [],
                    )
                tool_args["pre_event_window"] = window

            if prereq.requires_numeric_subjects:
                numeric_columns = [
                    name for name, column in profile.columns.items()
                    if "int" in column.dtype.lower() or "float" in column.dtype.lower()
                ]
                required_count = max(prereq.min_numeric_columns, 1)
                if len(numeric_columns) < required_count:
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="INSUFFICIENT_NUMERIC_COLUMNS",
                        error_message=f"Tool '{spec.name}' requires at least {required_count} numeric column(s).",
                    )

            if prereq.requires_target_column and not _resolve_explicit_column(
                profile, tool_args.get("target_column") or tool_args.get("target")
            ):
                return _clarification(
                    f"Tool '{spec.name}' requires an explicit target column.",
                    list(profile.target_column_candidates) + list(profile.binary_flag_candidates),
                )

            if prereq.requires_grouping_or_dimension:
                raw_grouping = next(
                    (
                        tool_args.get(key)
                        for key in ("group_by", "dimensions", "column", "columns")
                        if tool_args.get(key)
                    ),
                    None,
                )
                if not raw_grouping:
                    return _clarification(
                        f"Tool '{spec.name}' requires an explicit grouping or dimension column.",
                        list(profile.categorical_candidates),
                    )
                group_values = raw_grouping if isinstance(raw_grouping, list) else [raw_grouping]
                if not all(isinstance(value, str) and value.strip() for value in group_values):
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="INVALID_GROUP_BY",
                        error_message=f"Tool '{spec.name}' requires group_by to contain non-empty column names.",
                    )
                resolved_grouping = [_resolve_explicit_column(profile, value) for value in group_values]
                missing_grouping = [
                    str(value) for value, resolved in zip(group_values, resolved_grouping)
                    if resolved is None
                ]
                if missing_grouping:
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="COLUMN_NOT_FOUND",
                        error_message=(
                            f"Tool '{spec.name}' requested grouping column(s) "
                            f"{missing_grouping} that are not present in the active dataset."
                        ),
                    )
                # Preserve one canonical key for every later pipeline stage.
                tool_args["group_by"] = [str(value) for value in resolved_grouping]

            condition_argument_name = prereq.condition_argument_name
            if condition_argument_name:
                raw_condition = tool_args.get(condition_argument_name)
                if not isinstance(raw_condition, dict):
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="STRUCTURED_CONDITION_REQUIRED",
                        error_message=(
                            f"Tool '{spec.name}' requires a structured '{condition_argument_name}' "
                            "with column, operator, and value."
                        ),
                    )
                raw_column = raw_condition.get("column") or raw_condition.get("col") or raw_condition.get("field")
                resolved_column = _resolve_explicit_column(profile, raw_column)
                if not resolved_column:
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="COLUMN_NOT_FOUND",
                        error_message=(
                            f"Tool '{spec.name}' requested condition column '{raw_column}' "
                            "that is not present in the active dataset."
                        ),
                    )
                try:
                    parsed_condition = FilterSpec.model_validate({
                        **raw_condition,
                        "column": resolved_column,
                    })
                except ValueError as exc:
                    return FeasibilityCheckResult(
                        is_feasible=False,
                        error_code="STRUCTURED_CONDITION_REQUIRED",
                        error_message=str(exc),
                    )
                tool_args[condition_argument_name] = parsed_condition.model_dump(mode="json")

        # Scientific Feasibility Rules Gates (time-series grain, ML contract, statistical assumptions)
        from backend.ds_engine.feasibility_rules import FeasibilityRulesEngine

        for rule_func in (
            FeasibilityRulesEngine.check_time_series_grain,
            FeasibilityRulesEngine.check_supervised_ml_contract,
            FeasibilityRulesEngine.check_statistical_assumptions,
        ):
            rule_res = rule_func(primary_tool_name, tool_args, df)
            if rule_res and not rule_res.is_feasible:
                is_clarification = "CLARIFICATION" in (rule_res.error_code or "")
                return FeasibilityCheckResult(
                    is_feasible=False,
                    error_code="CLARIFICATION_REQUIRED" if is_clarification else (rule_res.error_code or "FEASIBILITY_CHECK_FAILED"),
                    error_message=rule_res.error_message,
                    clarification_prompt=rule_res.clarification_prompt or (rule_res.error_message if is_clarification else None),
                    candidate_columns=rule_res.candidate_columns,
                    suggested_adjustments=rule_res.suggested_adjustments,
                )

        return FeasibilityCheckResult(is_feasible=True)
