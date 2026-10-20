"""
DataPilot OS — Semantic Command Resolver & Grounding Engine.
Translates minimal LLM semantic commands / tool calls into fully validated,
type-checked, dataset-grounded DataPilot SemanticIntent and ExecutionPlan objects.

CORE PRINCIPLE:
Ollama understands WHAT the user means.
DataPilot decides HOW to execute it.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field

from backend.ds_engine.intent_schema import (
    SemanticIntent,
    IntentType,
    FilterSpec,
    FilterOperator,
    AggregationType,
    ChartType
)
from backend.ds_engine.semantic_command import (
    LLMSemanticCommand,
    ToolCall,
    SemanticTool
)
from backend.ds_engine.intent_resolver import (
    intent_resolver,
    AmbiguousIntentError,
    IntentResolutionError,
    ResolvedIntent
)
from backend.ds_engine.dataset_grounding import DatasetGrounding

logger = logging.getLogger(__name__)


# Comprehensive Natural Language Operator Synonym Mapping
OPERATOR_MAP: Dict[str, FilterOperator] = {
    # Greater than
    "gt": FilterOperator.GT,
    ">": FilterOperator.GT,
    "greater": FilterOperator.GT,
    "greater_than": FilterOperator.GT,
    "greater than": FilterOperator.GT,
    "above": FilterOperator.GT,
    "strictly above": FilterOperator.GT,
    "strictly_above": FilterOperator.GT,
    "exceeding": FilterOperator.GT,
    "more than": FilterOperator.GT,
    "more_than": FilterOperator.GT,
    "over": FilterOperator.GT,
    "higher than": FilterOperator.GT,
    "higher_than": FilterOperator.GT,
    "strictly_greater": FilterOperator.GT,

    # Greater than or equal
    "gte": FilterOperator.GTE,
    ">=": FilterOperator.GTE,
    "greater_or_equal": FilterOperator.GTE,
    "greater than or equal": FilterOperator.GTE,
    "greater than or equal to": FilterOperator.GTE,
    "at least": FilterOperator.GTE,
    "at_least": FilterOperator.GTE,
    "minimum": FilterOperator.GTE,
    "no less than": FilterOperator.GTE,
    "from": FilterOperator.GTE,

    # Less than
    "lt": FilterOperator.LT,
    "<": FilterOperator.LT,
    "less": FilterOperator.LT,
    "less_than": FilterOperator.LT,
    "less than": FilterOperator.LT,
    "below": FilterOperator.LT,
    "under": FilterOperator.LT,
    "lower than": FilterOperator.LT,
    "lower_than": FilterOperator.LT,
    "lesser than": FilterOperator.LT,
    "strictly_less": FilterOperator.LT,

    # Less than or equal
    "lte": FilterOperator.LTE,
    "<=": FilterOperator.LTE,
    "less_or_equal": FilterOperator.LTE,
    "less than or equal": FilterOperator.LTE,
    "less than or equal to": FilterOperator.LTE,
    "at most": FilterOperator.LTE,
    "at_most": FilterOperator.LTE,
    "maximum": FilterOperator.LTE,
    "up to": FilterOperator.LTE,
    "no more than": FilterOperator.LTE,

    # Equal
    "eq": FilterOperator.EQ,
    "=": FilterOperator.EQ,
    "==": FilterOperator.EQ,
    "equal": FilterOperator.EQ,
    "equals": FilterOperator.EQ,
    "is": FilterOperator.EQ,
    "exact": FilterOperator.EQ,
    "match": FilterOperator.EQ,
    "matches": FilterOperator.EQ,
    "same as": FilterOperator.EQ,

    # Not equal
    "neq": FilterOperator.NEQ,
    "!=": FilterOperator.NEQ,
    "<>": FilterOperator.NEQ,
    "not_equal": FilterOperator.NEQ,
    "not equal": FilterOperator.NEQ,
    "not equal to": FilterOperator.NEQ,
    "is not": FilterOperator.NEQ,
    "is_not": FilterOperator.NEQ,
    "does not equal": FilterOperator.NEQ,
    "different from": FilterOperator.NEQ,

    # Set membership
    "in": FilterOperator.IN,
    "isin": FilterOperator.IN,
    "within": FilterOperator.IN,
    "one of": FilterOperator.IN,
    "one_of": FilterOperator.IN,
    "any of": FilterOperator.IN,
    "inside": FilterOperator.IN,

    # Set non-membership
    "not_in": FilterOperator.NOT_IN,
    "notin": FilterOperator.NOT_IN,
    "not in": FilterOperator.NOT_IN,
    "none of": FilterOperator.NOT_IN,
    "outside": FilterOperator.NOT_IN,

    # Nullity / Missingness
    "is_null": FilterOperator.IS_NULL,
    "isnull": FilterOperator.IS_NULL,
    "isna": FilterOperator.IS_NULL,
    "null": FilterOperator.IS_NULL,
    "missing": FilterOperator.IS_NULL,
    "empty": FilterOperator.IS_NULL,
    "nan": FilterOperator.IS_NULL,
    "na": FilterOperator.IS_NULL,
    "none": FilterOperator.IS_NULL,

    # Not Null
    "is_not_null": FilterOperator.IS_NOT_NULL,
    "notnull": FilterOperator.IS_NOT_NULL,
    "notna": FilterOperator.IS_NOT_NULL,
    "not_null": FilterOperator.IS_NOT_NULL,
    "not null": FilterOperator.IS_NOT_NULL,
    "present": FilterOperator.IS_NOT_NULL,
    "not missing": FilterOperator.IS_NOT_NULL,
    "non_null": FilterOperator.IS_NOT_NULL,

    # String contains
    "contains": FilterOperator.CONTAINS,
    "like": FilterOperator.CONTAINS,
    "includes": FilterOperator.CONTAINS,
    "has": FilterOperator.CONTAINS,
    "substring": FilterOperator.CONTAINS
}

# Aggregation Function Synonym Mapping
AGGREGATION_MAP: Dict[str, AggregationType] = {
    "mean": AggregationType.MEAN,
    "avg": AggregationType.MEAN,
    "average": AggregationType.MEAN,
    "mean_value": AggregationType.MEAN,
    "sum": AggregationType.SUM,
    "total": AggregationType.SUM,
    "add": AggregationType.SUM,
    "summation": AggregationType.SUM,
    "count": AggregationType.COUNT,
    "number": AggregationType.COUNT,
    "freq": AggregationType.COUNT,
    "frequency": AggregationType.COUNT,
    "num_rows": AggregationType.COUNT,
    "min": AggregationType.MIN,
    "minimum": AggregationType.MIN,
    "lowest": AggregationType.MIN,
    "smallest": AggregationType.MIN,
    "max": AggregationType.MAX,
    "maximum": AggregationType.MAX,
    "highest": AggregationType.MAX,
    "largest": AggregationType.MAX,
    "greatest": AggregationType.MAX,
    "median": AggregationType.MEDIAN,
    "middle": AggregationType.MEDIAN,
    "p50": AggregationType.MEDIAN,
    "50th percentile": AggregationType.MEDIAN
}


class CommandResolutionResult(BaseModel):
    """
    Standard outcome returned by CommandResolver.
    """
    is_valid: bool
    semantic_intent: Optional[SemanticIntent] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    candidate_columns: List[str] = Field(default_factory=list)
    slot_filling: Optional[Dict[str, Any]] = None
    inferred_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    is_dataset_wide: bool = False


def _get_first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely retrieves the value of the first key that is present in dictionary d.
    Crucially preserves 0, False, and empty strings ("").
    """
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class CommandResolver:
    """
    Authoritative Command Resolver:
    Translates minimal LLM semantic commands / tool calls into valid SemanticIntent and ExecutionPlans.
    """

    @classmethod
    def resolve_operator(cls, raw_op: Any) -> Optional[FilterOperator]:
        """
        Normalizes any natural language condition, symbol, or string into a canonical FilterOperator.
        Returns None if unrecognized (no silent fallback).
        """
        if isinstance(raw_op, FilterOperator):
            return raw_op
        if raw_op is None or raw_op == "":
            return FilterOperator.EQ

        clean = str(raw_op).lower().strip()
        if clean in OPERATOR_MAP:
            op = OPERATOR_MAP[clean]
            logger.info(f"AUDIT_NORMALIZATION_OPERATOR | Raw: '{raw_op}' -> Canonical: {op.value}")
            return op
        elif ">=" in clean or "at least" in clean or "greater than or equal" in clean or "minimum of" in clean:
            return FilterOperator.GTE
        elif "<=" in clean or "at most" in clean or "less than or equal" in clean or "maximum of" in clean:
            return FilterOperator.LTE
        elif ">" in clean or "above" in clean or "greater" in clean or "more than" in clean or "higher than" in clean or "exceeding" in clean or "strictly above" in clean:
            return FilterOperator.GT
        elif "<" in clean or "below" in clean or "less" in clean or "under" in clean or "fewer than" in clean or "lower than" in clean or "strictly below" in clean:
            return FilterOperator.LT
        elif "!=" in clean or "<>" in clean or "not equal" in clean or "different from" in clean or "excluding" in clean:
            return FilterOperator.NEQ
        elif "contains" in clean or "like" in clean or "including" in clean:
            return FilterOperator.CONTAINS
        elif "missing" in clean or "is null" in clean or "null" in clean:
            return FilterOperator.IS_NULL
        elif "=" in clean or "equals" in clean or "is" in clean:
            return FilterOperator.EQ

        for k, v in OPERATOR_MAP.items():
            if len(k) > 2 and k in clean:
                logger.info(f"AUDIT_NORMALIZATION_OPERATOR_SUBSTRING | Raw: '{raw_op}' -> Canonical: {v.value}")
                return v
            elif re.search(rf'\b{re.escape(k)}\b', clean):
                logger.info(f"AUDIT_NORMALIZATION_OPERATOR_TOKEN | Raw: '{raw_op}' -> Canonical: {v.value}")
                return v

        logger.warning(f"AUDIT_NORMALIZATION_OPERATOR_UNRECOGNIZED | Raw: '{raw_op}'")
        return None

    @classmethod
    def resolve_aggregation(cls, raw_agg: Any) -> Optional[AggregationType]:
        """
        Normalizes any natural language aggregation name into canonical AggregationType.
        Returns None if unrecognized (no silent fallback).
        """
        if isinstance(raw_agg, AggregationType):
            return raw_agg
        if not raw_agg:
            return None

        clean = str(raw_agg).lower().strip()
        if clean in AGGREGATION_MAP:
            resolved = AGGREGATION_MAP[clean]
            logger.info(f"AUDIT_NORMALIZATION_AGGREGATION | Raw: '{raw_agg}' -> Canonical: {resolved.value}")
            return resolved

        for k, v in AGGREGATION_MAP.items():
            if k in clean:
                logger.info(f"AUDIT_NORMALIZATION_AGGREGATION_SUBSTRING | Raw: '{raw_agg}' -> Canonical: {v.value}")
                return v

        logger.warning(f"AUDIT_NORMALIZATION_AGGREGATION_UNRECOGNIZED | Raw: '{raw_agg}'")
        return None

    @classmethod
    def resolve_order(cls, raw_order: Any) -> Optional[str]:
        """
        Normalizes sort order to 'asc', 'desc', or None.
        """
        if raw_order is None:
            return None
        clean = str(raw_order).lower().strip()
        if clean in ["desc", "descending", "highest", "top", "largest", "decreasing", "downward"]:
            logger.info(f"AUDIT_NORMALIZATION_ORDER | Raw: '{raw_order}' -> Canonical: 'desc'")
            return "desc"
        if clean in ["asc", "ascending", "lowest", "bottom", "smallest", "increasing", "upward"]:
            logger.info(f"AUDIT_NORMALIZATION_ORDER | Raw: '{raw_order}' -> Canonical: 'asc'")
            return "asc"
        return None

    @classmethod
    def normalize_filter_spec(cls, f_item: Any) -> Optional[FilterSpec]:
        """
        Converts a structured filter object or FilterSpec into a validated FilterSpec.
        Strictly requires structured dictionary with column, operator, and value (preserves 0, False, '').
        Does NOT accept loose unstructured strings.
        """
        if isinstance(f_item, FilterSpec):
            return f_item

        if isinstance(f_item, dict):
            col = _get_first_present(f_item, ["column", "col", "field", "name", "feature"])
            if not col or not isinstance(col, str) or not col.strip():
                return None
            op_raw = _get_first_present(f_item, ["operator", "condition", "op"], default="==")
            op = cls.resolve_operator(op_raw)
            if op is None:
                logger.warning(f"AUDIT_NORMALIZATION_FILTER | Dropped invalid operator: '{op_raw}'")
                return None
            
            val = _get_first_present(f_item, ["value", "val", "target", "v"])
            if op not in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL] and val is None:
                logger.warning(f"AUDIT_NORMALIZATION_FILTER | Missing required value for operator {op.value}")
                return None

            spec = FilterSpec(
                column=str(col).strip(),
                operator=op,
                value=val
            )
            logger.info(f"AUDIT_NORMALIZATION_FILTER | Dict: {f_item} -> FilterSpec(col='{spec.column}', op={spec.operator.value}, val={spec.value})")
            return spec

        return None

    @classmethod
    def _extract_string_list(cls, val: Any, filter_aggregations: bool = False) -> List[str]:
        """Helper to extract a clean list of strings from list, string, or None."""
        if not val:
            return []
        if isinstance(val, str):
            items = [s.strip() for s in val.split(",") if s.strip()]
        elif isinstance(val, list):
            items = []
            for it in val:
                if isinstance(it, str):
                    items.extend([s.strip() for s in it.split(",") if s.strip()])
                elif it is not None:
                    items.append(str(it).strip())
        else:
            items = [str(val).strip()]

        # Filter out junk dummy values
        cleaned = []
        for it in items:
            it_low = it.lower()
            if it_low in ["none", "null", "n/a", "", "nan"]:
                continue
            if filter_aggregations and it_low in ["mean", "sum", "avg", "average", "count", "std", "min", "max", "median"]:
                continue
            cleaned.append(it)
        return cleaned

    @classmethod
    def get_effective_tool_intent_map(cls) -> Dict[str, IntentType]:
        """
        Dynamically returns tool mappings directly from CapabilityRegistry.
        Fails fast if CapabilityRegistry is empty or unavailable (zero fallback).
        """
        from modules.capability_registry import capability_registry
        reg_map = capability_registry.get_tool_intent_map()
        if not reg_map:
            raise RuntimeError("CAPABILITY_REGISTRY_UNAVAILABLE: No tool mappings found in CapabilityRegistry.")
        return reg_map

    @staticmethod
    def _validate_quantified_request_evidence(
        spec: Any,
        validated_args: Any,
        raw_query: str,
    ) -> Optional[str]:
        """Prove that a capability's numeric parameter came from the user.

        ToolSpecs declare this requirement themselves. The resolver does not
        classify natural language or route a query; it only checks an exact
        evidence span supplied by the model against the original request.
        """
        prereq = getattr(spec, "prerequisites", None)
        if not prereq or not getattr(prereq, "requires_quantified_request_evidence", False):
            return None

        # Query-less calls are internal, already-validated typed invocations.
        # Every LLM-originated command carries raw_query and must be grounded.
        if not raw_query:
            return None

        quantity_name = getattr(prereq, "quantity_argument_name", None)
        evidence_name = getattr(prereq, "quantity_evidence_argument_name", None)
        quantity = getattr(validated_args, quantity_name, None) if quantity_name else None
        evidence = getattr(validated_args, evidence_name, None) if evidence_name else None
        evidence_text = evidence.strip() if isinstance(evidence, str) else ""

        if quantity is None or not evidence_text:
            return (
                f"Tool '{spec.name}' requires an exact user-query evidence phrase for its "
                f"numeric '{quantity_name}' argument. Do not invent a number; choose a different "
                "capability or request clarification when the user did not specify one."
            )

        if evidence_text.casefold() not in raw_query.casefold():
            return (
                f"Tool '{spec.name}' supplied '{evidence_name}' that is not an exact phrase from "
                "the user's request. Numeric execution parameters must be grounded in the request."
            )

        # The exact evidence phrase must contain the numeric quantity used in
        # the plan. Boundaries prevent 3 from being treated as proof of 30.
        quantity_pattern = rf"(?<!\d){re.escape(str(quantity))}(?!\d)"
        if not re.search(quantity_pattern, evidence_text):
            return (
                f"Tool '{spec.name}' numeric '{quantity_name}'={quantity} is not present in its "
                f"declared evidence phrase '{evidence_text}'. Do not manufacture a horizon or count."
            )
        return None

    @staticmethod
    def _validate_numeric_measure_contract(
        spec: Any,
        validated_args: Any,
        df: Optional[pd.DataFrame],
    ) -> Optional[str]:
        """Validate registry-declared aggregation measures before execution.

        The registry supplies argument names and count-like exceptions.  This
        prevents a planner from binding a date, identifier, or category as the
        measure for a numeric aggregation while keeping the resolver unaware
        of any particular analytical domain or tool name.
        """
        prereq = getattr(spec, "prerequisites", None)
        if (
            df is None
            or not prereq
            or not getattr(prereq, "requires_numeric_measure", False)
        ):
            return None

        measure_names: List[str] = []
        for argument_name in getattr(prereq, "measure_argument_names", []):
            raw_value = getattr(validated_args, argument_name, None)
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                if isinstance(value, str) and value.strip() and value not in measure_names:
                    measure_names.append(value)

        # A count may be valid without a measure, but whenever the model did
        # name a measure it still has to be an exact active-dataset column.
        # Without this check a planner can emit display expressions such as
        # ``sum(metric)`` and defer a deterministic grounding failure until
        # after the expensive execution phase.
        missing = [name for name in measure_names if name not in df.columns]
        if missing:
            return (
                f"Tool '{spec.name}' requested measure column(s) {missing} that are not present in the active dataset. "
                "Provide exact column names only; aggregation expressions belong in the aggregation argument, not in a measure name."
            )

        aggregation_name = getattr(prereq, "aggregation_argument_name", None)
        aggregation = getattr(validated_args, aggregation_name, None) if aggregation_name else None
        allowed_non_numeric = {
            str(value).strip().lower()
            for value in getattr(prereq, "non_numeric_measure_aggregations", [])
        }
        if aggregation is not None and str(aggregation).strip().lower() in allowed_non_numeric:
            return None

        non_numeric = [
            name for name in measure_names
            if name in df.columns and not pd.api.types.is_numeric_dtype(df[name])
        ]
        if non_numeric:
            return (
                f"Tool '{spec.name}' requires numeric measure column(s) for aggregation "
                f"'{aggregation}'; non-numeric measure(s): {non_numeric}. "
                "Use a numeric measure from the dataset or choose an explicit count."
            )
        return None

    @staticmethod
    def _validate_visualization_contract(
        spec: Any,
        validated_args: Any,
    ) -> Optional[str]:
        """Reject a visual promise the selected typed operator cannot fulfil.

        ToolSpecs declare the physical output formats; this check performs no
        natural-language routing and never substitutes a chart.  Returning a
        normal command-resolution error lets the semantic planner repair its
        tool choice before any analysis executes.
        """
        supported = getattr(spec, "supported_chart_types", None)
        if supported is None:
            return None

        requested = getattr(validated_args, "chart_type", None)
        if requested is None:
            requested = getattr(validated_args, "chart", None)
        if requested is None:
            return None

        requested_name = str(getattr(requested, "value", requested)).strip().lower()
        if not requested_name or requested_name in {"none", "null"}:
            return None
        supported_names = {
            str(getattr(value, "value", value)).strip().lower()
            for value in supported
        }
        if requested_name not in supported_names:
            supported_text = ", ".join(sorted(supported_names)) or "none"
            return (
                f"Tool '{spec.name}' cannot generate chart '{requested_name}'. "
                f"Its registered output contract supports: {supported_text}. "
                "Choose a chart-capable tool or request a table-only result."
            )
        return None

    @classmethod
    def _validate_dataset_column_grounding(
        cls,
        spec: Any,
        validated_args: Any,
        df: Optional[pd.DataFrame],
    ) -> Optional[str]:
        """Validate that all referenced column arguments exist in the active dataset."""
        if df is None or not hasattr(df, "columns") or df.columns.empty:
            return None

        available_cols = [str(c) for c in df.columns]
        referenced_cols: List[str] = []

        # Check standard column argument names
        col_arg_keys = [
            "column", "columns", "metric", "metrics", "group_by",
            "target_column", "time_column", "sequence_column", "entity_id",
            "feature_columns", "features", "dimension", "dimensions",
        ]
        for key in col_arg_keys:
            val = getattr(validated_args, key, None)
            if val is not None:
                if isinstance(val, str) and val.strip():
                    referenced_cols.append(val.strip())
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item.strip():
                            referenced_cols.append(item.strip())

        # Check nested filter items if present
        filters_val = getattr(validated_args, "filters", None)
        if isinstance(filters_val, list):
            for f in filters_val:
                f_col = getattr(f, "column", None) or (f.get("column") if isinstance(f, dict) else None)
                if isinstance(f_col, str) and f_col.strip():
                    referenced_cols.append(f_col.strip())

        missing_with_suggestions = []
        for col_name in referenced_cols:
            match = DatasetGrounding.find_column_match(col_name, available_cols)
            if not match.found:
                if match.suggestions:
                    missing_with_suggestions.append(
                        f"'{col_name}' (did you mean: {', '.join(repr(s) for s in match.suggestions)}?)"
                    )
                else:
                    missing_with_suggestions.append(f"'{col_name}'")

        if missing_with_suggestions:
            unique_missing = list(dict.fromkeys(missing_with_suggestions))
            return (
                f"Tool '{spec.name}' referenced column(s) not found in the dataset: "
                + ", ".join(unique_missing)
                + ". Please specify exact column names."
            )

        return None

    @classmethod
    def resolve_command(
        cls,
        command_data: Union[LLMSemanticCommand, Dict[str, Any], List[Dict[str, Any]], str],
        df: Optional[pd.DataFrame] = None,
        raw_query: str = ""
    ) -> CommandResolutionResult:
        """
        Main entry point to resolve an LLM semantic command / tool calls into a validated SemanticIntent.
        """
        tool_intent_map = cls.get_effective_tool_intent_map()

        # Step 1: Normalize input into list of tool calls
        tool_calls: List[ToolCall] = []
        clarification_reason: Optional[str] = None

        if isinstance(command_data, LLMSemanticCommand):
            if command_data.clarification_reason:
                clarification_reason = command_data.clarification_reason
            if command_data.commands:
                tool_calls = list(command_data.commands)
                if command_data.tool:
                    top_tc = ToolCall(tool=command_data.tool, arguments=command_data.arguments or {})
                    if top_tc not in tool_calls:
                        tool_calls.insert(0, top_tc)
            elif command_data.tool:
                tool_calls = [ToolCall(tool=command_data.tool, arguments=command_data.arguments or {})]

        elif isinstance(command_data, list):
            for item in command_data:
                tool_name = item.get("tool") or item.get("name") if isinstance(item, dict) else None
                if isinstance(item, dict) and tool_name:
                    tool_calls.append(ToolCall(tool=tool_name, arguments=item.get("arguments", {})))
                elif isinstance(item, ToolCall):
                    tool_calls.append(item)

        elif isinstance(command_data, dict):
            # Check for multi-command list
            cmds_list = command_data.get("commands") if "commands" in command_data else command_data.get("tool_calls")
            if cmds_list and isinstance(cmds_list, list):
                top_name = command_data.get("tool") or command_data.get("name")
                if top_name:
                    tool_calls.append(ToolCall(tool=top_name, arguments=command_data.get("arguments", {})))
                for item in cmds_list:
                    tool_name = item.get("tool") or item.get("name") if isinstance(item, dict) else None
                    if isinstance(item, dict) and tool_name:
                        tool_calls.append(ToolCall(tool=tool_name, arguments=item.get("arguments", {})))
                    elif isinstance(item, ToolCall):
                        tool_calls.append(item)

            top_tool_name = command_data.get("tool") or command_data.get("name")
            if top_tool_name:
                top_tool = ToolCall(tool=top_tool_name, arguments=command_data.get("arguments", {}))
                if not any(tc.tool == top_tool.tool for tc in tool_calls):
                    tool_calls.insert(0, top_tool)

            if command_data.get("clarification_reason"):
                clarification_reason = command_data["clarification_reason"]
            elif command_data.get("ambiguity_reason"):
                clarification_reason = command_data["ambiguity_reason"]
        
        # Legacy fallback
        if not tool_calls and not clarification_reason and isinstance(command_data, dict) and "primary_intent" in command_data:
            return CommandResolutionResult(
                is_valid=False,
                error_code="UNSUPPORTED_OPERATION",
                error_message="Legacy unstructured intent dictionary is deprecated and rejected. All commands must use structured ToolCall conforming to CapabilityRegistry ToolSpec."
            )

        if not tool_calls and not clarification_reason:
            return CommandResolutionResult(
                is_valid=False,
                error_code="UNSUPPORTED_OPERATION",
                error_message="No valid semantic tool call or command was recognized in LLM output."
            )

        # Step 2a: Handle direct General Chat / Conversational Queries via ToolSpec
        from modules.capability_registry import capability_registry
        chat_calls = [
            tc for tc in tool_calls
            if (spec := capability_registry.get_tool_spec(tc.tool.lower().strip()))
            and spec.category == "general_chat"
        ]
        if chat_calls:
            tc = chat_calls[0]
            spec = capability_registry.get_tool_spec(tc.tool.lower().strip())
            state = {}
            if spec and spec.adapter:
                try:
                    validated_args = spec.input_model.model_validate(tc.arguments or {})
                    spec.adapter(validated_args, state)
                except Exception as val_err:
                    return CommandResolutionResult(
                        is_valid=False,
                        error_code="INVALID_ARGUMENTS",
                        error_message=f"Validation failed for tool '{tc.tool}': {val_err}"
                    )
            response_text = state.get("response_text") or (tc.arguments or {}).get("response", "")
            intent = SemanticIntent(
                intent_type=IntentType.GENERAL_CHAT,
                primary_intent=IntentType.GENERAL_CHAT,
                raw_query=raw_query or "",
                is_ambiguous=False,
                prompt_summary=response_text
            )
            return CommandResolutionResult(
                is_valid=True,
                semantic_intent=intent,
                is_ambiguous=False,
                error_message=response_text if response_text else None
            )

        # Step 2b: Handle explicit Clarification / Ambiguity via ToolSpec or clarification_reason
        clarify_calls = [
            tc for tc in tool_calls
            if (spec := capability_registry.get_tool_spec(tc.tool.lower().strip()))
            and spec.category == "clarification"
        ]
        if clarification_reason or clarify_calls:
            # A clarification is terminal conversational output, never an
            # execution-pipeline step. Reject a mixed response so the planner
            # repairs it instead of discarding a partially specified analysis.
            # This is a protocol rule, not a query or dataset heuristic.
            executable_calls = [
                tc for tc in tool_calls
                if not (
                    (spec := capability_registry.get_tool_spec(tc.tool.lower().strip()))
                    and spec.category == "clarification"
                )
            ]
            if executable_calls:
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="MIXED_CLARIFICATION_AND_ANALYSIS",
                    error_message=(
                        "A planner response must contain either one complete analytical tool call "
                        "or clarification_needed, never both. Rebuild one complete grounded analysis command."
                    ),
                )
            reason = clarification_reason
            if clarify_calls:
                tc = clarify_calls[0]
                spec = capability_registry.get_tool_spec(tc.tool.lower().strip())
                state = {}
                if spec and spec.adapter:
                    try:
                        validated_args = spec.input_model.model_validate(tc.arguments or {})
                        spec.adapter(validated_args, state)
                    except Exception as val_err:
                        return CommandResolutionResult(
                            is_valid=False,
                            error_code="INVALID_ARGUMENTS",
                            error_message=f"Validation failed for tool '{tc.tool}': {val_err}"
                        )
                reason = state.get("clarification_reason") or reason or (tc.arguments or {}).get("reason")

            reason = reason or "Query is ambiguous or underspecified. Please specify an analytical goal."
            intent = SemanticIntent(
                intent_type=IntentType.EDA,
                primary_intent=IntentType.EDA,
                raw_query=raw_query or "",
                is_ambiguous=True,
                ambiguity_reason=reason
            )
            return CommandResolutionResult(
                is_valid=True,
                semantic_intent=intent,
                is_ambiguous=True,
                ambiguity_reason=reason
            )

        # Step 2c: Apply data-aware intent completion if df is present
        if df is not None and isinstance(df, pd.DataFrame):
            from backend.ds_engine.dataset_profiler import DatasetProfiler
            from backend.ds_engine.intent_completion import IntentCompletionService
            profile_for_cmd = DatasetProfiler.profile(df)
            for tc in tool_calls:
                comp = IntentCompletionService.complete_intent(
                    tool_name=tc.tool,
                    arguments=tc.arguments or {},
                    profile=profile_for_cmd,
                    query_raw=raw_query,
                )
                if comp.action == "clarify":
                    clarif_intent = SemanticIntent(
                        intent_type=IntentType.CLARIFICATION,
                        primary_intent=IntentType.CLARIFICATION,
                        raw_query=raw_query or "",
                        is_ambiguous=True,
                        ambiguity_reason=comp.clarification_prompt or "Clarification required.",
                        candidate_columns=comp.candidate_columns,
                        slot_filling=comp.slot_filling.model_dump() if comp.slot_filling else None,
                        tool_name=tc.tool,
                    )
                    return CommandResolutionResult(
                        is_valid=True,
                        semantic_intent=clarif_intent,
                        is_ambiguous=True,
                        ambiguity_reason=comp.clarification_prompt or "Clarification required.",
                        candidate_columns=comp.candidate_columns,
                        slot_filling=comp.slot_filling.model_dump() if comp.slot_filling else None,
                    )
                elif comp.action in ("reject", "unsupported"):
                    return CommandResolutionResult(
                        is_valid=False,
                        error_code=comp.error_code or "UNSUPPORTED_OPERATION",
                        error_message=comp.error_message or "Unsupported operation for the active dataset.",
                    )
                elif comp.action == "execute":
                    tc.arguments = comp.completed_arguments
                    tc.is_dataset_wide = comp.is_dataset_wide
                    tc.inferred_bindings = [b.model_dump() for b in comp.inferred_bindings]

        # Step 2d: Validate all tool names and arguments against ToolSpec
        for tc in tool_calls:
            t_name = tc.tool.lower().strip()
            spec = capability_registry.get_tool_spec(t_name)
            if not spec:
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="UNSUPPORTED_OPERATION",
                    error_message=f"Unknown or unsupported semantic tool: '{tc.tool}'"
                )
            if not spec.adapter:
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="UNSUPPORTED_OPERATION",
                    error_message=f"Tool '{tc.tool}' has no registered argument adapter in CapabilityRegistry."
                )
            try:
                validated_args = spec.input_model.model_validate(tc.arguments or {})
            except Exception as val_err:
                logger.warning(f"TOOL_ARGUMENT_VALIDATION_FAILED | Tool: '{t_name}' | Error: {val_err}")
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="INVALID_ARGUMENTS",
                    error_message=f"Validation failed for tool '{tc.tool}': {val_err}"
                )
            visualization_error = cls._validate_visualization_contract(
                spec=spec,
                validated_args=validated_args,
            )
            if visualization_error:
                logger.warning("TOOL_VISUALIZATION_CONTRACT_FAILED | Tool: '%s' | %s", t_name, visualization_error)
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="UNSUPPORTED_VISUALIZATION",
                    error_message=visualization_error,
                )
            grounding_error = cls._validate_quantified_request_evidence(
                spec=spec,
                validated_args=validated_args,
                raw_query=raw_query,
            )
            if grounding_error:
                logger.warning("TOOL_ARGUMENT_GROUNDING_FAILED | Tool: '%s' | %s", t_name, grounding_error)
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="UNGROUNDED_REQUEST_ARGUMENT",
                    error_message=grounding_error,
                )
            measure_error = cls._validate_numeric_measure_contract(
                spec=spec,
                validated_args=validated_args,
                df=df,
            )
            if measure_error:
                logger.warning("TOOL_NUMERIC_MEASURE_VALIDATION_FAILED | Tool: '%s' | %s", t_name, measure_error)
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="INVALID_NUMERIC_MEASURE",
                    error_message=measure_error,
                )
            grounding_col_error = cls._validate_dataset_column_grounding(
                spec=spec,
                validated_args=validated_args,
                df=df,
            )
            if grounding_col_error:
                logger.warning("TOOL_COLUMN_GROUNDING_FAILED | Tool: '%s' | %s", t_name, grounding_col_error)
                err_code = "COLUMN_NOT_FOUND"
                err_msg = grounding_col_error
                if t_name in ("forecast_data", "time_series_forecast", "decompose_time_series"):
                    has_temporal = False
                    if df is not None:
                        try:
                            from backend.ds_engine.dataset_profiler import DatasetProfiler
                            prof = DatasetProfiler.profile(df)
                            has_temporal = len(prof.ranked_temporal_columns) > 0
                        except Exception:
                            has_temporal = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
                    if not has_temporal:
                        err_code = "NO_TEMPORAL_COLUMN"
                        err_msg = f"Forecasting requires a datetime/temporal column, but the dataset contains no temporal columns. {grounding_col_error}"
                available_cols = [str(c) for c in df.columns] if df is not None and hasattr(df, "columns") else []
                return CommandResolutionResult(
                    is_valid=False,
                    error_code=err_code,
                    error_message=err_msg,
                    candidate_columns=available_cols,
                )

        # Step 3: Parse and merge tool calls into primary intent + prerequisites
        def _tool_priority(tc: ToolCall) -> int:
            spec = capability_registry.get_tool_spec(tc.tool.lower().strip())
            return spec.execution_priority if spec else 2

        ordered_tools = sorted(tool_calls, key=_tool_priority)
        non_filter_tools = [
            tc for tc in ordered_tools
            if not capability_registry.get_tool_spec(tc.tool.lower().strip()).is_prerequisite_only
        ]
        if non_filter_tools:
            primary_tool_call = non_filter_tools[-1]
        else:
            primary_tool_call = ordered_tools[-1]

        primary_tool_name = primary_tool_call.tool.lower().strip()
        if primary_tool_name not in tool_intent_map:
            raise RuntimeError(f"REGISTRY_CONTRACT_INVALID: Tool '{primary_tool_name}' is not mapped to an IntentType in tool_intent_map.")
        primary_intent_type = tool_intent_map[primary_tool_name]
        prereq_intent_types: List[IntentType] = []

        state: Dict[str, Any] = {
            "filters": [],
            "dimensions": [],
            "metrics": [],
            "feature_columns": [],
            "target_column": None,
            "time_column": None,
            "sequence_column": None,
            "entity_id": None,
            "aggregation": None,
            "order": None,
            "limit": None,
            "horizon": None,
            "chart_type": ChartType.NONE,
            "has_filter_spec": False
        }

        # Process all tool calls (both prerequisites and primary) strictly via ToolSpec adapter
        for tc in ordered_tools:
            t_name = tc.tool.lower().strip()
            if t_name not in tool_intent_map:
                raise RuntimeError(f"REGISTRY_CONTRACT_INVALID: Tool '{t_name}' is not mapped to an IntentType in tool_intent_map.")
            intent_type = tool_intent_map[t_name]

            if tc != primary_tool_call:
                if intent_type not in prereq_intent_types:
                    prereq_intent_types.append(intent_type)

            spec = capability_registry.get_tool_spec(t_name)
            validated_args = spec.input_model.model_validate(tc.arguments or {})
            spec.adapter(validated_args, state)

        filters = state["filters"]
        dimensions = state["dimensions"]
        metrics = state["metrics"]
        feature_columns = state["feature_columns"]
        target_column = state["target_column"]
        time_column = state["time_column"]
        sequence_column = state["sequence_column"]
        entity_id = state["entity_id"]
        aggregation = state["aggregation"]
        order = state["order"]
        limit = state["limit"]
        horizon = state["horizon"]
        chart_type = state["chart_type"]

        if state.get("filter_error"):
            return CommandResolutionResult(
                is_valid=False,
                error_code="INVALID_FILTER_SPEC",
                error_message=state["filter_error"]
            )

        # Validate filter_data command has filters
        primary_spec = capability_registry.get_tool_spec(primary_tool_call.tool.lower().strip())
        if primary_spec and primary_spec.requires_filter_conditions and not filters and not state.get("has_filter_spec"):
            return CommandResolutionResult(
                is_valid=False,
                error_code="INVALID_FILTER_SPEC",
                error_message=f"filter_data requires at least one valid filter condition or column: {primary_tool_call.arguments}"
            )

        # Automatic prerequisite injection:
        # If filters exist on a primary intent other than FILTER_QUERY / DATA_CLEAN, ensure FILTER_QUERY is prerequisite
        if filters and primary_intent_type not in (IntentType.FILTER_QUERY, IntentType.DATA_CLEAN):
            if IntentType.FILTER_QUERY not in prereq_intent_types:
                prereq_intent_types.insert(0, IntentType.FILTER_QUERY)

        # Grouping is a ToolSpec contract, not a blanket property of the
        # shared GROUP_AGGREGATE intent family.  ``aggregate_data`` is a
        # legitimate overall summary with no dimensions; ``group_data`` and
        # conditional rates still require explicit dimensions.
        if primary_intent_type == IntentType.GROUP_AGGREGATE:
            requires_grouping = bool(
                primary_spec
                and primary_spec.prerequisites
                and primary_spec.prerequisites.requires_grouping_or_dimension
            )
            if requires_grouping and not dimensions:
                return CommandResolutionResult(
                    is_valid=False,
                    error_code="MISSING_REQUIRED_INFORMATION",
                    error_message="Group and aggregate query requires at least one group_by dimension column."
                )
            if not aggregation:
                if not metrics:
                    aggregation = AggregationType.COUNT
                else:
                    return CommandResolutionResult(
                        is_valid=False,
                        error_code="MISSING_REQUIRED_INFORMATION",
                        error_message=f"Group query specifies metric {metrics} but is missing an explicit aggregation function (e.g. 'sum', 'mean', 'count', 'min', 'max', 'median')."
                    )

        pre_event_window = state.get("pre_event_window")
        event_predicate = state.get("event_predicate")

        time_ctx = None
        if time_column or sequence_column or target_column or pre_event_window is not None or event_predicate is not None:
            from backend.ds_engine.intent_schema import TimeContext
            time_ctx = TimeContext(
                time_column=time_column,
                sequence_column=sequence_column,
                event_column=target_column,
                entity_id=entity_id,
                event_predicate=event_predicate,
                pre_event_window=pre_event_window,
                horizon=horizon
            )

        # Build execution plan and tool args dictionary
        tool_args = dict(primary_tool_call.arguments or {})
        if "arguments" in state and isinstance(state["arguments"], dict):
            tool_args.update(state["arguments"])
        for k in ["n_clusters", "k", "max_lags", "n_components", "period", "model", "pre_event_window", "event_predicate", "entity_id", "sequence_column"]:
            if k in state and state[k] is not None:
                tool_args[k] = state[k]

        exec_plan = [{"tool": primary_tool_call.tool, "arguments": tool_args, "state": state}]

        inferred_bindings = getattr(primary_tool_call, "inferred_bindings", []) or (primary_tool_call.arguments or {}).get("inferred_bindings", [])
        is_ds_wide = bool(getattr(primary_tool_call, "is_dataset_wide", False) or (primary_tool_call.arguments or {}).get("is_dataset_wide", False))

        # Build clean SemanticIntent
        intent = SemanticIntent(
            intent_type=primary_intent_type,
            primary_intent=primary_intent_type,
            tool_name=primary_tool_call.tool,
            prerequisite_operations=prereq_intent_types,
            target_column=target_column,
            time_column=time_column,
            sequence_column=sequence_column,
            time_context=time_ctx,
            feature_columns=feature_columns,
            dimensions=dimensions,
            metrics=metrics,
            aggregation=aggregation,
            group_by=dimensions,
            filters=filters,
            order=order,
            limit=limit,
            horizon=horizon,
            chart_type=chart_type,
            prompt_summary=raw_query,
            raw_query=raw_query,
            confidence=1.0,
            is_ambiguous=False,
            execution_plan=exec_plan,
            inferred_bindings=inferred_bindings,
            is_dataset_wide=is_ds_wide,
        )

        return CommandResolutionResult(
            is_valid=True,
            semantic_intent=intent,
            is_ambiguous=False,
            inferred_bindings=inferred_bindings,
            is_dataset_wide=is_ds_wide,
        )


command_resolver = CommandResolver()
