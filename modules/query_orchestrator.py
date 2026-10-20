"""
DataPilot OS — Single Authoritative Query Pipeline Orchestrator.
V2 Capability-Based Data Science Operating System (DS-OS) Pipeline.

Lifecycle States: IDLE -> PLANNING -> PLAN_VALIDATION -> EXECUTING -> VALIDATING_RESULT -> SUCCEEDED / FAILED

Zero Fallback Guarantees:
- DatasetProfiler & AnalysisIntentV2 ground LLM intent extraction.
- Authoritative execution handled directly by Deterministic Typed Operators (TypedOperatorRegistry).
- SemanticResultValidator validates ExecutionManifest parity against AnalysisIntentV2 before returning 200 OK.
- NO silent fallbacks to legacy code generator or subprocess sandbox when V2 validation or execution halts.
"""

import logging
import time
import uuid
import difflib
import re
import pandas as pd
from typing import Optional, Dict, Any, List

from backend.intelligence.ollama_service import ollama_service
from modules.semantic_planner import SemanticPlanner
from backend.ds_engine.dataset_profiler import DatasetProfiler, DatasetProfile
from backend.ds_engine.intent_schema import (
    SemanticIntent, IntentType, ChartType, AggregationType, FilterSpec, FilterOperator,
    AnalysisIntentV2, CorrelationSpec, TaskFamily, TimeContext, VisualizationSpec, ExecutionManifest
)
from backend.ds_engine.typed_operators import TypedOperatorRegistry, OperatorOutput
from backend.ds_engine.target_leakage import TargetLeakageDetector
from backend.ds_engine.result_validator import SemanticResultValidator, ResultValidator
from backend.ds_engine.semantic_feasibility import SemanticFeasibilityValidator
from backend.ds_engine.execution_graph import ExecutionGraphCompiler
from modules.models.execution_result import (
    CanonicalQueryResponse,
    QueryErrorDetail,
    CanonicalError,
    LifecycleState,
)
from backend.core.request_context import get_request_context, set_request_context, RequestContext
from backend.core.error_states import ErrorState
from backend.core.session_telemetry import (
    log_query_received,
    log_grounding_completed,
    log_planning_completed,
    log_feasibility_gate,
    log_operator_execution,
    log_result_verification,
    log_query_success,
    log_query_clarification,
    log_query_halted,
)

logger = logging.getLogger(__name__)


def humanize_clarification_message(message: str, candidate_columns: Optional[List[str]] = None) -> str:
    """Converts technical ambiguity and slot-filling messages into concise, friendly clarification questions."""
    if not message:
        if candidate_columns:
            cols = ", ".join(f"`{c}`" for c in candidate_columns)
            return f"Which column would you like to use? Candidates: {cols}."
        return "Could you please clarify your request?"

    # Pattern: Ambiguous column reference 'x': matches multiple dataset columns (a, b)
    ambig_match = re.search(r"Ambiguous column reference '([^']+)': matches multiple dataset columns \(([^)]+)\)", message)
    if ambig_match:
        ref = ambig_match.group(1)
        matches = [m.strip() for m in ambig_match.group(2).split(",")]
        cols = ", ".join(f"`{c}`" for c in matches)
        return f"Multiple columns match '{ref}': {cols}. Which one would you like to analyze?"

    # Pattern: Statistical test requires explicit columns to compare; multiple categorical columns found
    if "multiple categorical columns found" in message and candidate_columns:
        cols = ", ".join(f"`{c}`" for c in candidate_columns)
        return f"Which category would you like to compare across? Candidates: {cols}."

    # Pattern: Statistical test requires explicit metric columns; multiple numeric columns found
    if "multiple numeric columns found" in message and candidate_columns:
        cols = ", ".join(f"`{c}`" for c in candidate_columns)
        return f"Which metric would you like to test? Candidates: {cols}."

    return message


def _map_intent_to_v2(
    intent: SemanticIntent,
    raw_query: str,
    df: Optional[pd.DataFrame] = None,
    validated_tool_args: Optional[Dict[str, Any]] = None,
) -> AnalysisIntentV2:
    from modules.capability_registry import capability_registry

    i_type = getattr(intent, "intent_type", None)
    p_intent = getattr(intent, "primary_intent", None)
    t_name = str(getattr(intent, "tool_name", "") or "").lower().strip()

    family: Optional[TaskFamily] = None

    # Ambiguity / Clarification check
    if getattr(intent, "is_ambiguous", False) or i_type == IntentType.CLARIFICATION or p_intent == IntentType.CLARIFICATION:
        family = TaskFamily.CLARIFICATION
    elif t_name:
        # Query CapabilityRegistry dynamically for tool_spec.task_family
        spec = capability_registry.get_tool_spec(t_name)
        if spec and spec.task_family and spec.task_family != TaskFamily.UNSUPPORTED:
            family = spec.task_family
        else:
            family = TaskFamily.UNSUPPORTED
    else:
        target_intent = i_type or p_intent
        if target_intent:
            intent_str = target_intent.value if hasattr(target_intent, "value") else str(target_intent)
            clean_upper = intent_str.upper().strip()
            clean_lower = intent_str.lower().strip()
            if hasattr(TaskFamily, clean_upper):
                family = getattr(TaskFamily, clean_upper)
            elif clean_lower in [f.value for f in TaskFamily]:
                family = TaskFamily(clean_lower)
            else:
                spec = capability_registry.get_tool_spec(clean_lower)
                if spec and spec.task_family and spec.task_family != TaskFamily.UNSUPPORTED:
                    family = spec.task_family
                else:
                    family = TaskFamily.UNSUPPORTED
        else:
            family = TaskFamily.UNSUPPORTED

    if family is None:
        family = TaskFamily.UNSUPPORTED

    # Preserve the selected ToolSpec's execution contract into the typed
    # intent.  A task family may contain both grouped and overall operations;
    # only the registry determines which one requires dimensions.
    selected_spec = capability_registry.get_tool_spec(t_name) if t_name else None
    requires_grouping = not bool(
        selected_spec
        and selected_spec.prerequisites
        and selected_spec.prerequisites.requires_grouping_or_dimension is False
    )

    # Extract time_context, event_predicate, entity_id, and pre_event_window directly
    time_ctx_obj = getattr(intent, "time_context", None)
    tool_args = getattr(intent, "arguments", {}) or getattr(intent, "tool_call_arguments", {}) or {}
    if not isinstance(tool_args, dict):
        tool_args = {}

    if intent.execution_plan and isinstance(intent.execution_plan, list) and len(intent.execution_plan) > 0:
        ep = intent.execution_plan[0]
        if isinstance(ep, dict):
            if "arguments" in ep and isinstance(ep["arguments"], dict):
                tool_args.update(ep["arguments"])
            if "state" in ep and isinstance(ep["state"], dict):
                tool_args.update(ep["state"].get("arguments", {}))
                for k in ["n_clusters", "k", "max_lags", "n_components", "period", "model", "pre_event_window", "event_predicate", "entity_id", "sequence_column"]:
                    if k in ep["state"] and ep["state"][k] is not None:
                        tool_args[k] = ep["state"][k]
    if validated_tool_args:
        tool_args.update(validated_tool_args)

    if family == TaskFamily.DATA_CLEANING:
        clean_method = str(tool_args.get("method", "")).lower()
        if clean_method in (
            "minmax_scale", "normalize_minmax", "standardize", "z_score", "one_hot_encode",
            "yeo_johnson", "power_transform", "log_transform",
            "lag", "create_lag", "rolling", "create_rolling",
            "difference", "create_difference", "calendar_features", "create_calendar_features"
        ):
            family = TaskFamily.FEATURE_ENGINEERING

    if family == TaskFamily.SUPERVISED_ML:
        # Supervised ML has two distinct roles: the label and its predictors.
        # Preserve the LLM-selected target first, followed by every explicit
        # feature (including categorical features that the operator encodes).
        subjects = []
        if intent.target_column:
            subjects.append(intent.target_column)
        for feature in intent.feature_columns or []:
            if feature not in subjects:
                subjects.append(feature)
    elif family in (TaskFamily.DATA_CLEANING, TaskFamily.FEATURE_ENGINEERING):
        subjects = list(intent.feature_columns or [])
        if not subjects and intent.target_column:
            subjects = [intent.target_column]
        if not subjects and tool_args:
            cols = tool_args.get("columns") or tool_args.get("column")
            if cols:
                subjects = [cols] if isinstance(cols, str) else list(cols)
    else:
        subjects = intent.metrics or (intent.feature_columns if intent.feature_columns else ([] if not intent.target_column else [intent.target_column]))

    def _get_val(key_name: str) -> Any:
        if time_ctx_obj:
            val = getattr(time_ctx_obj, key_name, None) if hasattr(time_ctx_obj, key_name) else (time_ctx_obj.get(key_name) if isinstance(time_ctx_obj, dict) else None)
            if val is not None:
                return val
        val = getattr(intent, key_name, None)
        if val is not None:
            return val
        return tool_args.get(key_name)

    time_col = getattr(intent, "time_column", None) or _get_val("time_column")
    sequence_col = getattr(intent, "sequence_column", None) or _get_val("sequence_column")

    event_col = _get_val("event_column")

    event_predicate = _get_val("event_predicate")
    if event_predicate is None and intent.filters and event_col:
        for f in intent.filters:
            col_name = getattr(f, "column", None) if hasattr(f, "column") else (f.get("column") if isinstance(f, dict) else None)
            if col_name == event_col:
                event_predicate = f
                break
    if not event_col and isinstance(event_predicate, dict):
        event_col = event_predicate.get("column")

    entity_id = _get_val("entity_id")
    pre_event_window = _get_val("pre_event_window") or _get_val("window")

    time_ctx = TimeContext(
        entity_id=entity_id,
        time_column=time_col,
        sequence_column=sequence_col,
        event_column=event_col,
        event_predicate=event_predicate,
        pre_event_window=pre_event_window,
        horizon=intent.horizon
    )

    correlation_spec = None
    if family == TaskFamily.CORRELATION_ANALYSIS:
        correlation_fields = (
            "analysis_mode", "relationship_form", "correlation_methods",
            "chart_type", "smoother", "expected_direction",
        )
        correlation_spec = CorrelationSpec.model_validate({
            field: tool_args.get(field) for field in correlation_fields
        })

    # Visualization
    chart_kind = (
        correlation_spec.chart_type if correlation_spec is not None
        else (
            tool_args.get("evaluation_artifact")
            if family == TaskFamily.SUPERVISED_ML and tool_args.get("evaluation_artifact")
            # A ToolSpec's typed chart request is the authoritative contract.
            # Fall back only to the already-resolved SemanticIntent field when
            # the ToolSpec has no chart field at all.
            else (
                tool_args.get("chart_type")
                or (intent.chart_type.value if hasattr(intent.chart_type, "value") else str(intent.chart_type or "none"))
            )
        )
    )
    viz_spec = VisualizationSpec(type=chart_kind)

    # Statistical requirements
    stat_reqs = []
    if intent.aggregation:
        stat_reqs.append(intent.aggregation.value if hasattr(intent.aggregation, "value") else str(intent.aggregation))
    if family in (TaskFamily.DISTRIBUTION_COMPARISON, TaskFamily.EVENT_WINDOW_ANALYSIS):
        stat_reqs.extend(["variance", "quantiles", "iqr"])

    return AnalysisIntentV2(
        task_family=family,
        operations=[family.value],
        subjects=subjects,
        target_column=intent.target_column,
        time_column=time_col,
        sequence_column=sequence_col,
        horizon=intent.horizon,
        conditions=intent.filters or [],
        grouping=intent.dimensions or intent.group_by or [],
        requires_grouping=requires_grouping,
        time_context=time_ctx,
        statistical_requirements=stat_reqs,
        visualization=viz_spec,
        correlation_spec=correlation_spec,
        prompt_summary=intent.prompt_summary or intent.ambiguity_reason,
        raw_query=intent.raw_query,
        arguments=tool_args
    )



class QueryOrchestrator:
    @classmethod
    def process_query(
        cls,
        query_raw: str,
        df: Optional[pd.DataFrame] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> CanonicalQueryResponse:
        t_start = time.time()
        request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"
        query_raw = query_raw.strip()
        logger.info(f"[request_id={request_id}] QUERY_RECEIVED | Query: '{query_raw}'")
        ds_name = (
            getattr(df, "name", None)
            or (df.attrs.get("name") if hasattr(df, "attrs") else None)
            or (context.get("dataset_name") if isinstance(context, dict) else None)
        )
        ds_hash = context.get("dataset_hash") if isinstance(context, dict) else None
        log_query_received(
            request_id=request_id,
            query=query_raw,
            dataset_name=ds_name,
            row_count=len(df) if df is not None and isinstance(df, pd.DataFrame) else None,
            col_count=len(df.columns) if df is not None and isinstance(df, pd.DataFrame) else None,
            dataset_hash=ds_hash,
        )

        req_ctx = RequestContext(request_id=request_id)
        set_request_context(req_ctx)
        req_ctx.start_phase("planning")

        # ── 1. SECURITY INTERDICT CHECK ──────────────────────────────────────────────
        forbidden_patterns = ["__import__", "subprocess", "socket", "os.system", "rm -rf", "exec(", "eval("]
        if any(k in query_raw.lower() for k in forbidden_patterns):
            logger.warning(f"[request_id={request_id}] SECURITY_VIOLATION | Blocked prompt payload")
            log_query_halted(request_id, code="SECURITY_VIOLATION", message="Security Interdict Triggered: Disallowed system command or payload in query.", stage="PLANNING", required_action="Remove shell commands from query.")
            req_ctx.end_phase("planning")
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=LifecycleState.PLANNING.value,
                plan=None,
                result=None,
                error=QueryErrorDetail(
                    code="SECURITY_VIOLATION",
                    message="Security Interdict Triggered: Disallowed system command or payload detected in query prompt.",
                    phase=LifecycleState.PLANNING.value,
                    request_id=request_id
                ),
                diagnostics={"phase_timings": req_ctx.phase_timings}
            )

        # ── 2. OLLAMA READINESS CHECK ─────────────────────────────────────────────────
        readiness = ollama_service.verify_ready()
        if not readiness.ready:
            err_code = readiness.error_code or "INTENT_PLANNER_UNAVAILABLE"
            err_msg = readiness.error_message or "Ollama Intent Planner is unavailable."
            logger.warning(f"[request_id={request_id}] PLANNING_FAILED | code={err_code} | msg={err_msg}")
            log_query_halted(request_id, code=err_code, message=err_msg, stage="PLANNING", required_action="Verify Ollama daemon is running.")
            req_ctx.end_phase("planning")
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=LifecycleState.PLANNING.value,
                plan=None,
                result=None,
                error=QueryErrorDetail(
                    code=err_code,
                    message=err_msg,
                    phase=LifecycleState.PLANNING.value,
                    request_id=request_id
                ),
                diagnostics={"phase_timings": req_ctx.phase_timings}
            )

        # ── 3. DATASET PROFILING ──────────────────────────────────────────────────────
        profile: Optional[DatasetProfile] = None
        if df is not None and isinstance(df, pd.DataFrame):
            try:
                profile = DatasetProfiler.profile(df, dataset_name=ds_name or "active_dataset")
                logger.info(f"[request_id={request_id}] DATASET_PROFILED | Hash: {profile.dataset_hash[:8]}")
                log_grounding_completed(
                    request_id=request_id,
                    temporal_cols=list(profile.temporal_column_candidates) if profile else [],
                    numeric_cols=list(profile.continuous_numeric_candidates) if profile else [],
                    categorical_cols=list(profile.categorical_candidates) if profile else [],
                    dataset_hash=profile.dataset_hash if profile else None,
                )
            except Exception as prof_err:
                logger.warning(f"[request_id={request_id}] DATASET_PROFILING_WARNING: {prof_err}")

        # ── 4. OLLAMA INTENT EXTRACTION ──────────────────────────────────────────────
        t_llm_start = time.time()
        intent_dict = SemanticPlanner.extract_semantic_intent(
            query_raw=query_raw,
            df=df,
            conversation_history=conversation_history,
            context=context
        )
        llm_duration_ms = (time.time() - t_llm_start) * 1000
        req_ctx.end_phase("planning")

        # Check if planner or repair budget halted execution with an error code
        if isinstance(intent_dict, dict) and intent_dict.get("error_code"):
            code = intent_dict["error_code"]
            msg = intent_dict.get("error_message") or "Planning halted."
            phase = LifecycleState.PLANNING.value if code == "INTENT_PLANNER_UNAVAILABLE" else LifecycleState.PLAN_VALIDATION.value
            logger.warning(f"[request_id={request_id}] PLANNING_HALTED | code={code} | msg={msg}")
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=phase,
                plan=intent_dict,
                result={"repair_audit": intent_dict.get("repair_history", [])},
                error=QueryErrorDetail(
                    code=code,
                    message=msg,
                    phase=phase,
                    request_id=request_id,
                    diagnostics={"repair_history": intent_dict.get("repair_history", [])}
                ),
                diagnostics={"repair_audit": intent_dict.get("repair_history", []), "phase_timings": req_ctx.phase_timings}
            )

        if not intent_dict:
            # Missing model output is a planner failure, not evidence that the
            # user omitted arguments. Never diagnose an invented empty call.
            logger.warning(f"[request_id={request_id}] PLANNING_FAILED | code=INTENT_PLAN_INVALID | No structured JSON produced")
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=LifecycleState.PLANNING.value,
                plan=None,
                result=None,
                error=QueryErrorDetail(
                    code="INTENT_PLAN_INVALID",
                    message="Ollama Intent Planner was unable to parse natural language query into a valid structured intent.",
                    phase=LifecycleState.PLANNING.value,
                    request_id=request_id
                )
            )

        if isinstance(intent_dict, dict) and isinstance(intent_dict.get("commands"), list):
            commands = intent_dict["commands"]
            from modules.capability_registry import capability_registry
            has_non_dataset_tool = any(
                isinstance(c, dict)
                and (spec := capability_registry.get_tool_spec(c.get("tool", "")))
                and not spec.requires_dataset
                for c in commands
            )
            if not has_non_dataset_tool and len(commands) > 1:
                return cls._process_sequence(query_raw, df, commands,
                                             conversation_history, context, request_id, t_start)
        return cls._process_intent(query_raw, df, intent_dict, profile,
                                   conversation_history, context, request_id, t_start,
                                   output_frames=kwargs.get("output_frames"))

    @classmethod
    def _process_sequence(cls, query_raw, df, commands, conversation_history, context, request_id, t_start):
        from modules.capability_registry import capability_registry
        def fail(code, message, steps):
            return CanonicalQueryResponse(
                request_id=request_id, status="FAILED", phase=LifecycleState.PLAN_VALIDATION.value,
                plan={"commands": commands, "execution_plan": steps}, result=None,
                error=QueryErrorDetail(code=code, message=message,
                                       phase=LifecycleState.PLAN_VALIDATION.value, request_id=request_id))
        try:
            graph = ExecutionGraphCompiler.compile(commands, capability_registry)
        except (ValueError, TypeError) as exc:
            message = str(exc)
            code = message.split(":", 1)[0] if ":" in message else "INVALID_ARGUMENTS"
            if code not in {"INVALID_PLAN_SEQUENCE", "UNSUPPORTED_OPERATION"}:
                code = "INVALID_ARGUMENTS"
            return fail(code, message, [])
        current = df.copy(deep=True) if df is not None else None
        steps, artifacts = [], []
        tool_intent_map = capability_registry.get_tool_intent_map()
        for index, node in enumerate(graph.nodes):
            command = {"tool": node.tool, "arguments": node.arguments}
            profile = DatasetProfiler.profile(current) if current is not None else None
            output_frames = []
            response = cls._process_intent(query_raw, current, command, profile,
                                           conversation_history, context, request_id, t_start, output_frames)
            step_spec = capability_registry.get_tool_spec(command["tool"])
            step_result = response.result or {}
            compatibility_intent = tool_intent_map[node.tool]
            steps.append({
                "step_index": index + 1,
                "operation_id": node.operation_id,
                "depends_on": node.depends_on,
                "tool": command["tool"],
                "intent_type": (
                    compatibility_intent.value
                    if hasattr(compatibility_intent, "value")
                    else str(compatibility_intent)
                ),
                "task_family": node.task_family,
                "input_contract": node.input_contract,
                "result_contract": node.output_contract,
                "arguments": command["arguments"],
                "status": response.status,
                "input_profile": {
                    "dataset_hash": profile.dataset_hash if profile else None,
                    "row_count": profile.row_count if profile else None,
                    "columns": list(profile.columns) if profile else [],
                },
                "manifest": step_result.get("manifest"),
                "validation": step_result.get("validation"),
            })
            if response.status != "SUCCEEDED":
                response.plan = {"commands": commands, "execution_plan": steps, "failed_step": index + 1}
                return response
            artifacts.append(response.result)
            if index + 1 < len(graph.nodes):
                if not output_frames or not isinstance(output_frames[0], pd.DataFrame):
                    return fail("MISSING_INTERMEDIATE_DATAFRAME", f"Step {index + 1} cannot supply the next step.", steps)
                current = output_frames[0].copy(deep=True)
        response.plan = {
            "commands": commands,
            "graph_version": graph.graph_version,
            "execution_plan": steps,
        }
        response.result["steps"] = artifacts[:-1] + [dict(response.result)]
        code_blocks = ['import pandas as pd', 'df = df.copy(deep=True)']
        for index, artifact in enumerate(artifacts):
            code_blocks.extend([f'# Step {index + 1}', 'result_df = None', artifact.get('code') or ''])
            if index + 1 < len(artifacts):
                code_blocks.extend([
                    "if not isinstance(result_df, pd.DataFrame):",
                    "    raise ValueError('Step did not produce a reproducible intermediate dataframe')",
                    "df = result_df.copy(deep=True)",
                ])
        response.result['code'] = '\n'.join(code_blocks)
        response.result["artifacts"]["execution_time_sec"] = round(time.time() - t_start, 3)
        return response

    @classmethod
    def _process_intent(cls, query_raw, df, intent_dict, profile, conversation_history,
                        context, request_id, t_start, output_frames=None):
        req_ctx = get_request_context()
        # Resolve command into validated SemanticIntent
        if isinstance(intent_dict, dict) and ("primary_intent" in intent_dict or "intent_type" in intent_dict):
            try:
                intent = SemanticIntent.model_validate(intent_dict)
                resolution = None
            except Exception:
                from backend.ds_engine.command_resolver import command_resolver
                resolution = command_resolver.resolve_command(intent_dict, df=df, raw_query=query_raw)
        else:
            from backend.ds_engine.command_resolver import command_resolver
            resolution = command_resolver.resolve_command(intent_dict, df=df, raw_query=query_raw)

        if resolution is not None:
            if resolution.is_ambiguous:
                raw_reason = resolution.ambiguity_reason or resolution.error_message or "Query requires clarification."
                candidate_cols = resolution.candidate_columns or []
                reason = humanize_clarification_message(raw_reason, candidate_cols)
                logger.warning(f"[request_id={request_id}] AMBIGUOUS_INTENT | Reason: {reason}")
                return CanonicalQueryResponse(
                    request_id=request_id,
                    status=LifecycleState.NEEDS_CLARIFICATION.value,
                    phase=LifecycleState.PLAN_VALIDATION.value,
                    plan=intent_dict if isinstance(intent_dict, dict) else None,
                    result=None,
                    clarification=reason,
                    candidate_columns=candidate_cols,
                    slot_filling=resolution.slot_filling,
                    error=CanonicalError(
                        code="CLARIFICATION_REQUIRED",
                        message=reason,
                        phase=LifecycleState.PLAN_VALIDATION.value,
                        request_id=request_id,
                        candidate_columns=candidate_cols,
                        slot_filling=resolution.slot_filling,
                    )
                )
            if not resolution.is_valid or not resolution.semantic_intent:
                err_code = resolution.error_code or "INTENT_PLAN_INVALID"
                err_msg = resolution.error_message or "Unable to resolve query into a valid semantic intent."
                failure_phase = LifecycleState.PLAN_VALIDATION.value
                tool_called = (intent_dict.get("tool") if isinstance(intent_dict, dict) else None) or ""
                is_clean_transform = str(tool_called).lower() in ("clean_data", "transform_data", "data_clean", "data_transform", "transform_features", "feature_engineering")
                if err_code == "COLUMN_NOT_FOUND" and is_clean_transform:
                    candidate_cols = resolution.candidate_columns or (list(df.columns) if df is not None and hasattr(df, "columns") else [])
                    logger.warning(f"[request_id={request_id}] COLUMN_NOT_FOUND -> CLARIFICATION | Reason: {err_msg}")
                    return CanonicalQueryResponse(
                        request_id=request_id,
                        status=LifecycleState.NEEDS_CLARIFICATION.value,
                        phase=failure_phase,
                        plan=intent_dict if isinstance(intent_dict, dict) else None,
                        result=None,
                        clarification=err_msg,
                        candidate_columns=candidate_cols,
                        slot_filling=resolution.slot_filling,
                        error=CanonicalError(
                            code="COLUMN_NOT_FOUND",
                            message=err_msg,
                            phase=failure_phase,
                            request_id=request_id,
                            candidate_columns=candidate_cols,
                            slot_filling=resolution.slot_filling,
                        )
                    )
                # The model produced a structured command; rejecting its tool,
                # arguments, or dataset bindings is plan validation, not an LLM
                # availability/planning failure.
                logger.warning(f"[request_id={request_id}] PLAN_VALIDATION_FAILED | code={err_code} | msg={err_msg}")
                return CanonicalQueryResponse(
                    request_id=request_id,
                    status=LifecycleState.FAILED.value,
                    phase=failure_phase,
                    plan=None,
                    result=None,
                    error=QueryErrorDetail(
                        code=err_code,
                        message=err_msg,
                        phase=failure_phase,
                        request_id=request_id,
                        candidate_columns=resolution.candidate_columns or [],
                        slot_filling=resolution.slot_filling,
                    )
                )
            intent = resolution.semantic_intent

        intent_dict_full = intent.model_dump()
        intent_dict = intent_dict_full

        # Ambiguity Interdict
        if intent.is_ambiguous or (resolution and resolution.is_ambiguous):
            raw_reason = intent.ambiguity_reason or (resolution.ambiguity_reason if resolution else None) or "Query is ambiguous or underspecified."
            candidate_columns = getattr(intent, "candidate_columns", None) or (resolution.candidate_columns if resolution else None) or []
            slot_filling = getattr(intent, "slot_filling", None) or (resolution.slot_filling if resolution else None)
            reason = humanize_clarification_message(raw_reason, candidate_columns)

            logger.warning(f"[request_id={request_id}] AMBIGUOUS_INTENT | Reason: {reason}")
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.NEEDS_CLARIFICATION.value,
                phase=LifecycleState.PLAN_VALIDATION.value,
                plan=intent_dict,
                result=None,
                clarification=reason,
                candidate_columns=candidate_columns,
                slot_filling=slot_filling,
                error=CanonicalError(
                    code="CLARIFICATION_REQUIRED",
                    message=reason,
                    phase=LifecycleState.PLAN_VALIDATION.value,
                    request_id=request_id,
                    candidate_columns=candidate_columns,
                    slot_filling=slot_filling,
                )
            )

        # ── 4b. SEMANTIC FEASIBILITY & GROUNDING VALIDATION ──────────────────────────
        primary_tool_name = intent.tool_name or (intent.execution_plan[0].get("tool") if intent.execution_plan and isinstance(intent.execution_plan[0], dict) else None) or (intent.primary_intent.value if hasattr(intent.primary_intent, "value") else str(intent.primary_intent or ""))
        tool_args = dict(getattr(intent, "arguments", {}) or getattr(intent, "tool_call_arguments", {}) or {})
        if intent.execution_plan and isinstance(intent.execution_plan[0], dict):
            tool_args.update(intent.execution_plan[0].get("arguments", {}))
        if intent.target_column and "target_column" not in tool_args:
            tool_args["target_column"] = intent.target_column
        if intent.time_column and "time_column" not in tool_args:
            tool_args["time_column"] = intent.time_column
        if intent.filters and "filters" not in tool_args:
            tool_args["filters"] = intent.filters
        if intent.dimensions and "group_by" not in tool_args:
            tool_args["group_by"] = intent.dimensions
        if intent.time_context:
            tc_pred = getattr(intent.time_context, "event_predicate", None) or (intent.time_context.get("event_predicate") if isinstance(intent.time_context, dict) else None)
            if tc_pred and "event_predicate" not in tool_args:
                tool_args["event_predicate"] = tc_pred
            tc_event_col = getattr(intent.time_context, "event_column", None) or (intent.time_context.get("event_column") if isinstance(intent.time_context, dict) else None)
            if tc_event_col and "event_column" not in tool_args:
                tool_args["event_column"] = tc_event_col
            tc_time_col = getattr(intent.time_context, "time_column", None) or (intent.time_context.get("time_column") if isinstance(intent.time_context, dict) else None)
            if tc_time_col and "time_column" not in tool_args:
                tool_args["time_column"] = tc_time_col

        log_planning_completed(
            request_id=request_id,
            tool_name=primary_tool_name or "unknown",
            task_family=str(getattr(intent, "primary_intent", "UNKNOWN")),
            arguments=tool_args,
            inferred_bindings=getattr(intent, "inferred_bindings", None),
            attempt=(getattr(intent, "repair_attempts", None) or 0) + 1,
        )

        req_ctx.start_phase("feasibility")
        feasibility_res = SemanticFeasibilityValidator.validate_plan(
            primary_tool_name=primary_tool_name,
            tool_args=tool_args,
            profile=profile,
            query_raw=query_raw,
            df=df,
        )
        req_ctx.end_phase("feasibility")

        if not feasibility_res.is_feasible:
            logger.warning(f"[request_id={request_id}] FEASIBILITY_CHECK_FAILED | code={feasibility_res.error_code} | msg={feasibility_res.error_message}")
            if feasibility_res.error_code == "CLARIFICATION_REQUIRED":
                clarif_msg = humanize_clarification_message(
                    feasibility_res.clarification_prompt or feasibility_res.error_message or "Clarification required.",
                    feasibility_res.candidate_columns
                )
                cand_cols = feasibility_res.candidate_columns or []
                slot_fill = getattr(intent, "slot_filling", None)
                log_feasibility_gate(
                    request_id=request_id,
                    decision="clarification_needed",
                    status_code=422,
                    warnings=[clarif_msg],
                )
                log_query_clarification(
                    request_id=request_id,
                    reason=clarif_msg,
                    candidate_columns=cand_cols,
                    suggested_options=getattr(feasibility_res, "suggested_options", None),
                )
                return CanonicalQueryResponse(
                    request_id=request_id,
                    status=LifecycleState.NEEDS_CLARIFICATION.value,
                    phase=LifecycleState.PLAN_VALIDATION.value,
                    plan=intent_dict,
                    result=None,
                    clarification=clarif_msg,
                    candidate_columns=cand_cols,
                    slot_filling=slot_fill,
                    error=CanonicalError(
                        code="CLARIFICATION_REQUIRED",
                        message=clarif_msg,
                        phase=LifecycleState.PLAN_VALIDATION.value,
                        request_id=request_id,
                        candidate_columns=cand_cols,
                        slot_filling=slot_fill,
                    )
                )
            else:
                log_feasibility_gate(
                    request_id=request_id,
                    decision="rejected",
                    status_code=400,
                    warnings=[feasibility_res.error_message or "Plan failed semantic feasibility validation."],
                )
                log_query_halted(
                    request_id=request_id,
                    code=feasibility_res.error_code or "FEASIBILITY_VALIDATION_FAILED",
                    message=feasibility_res.error_message or "Plan failed semantic feasibility validation.",
                    stage="FEASIBILITY",
                    required_action="Modify query arguments to conform to schema constraints."
                )
                return CanonicalQueryResponse(
                    request_id=request_id,
                    status=LifecycleState.FAILED.value,
                    phase=LifecycleState.PLAN_VALIDATION.value,
                    plan=intent_dict,
                    result=None,
                    error=CanonicalError(
                        code=feasibility_res.error_code or "FEASIBILITY_VALIDATION_FAILED",
                        message=feasibility_res.error_message or "Plan failed semantic feasibility validation.",
                        phase=LifecycleState.PLAN_VALIDATION.value,
                        request_id=request_id,
                        candidate_columns=feasibility_res.candidate_columns
                    )
                )
        else:
            log_feasibility_gate(
                request_id=request_id,
                decision="passed",
                status_code=200,
                checks_passed=["schema_consistency", "role_compatibility", "cardinality_safety"],
                warnings=feasibility_res.warnings if hasattr(feasibility_res, "warnings") and feasibility_res.warnings else None,
            )

        # ── 5. GENERAL CHAT SHORT-CIRCUIT ─────────────────────────────────────────────
        if intent.intent_type == IntentType.GENERAL_CHAT or intent.primary_intent == IntentType.GENERAL_CHAT:
            try:
                messages = []
                if conversation_history:
                    for m in conversation_history[-6:]:
                        role = "user" if m.get("role") in ["user", "human"] else "assistant"
                        messages.append({"role": role, "content": m.get("content", "")})
                messages.append({"role": "user", "content": query_raw})

                sys_prompt = "You are DataPilot OS, an intelligent data science assistant. Answer the user's conversational, conceptual, or educational question clearly and concisely in markdown."
                gen_res = ollama_service.chat_completion(messages=messages, system_prompt=sys_prompt)
                chat_response = gen_res.get("content", "") if isinstance(gen_res, dict) else str(gen_res)
            except Exception as e:
                return CanonicalQueryResponse(
                    request_id=request_id,
                    status=LifecycleState.FAILED.value,
                    phase=LifecycleState.PLANNING.value,
                    plan=intent_dict,
                    result=None,
                    error=QueryErrorDetail(
                        code="OLLAMA_TIMEOUT" if "timeout" in str(e).lower() else "INTENT_PLANNER_UNAVAILABLE",
                        message=f"Failed to generate conversational answer: {str(e)}",
                        phase=LifecycleState.PLANNING.value,
                        request_id=request_id
                    )
                )

            chat_dt = round(time.time() - t_start, 3)
            log_query_success(
                request_id=request_id,
                duration_sec=chat_dt,
                summary_text=chat_response[:140],
                output_shape=None,
            )
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.SUCCEEDED.value,
                phase=LifecycleState.SUCCEEDED.value,
                plan=intent_dict,
                result={
                    "response": chat_response,
                    "answer": chat_response,
                    "data": {"records": [], "columns": []},
                    "type": "text",
                    "code": "",
                    "plot_image": None,
                    "artifacts": {
                        "execution_time_sec": chat_dt,
                        "stdout": chat_response,
                        "stderr": None
                    }
                },
                error=None
            )

        # ── 6. MAP TO ANALYSIS INTENT V2 ─────────────────────────────────────────────
        try:
            analysis_intent_v2 = _map_intent_to_v2(
                intent,
                raw_query=query_raw,
                df=df,
                validated_tool_args=tool_args,
            )
        except ValueError as map_error:
            logger.warning(
                "[request_id=%s] ANALYSIS_CONTRACT_INVALID | %s",
                request_id,
                map_error,
            )
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=LifecycleState.PLAN_VALIDATION.value,
                plan=intent_dict,
                result=None,
                error=CanonicalError(
                    code="INVALID_ANALYSIS_CONTRACT",
                    message=str(map_error),
                    phase=LifecycleState.PLAN_VALIDATION.value,
                    request_id=request_id,
                ),
            )
        logger.info(f"[request_id={request_id}] V2_INTENT_COMPILED | TaskFamily: {analysis_intent_v2.task_family.value}")

        # ── 7. EXECUTE VIA DETERMINISTIC TYPED OPERATORS ENGINE (DS-OS V2) ───────────
        if df is None or not isinstance(df, pd.DataFrame):
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=LifecycleState.EXECUTING.value,
                plan=intent_dict,
                result=None,
                error=QueryErrorDetail(
                    code="NO_DATASET_LOADED",
                    message="An active pandas DataFrame is required to execute analytical operations.",
                    phase=LifecycleState.EXECUTING.value,
                    request_id=request_id
                )
            )

        # ── 7a. TARGET-LEAKAGE INTERDICTION (SUPERVISED CLASSIFICATION) ───────────
        # This is schema- and evidence-based: no feature names, domain terms, or
        # dataset-specific rules are used.  A prospective prediction must not
        # report an apparently excellent held-out score when a feature is an
        # observed deterministic encoding of its label.
        leakage_assessment = None
        if (
            analysis_intent_v2.task_family == TaskFamily.SUPERVISED_ML
            and analysis_intent_v2.arguments.get("task_type") == "classification"
            and analysis_intent_v2.target_column
        ):
            feature_columns = [
                column for column in analysis_intent_v2.subjects
                if column != analysis_intent_v2.target_column
            ]
            leakage_assessment = TargetLeakageDetector.assess_classification(
                df,
                target_column=analysis_intent_v2.target_column,
                feature_columns=feature_columns,
            )
            analysis_intent_v2.arguments["target_leakage_assessment"] = leakage_assessment.to_dict()
            if (
                leakage_assessment.has_deterministic_signal
                and analysis_intent_v2.arguments.get("prediction_context") == "prospective_prediction"
            ):
                # Data-scientist workflow: retain the request, create a safe
                # baseline from the remaining explicitly selected predictors,
                # and preserve an audit trail.  This is deliberately not a
                # silent fallback: the excluded columns and numeric evidence
                # are placed in both the plan and execution manifest.
                excluded = {signal.feature for signal in leakage_assessment.signals}
                safe_features = [column for column in feature_columns if column not in excluded]
                evidence = "; ".join(
                    f"`{signal.feature}` ({signal.mapping_purity:.1%} mapping purity; "
                    f"{signal.held_out_balanced_accuracy:.1%} held-out balanced accuracy)"
                    for signal in leakage_assessment.signals
                )
                if not safe_features:
                    clarification = (
                        "Every selected predictor has deterministic target-leakage evidence: " + evidence + ". "
                        "Provide at least one predictor available before the outcome to create a prospective model."
                    )
                    logger.warning("[request_id=%s] TARGET_LEAKAGE_NO_SAFE_FEATURES | %s", request_id, evidence)
                    return CanonicalQueryResponse(
                        request_id=request_id,
                        status=LifecycleState.NEEDS_CLARIFICATION.value,
                        phase=LifecycleState.PLAN_VALIDATION.value,
                        plan=intent_dict,
                        result=None,
                        clarification=clarification,
                        error=CanonicalError(
                            code="TARGET_LEAKAGE_RISK",
                            message=clarification,
                            phase=LifecycleState.PLAN_VALIDATION.value,
                            request_id=request_id,
                            candidate_columns=sorted(excluded),
                            details=leakage_assessment.to_dict(),
                        ),
                    )

                analysis_intent_v2.subjects = [analysis_intent_v2.target_column] + safe_features
                analysis_intent_v2.arguments["excluded_leakage_features"] = sorted(excluded)
                analysis_intent_v2.arguments["leakage_policy"] = "excluded_from_prospective_baseline"
                intent_dict["safety_audit"] = {
                    "policy": "excluded_from_prospective_baseline",
                    "excluded_features": sorted(excluded),
                    "retained_features": safe_features,
                    "evidence": leakage_assessment.to_dict(),
                }
                logger.warning(
                    "[request_id=%s] TARGET_LEAKAGE_EXCLUDED_FROM_BASELINE | excluded=%s retained=%s",
                    request_id, sorted(excluded), safe_features,
                )

        req_ctx.start_phase("execution")
        op_out: OperatorOutput = TypedOperatorRegistry.execute_intent(
            df=df,
            intent=analysis_intent_v2,
            profile=profile
        )
        req_ctx.end_phase("execution")

        if op_out.status != "SUCCEEDED":
            err_code = op_out.error_code or "UNSUPPORTED_OPERATION"
            err_msg = op_out.error_message or f"Execution failed for TaskFamily '{analysis_intent_v2.task_family.value}'."
            phase_val = LifecycleState.PLAN_VALIDATION.value if err_code in ("COLUMN_NOT_FOUND", "NO_TEMPORAL_COLUMN", "AMBIGUOUS_COLUMN", "SUBJECT_COLUMNS_REQUIRED", "EVENT_COLUMN_REQUIRED", "INSUFFICIENT_NUMERIC_COLUMNS") else LifecycleState.EXECUTING.value
            candidate_columns: List[str] = []
            if err_code == "COLUMN_NOT_FOUND":
                available_columns = [str(column) for column in df.columns]
                requested_columns = list(dict.fromkeys(
                    [str(column) for column in (
                        list(analysis_intent_v2.subjects or [])
                        + list(analysis_intent_v2.grouping or [])
                        + ([analysis_intent_v2.target_column] if analysis_intent_v2.target_column else [])
                        + ([analysis_intent_v2.time_column] if analysis_intent_v2.time_column else [])
                    )]
                ))
                missing_columns = [column for column in requested_columns if column not in available_columns]
                for missing in missing_columns:
                    for match in difflib.get_close_matches(missing, available_columns, n=3, cutoff=0.45):
                        if match not in candidate_columns:
                            candidate_columns.append(match)
                if candidate_columns:
                    err_msg += " Similar existing column(s): " + ", ".join(
                        f"`{column}`" for column in candidate_columns
                    ) + ". No substitution was applied."
            if err_code == "AMBIGUOUS_COLUMN":
                cand_cols = candidate_columns or getattr(op_out, "candidate_columns", []) or (list(df.columns) if df is not None and hasattr(df, "columns") else [])
                humanized = humanize_clarification_message(err_msg, cand_cols)
                logger.warning(f"[request_id={request_id}] OPERATOR_{err_code} | Reason: {humanized}")
                return CanonicalQueryResponse(
                    request_id=request_id,
                    status=LifecycleState.NEEDS_CLARIFICATION.value,
                    phase=phase_val,
                    plan=intent_dict,
                    result=None,
                    clarification=humanized,
                    candidate_columns=cand_cols,
                    error=CanonicalError(
                        code=err_code,
                        message=humanized,
                        phase=phase_val,
                        request_id=request_id,
                        candidate_columns=cand_cols,
                    ),
                    diagnostics={"phase_timings": dict(req_ctx.phase_timings)}
                )
            logger.warning(f"[request_id={request_id}] OPERATOR_EXECUTION_FAILED | code={err_code} | msg={err_msg}")
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=phase_val,
                plan=intent_dict,
                result=None,
                error=QueryErrorDetail(
                    code=err_code,
                    message=err_msg,
                    phase=phase_val,
                    request_id=request_id,
                    candidate_columns=candidate_columns,
                ),
                diagnostics={"phase_timings": dict(req_ctx.phase_timings)}
            )

        # ── 8. SEMANTIC RESULT VALIDATION ────────────────────────────────────────────
        log_operator_execution(
            request_id=request_id,
            operator=op_out.manifest.operator if op_out.manifest else str(primary_tool_name),
            rows_in=op_out.manifest.row_count_in if op_out.manifest else (len(df) if df is not None else 0),
            rows_out=op_out.manifest.row_count_out if op_out.manifest else 0,
            duration_sec=round(time.time() - t_start, 3),
            chart_type=op_out.manifest.chart_type if op_out.manifest else "none",
            chart_generated=bool(op_out.plot_image_base64 or op_out.plotly_spec),
        )

        req_ctx.start_phase("verification")
        val_res = SemanticResultValidator.validate_output(analysis_intent_v2, op_out, input_row_count=len(df))
        req_ctx.end_phase("verification")

        log_result_verification(
            request_id=request_id,
            is_valid=val_res.is_valid,
            issues=val_res.issues,
        )

        if not val_res.is_valid:
            err_msg = "Result Validation Violations: " + "; ".join(val_res.issues)
            logger.warning(f"[request_id={request_id}] SEMANTIC_VALIDATION_FAILED | Issues: {val_res.issues}")
            log_query_halted(request_id, code="RESULT_VALIDATION_FAILED", message=err_msg, stage="VALIDATING_RESULT", required_action="Review output constraints.")
            return CanonicalQueryResponse(
                request_id=request_id,
                status=LifecycleState.FAILED.value,
                phase=LifecycleState.VALIDATING_RESULT.value,
                plan=intent_dict,
                result=None,
                error=QueryErrorDetail(
                    code="RESULT_VALIDATION_FAILED",
                    message=err_msg,
                    phase=LifecycleState.VALIDATING_RESULT.value,
                    request_id=request_id
                ),
                diagnostics={"phase_timings": dict(req_ctx.phase_timings)}
            )

        # ── 9. CONSTRUCT CANONICAL RESPONSE PAYLOAD ──────────────────────────────────
        if output_frames is not None:
            output_frames.append(op_out.dataframe)
        dt_total = round(time.time() - t_start, 3)

        intent_dict["execution_plan"] = [
            {"step_index": 1, "intent_type": intent.primary_intent.value if hasattr(intent.primary_intent, "value") else str(intent.primary_intent or "ANALYSIS"), "status": "SUCCEEDED"}
        ]
        if intent.filters and (intent.primary_intent != IntentType.FILTER_QUERY and intent.intent_type != IntentType.FILTER_QUERY):
            intent_dict["execution_plan"].insert(0, {"step_index": 1, "intent_type": "FILTER_QUERY", "status": "SUCCEEDED"})
            intent_dict["execution_plan"][1]["step_index"] = 2

        repair_history_list = [r.model_dump() for r in req_ctx.repair_budget_manager.history] if req_ctx.repair_budget_manager else []
        manifest_dict = op_out.manifest.model_dump()
        inferred_bindings = getattr(intent, "inferred_bindings", []) or []
        is_ds_wide = bool(getattr(intent, "is_dataset_wide", False))
        if inferred_bindings:
            manifest_dict["inferred_bindings"] = inferred_bindings
        if is_ds_wide:
            manifest_dict["is_dataset_wide"] = is_ds_wide

        result_payload = {
            "response": op_out.written_finding,
            "answer": op_out.written_finding,
            "data": op_out.data,
            "type": "chart" if op_out.plot_image_base64 or op_out.plotly_spec else "table",
            "code": op_out.reproducible_code,
            "plot_image": op_out.plot_image_base64,
            "plotly_spec": op_out.plotly_spec,
            "manifest": manifest_dict,
            "validation": val_res.model_dump(),
            "inferred_bindings": inferred_bindings,
            "is_dataset_wide": is_ds_wide,
            "artifacts": {
                "execution_time_sec": dt_total,
                "stdout": op_out.written_finding,
                "stderr": None,
                "dataset_hash": profile.dataset_hash if profile else None,
                "target_leakage_assessment": leakage_assessment.to_dict() if leakage_assessment else None,
                "phase_timings": dict(req_ctx.phase_timings),
                "repair_audit": repair_history_list,
                "inferred_bindings": inferred_bindings,
                "is_dataset_wide": is_ds_wide,
            }
        }

        out_shape = (
            op_out.dataframe.shape
            if getattr(op_out, "dataframe", None) is not None
            else (manifest_dict.get("row_count_out", 0), len(df.columns) if df is not None else 0)
        )
        logger.info(f"[request_id={request_id}] QUERY_SUCCEEDED | Duration: {dt_total}s")
        log_query_success(
            request_id=request_id,
            duration_sec=dt_total,
            summary_text=op_out.written_finding or "Query succeeded.",
            output_shape=out_shape,
        )
        return CanonicalQueryResponse(
            request_id=request_id,
            status=LifecycleState.SUCCEEDED.value,
            phase=LifecycleState.SUCCEEDED.value,
            plan=intent_dict,
            result=result_payload,
            inferred_bindings=inferred_bindings,
            error=None,
            diagnostics={
                "phase_timings": dict(req_ctx.phase_timings),
                "repair_audit": repair_history_list,
            }
        )
