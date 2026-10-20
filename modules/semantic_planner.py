"""
DataPilot OS — Semantic Planner.
Extracts minimal semantic tool commands from user queries (English/Hinglish)
and resolves them into validated, grounded SemanticIntent objects via CommandResolver.
"""

import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional, List
import pandas as pd

from backend.intelligence.ollama_service import ollama_service, OllamaRequest, OLLAMA_TIMEOUT
from backend.ds_engine.intent_schema import SemanticIntent, IntentType, ChartType, AggregationType
from backend.ds_engine.semantic_command import MINIMAL_COMMAND_JSON_SCHEMA, LLMSemanticCommand, SCHEMA_VERSION, PROMPT_VERSION
from backend.ds_engine.command_resolver import command_resolver
from modules.capability_registry import capability_registry
from modules.planner_learning_memory import planner_learning_memory
from modules.planner_prompt_contract import render_tool_contract, render_user_prompt, render_system_prompt, eligible_tools, render_runtime_system
from backend.core.error_states import ErrorState
from backend.core.request_context import get_request_context
from backend.ds_engine.repair_budget import RepairBudgetManager
from modules.planner_audit import planner_auditor, PlannerAuditRecord, PlannerAuditor as PlannerAuditorCls

logger = logging.getLogger(__name__)


class SemanticPlanner:
    """
    Ollama Semantic Brain: Parses natural language queries into
    minimal semantic tool commands, resolved deterministically by DataPilot.
    """

    _selected_tool_contract = staticmethod(render_tool_contract)

    @staticmethod
    def _uses_only_selected_tools(parsed: Dict[str, Any], allowed_tools: set[str]) -> bool:
        """Reject calls outside the retrieved ToolSpec set before resolution."""
        calls = parsed.get("commands") if isinstance(parsed, dict) else None
        if calls is None:
            calls = [parsed]
        if not isinstance(calls, list) or not calls:
            return False
        names = [call.get("tool") for call in calls if isinstance(call, dict)]
        return bool(names) and len(names) == len(calls) and all(name in allowed_tools for name in names)

    @staticmethod
    def _find_candidate_analytical_tool(parsed: Dict[str, Any]) -> Optional[str]:
        """Identify the primary analytical tool candidate from a parsed response."""
        if not isinstance(parsed, dict):
            return None
        calls = parsed.get("commands")
        if not isinstance(calls, list):
            calls = [parsed]
        for call in calls:
            if isinstance(call, dict):
                t_name = call.get("tool")
                if isinstance(t_name, str) and t_name.strip():
                    spec = capability_registry.get_tool_spec(t_name.strip())
                    if spec and spec.category != "clarification":
                        return spec.name
        return None

    @staticmethod
    def _normalise_tool_protocol(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert provider function-call wire shapes to the canonical ToolSpec shape.

        Some local Ollama models serialize a function invocation as
        ``{"name": "tool_name", "arguments": {...}}`` even when the prompt
        asks for the application's canonical ``tool`` field.  ``name`` is a
        protocol alias, not a semantic alias: it is accepted only when there is
        no conflicting ``tool`` value.  Tool membership and Pydantic argument
        validation still happen immediately afterwards.
        """
        if not isinstance(parsed, dict):
            return parsed

        def normalise_call(call: Any) -> Any:
            if not isinstance(call, dict):
                return call
            canonical = dict(call)
            tool = canonical.get("tool")
            name = canonical.get("name")
            if tool is None and isinstance(name, str) and name.strip():
                canonical["tool"] = name.strip()
            # A response that identifies two different operations is malformed;
            # leave it unnormalised so the strict selected-tool gate rejects it.
            elif tool is not None and name is not None and tool != name:
                canonical.pop("tool", None)
            return canonical

        canonical = normalise_call(parsed)
        commands = canonical.get("commands")
        if isinstance(commands, list):
            canonical["commands"] = [normalise_call(call) for call in commands]
        return canonical

    @staticmethod
    def _trusted_grounding_text(
        query_raw: str,
        conversation_history: Optional[List[Dict[str, str]]],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Combine only user-authored text and explicit UI selections.

        Assistant/model messages are deliberately excluded: a hallucinated
        column in an earlier response must never become evidence for itself.
        """
        parts = [query_raw]
        for message in conversation_history or []:
            if isinstance(message, dict) and str(message.get("role", "")).casefold() == "user":
                parts.append(str(message.get("content", "")))
        if isinstance(context, dict):
            selected = context.get("selected_columns") or []
            if isinstance(selected, str):
                selected = [selected]
            if isinstance(selected, list):
                parts.extend(str(value) for value in selected)
        return "\n".join(parts)

    @staticmethod
    def _dataset_column_grounding_error(
        parsed: Dict[str, Any],
        df: Optional[pd.DataFrame],
        evidence_text: str,
    ) -> Optional[str]:
        """Reject invented schema bindings without language keyword routing.

        The check uses the live dataframe schema only. Every source column
        selected by an executable plan must occur in user-authored evidence
        (underscores and punctuation are treated as spaces). This prevents a
        plan from smuggling invented columns alongside one grounded column.
        """
        if df is None or not isinstance(parsed, dict):
            return None
        calls = parsed.get("commands")
        if not isinstance(calls, list):
            calls = [parsed]
        normalized_evidence = " " + re.sub(r"[_\W]+", " ", evidence_text.casefold()).strip() + " "
        schema = {str(column).casefold(): str(column) for column in df.columns}
        selected: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, str) and value.casefold() in schema:
                selected.add(schema[value.casefold()])

        executable = False
        inferred_columns: set[str] = set()
        for call in calls:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool", "")).lower().strip()
            # Dataset-wide operations without columns are permitted
            if tool_name in ("eda_data", "eda"):
                eda_cols = call.get("arguments", {}).get("columns") if isinstance(call.get("arguments"), dict) else None
                if not eda_cols:
                    continue
            if call.get("is_dataset_wide") or (isinstance(call.get("arguments"), dict) and call["arguments"].get("is_dataset_wide")):
                continue
            for b in call.get("inferred_bindings", []):
                if isinstance(b, dict):
                    col_val = b.get("column")
                    if isinstance(col_val, list):
                        inferred_columns.update(str(c) for c in col_val)
                    elif isinstance(col_val, str):
                        inferred_columns.add(str(col_val))
            spec = capability_registry.get_tool_spec(tool_name)
            if spec and spec.requires_dataset:
                executable = True
                visit(call.get("arguments", {}))
        if not executable or not selected:
            return None
        grounded: set[str] = set()
        
        # Schema-level semantic categories explicitly requested by user (zero domain guessing)
        has_numeric_selector = bool(re.search(r"\b(numeric|numerical|number|numbers|metric|metrics|quantitative)\s+columns?\b|\ball\s+(numeric|numerical|numbers|metrics)\b", normalized_evidence))
        has_categorical_selector = bool(re.search(r"\b(categorical|category|categories|string|strings|text)\s+columns?\b|\ball\s+(categorical|categories)\b", normalized_evidence))
        has_datetime_selector = bool(re.search(r"\b(date|datetime|time|temporal)\s+columns?\b|\ball\s+(date|datetime|temporal)\b", normalized_evidence))
        has_boolean_selector = bool(re.search(r"\b(bool|boolean)\s+columns?\b|\ball\s+bool(ean)?s?\b", normalized_evidence))
        has_missing_selector = bool(re.search(r"\b(missing|null|nulls|nan|nans)\b", normalized_evidence))

        for column in selected:
            if column in inferred_columns:
                grounded.add(column)
                continue
            normalized_column = re.sub(r"[_\W]+", " ", column.casefold()).strip()
            if normalized_column and f" {normalized_column} " in normalized_evidence:
                grounded.add(column)
                continue
            
            # Check schema-level deterministic grounded criteria
            if column in df.columns:
                series = df[column]
                if has_numeric_selector and pd.api.types.is_numeric_dtype(series):
                    grounded.add(column)
                    continue
                if has_categorical_selector and (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)):
                    grounded.add(column)
                    continue
                if has_datetime_selector and pd.api.types.is_datetime64_any_dtype(series):
                    grounded.add(column)
                    continue
                if has_boolean_selector and pd.api.types.is_bool_dtype(series):
                    grounded.add(column)
                    continue
                if has_missing_selector and series.isna().any():
                    grounded.add(column)
                    continue

        ungrounded = selected - grounded
        if not ungrounded:
            return None
        return (
            "UNGROUNDED_COLUMN_SELECTION: The plan selected dataset column(s) "
            f"{sorted(ungrounded)} without support in the current or prior user-authored request "
            "or explicit UI column selection. Ask which columns/metrics to use instead of inventing them."
        )

    @classmethod
    def extract_semantic_intent(
        cls,
        query_raw: str,
        df: Optional[pd.DataFrame] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_sec: float = OLLAMA_TIMEOUT,
        planner_model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        t_start = time.time()
        req_ctx = get_request_context()
        request_id = req_ctx.request_id or "ephemeral"
        if not req_ctx.request_id or req_ctx.request_id == "ephemeral":
            repair_budget_mgr = RepairBudgetManager(max_attempts=2)
        else:
            repair_budget_mgr = req_ctx.get_repair_budget_manager(max_attempts=2)
        initial_raw: Optional[str] = None
        previous_candidate: Optional[Dict[str, Any]] = None
        validation_failures: List[str] = []
        candidate_tool_name: Optional[str] = None

        prompt = render_user_prompt(query_raw, df, conversation_history, context)

        planner_specs = capability_registry.select_planner_tools(
            query_raw,
            has_dataset=df is not None,
        )
        available_specs = eligible_tools(capability_registry, df is not None)

        max_repair_attempts = 2
        last_error_reason = None
        current_prompt = prompt

        for attempt in range(1, max_repair_attempts + 2):
            system_prompt = render_runtime_system(planner_specs, available_specs, query_raw)

            req = OllamaRequest(
                prompt=current_prompt,
                system_prompt=system_prompt,
                # Intent compilation is structured extraction, not an agent
                # loop.  Use exactly one Ollama protocol: JSON-schema format.
                # Combining `tools` and `format` produces ambiguous/empty
                # tool-call behaviour on local runtimes and defeats a locked
                # schema.  Execution happens only after this response passes
                # the registry resolver.
                tools=None,
                response_schema=capability_registry.get_dynamic_command_json_schema(
                    planner_specs,
                    allow_command_sequence=True,
                ),
                temperature=0.0,
                top_p=0.1,
                timeout_sec=timeout_sec,
                think=False,
                # A tool call is a short, structured action. Keeping this
                # compact avoids wasting CPU-only local inference on prose.
                max_tokens=int(os.getenv("OLLAMA_PLANNER_MAX_TOKENS", "256")),
                # Benchmarking may compare a candidate model without mutating
                # the model selected for the live desktop session.
                model=planner_model,
            )

            try:
                logger.info(f"OLLAMA_SEMANTIC_PLANNER_ATTEMPT | Attempt #{attempt} | Query: '{query_raw}'")
                response = ollama_service.structured_request(req)
                if not response.success:
                    logger.warning(f"OLLAMA_SERVICE_FAILED | Attempt #{attempt} | Code: {response.error_code} | Msg: {response.error_message}")
                    last_error_reason = response.error_code or "OLLAMA_UNAVAILABLE"
                    continue

                raw_content = response.content or ""
                if initial_raw is None:
                    initial_raw = raw_content

                parsed = response.parsed_json or {}
                if not parsed and raw_content:
                    try:
                        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group(0))
                    except Exception:
                        parsed = {}

                # If in self-repair loop (attempt > 1), enforce finite budget & diff-checking
                if attempt > 1:
                    current_candidate = parsed if parsed else {"raw": raw_content}
                    is_clarification_attempt = (
                        isinstance(parsed, dict)
                        and (
                            parsed.get("tool") == "clarification_needed"
                            or (isinstance(parsed.get("commands"), list) and any(c.get("tool") == "clarification_needed" for c in parsed.get("commands", [])))
                        )
                    )
                    if not is_clarification_attempt and parsed and isinstance(previous_candidate, dict) and not previous_candidate.get("raw"):
                        orig_hash = repair_budget_mgr.compute_plan_hash(previous_candidate or {})
                        prop_hash = repair_budget_mgr.compute_plan_hash(current_candidate)
                        if orig_hash == prop_hash:
                            progress_err = repair_budget_mgr.check_progress_and_record(
                                original_plan=previous_candidate or {},
                                proposed_plan=current_candidate,
                                error_diagnosed=last_error_reason or "",
                                success=False,
                            )
                            if progress_err is not None:
                                logger.warning(f"REPAIR_BUDGET_ABORT | {progress_err.code} | {progress_err.message}")
                            validation_failures.append(progress_err.message)
                            planner_auditor.record(PlannerAuditRecord(
                                request_id=request_id,
                                query_fingerprint=PlannerAuditorCls.fingerprint_query(query_raw),
                                model_name=planner_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
                                retrieved_tools=[s.name for s in planner_specs],
                                locked_tool=candidate_tool_name,
                                initial_raw_response=initial_raw,
                                validation_failures=validation_failures,
                                repair_attempts=repair_budget_mgr.attempts_used,
                                repair_history=[r.model_dump() for r in repair_budget_mgr.history],
                                final_decision="clarify" if progress_err.code != ErrorState.LOOP_BUDGET_EXCEEDED else "failed",
                                stage_latencies_ms={"planning_ms": round((time.time() - t_start) * 1000, 2)},
                            ))
                            return {
                                "intent_type": "UNSUPPORTED" if progress_err.code == ErrorState.LOOP_BUDGET_EXCEEDED else IntentType.CLARIFICATION.value,
                                "primary_intent": "UNSUPPORTED" if progress_err.code == ErrorState.LOOP_BUDGET_EXCEEDED else IntentType.CLARIFICATION.value,
                                "raw_query": query_raw,
                                "is_ambiguous": True if progress_err.code != ErrorState.LOOP_BUDGET_EXCEEDED else False,
                                "ambiguity_reason": progress_err.message,
                                "error_code": progress_err.code.value if hasattr(progress_err.code, "value") else str(progress_err.code),
                                "error_message": progress_err.message,
                                "schema_version": SCHEMA_VERSION,
                                "prompt_version": PROMPT_VERSION,
                                "raw_llm_output": raw_content,
                                "repair_attempts": repair_budget_mgr.attempts_used,
                                "repair_history": [r.model_dump() for r in repair_budget_mgr.history],
                            }

                if not parsed:
                    last_error_reason = "Empty or malformed JSON returned by LLM."
                    logger.warning(f"SEMANTIC_PLANNER_MALFORMED_JSON | Attempt #{attempt} | Content: {raw_content[:200]}")
                    validation_failures.append(last_error_reason)
                    previous_candidate = {"raw": raw_content}
                    if attempt <= max_repair_attempts:
                        current_prompt = (
                            f"{prompt}\n\n"
                            "REPAIR NOTICE: Your previous output was not valid JSON. "
                            "Please return ONLY a strict JSON object matching the registered tool schema."
                        )
                    continue

                # Canonicalise only the provider-level function-call envelope.
                # This never selects a tool, fills arguments, or relaxes the
                # registry contract; it merely makes Ollama's `name` spelling
                # interoperable with DataPilot's `tool` spelling.
                parsed = cls._normalise_tool_protocol(parsed)
                logger.info(
                    "SEMANTIC_PLANNER_CANDIDATE_COMMAND | Attempt #%s | %s",
                    attempt,
                    json.dumps(parsed, ensure_ascii=False)[:4000],
                )

                allowed_tool_names = {spec.name for spec in planner_specs}
                if not cls._uses_only_selected_tools(parsed, allowed_tool_names):
                    invalid_names = []
                    calls = parsed.get("commands") if isinstance(parsed, dict) else None
                    for call in calls if isinstance(calls, list) else [parsed]:
                        if isinstance(call, dict) and call.get("tool") not in allowed_tool_names:
                            invalid_names.append(str(call.get("tool")))
                    last_error_reason = (
                        "Planner selected a tool outside the retrieved capability contract: "
                        + ", ".join(invalid_names or ["missing tool name"])
                    )
                    logger.warning("SEMANTIC_PLANNER_OUT_OF_CONTRACT_TOOL | %s", last_error_reason)
                    validation_failures.append(last_error_reason)
                    previous_candidate = parsed
                    if attempt <= max_repair_attempts:
                        current_prompt = (
                            f"{prompt}\n\nREPAIR NOTICE: Your previous output produced an error: {last_error_reason}. "
                            f"Return a call only from: {', '.join(sorted(allowed_tool_names))}."
                        )
                # Data-aware intent completion for dataset-wide modes and missing role slots
                if df is not None and isinstance(df, pd.DataFrame):
                    from backend.ds_engine.dataset_profiler import DatasetProfiler
                    from backend.ds_engine.intent_completion import IntentCompletionService
                    profile_for_comp = DatasetProfiler.profile(df)
                    primary_tool = cls._find_candidate_analytical_tool(parsed)
                    if primary_tool:
                        calls_list = parsed.get("commands") if isinstance(parsed, dict) and isinstance(parsed.get("commands"), list) else [parsed]
                        target_call = next((c for c in calls_list if isinstance(c, dict) and c.get("tool") == primary_tool), None)
                        if target_call is not None:
                            completion_decision = IntentCompletionService.complete_intent(
                                tool_name=primary_tool,
                                arguments=target_call.get("arguments", {}),
                                profile=profile_for_comp,
                                query_raw=query_raw,
                                conversation_history=conversation_history,
                                context=context,
                            )
                            if completion_decision.action == "clarify":
                                planner_auditor.record(PlannerAuditRecord(
                                    request_id=request_id,
                                    query_fingerprint=PlannerAuditorCls.fingerprint_query(query_raw),
                                    model_name=planner_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
                                    retrieved_tools=[s.name for s in planner_specs],
                                    locked_tool=candidate_tool_name,
                                    initial_raw_response=initial_raw,
                                    validation_failures=validation_failures,
                                    repair_attempts=repair_budget_mgr.attempts_used,
                                    repair_history=[r.model_dump() for r in repair_budget_mgr.history],
                                    final_decision="clarify",
                                    final_tool_call=parsed,
                                    stage_latencies_ms={"planning_ms": round((time.time() - t_start) * 1000, 2)},
                                ))
                                return {
                                    "intent_type": IntentType.CLARIFICATION.value,
                                    "primary_intent": IntentType.CLARIFICATION.value,
                                    "raw_query": query_raw,
                                    "is_ambiguous": True,
                                    "ambiguity_reason": completion_decision.clarification_prompt or "Clarification required.",
                                    "candidate_columns": completion_decision.candidate_columns,
                                    "slot_filling": completion_decision.slot_filling.model_dump() if completion_decision.slot_filling else None,
                                    "schema_version": SCHEMA_VERSION,
                                    "prompt_version": PROMPT_VERSION,
                                    "raw_llm_output": raw_content,
                                    "repair_attempts": attempt - 1,
                                    "repair_history": [r.model_dump() for r in repair_budget_mgr.history],
                                }
                            elif completion_decision.action == "execute":
                                target_call["arguments"] = completion_decision.completed_arguments
                                target_call["is_dataset_wide"] = completion_decision.is_dataset_wide
                                target_call["inferred_bindings"] = [b.model_dump() for b in completion_decision.inferred_bindings]

                grounding_error = cls._dataset_column_grounding_error(
                    parsed,
                    df,
                    cls._trusted_grounding_text(query_raw, conversation_history, context),
                )
                if grounding_error:
                    last_error_reason = grounding_error
                    logger.warning("SEMANTIC_PLANNER_COLUMN_GROUNDING_FAILED | %s", grounding_error)
                    validation_failures.append(last_error_reason)
                    previous_candidate = parsed
                    if attempt <= max_repair_attempts:
                        current_prompt = (
                            f"{prompt}\n\nREPAIR NOTICE: {grounding_error} "
                            "Return clarification_needed unless the original user-authored evidence explicitly names a valid source column."
                        )
                    continue

                # Preserve ordered plans. Ground each step on its actual intermediate
                # dataframe in the orchestrator, never merge their argument state.
                if (isinstance(parsed.get("commands"), list) and len(parsed["commands"]) > 1
                        and all(capability_registry.get_tool_spec(c['tool']).requires_dataset for c in parsed['commands'])):
                    calls = parsed["commands"]
                    if len(calls) > 8:
                        raise ValueError("INVALID_PLAN_SEQUENCE: maximum 8 steps")
                    for call in calls:
                        spec = capability_registry.get_tool_spec(call["tool"])
                        if not spec.requires_dataset:
                            raise ValueError("INVALID_PLAN_SEQUENCE: conversation cannot be mixed with execution")
                        capability_registry.validate_tool_call(call["tool"], call.get("arguments", {}))

                    planner_auditor.record(PlannerAuditRecord(
                        request_id=request_id,
                        query_fingerprint=PlannerAuditorCls.fingerprint_query(query_raw),
                        model_name=planner_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
                        retrieved_tools=[s.name for s in planner_specs],
                        locked_tool=candidate_tool_name,
                        initial_raw_response=initial_raw,
                        validation_failures=validation_failures,
                        repair_attempts=repair_budget_mgr.attempts_used,
                        repair_history=[r.model_dump() for r in repair_budget_mgr.history],
                        final_decision="execute",
                        final_tool_call=parsed,
                        stage_latencies_ms={"planning_ms": round((time.time() - t_start) * 1000, 2)},
                    ))
                    return {"commands": calls, "schema_version": SCHEMA_VERSION,
                            "raw_llm_output": raw_content, "repair_attempts": attempt - 1,
                            "repair_history": [r.model_dump() for r in repair_budget_mgr.history]}

                # Resolve minimal command or legacy dictionary into validated SemanticIntent
                resolution = command_resolver.resolve_command(parsed, df=df, raw_query=query_raw)

                if resolution.is_valid and resolution.semantic_intent:
                    # Local models can overlook evidence already present in
                    # the request. Confirm an initial clarification through
                    # the same strict contract before displaying it; a second
                    # clarification remains a genuine user-facing question.
                    if resolution.is_ambiguous and attempt == 1 and df is not None:
                        last_error_reason = resolution.ambiguity_reason or "The planner requested clarification."
                        logger.warning(
                            "SEMANTIC_PLANNER_CLARIFICATION_CONFIRMATION | Attempt #%s | Reason: %s",
                            attempt,
                            last_error_reason,
                        )
                        validation_failures.append(last_error_reason)
                        previous_candidate = parsed
                        current_prompt = (
                            f"{prompt}\n\n"
                            "CLARIFICATION CONFIRMATION: Re-read the entire original USER QUERY and live dataset schema. "
                            "If it already states an exact column, metric, aggregation, grouping, date parse, ordering, "
                            "horizon, filter, or chart, bind it to the registered tool arguments now. Return "
                            "clarification_needed only when a required value is genuinely absent; otherwise return exactly "
                            "one complete schema-valid analysis command. Never mix clarification_needed with analysis tools."
                        )
                        continue

                    if attempt > 1:
                        repair_budget_mgr.check_progress_and_record(
                            original_plan=previous_candidate or {},
                            proposed_plan=parsed,
                            error_diagnosed="",
                            success=True,
                        )

                    planner_auditor.record(PlannerAuditRecord(
                        request_id=request_id,
                        query_fingerprint=PlannerAuditorCls.fingerprint_query(query_raw),
                        model_name=planner_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
                        retrieved_tools=[s.name for s in planner_specs],
                        locked_tool=candidate_tool_name,
                        initial_raw_response=initial_raw,
                        validation_failures=validation_failures,
                        repair_attempts=repair_budget_mgr.attempts_used,
                        repair_history=[r.model_dump() for r in repair_budget_mgr.history],
                        final_decision="clarify" if resolution.is_ambiguous else "execute",
                        final_tool_call=parsed,
                        stage_latencies_ms={"planning_ms": round((time.time() - t_start) * 1000, 2)},
                    ))

                    intent_obj = resolution.semantic_intent
                    intent_dict = intent_obj.model_dump()
                    intent_dict["schema_version"] = SCHEMA_VERSION
                    intent_dict["prompt_version"] = PROMPT_VERSION
                    intent_dict["raw_llm_output"] = raw_content
                    intent_dict["llm_latency_ms"] = response.latency_ms
                    intent_dict["prompt_eval_count"] = response.prompt_eval_count
                    intent_dict["eval_count"] = response.eval_count
                    intent_dict["repair_attempts"] = attempt - 1
                    intent_dict["repair_history"] = [r.model_dump() for r in repair_budget_mgr.history]

                    if resolution.is_ambiguous:
                        intent_dict["is_ambiguous"] = True
                        intent_dict["ambiguity_reason"] = resolution.ambiguity_reason
                    return intent_dict

                if resolution.is_ambiguous:
                    planner_auditor.record(PlannerAuditRecord(
                        request_id=request_id,
                        query_fingerprint=PlannerAuditorCls.fingerprint_query(query_raw),
                        model_name=planner_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
                        retrieved_tools=[s.name for s in planner_specs],
                        locked_tool=candidate_tool_name,
                        initial_raw_response=initial_raw,
                        validation_failures=validation_failures,
                        repair_attempts=repair_budget_mgr.attempts_used,
                        repair_history=[r.model_dump() for r in repair_budget_mgr.history],
                        final_decision="clarify",
                        final_tool_call=parsed,
                        stage_latencies_ms={"planning_ms": round((time.time() - t_start) * 1000, 2)},
                    ))
                    return {
                        "intent_type": IntentType.EDA.value,
                        "primary_intent": IntentType.EDA.value,
                        "raw_query": query_raw,
                        "is_ambiguous": True,
                        "ambiguity_reason": resolution.ambiguity_reason or "Query is ambiguous.",
                        "schema_version": SCHEMA_VERSION,
                        "prompt_version": PROMPT_VERSION,
                        "raw_llm_output": raw_content,
                        "llm_latency_ms": response.latency_ms,
                        "prompt_eval_count": response.prompt_eval_count,
                        "eval_count": response.eval_count,
                        "repair_attempts": attempt - 1,
                        "repair_history": [r.model_dump() for r in repair_budget_mgr.history]
                    }

                # If resolution rejected the command (e.g. unknown tool, validation failure, missing fields, mixed clarification), attempt repair
                last_error_reason = resolution.error_message or "Command resolution failed."
                if resolution.error_code:
                    last_error_reason = f"{resolution.error_code}: {last_error_reason}"
                logger.warning(f"SEMANTIC_PLANNER_REPAIR_TRIGGERED | Attempt #{attempt} | Reason: {last_error_reason}")
                validation_failures.append(last_error_reason)
                previous_candidate = parsed

                if attempt <= max_repair_attempts:
                    if resolution.error_code == "UNSUPPORTED_VISUALIZATION":
                        # If a real operator contract rejected a requested chart,
                        # dynamically expose only registry-declared chart-capable
                        # alternatives for the repair.
                        planner_specs = capability_registry.expand_tools_for_requested_visualizations(
                            planner_specs,
                            parsed,
                        )
                        recognized_tools = ", ".join([s.name for s in planner_specs])
                        current_prompt = (
                            f"{prompt}\n\n"
                            f"REPAIR NOTICE: Your previous output produced an error: {last_error_reason}. "
                            "Re-read the original USER QUERY and preserve every value it explicitly states. "
                            "Do not ask for clarification when the requested metric, grouping, aggregate, or other required value is already stated there. "
                            f"Invoke exactly one of the registered tools with schema-valid arguments: {recognized_tools}."
                        )
                    else:
                        candidate_tool_name = cls._find_candidate_analytical_tool(parsed)
                        if candidate_tool_name:
                            locked_spec = capability_registry.get_tool_spec(candidate_tool_name)
                            if locked_spec:
                                planner_specs = [locked_spec]
                                available_specs = [locked_spec]

                        rejected_cmd = json.dumps(parsed, ensure_ascii=False)
                        recognized_tools = ", ".join([s.name for s in planner_specs])
                        current_prompt = (
                            f"{prompt}\n\n"
                            f"REPAIR NOTICE: Your previous output produced an error: {last_error_reason}.\n"
                            f"Rejected candidate command: {rejected_cmd}\n"
                            "Re-read the original USER QUERY and live dataset schema. "
                            "Preserve every value explicitly stated in the query (metric, dimensions, aggregation, datetime parsing if requested, and chart type). "
                            "Do not ask for clarification when the requested metric, grouping, aggregate, or other required value is already stated there. "
                            f"Invoke ONLY the registered tool with schema-valid arguments: {recognized_tools}."
                        )

            except Exception as e:
                logger.error(f"OLLAMA_SEMANTIC_PLANNER_EXCEPTION | Attempt #{attempt}: {e}")
                last_error_reason = str(e)
                validation_failures.append(last_error_reason)
                previous_candidate = parsed if 'parsed' in locals() and parsed else {"raw": raw_content if 'raw_content' in locals() else ""}
                current_prompt = f"{prompt}\n\nREPAIR NOTICE: {last_error_reason}. Preserve the query and return a schema-valid plan."

        planner_auditor.record(PlannerAuditRecord(
            request_id=request_id,
            query_fingerprint=PlannerAuditorCls.fingerprint_query(query_raw),
            model_name=planner_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            retrieved_tools=[s.name for s in planner_specs],
            locked_tool=candidate_tool_name,
            initial_raw_response=initial_raw,
            validation_failures=validation_failures,
            repair_attempts=repair_budget_mgr.attempts_used,
            repair_history=[r.model_dump() for r in repair_budget_mgr.history],
            final_decision="failed",
            stage_latencies_ms={"planning_ms": round((time.time() - t_start) * 1000, 2)},
        ))
        logger.error(f"SEMANTIC_PLANNER_EXHAUSTED_RETRIES | Query: '{query_raw}' | Last Reason: {last_error_reason}")
        if last_error_reason in ["OLLAMA_UNAVAILABLE", "OLLAMA_MODEL_NOT_FOUND", "OLLAMA_TIMEOUT", "INTENT_PLANNER_UNAVAILABLE"]:
            return {
                "error_code": "INTENT_PLANNER_UNAVAILABLE",
                "error_message": f"Planning engine is offline or unreachable ({last_error_reason}).",
                "repair_history": [r.model_dump() for r in repair_budget_mgr.history],
            }
        return None
