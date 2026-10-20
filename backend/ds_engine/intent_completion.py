"""
DataPilot OS — Registry-Driven Intent Completion & Conversational Slot Filling Service.
Domain-agnostic inference engine that automatically completes partial analytical intents
using pure statistical and type-theoretic evidence from DatasetProfile and declarative
CompletionPolicy attached to capability ToolSpecs, or emits structured conversational slot-filling cards.

Core Invariants:
1. Zero domain hardcoding, zero phrase matching lists, zero regex routers, zero first-column arbitrary defaults.
2. 100% registry-driven: policies declared directly on ToolSpec in modules/capability_registry.py.
3. Every inferred binding records provenance: source='profile_inference', confidence, evidence.
4. Dataset-wide mode allowed ONLY when policy.allowed_dataset_wide_mode is True and all policy conditions are met
   (eda_data, detect_anomalies across eligible numeric features with explicit parameters, correlate_data in matrix mode).
5. Never dataset-wide for forecasting, supervised ML, causal analysis, or event-window analysis.
6. Target selection for supervised ML / forecasting is NEVER auto-inferred from profile candidates.
7. Algorithmic methods and thresholds (e.g. anomaly detection IQR/Z-score/sensitivity) are NEVER silently defaulted.
8. Zero tool-name branching: all capability behavior is driven through ToolSpec metadata and CompletionPolicy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from backend.ds_engine.dataset_profiler import DatasetProfile, RoleCandidate

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.85
AMBIGUITY_MARGIN = 0.15


class InferredBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: str
    column: Union[str, List[str], Any]
    source: Literal["profile_inference", "user_explicit", "dataset_wide_default", "policy_default"] = "profile_inference"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class SlotDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    label: str
    options: List[str] = Field(default_factory=list)
    slot_name: Optional[str] = None
    candidates: List[str] = Field(default_factory=list)
    slot_type: str = "select"  # select | multiselect
    required: bool = True
    default_value: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.slot_name:
            self.slot_name = self.name
        if not self.candidates and self.options:
            self.candidates = list(self.options)
        elif not self.options and self.candidates:
            self.options = list(self.candidates)


class SlotFillingCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slot_type: str  # target_selection | forecast_slots | pairwise_slots | grouped_slots | column_selection | method_selector
    card_type: Optional[str] = None
    title: str
    prompt: str
    slots: List[SlotDefinition] = Field(default_factory=list)
    default_query_template: str = ""
    candidate_columns: List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.card_type:
            if self.slot_type in ("target_selection", "target_selector"):
                self.card_type = "target_selector"
            elif self.slot_type in ("forecast_slots", "forecast_selector"):
                self.card_type = "forecast_selector"
            elif self.slot_type in ("pairwise_slots", "correlation_selector"):
                self.card_type = "correlation_selector"
            elif self.slot_type in ("grouped_slots", "dimension_selector"):
                self.card_type = "dimension_selector"
            elif self.slot_type in ("method_selector", "method_selection"):
                self.card_type = "method_selector"
            else:
                self.card_type = self.slot_type or "generic"
        if not self.suggested_queries and self.default_query_template and self.candidate_columns:
            queries = []
            safe_defaults = {
                "target_column": "",
                "metric": "",
                "dimension": "",
                "time_column": "",
                "col_x": "",
                "col_y": "",
                "method": "",
                "sensitivity": "",
                "group_by": "",
            }
            for col in self.candidate_columns[:3]:
                ctx = dict(safe_defaults, target_column=col, metric=col, dimension=col, time_column=col, col_x=col, col_y=col, group_by=col)
                try:
                    queries.append(self.default_query_template.format(**ctx))
                except (KeyError, ValueError, IndexError) as fmt_err:
                    logger.debug("Failed to interpolate suggested query template: %s", fmt_err)
            self.suggested_queries = queries


class IntentCompletionDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: Literal["execute", "clarify", "unsupported", "unsafe", "reject"]
    completed_tool: str
    completed_arguments: Dict[str, Any] = Field(default_factory=dict)
    inferred_bindings: List[InferredBinding] = Field(default_factory=list)
    is_dataset_wide: bool = False
    is_ambiguous: bool = False
    clarification_prompt: Optional[str] = None
    candidate_columns: List[str] = Field(default_factory=list)
    slot_filling: Optional[SlotFillingCard] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.action == "clarify":
            self.is_ambiguous = True


class IntentCompletionService:
    """
    Statistically grounded, 100% Registry-Driven Intent Completion & Conversational Slot-Filling Engine.
    Evaluates whether a partial or ungrounded query can be safely executed or needs clarification,
    driven entirely by ToolSpec.completion_policy in CapabilityRegistry.
    Contains zero hardcoded tool names, zero phrase lists, and zero regex routers.
    """

    @classmethod
    def _extract_candidates(
        cls,
        profile: DatasetProfile,
        role_source: Optional[str],
        limit: int = 5,
    ) -> List[str]:
        if not role_source:
            return []
        attr_val = getattr(profile, role_source, None)
        if attr_val is None:
            return []
        if isinstance(attr_val, dict):
            return list(attr_val.keys())[:limit]
        res: List[str] = []
        if isinstance(attr_val, (list, tuple, set)):
            for item in attr_val:
                if hasattr(item, "column"):
                    res.append(str(item.column))
                elif isinstance(item, str):
                    res.append(str(item))
                if len(res) >= limit:
                    break
        return res

    @classmethod
    def complete_intent(
        cls,
        tool_name: Any,
        arguments: Optional[Dict[str, Any]] = None,
        profile: Optional[DatasetProfile] = None,
        query_raw: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> IntentCompletionDecision:
        # Support passing a ToolCall directly as the first argument
        if hasattr(tool_name, "tool"):
            t_name = str(tool_name.tool or "").lower().strip()
            args = dict(tool_name.arguments or {})
            if isinstance(arguments, DatasetProfile):
                profile = arguments
            elif "profile" in kwargs:
                profile = kwargs["profile"]
            if isinstance(arguments, str) and not query_raw:
                query_raw = arguments
            elif "query_raw" in kwargs:
                query_raw = kwargs["query_raw"]
        elif isinstance(tool_name, dict) and ("tool" in tool_name or "name" in tool_name):
            t_name = str(tool_name.get("tool") or tool_name.get("name") or "").lower().strip()
            args = dict(tool_name.get("arguments", {}))
            if isinstance(arguments, DatasetProfile):
                profile = arguments
            elif "profile" in kwargs:
                profile = kwargs["profile"]
            if isinstance(arguments, str) and not query_raw:
                query_raw = arguments
            elif "query_raw" in kwargs:
                query_raw = kwargs["query_raw"]
        else:
            t_name = str(tool_name or "").lower().strip()
            args = dict(arguments or {})

        if profile is None and "profile" in kwargs:
            profile = kwargs["profile"]
        if not query_raw and "query_raw" in kwargs:
            query_raw = kwargs["query_raw"]

        if profile is None:
            # Without a dataset profile, cannot perform data-aware completion
            return IntentCompletionDecision(
                action="execute",
                completed_tool=t_name,
                completed_arguments=args,
            )

        # ── 1. REGISTRY METADATA LOOKUP (MANDATORY POLICY) ──────────────────
        from modules.capability_registry import capability_registry, CompletionPolicy, SlotPresentationPolicy
        spec = capability_registry.get_tool_spec(t_name)
        if spec is None:
            return IntentCompletionDecision(
                action="unsupported",
                completed_tool=t_name,
                error_code="UNSUPPORTED_OPERATION",
                error_message=f"Capability '{t_name}' is not registered in CapabilityRegistry.",
            )

        policy: CompletionPolicy = spec.completion_policy

        # ── 1b. ARGUMENT ALIAS CANONICALIZATION ────────────────────────────
        if policy.argument_aliases:
            for canonical_slot, aliases in policy.argument_aliases.items():
                if canonical_slot not in args or args[canonical_slot] is None:
                    for alias in aliases:
                        if alias != canonical_slot and alias in args and args[alias] is not None:
                            val = args[alias]
                            pres = policy.slot_presentations.get(canonical_slot)
                            if pres and pres.slot_type == "multiselect":
                                if isinstance(val, (list, tuple, set)):
                                    args[canonical_slot] = list(val)
                                else:
                                    args[canonical_slot] = [val]
                            else:
                                args[canonical_slot] = val
                            break

        # ── 2. CLARIFICATION TOOL: METADATA-DRIVEN DISPATCH ─────────────────
        if policy.is_clarification_tool or policy.clarification_category == "clarification":
            top_cols = [c.column for c in profile.ranked_numeric_measures[:5]] or list(profile.columns.keys())[:5]
            clarif_text = args.get("clarification") or "Query is too broad. Please specify target column or analytical task."
            return IntentCompletionDecision(
                action="clarify",
                completed_tool=spec.name,
                clarification_prompt=clarif_text,
                candidate_columns=top_cols,
                slot_filling=SlotFillingCard(
                    slot_type="target_selection",
                    title="Specify Analytical Task",
                    prompt=clarif_text,
                    candidate_columns=top_cols,
                    suggested_queries=[
                        "Overview and summary statistics of dataset",
                        f"Detect anomalies in {top_cols[0]}" if top_cols else "Detect anomalies in numeric columns",
                        f"Correlation between {top_cols[0]} and {top_cols[1]}" if len(top_cols) >= 2 else "Correlation matrix",
                    ]
                )
            )

        # ── 3. ALGORITHMIC EXPLICITNESS (NO SILENT METHOD/THRESHOLD DEFAULTS)
        method_slot = policy.required_method_slot or "method"
        if policy.requires_explicit_method and not args.get(method_slot):
            allowed_methods = list(policy.allowed_methods)
            card = SlotFillingCard(
                slot_type="method_selector",
                title=policy.slot_filling_title or f"Select Method for {spec.name}",
                prompt=policy.slot_filling_prompt or f"Please select a method to run {spec.name}:",
                slots=[
                    SlotDefinition(
                        name=method_slot,
                        label="Method",
                        options=allowed_methods,
                        default_value=None,  # NEVER silent default!
                    )
                ],
                default_query_template=policy.default_query_template or f"Run {spec.name} with {{{method_slot}}}",
                suggested_queries=[f"Run using {m}" for m in allowed_methods],
            )
            return IntentCompletionDecision(
                action="clarify",
                completed_tool=spec.name,
                completed_arguments=args,
                clarification_prompt=card.prompt,
                slot_filling=card,
            )

        sens_slot = policy.required_sensitivity_slot or "sensitivity"
        if policy.requires_explicit_sensitivity and not args.get(sens_slot):
            allowed_sens = [str(s) for s in policy.allowed_sensitivities]
            card = SlotFillingCard(
                slot_type="method_selector",
                title="Select Sensitivity Threshold",
                prompt="Please select a sensitivity threshold:",
                slots=[
                    SlotDefinition(
                        name=sens_slot,
                        label="Sensitivity",
                        options=allowed_sens,
                        default_value=None,  # NEVER silent default!
                    )
                ],
                suggested_queries=[f"Sensitivity {s}" for s in allowed_sens],
            )
            return IntentCompletionDecision(
                action="clarify",
                completed_tool=spec.name,
                completed_arguments=args,
                clarification_prompt=card.prompt,
                slot_filling=card,
            )

        # ── 4. DATASET-WIDE EVALUATION ──────────────────────────────────────
        target_arg = policy.dataset_wide_target_arg or "columns"
        existing_cols = args.get(target_arg)
        if existing_cols is None and policy.argument_aliases:
            for alias in policy.argument_aliases.get(target_arg, []):
                if alias in args and args[alias] is not None:
                    c_val = args[alias]
                    existing_cols = [c_val] if isinstance(c_val, str) else list(c_val)
                    break
        elif isinstance(existing_cols, str):
            existing_cols = [existing_cols]

        min_cols_req = policy.minimum_items_by_slot.get(target_arg, 1)
        has_insufficient_cols = (
            existing_cols is None or len(existing_cols) == 0
            or len(existing_cols) < min_cols_req
        )

        inferred_bindings: List[InferredBinding] = []

        if has_insufficient_cols and policy.allowed_dataset_wide_mode:
            # Check conditional requirement (e.g. analysis_mode == "matrix" for correlation)
            mode_ok = True
            if policy.dataset_wide_required_mode_arg:
                req_key, req_val = policy.dataset_wide_required_mode_arg
                mode_ok = (args.get(req_key) == req_val)

            if mode_ok:
                if policy.dataset_wide_role:
                    role_cands = getattr(profile, policy.dataset_wide_role, [])
                    eligible = [
                        c.column for c in role_cands
                        if c.confidence >= 0.70 and c.cardinality > 2
                    ]
                    if not eligible and policy.allow_role_candidate_fallback and policy.fallback_role_name:
                        fallback_cands = getattr(profile, policy.fallback_role_name, [])
                        if isinstance(fallback_cands, dict):
                            eligible = list(fallback_cands.keys())
                        elif isinstance(fallback_cands, (list, tuple, set)):
                            eligible = [
                                c.column if hasattr(c, "column") else c
                                for c in fallback_cands
                            ]

                    if len(eligible) >= policy.dataset_wide_min_columns:
                        args[target_arg] = eligible

                        binding = InferredBinding(
                            slot=target_arg,
                            column=eligible,
                            source="dataset_wide_default",
                            confidence=0.90,
                            evidence=[f"selected all {len(eligible)} eligible features for dataset-wide operation"],
                        )
                        inferred_bindings.append(binding)

                        # Apply declarative dataset-wide defaults if specified
                        if policy.dataset_wide_default_args:
                            for def_k, def_v in policy.dataset_wide_default_args.items():
                                if def_k not in args or args[def_k] is None:
                                    args[def_k] = def_v
                                    inferred_bindings.append(InferredBinding(
                                        slot=def_k,
                                        column=def_v,
                                        source="policy_default",
                                        confidence=1.0,
                                        evidence=[f"applied declared dataset-wide default '{def_k}={def_v}' from ToolSpec('{spec.name}') CompletionPolicy"],
                                    ))

                        return IntentCompletionDecision(
                            action="execute",
                            completed_tool=spec.name,
                            completed_arguments=args,
                            inferred_bindings=inferred_bindings,
                            is_dataset_wide=True,
                        )
                    else:
                        return IntentCompletionDecision(
                            action="unsafe",
                            completed_tool=spec.name,
                            error_code="INSUFFICIENT_COLUMNS",
                            error_message=f"Dataset contains fewer than {policy.dataset_wide_min_columns} eligible columns for {spec.name}.",
                        )
                else:
                    # Dataset-wide tool that requires no columns (e.g. eda_data)
                    return IntentCompletionDecision(
                        action="execute",
                        completed_tool=spec.name,
                        completed_arguments=args,
                        is_dataset_wide=True,
                    )

        # ── 5. TEMPORAL PRECONDITION GUARD ──────────────────────────────────
        if policy.requires_temporal_column:
            if not profile.ranked_temporal_columns and not profile.temporal_column_candidates:
                return IntentCompletionDecision(
                    action="reject",
                    completed_tool=spec.name,
                    error_code="NO_TEMPORAL_COLUMN",
                    error_message=f"{spec.name} requires a temporal column (date/time), but none was found in the dataset.",
                )

        # ── 6. SLOT RESOLUTION VIA ROLE CANDIDATES ──────────────────────────
        missing_slots: List[str] = []
        for slot in policy.required_slots:
            slot_val = args.get(slot)
            if not slot_val and policy.argument_aliases:
                for alias in policy.argument_aliases.get(slot, []):
                    alias_val = args.get(alias)
                    if alias_val is not None:
                        slot_val = alias_val
                        break

            req_items = policy.minimum_items_by_slot.get(slot, 1)
            is_slot_missing = (
                not slot_val
                or (isinstance(slot_val, list) and len(slot_val) < req_items)
            )

            if is_slot_missing:
                inferred = False
                if slot in policy.inferable_slots:
                    role_name = policy.inferable_slots[slot]
                    top_cand, _ = profile.get_top_candidate(
                        role_name,
                        threshold=policy.confidence_threshold,
                        ambiguity_margin=policy.ambiguity_margin,
                    )
                    if top_cand:
                        inferred = True
                        pres = policy.slot_presentations.get(slot)
                        if pres and pres.slot_type == "multiselect":
                            args[slot] = [top_cand.column]
                        else:
                            args[slot] = top_cand.column
                        inferred_bindings.append(InferredBinding(
                            slot=slot,
                            column=top_cand.column,
                            source="profile_inference",
                            confidence=top_cand.confidence,
                            evidence=top_cand.evidence,
                        ))
                if not inferred:
                    missing_slots.append(slot)

        # ── 7. CONVERSATIONAL SLOT FILLING WHEN SLOTS REMAIN MISSING ────────
        if missing_slots:
            slot_defs: List[SlotDefinition] = []
            all_candidate_cols: List[str] = []

            for ms in missing_slots:
                pres = policy.slot_presentations.get(ms)
                if pres:
                    label = pres.label
                    limit = pres.max_candidates or 5
                    opts = cls._extract_candidates(profile, pres.role_source, limit=limit)
                    if not opts and pres.fallback_roles:
                        fb_limit = max(1, limit // len(pres.fallback_roles))
                        for fb_role in pres.fallback_roles:
                            fb_opts = cls._extract_candidates(profile, fb_role, limit=fb_limit)
                            for o in fb_opts:
                                if o not in opts:
                                    opts.append(o)
                            if len(opts) >= limit:
                                break
                    opts = opts[:limit]

                    slot_name = pres.presentation_slot_name or ms
                    if pres.is_pairwise and pres.pairwise_slots:
                        for p_name, p_label in pres.pairwise_slots:
                            slot_defs.append(SlotDefinition(
                                name=p_name,
                                label=p_label,
                                options=opts[:4] if p_name == pres.pairwise_slots[0][0] else opts[1:5],
                                slot_type=pres.slot_type,
                                required=True,
                                default_value=pres.default_value,
                            ))
                    else:
                        slot_defs.append(SlotDefinition(
                            name=slot_name,
                            label=label,
                            options=opts,
                            slot_type=pres.slot_type,
                            required=True,
                            default_value=pres.default_value,
                        ))
                    all_candidate_cols.extend(opts)
                else:
                    opts = list(profile.columns.keys())[:5]
                    slot_defs.append(SlotDefinition(
                        name=ms,
                        label=ms.replace("_", " ").title(),
                        options=opts,
                        required=True,
                        default_value=None,
                    ))
                    all_candidate_cols.extend(opts)

            card = SlotFillingCard(
                slot_type=policy.slot_filling_type or "column_selection",
                title=policy.slot_filling_title or f"Specify Parameters for {spec.name}",
                prompt=policy.slot_filling_prompt or f"Please specify required inputs for {spec.name}:",
                slots=slot_defs,
                candidate_columns=all_candidate_cols,
                default_query_template=policy.default_query_template,
            )
            return IntentCompletionDecision(
                action="clarify",
                completed_tool=spec.name,
                completed_arguments=args,
                clarification_prompt=card.prompt,
                candidate_columns=all_candidate_cols,
                slot_filling=card,
            )

        # ── 8. SAFE POST-COMPLETION NORMALIZATION (DECLARATIVE) ──────────────
        if policy.post_completion_defaults:
            for def_k, def_v in policy.post_completion_defaults.items():
                if not args.get(def_k):
                    args[def_k] = def_v
                    inferred_bindings.append(InferredBinding(
                        slot=def_k,
                        column=def_v,
                        source="policy_default",
                        confidence=1.0,
                        evidence=[f"applied declared post-completion default '{def_k}={def_v}' from ToolSpec('{spec.name}') CompletionPolicy"],
                    ))

        freq_slot = getattr(policy, "frequency_slot_name", None) or "frequency"
        if (policy.allow_inferred_frequency or policy.infers_frequency_from_temporal) and not args.get(freq_slot):
            time_slot = policy.temporal_slot_name or "time_column"
            time_col = args.get(time_slot)
            top_temp = profile.columns.get(time_col) if time_col else None
            inferred_freq = getattr(top_temp, "inferred_frequency", None) if top_temp else None
            freq_confidence = getattr(top_temp, "datetime_parse_ratio", 0.0) if top_temp else 0.0

            if inferred_freq and freq_confidence >= policy.frequency_confidence_threshold:
                args[freq_slot] = inferred_freq
                inferred_bindings.append(InferredBinding(
                    slot=freq_slot,
                    column=str(inferred_freq),
                    source="profile_inference",
                    confidence=float(freq_confidence),
                    evidence=[f"inferred calendar frequency '{inferred_freq}' from temporal column '{time_col}' with confidence {freq_confidence:.2f}"],
                ))
            else:
                pres = policy.slot_presentations.get(freq_slot)
                freq_label = (pres.label if pres and pres.label else "Calendar Frequency")
                freq_options = list(pres.options) if pres and pres.options else []
                raw_prompt = (pres.prompt_template if pres and pres.prompt_template else "Could not determine calendar frequency for temporal column '{time_col}'. Please specify a frequency:")
                freq_prompt = raw_prompt.format(time_col=time_col or time_slot)
                freq_title = (pres.title if pres and pres.title else "Specify Calendar Frequency")
                suggested_queries = (list(pres.suggested_queries) if pres and pres.suggested_queries else [f"Forecast with frequency {opt}" for opt in freq_options[:3]])

                card = SlotFillingCard(
                    slot_type="frequency_selector",
                    title=freq_title,
                    prompt=freq_prompt,
                    slots=[
                        SlotDefinition(
                            name=freq_slot,
                            label=freq_label,
                            options=freq_options,
                            default_value=None,
                        )
                    ],
                    suggested_queries=suggested_queries,
                )
                return IntentCompletionDecision(
                    action="clarify",
                    completed_tool=spec.name,
                    completed_arguments=args,
                    clarification_prompt=card.prompt,
                    candidate_columns=[time_col] if time_col else [],
                    slot_filling=card,
                )

        if policy.multi_column_threshold is not None:
            cols = args.get("columns") or []
            if len(cols) > policy.multi_column_threshold:
                if policy.multi_column_first_item_slots:
                    for slot_name in policy.multi_column_first_item_slots:
                        val = args.get(slot_name)
                        if isinstance(val, list) and len(val) > 1:
                            args[slot_name] = [val[0]]
                            inferred_bindings.append(InferredBinding(
                                slot=slot_name,
                                column=val[0],
                                source="policy_default",
                                confidence=1.0,
                                evidence=[f"constrained '{slot_name}' to first item for multi-column operation (> {policy.multi_column_threshold} columns) per ToolSpec('{spec.name}') policy"],
                            ))
                if policy.multi_column_overrides:
                    for ov_k, ov_v in policy.multi_column_overrides.items():
                        args[ov_k] = ov_v
                        inferred_bindings.append(InferredBinding(
                            slot=ov_k,
                            column=ov_v,
                            source="policy_default",
                            confidence=1.0,
                            evidence=[f"applied multi-column override '{ov_k}={ov_v}' (> {policy.multi_column_threshold} columns) from ToolSpec('{spec.name}') CompletionPolicy"],
                        ))

        return IntentCompletionDecision(
            action="execute",
            completed_tool=spec.name,
            completed_arguments=args,
            inferred_bindings=inferred_bindings,
        )


intent_completion_service = IntentCompletionService()
