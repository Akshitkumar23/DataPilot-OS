"""
DataPilot OS — Execution Plan Compiler.
Compiles SemanticIntent into an ordered, multi-step ExecutionPlan.
"""

import logging
import pandas as pd
from typing import Optional, List, Dict, Any

from backend.ds_engine.intent_schema import SemanticIntent, IntentType
from backend.ds_engine.intent_resolver import intent_resolver, ResolvedIntent
from backend.ds_engine.code_generator import code_generator
from modules.models.execution_plan import ExecutionPlan, ExecutionStep

logger = logging.getLogger(__name__)

class ExecutionPlanCompiler:
    @classmethod
    def compile_plan(
        cls,
        semantic_intent: SemanticIntent,
        df: Optional[pd.DataFrame] = None,
        request_id: str = "req-000"
    ) -> ExecutionPlan:
        """
        Compiles a SemanticIntent into an ordered multi-step ExecutionPlan.
        Prerequisite steps (e.g. FILTER_QUERY, DATA_CLEAN) come first,
        followed by the Primary Intent step.
        """
        prereqs = list(semantic_intent.prerequisite_operations or [])
        primary = semantic_intent.primary_intent or semantic_intent.intent_type

        # Ensure that if filters are present on a non-FILTER_QUERY primary intent, FILTER_QUERY is executed as prerequisite
        primary_val = primary.value if hasattr(primary, "value") else str(primary)
        if semantic_intent.filters and primary_val != "FILTER_QUERY":
            has_filter_prereq = any((p.value if hasattr(p, "value") else str(p)) == "FILTER_QUERY" for p in prereqs)
            if not has_filter_prereq:
                prereqs.insert(0, IntentType.FILTER_QUERY)

        steps: List[ExecutionStep] = []
        step_idx = 1

        # 1. Compile Prerequisite Steps
        for pre_intent in prereqs:
            # Avoid duplicating primary intent as a prerequisite
            if pre_intent == primary:
                continue

            # Create transient SemanticIntent for prerequisite step
            pre_semantic = SemanticIntent(
                intent_type=pre_intent,
                primary_intent=pre_intent,
                time_column=semantic_intent.time_column,
                target_column=semantic_intent.target_column,
                feature_columns=semantic_intent.feature_columns,
                dimensions=semantic_intent.dimensions,
                metrics=semantic_intent.metrics,
                aggregation=semantic_intent.aggregation,
                filters=semantic_intent.filters,
                order=semantic_intent.order,
                limit=semantic_intent.limit,
                raw_query=semantic_intent.raw_query
            )

            resolved_pre = intent_resolver.resolve(pre_semantic, df=df)
            code_pre = code_generator.generate_from_resolved_intent(resolved_pre)

            steps.append(ExecutionStep(
                step_index=step_idx,
                intent_type=pre_intent,
                description=f"Prerequisite Step {step_idx}: {pre_intent.value}",
                resolved_intent=resolved_pre,
                code_version=code_pre,
                input_shape=df.shape if df is not None else None,
                is_prerequisite=True
            ))
            step_idx += 1

        # 2. Compile Primary Intent Step
        resolved_primary = intent_resolver.resolve(semantic_intent, df=df)
        code_primary = code_generator.generate_from_resolved_intent(resolved_primary)

        steps.append(ExecutionStep(
            step_index=step_idx,
            intent_type=primary,
            description=f"Primary Step {step_idx}: {primary.value}",
            resolved_intent=resolved_primary,
            code_version=code_primary,
            input_shape=df.shape if df is not None else None,
            is_prerequisite=False
        ))

        plan = ExecutionPlan(
            request_id=request_id,
            primary_intent=primary,
            prerequisites=prereqs,
            steps=steps,
            is_compiled=True,
            execution_status="COMPILED"
        )
        return plan
