import logging
import pandas as pd
from typing import Optional

from modules.schema_analyzer import SchemaAnalyzer
from modules.semantic_planner import SemanticPlanner
from modules.plan_normalizer import PlanNormalizer
from modules.plan_validator import PlanValidator
from modules.query_repair import QueryRepair
from modules.result_validator import ResultValidator
from modules.emergency_fallback import EmergencyFallback
from modules.capability_registry import capability_registry
from modules.models.execution_result import ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)

class QueryOrchestrator:
    """
    Central Controller for DataPilot OS Query Execution Lifecycle:
    1. Schema Analysis & Caching
    2. Ollama Semantic Planning (Meaning Extraction)
    3. Plan Normalization (SemanticPlan -> ExecutionPlan)
    4. Plan Validation & Goal-Preserving Repair Loop (Max 2 attempts)
    5. Jinja2 Code Rendering & Subprocess Sandbox Execution
    6. Structural Result Validation
    7. Emergency Recovery Net (Safety Net ONLY)
    """

    @classmethod
    def process_query(cls, query_raw: str, df: Optional[pd.DataFrame] = None, **kwargs) -> ExecutionResult:
        query_raw = query_raw.strip()
        logger.info(f"ORCHESTRATOR_START | Query: '{query_raw}'")

        # 1. Cached Schema Analysis
        schema = SchemaAnalyzer.analyze(df)

        # 2. Ollama Semantic Planner
        semantic_plan = SemanticPlanner.plan(query_raw, schema)

        exec_plan = None
        if semantic_plan:
            # 3. Plan Normalization
            exec_plan = PlanNormalizer.normalize(semantic_plan)
            
            # 4. Plan Validation
            val_res = PlanValidator.validate(exec_plan, df=df)

            if not val_res.is_valid:
                logger.warning(f"PLAN_INVALID | Validation failed. Triggering Self-Repair Loop...")
                # 5. Ollama Self-Repair Loop (Max 2 Attempts)
                exec_plan = QueryRepair.repair_loop(
                    query_raw=query_raw,
                    initial_plan=semantic_plan,
                    validation_res=val_res,
                    schema=schema,
                    df=df
                )
                if exec_plan:
                    logger.info("PLAN_REPAIRED | Successfully recovered plan via Self-Repair Loop.")

        # 6. Fallback if Ollama unavailable or Repair failed
        if not exec_plan:
            logger.warning("EMERGENCY_RECOVERY | Planning failed or Ollama offline. Routing to emergency fallback...")
            return EmergencyFallback.execute_fallback(query_raw, df=df, reason="Semantic planning or self-repair failed")

        # 7. Execute Validated Plan Steps
        results_list = []
        for step in exec_plan.steps:
            cap_id = step.capability
            step_params = step.parameters

            try:
                if cap_id in capability_registry._registry:
                    res_dict = capability_registry.execute_capability(cap_id, df=df, query=query_raw, **step_params)
                else:
                    res_dict = capability_registry.route_and_dispatch(query_raw, df=df)
                
                # 8. Structural Result Validation
                validated_res = ResultValidator.validate_result(res_dict)
                results_list.append(validated_res)

            except Exception as err:
                logger.error(f"EXECUTION_ERROR | Step execution failed for '{cap_id}': {err}", exc_info=True)
                return EmergencyFallback.execute_fallback(query_raw, df=df, reason=f"Subprocess sandbox execution error in step '{cap_id}'")

        if len(results_list) == 1:
            return results_list[0]
        
        # Combine compound query step outputs
        combined_response = "\n\n".join([f"### Step {idx+1}:\n{r.response}" for idx, r in enumerate(results_list)])
        combined_code = "\n\n".join([f"# Step {idx+1}\n{r.code}" for idx, r in enumerate(results_list) if r.code])
        plots = [r.plot_image for r in results_list if r.plot_image]

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response=combined_response,
            code=combined_code,
            plot_image=plots[0] if plots else None,
            artifacts={"plots": plots}
        )
