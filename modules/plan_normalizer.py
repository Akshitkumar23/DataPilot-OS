import logging
from typing import Dict, Any, List
from modules.models.semantic_plan import SemanticPlan
from modules.models.execution_plan import ExecutionPlan, ExecutionStep

logger = logging.getLogger(__name__)

class PlanNormalizer:
    """
    Decouples Semantic Intent from Execution Capabilities.
    Normalizes abstract SemanticPlan into concrete ExecutionPlan steps.
    """

    OPERATION_MAPPING = {
        "distribution": "sig_hist",
        "histogram": "sig_hist",
        "scatter": "sig_scatter",
        "relationship": "sig_scatter",
        "top_n": "sig_top",
        "ranking": "sig_top",
        "forecast": "sig_forecast",
        "time_series": "sig_forecast",
        "eda": "sig_eda_dashboard",
        "overview": "sig_eda_dashboard",
        "heatmap": "sig_heatmap",
        "correlation": "sig_heatmap",
        "clean": "sig_clean",
        "impute": "sig_clean",
        "ml_train": "sig_ml",
        "model": "sig_ml",
        "sql": "sig_sql",
        "query": "sig_sql",
        "describe": "sig_describe",
        "profile": "sig_describe",
        "pca": "sig_pca",
        "cluster": "sig_cluster",
        "group_compare": "sig_top"
    }

    @classmethod
    def normalize(cls, semantic_plan: SemanticPlan) -> ExecutionPlan:
        steps: List[ExecutionStep] = []
        for idx, op in enumerate(semantic_plan.operations, 1):
            op_name = op.operation.lower().strip()
            capability = cls.OPERATION_MAPPING.get(op_name, "sig_eda_dashboard")

            params: Dict[str, Any] = dict(op.parameters)
            if op.target:
                params["target"] = op.target
                params["column"] = op.target
                params["c_num"] = op.target
            if op.group_by and len(op.group_by) > 0:
                params["group_by"] = op.group_by[0]
                params["c_cat"] = op.group_by[0]
                params["group_by_list"] = op.group_by

            # Propagate top-level semantic plan filters
            if semantic_plan.filters and len(semantic_plan.filters) > 0:
                params["filters"] = [f.model_dump() if hasattr(f, "model_dump") else dict(f) for f in semantic_plan.filters]

            # Map explicit operation dependencies or fall back to default
            if hasattr(op, "depends_on") and op.depends_on:
                step_deps = [f"step_{d}" for d in op.depends_on if isinstance(d, int)]
            else:
                step_deps = [f"step_{idx-1}"] if idx > 1 else []

            step = ExecutionStep(
                id=f"step_{idx}",
                capability=capability,
                parameters=params,
                depends_on=step_deps
            )
            steps.append(step)


        if not steps:
            steps.append(ExecutionStep(id="step_1", capability="sig_eda_dashboard", parameters={}))

        logger.info(f"PLAN_NORMALIZED | Converted {len(semantic_plan.operations)} operations into {len(steps)} execution step(s).")
        return ExecutionPlan(steps=steps)
