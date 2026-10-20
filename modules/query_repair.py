import json
import logging
import os
import pandas as pd
from typing import Dict, Any, Optional
from modules.models.semantic_plan import SemanticPlan
from modules.models.execution_plan import ExecutionPlan
from modules.models.validation_result import ValidationResult
from modules.semantic_planner import SemanticPlanner
from modules.plan_normalizer import PlanNormalizer
from modules.plan_validator import PlanValidator

logger = logging.getLogger(__name__)

class QueryRepair:
    """
    Manages the Goal-Preserving Ollama Self-Repair Loop (MAX_REPAIR_ATTEMPTS = 2).
    Sends validation feedback to Ollama instructing it to fix invalid parameters
    WITHOUT changing the user's analytical goal.
    """
    MAX_REPAIR_ATTEMPTS = 2

    @classmethod
    def repair_loop(
        cls,
        query_raw: str,
        initial_plan: SemanticPlan,
        validation_res: ValidationResult,
        schema: Dict[str, Any],
        df: Optional[pd.DataFrame] = None
    ) -> Optional[ExecutionPlan]:
        
        current_val_res = validation_res
        current_plan = initial_plan

        for attempt in range(1, cls.MAX_REPAIR_ATTEMPTS + 1):
            logger.info(f"PLAN_REPAIRED_ATTEMPT | Initiating Repair Loop Attempt #{attempt}/{cls.MAX_REPAIR_ATTEMPTS}")
            
            repaired_semantic_plan = cls._call_ollama_repair(
                query_raw=query_raw,
                invalid_plan=current_plan,
                val_res=current_val_res,
                schema=schema
            )

            if repaired_semantic_plan:
                exec_plan = PlanNormalizer.normalize(repaired_semantic_plan)
                val_res = PlanValidator.validate(exec_plan, df=df)
                
                if val_res.is_valid:
                    logger.info(f"PLAN_REPAIRED_SUCCESS | Self-repair succeeded on Attempt #{attempt}!")
                    return exec_plan
                else:
                    current_val_res = val_res
                    current_plan = repaired_semantic_plan

        logger.warning("PLAN_REPAIR_EXHAUSTED | Max repair attempts reached without valid plan.")
        return None

    @classmethod
    def _call_ollama_repair(
        cls,
        query_raw: str,
        invalid_plan: SemanticPlan,
        val_res: ValidationResult,
        schema: Dict[str, Any],
        timeout_sec: float = 2.0
    ) -> Optional[SemanticPlan]:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

        prompt = (
            "You are a Data Science Query Repair Assistant.\n"
            f"ORIGINAL QUERY: \"{query_raw}\"\n"
            f"YOUR INVALID PLAN: {invalid_plan.model_dump_json()}\n"
            f"VALIDATION ERROR: {val_res.error_msg}\n"
            f"AVAILABLE DATASET SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
            "TASK: Repair ONLY the invalid parts of your JSON plan to match available columns and types.\n"
            "CRITICAL INSTRUCTION: Do NOT change the user's analytical goal.\n"
            "Output strictly valid JSON matching the SemanticPlan schema:"
        )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False
        }

        try:
            import urllib.request
            req_data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(ollama_url, data=req_data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    content = data.get("message", {}).get("content", "")
                    return SemanticPlanner.parse_and_validate_json(content)
        except Exception as err:
            logger.warning(f"OLLAMA_REPAIR_NOTICE | Ollama repair call failed: {err}")

        return None
