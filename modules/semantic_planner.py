import json
import logging
import os
from typing import Dict, Any, Optional
from modules.models.semantic_plan import SemanticPlan

logger = logging.getLogger(__name__)

class SemanticPlanner:
    """
    Calls Ollama LLM as the primary intelligence brain to extract high-level
    analytical meaning (SemanticPlan) without binding to execution capabilities.
    """

    @classmethod
    def plan(cls, query_raw: str, schema: Dict[str, Any], timeout_sec: float = 2.0) -> Optional[SemanticPlan]:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        schema_json = json.dumps(schema, indent=2)

        prompt = (
            "You are a Data Science Query Semantic Planner. Analyze the user query against the dataset schema and return an abstract analytical SemanticPlan.\n"
            "Do NOT write code or use execution capability names (like sig_hist). Output strictly valid JSON matching this schema:\n"
            "{\n"
            '  "goal": "description of analytical goal",\n'
            '  "operations": [{"operation": "distribution|scatter|top_n|forecast|group_compare|clean|ml_train|sql|describe|pca|cluster", "target": "ColName", "aggregation": "mean|sum|count", "group_by": ["ColName"], "parameters": {"n": 10, "horizon": 30}}],\n'
            '  "filters": [],\n'
            '  "group_by": [],\n'
            '  "visualization": {"requested": true, "type": "auto"}\n'
            "}\n\n"
            f"DATASET SCHEMA:\n{schema_json}\n\n"
            f'USER QUERY: "{query_raw}"\nJSON OUTPUT:'
        )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False
        }

        try:
            logger.info(f"OLLAMA_SEMANTIC_PLANNER_ATTEMPT | Query: '{query_raw}'")
            import urllib.request
            req_data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(ollama_url, data=req_data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    content = data.get("message", {}).get("content", "")
                    return cls.parse_and_validate_json(content)
        except Exception as err:
            logger.warning(f"OLLAMA_UNAVAILABLE | Semantic planner notice: {err}")
        
        return None

    @classmethod
    def parse_and_validate_json(cls, content: str) -> Optional[SemanticPlan]:
        try:
            parsed = json.loads(content)
            plan = SemanticPlan(**parsed)
            logger.info(f"OLLAMA_SEMANTIC_PLAN_SUCCESS | Goal: '{plan.goal}', Operations: {len(plan.operations)}")
            return plan
        except Exception as err:
            logger.warning(f"PLAN_INVALID | Failed to parse SemanticPlan JSON: {err}")
            return None
