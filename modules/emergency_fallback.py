import logging
import pandas as pd
from typing import Optional
from modules.models.execution_result import ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)

class EmergencyFallback:
    """
    Emergency Recovery Net invoked ONLY when Ollama is offline or repairs fail:
    1. Safe deterministic recovery via rule-based capability registry dispatch.
    2. Prompt user for clarification if ambiguous.
    3. Clean rejection for unsupported requests.
    """

    @classmethod
    def execute_fallback(
        cls,
        query_raw: str,
        df: Optional[pd.DataFrame] = None,
        reason: str = "Ollama offline or repair loop exhausted"
    ) -> ExecutionResult:
        logger.warning(f"EMERGENCY_RECOVERY | Engaged emergency fallback for query '{query_raw}'. Reason: {reason}")

        from modules.capability_registry import capability_registry
        try:
            res = capability_registry.route_and_dispatch(query_raw, df=df)
            if res.get("status") == "success" or res.get("response"):
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    response=res.get("response") or res.get("answer") or "Executed via fallback.",
                    code=res.get("code"),
                    plot_image=res.get("plot_image"),
                    artifacts=res.get("artifacts", {})
                )
        except Exception as err:
            logger.error(f"EMERGENCY_FALLBACK_FAILED | Fallback execution error: {err}")

        # Clarify or reject if deterministic router is also unable to proceed
        reject_msg = f"⚠️ Could not interpret query: '{query_raw}'. Please clarify which dataset column or capability to use."
        return ExecutionResult(
            status=ExecutionStatus.UNSUPPORTED,
            response=reject_msg,
            code=None,
            plot_image=None
        )
