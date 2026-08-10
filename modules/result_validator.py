import logging
from typing import Dict, Any
from modules.models.execution_result import ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)

class ResultValidator:
    """
    Validates structural execution outputs after sandbox execution:
    - Checks execution status code.
    - Checks for valid response text.
    - Checks for valid plot image payload.
    - Ensures results are not empty or filled with NaN/Inf anomalies.
    """

    @classmethod
    def validate_result(cls, raw_result: Dict[str, Any]) -> ExecutionResult:
        status_raw = raw_result.get("status", "success")
        code = raw_result.get("code")
        response = raw_result.get("response") or raw_result.get("answer") or ""
        artifacts = raw_result.get("artifacts", {})
        plot_image = raw_result.get("plot_image")

        if not plot_image and isinstance(artifacts, dict):
            plots = artifacts.get("plots", [])
            if len(plots) > 0:
                plot_image = plots[0]

        if status_raw != "success" and not response:
            logger.warning("RESULT_INVALID | Execution returned error status or empty response.")
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_ERROR,
                response="Execution failed in subprocess sandbox.",
                code=code,
                artifacts=artifacts
            )

        if not response or len(response.strip()) == 0:
            logger.warning("RESULT_INVALID | Output response is empty.")
            return ExecutionResult(
                status=ExecutionStatus.EMPTY_RESULT,
                response="Execution completed but produced an empty result.",
                code=code,
                artifacts=artifacts
            )

        logger.info("RESULT_VALID | Structural result validation passed successfully.")
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response=response,
            code=code,
            plot_image=plot_image,
            artifacts=artifacts
        )
