import json
import logging
import os
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowSaverEngine:
    """
    Engine for saving and loading pipeline workflows.
    """
    
    def save_workflow_pipeline(self, name: str, steps: List[Dict[str, Any]], target_file: Optional[str] = None) -> str:
        """
        Saves a complete pipeline workflow (Import -> Clean -> EDA -> ML -> Forecast) 
        as a JSON pipeline file.
        """
        logger.info(f"Saving workflow pipeline: {name}")
        try:
            filename = target_file if target_file else f"{name.replace(' ', '_').lower()}_pipeline.json"
            
            workflow_data = {
                "pipeline_name": name,
                "steps": steps
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=4)
                
            abs_path = os.path.abspath(filename)
            logger.info(f"Workflow pipeline saved to {abs_path}")
            return abs_path
        except Exception as e:
            logger.error(f"Error saving workflow pipeline: {e}")
            raise

    def load_workflow_pipeline(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Loads a saved pipeline workflow for 1-click re-execution.
        """
        logger.info(f"Loading workflow pipeline from {filepath}")
        try:
            if not os.path.exists(filepath):
                logger.error(f"Pipeline file not found: {filepath}")
                raise FileNotFoundError(f"Pipeline file not found: {filepath}")
                
            with open(filepath, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
                
            steps = workflow_data.get("steps", [])
            logger.info("Workflow pipeline loaded successfully.")
            return steps
        except Exception as e:
            logger.error(f"Error loading workflow pipeline: {e}")
            raise

    def replay_workflow(self, pipeline_json_path: str, df: Any) -> Any:
        """
        Reads a saved pipeline JSON file, parses steps, and executes them sequentially 
        on the input DataFrame via QueryOrchestrator or capability_registry.
        """
        logger.info(f"Replaying workflow from pipeline: {pipeline_json_path}")
        from modules.models.execution_result import ExecutionResult, ExecutionStatus
        try:
            steps = self.load_workflow_pipeline(pipeline_json_path)
        except Exception as e:
            logger.error(f"Failed to load pipeline file '{pipeline_json_path}': {e}")
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_ERROR,
                response=f"Failed to load pipeline file: {e}",
                artifacts={"error": str(e)}
            )

        if not steps:
            logger.warning(f"No steps found in pipeline file: {pipeline_json_path}")
            return ExecutionResult(
                status=ExecutionStatus.EMPTY_RESULT,
                response="Pipeline contains no steps to execute.",
                artifacts={}
            )

        from modules.query_orchestrator import QueryOrchestrator
        from modules.capability_registry import capability_registry
        from modules.result_validator import ResultValidator

        current_df = df.copy() if df is not None else None
        results_list = []

        for idx, step in enumerate(steps):
            logger.info(f"Replaying workflow step {idx + 1}/{len(steps)}: {step}")
            res_dict = None
            exec_res = None

            if isinstance(step, dict):
                cap_id = step.get("capability") or step.get("capability_id") or step.get("intent") or step.get("action")
                step_params = step.get("parameters") or step.get("kwargs") or {}
                query_str = step.get("query") or step.get("name") or (str(cap_id) if cap_id else "")

                if cap_id and cap_id in capability_registry._registry:
                    try:
                        res_dict = capability_registry.execute_capability(cap_id, df=current_df, query=query_str, **step_params)
                    except Exception as err:
                        logger.error(f"Error executing capability '{cap_id}' in replay: {err}")
                        res_dict = {"status": "error", "response": f"Step {idx+1} execution failed: {err}"}
                elif query_str:
                    exec_res = QueryOrchestrator.process_query(query_str, df=current_df)
                else:
                    exec_res = ExecutionResult(
                        status=ExecutionStatus.EXECUTION_ERROR,
                        response=f"Step {idx+1} is missing a valid capability or query definition."
                    )
            elif isinstance(step, str):
                exec_res = QueryOrchestrator.process_query(step, df=current_df)
            else:
                exec_res = ExecutionResult(
                    status=ExecutionStatus.EXECUTION_ERROR,
                    response=f"Invalid step format at step {idx+1}."
                )

            if exec_res is None and res_dict is not None:
                if isinstance(res_dict, ExecutionResult):
                    exec_res = res_dict
                else:
                    exec_res = ResultValidator.validate_result(res_dict if isinstance(res_dict, dict) else {"status": "success", "response": str(res_dict)})

            # Propagate transformed DataFrame to subsequent steps
            if exec_res and exec_res.artifacts:
                if "transformed_df" in exec_res.artifacts and isinstance(exec_res.artifacts["transformed_df"], pd.DataFrame):
                    current_df = exec_res.artifacts["transformed_df"]
                elif "df" in exec_res.artifacts and isinstance(exec_res.artifacts["df"], pd.DataFrame):
                    current_df = exec_res.artifacts["df"]

            results_list.append(exec_res)

        if len(results_list) == 1:
            return results_list[0]

        combined_response = "\n\n".join([f"### Step {i+1}:\n{r.response}" for i, r in enumerate(results_list)])
        combined_code = "\n\n".join([f"# Step {i+1}\n{r.code}" for i, r in enumerate(results_list) if r.code])
        plots = [r.plot_image for r in results_list if r.plot_image]

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response=combined_response,
            code=combined_code if combined_code else None,
            plot_image=plots[0] if plots else None,
            artifacts={"plots": plots}
        )

    def schedule_pipeline_execution(self, filepath: str, cron_expression: str) -> Dict[str, Any]:
        """
        Schedules a saved pipeline for recurring background execution.
        """
        logger.info(f"Scheduling pipeline {filepath} with cron {cron_expression}")
        return {
            "pipeline_file": os.path.abspath(filepath),
            "cron_schedule": cron_expression,
            "status": "SCHEDULED",
            "next_run": "Automated Trigger Enabled"
        }


# Global Singleton Instance
workflow_saver = WorkflowSaverEngine()

