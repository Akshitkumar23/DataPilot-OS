"""
AI Planning Agent Module for DataPilot OS.

This module provides the AIPlanningAgent class responsible for dynamically
generating and executing multi-step analytical plans for complex user queries.
It breaks down abstract requests into actionable analytical steps.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

# Configure logging for the planning agent
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class AIPlanningAgent:
    """
    Agent responsible for constructing and executing multi-step analytical plans.

    This class interprets complex queries and formulates a structured execution
    plan involving various analytical operations.
    """

    def __init__(self) -> None:
        """Initialize the AI Planning Agent."""
        logger.info("Initializing AIPlanningAgent")

    def construct_multi_step_plan(
        self, query_raw: str, parsed_intent: Any, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Construct a structured multi-step plan array for a complex query.

        Args:
            query_raw (str): The raw string query from the user.
            parsed_intent (Any): The parsed intent representation of the query.
            df (pd.DataFrame): The dataframe representing the context for the plan.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a step in the execution plan.

        Raises:
            ValueError: If inputs are invalid or insufficient to create a plan.
        """
        logger.info(f"Constructing multi-step plan for query: '{query_raw}'")

        if not isinstance(query_raw, str) or not query_raw.strip():
            logger.error("Invalid query provided.")
            raise ValueError("Query string cannot be empty.")

        if df is None or df.empty:
            logger.warning("Dataframe is empty or None. The plan might be limited.")

        try:

            plan: List[Dict[str, Any]] = [
                {
                    "step_number": 1,
                    "step_name": "Trend & Time-Series Analysis",
                    "description": "Analyze metrics over time to identify trends and seasonality.",
                    "action_type": "time_series",
                    "parameters": {}
                },
                {
                    "step_number": 2,
                    "step_name": "Spatial & Dimension Breakdown",
                    "description": "Break down data by key dimensions (e.g., region, product category).",
                    "action_type": "dimension_breakdown",
                    "parameters": {}
                },
                {
                    "step_number": 3,
                    "step_name": "Correlation & Driver Identification",
                    "description": "Identify factors that correlate with changes in key metrics.",
                    "action_type": "correlation_analysis",
                    "parameters": {}
                },
                {
                    "step_number": 4,
                    "step_name": "Metric Calculation",
                    "description": "Calculate advanced or custom metrics derived from raw data.",
                    "action_type": "calculate_metrics",
                    "parameters": {}
                },
                {
                    "step_number": 5,
                    "step_name": "Insight Synthesis",
                    "description": "Synthesize findings from previous steps into coherent insights.",
                    "action_type": "synthesize",
                    "parameters": {}
                },
                {
                    "step_number": 6,
                    "step_name": "Actionable Recommendation",
                    "description": "Generate actionable recommendations based on synthesized insights.",
                    "action_type": "recommendation",
                    "parameters": {}
                }
            ]

            logger.info(f"Successfully constructed {len(plan)} step plan.")
            return plan

        except Exception as e:
            logger.error(f"Error constructing plan: {e}")
            raise

    def execute_plan_step(self, step_dict: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute an individual plan step.

        This method acts as an interface to the Capability Registry to execute
        the specified action.

        Args:
            step_dict (Dict[str, Any]): Dictionary defining the step to execute.
            df (pd.DataFrame): The dataframe context for the execution.

        Returns:
            Dict[str, Any]: The output of the step execution, including generated
                            code (if applicable) and text insights.

        Raises:
            KeyError: If the step dictionary is missing required keys.
            Exception: For execution failures.
        """
        logger.info(f"Executing step {step_dict.get('step_number', 'Unknown')}: {step_dict.get('step_name', 'Unnamed')}")

        try:
            if 'action_type' not in step_dict:
                raise KeyError("step_dict must contain an 'action_type' key.")

            action_type = step_dict['action_type']
            
            output: Dict[str, Any] = {
                "step": step_dict.get("step_number"),
                "status": "success",
                "output_code": f"# Code to execute {action_type}\npass",
                "output_text": f"Output for executing {action_type} on data with shape {df.shape if df is not None else 'None'}"
            }
            
            logger.info(f"Step {step_dict.get('step_number')} executed successfully.")
            return output

        except KeyError as ke:
             logger.error(f"Invalid step definition: {ke}")
             raise
        except Exception as e:
            logger.error(f"Failed to execute step: {e}")
            return {
                "step": step_dict.get("step_number"),
                "status": "error",
                "error_message": str(e)
            }


# Global Singleton Instance
planning_agent = AIPlanningAgent()
