"""
DataPilot OS — Intent Engine.
Converts natural language user queries into structured analytical intents
using Ollama JSON schema-constrained structured output.

NO FALLBACK. If Ollama is unavailable, return INTENT_INCOMPLETE error.
NO keyword matching. NO template substitution.

Different phrasings of the same analytical request MUST produce equivalent intents.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import logging
from backend.core.error_states import ErrorState, DataPilotError
# Assuming ollama_service exists and can be imported
from backend.intelligence.ollama_service import ollama_service

logger = logging.getLogger(__name__)

class IntentSchema(BaseModel):
    objective: str = ""                       # What the user wants to achieve
    analysis_type: str = "analytics"          # e.g. 'time_series', 'comparison', 'correlation', 'distribution'
    metrics: List[str] = Field(default_factory=list)  # Column names or computed metrics to measure
    dimensions: List[str] = Field(default_factory=list)  # Grouping dimensions (product, region, customer_type)
    time_dimension: Optional[str] = None      # Date/time column if temporal analysis
    operations: List[str] = Field(default_factory=list)  # e.g. ['trend_analysis', 'group_comparison', 'ranking']
    visualizations: List[str] = Field(default_factory=list)  # e.g. ['line_chart', 'bar_chart', 'heatmap']
    filters: List[Dict[str, Any]] = Field(default_factory=list)  # e.g. [{"column": "region", "operator": "==", "value": "North"}]
    requires_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    intent_complete: bool = True
    ambiguities: List[str] = Field(default_factory=list)  # Detected ambiguities
    assumptions: List[str] = Field(default_factory=list)  # Assumptions made (shown to user)
    confidence: str = "high"            # 'high', 'medium', 'low'

class IntentEngine:
    def _build_dataset_context_prompt(self, dataset_context: dict) -> str:
        """Formats dataset context for Ollama."""
        prompt_parts = ["Dataset Context:"]
        
        columns = dataset_context.get("columns", [])
        if columns:
            prompt_parts.append("Columns:")
            for col in columns:
                if isinstance(col, str):
                    name = col
                    dtype = "Unknown"
                    semantic = ""
                    sample = []
                elif isinstance(col, dict):
                    name = col.get("name", "Unknown")
                    dtype = col.get("type", "Unknown")
                    semantic = col.get("semantic_type", "")
                    sample = col.get("sample_values", [])
                else:
                    name = str(col)
                    dtype = "Unknown"
                    semantic = ""
                    sample = []
                
                col_str = f" - {name} ({dtype})"
                if semantic:
                    col_str += f" | Semantic: {semantic}"
                if sample:
                    col_str += f" | Samples: {sample[:3]}"
                prompt_parts.append(col_str)
        else:
            prompt_parts.append("No column information available.")
            
        return "\n".join(prompt_parts)

    def _validate_intent_against_dataset(self, intent: IntentSchema, dataset_context: dict) -> Tuple[bool, List[str]]:
        """Checks if mentioned columns/metrics actually exist in the dataset."""
        is_valid = True
        issues = []
        
        available_columns = [col.get("name", "") for col in dataset_context.get("columns", [])]
        available_columns_lower = {col.lower(): col for col in available_columns if col}
        
        # Check metrics
        validated_metrics = []
        for metric in intent.metrics:
            if metric.lower() in available_columns_lower:
                validated_metrics.append(available_columns_lower[metric.lower()])
            else:
                issues.append(f"Metric column '{metric}' not found in dataset.")
                is_valid = False
        if is_valid:
            intent.metrics = validated_metrics

        # Check dimensions
        validated_dimensions = []
        for dim in intent.dimensions:
            if dim.lower() in available_columns_lower:
                validated_dimensions.append(available_columns_lower[dim.lower()])
            else:
                issues.append(f"Dimension column '{dim}' not found in dataset.")
                is_valid = False
        if is_valid:
            intent.dimensions = validated_dimensions

        # Check time dimension
        if intent.time_dimension:
            if intent.time_dimension.lower() in available_columns_lower:
                intent.time_dimension = available_columns_lower[intent.time_dimension.lower()]
            else:
                issues.append(f"Time dimension column '{intent.time_dimension}' not found in dataset.")
                is_valid = False

        return is_valid, issues

    def analyze(
        self, 
        user_query: str, 
        dataset_context: dict, 
        conversation_history: List[Dict] = None, 
        available_capabilities: List[str] = None
    ) -> Union[IntentSchema, DataPilotError]:
        if not dataset_context:
            return DataPilotError(
                code=ErrorState.DATASET_NOT_CONNECTED,
                message="No active dataset context provided.",
                stage="intent_engine",
                recoverable=True,
                required_action="Connect a dataset first."
            )
            
        context_str = self._build_dataset_context_prompt(dataset_context)
        
        try:
            # We assume ollama_service has a generate_intent method that takes these params
            # and uses JSON schema constraint.
            response = ollama_service.generate_intent(
                user_query=user_query,
                dataset_context=dataset_context,
                conversation_history=conversation_history
            )
            
            if not response.success:
                return DataPilotError(
                    code=getattr(ErrorState, response.error_code, ErrorState.OLLAMA_INVALID_RESPONSE) if response.error_code else ErrorState.OLLAMA_INVALID_RESPONSE,
                    message=response.error_message or "Ollama intent generation failed",
                    stage="intent_engine",
                    recoverable=True,
                    required_action="Retry query or check Ollama model status"
                )
                
            intent = IntentSchema.model_validate(response.parsed_json or {})
            
        except Exception as e:
            logger.error(f"Failed to generate intent: {e}")
            return DataPilotError(
                code=ErrorState.INTENT_INCOMPLETE,
                message=f"Error processing intent: {str(e)}",
                stage="intent_engine",
                recoverable=True,
                required_action="Provide a clearer query"
            )

        is_valid, issues = self._validate_intent_against_dataset(intent, dataset_context)
        
        if not is_valid:
            intent.ambiguities.extend(issues)
            intent.requires_clarification = True
            intent.intent_complete = False

        if intent.requires_clarification:
            # Return as is for ClarificationEngine to handle
            return intent
            
        return intent

    def is_equivalent(self, intent_a: IntentSchema, intent_b: IntentSchema) -> bool:
        """Returns True if two intents have the same analysis_type and overlapping operations."""
        if intent_a.analysis_type != intent_b.analysis_type:
            return False
            
        ops_a = set(intent_a.operations)
        ops_b = set(intent_b.operations)
        
        return bool(ops_a.intersection(ops_b))

intent_engine = IntentEngine()
