"""
DataPilot OS — Minimal Semantic Command Contract.
Defines the clean, concise tool/command protocol for LLM intent expression.
The LLM describes WHAT the user means; DataPilot's CommandResolver decides HOW to execute it.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class SemanticTool(str, Enum):
    FILTER_DATA = "filter_data"
    GROUP_DATA = "group_data"
    AGGREGATE_DATA = "aggregate_data"
    SORT_DATA = "sort_data"
    CORRELATE_DATA = "correlate_data"
    FORECAST_DATA = "forecast_data"
    CLEAN_DATA = "clean_data"
    DETECT_ANOMALIES = "detect_anomalies"
    EDA_DATA = "eda_data"
    TRAIN_MODEL = "train_model"
    CLUSTER_DATA = "cluster_data"
    STATISTICAL_TEST = "statistical_test"
    PCA_ANALYSIS = "pca_analysis"
    AUTOCORRELATION = "autocorrelation"
    FEATURE_ENGINEERING = "feature_engineering"
    CLARIFICATION_NEEDED = "clarification_needed"
    GENERAL_CHAT = "general_chat"


class ToolCall(BaseModel):
    """
    A single minimal semantic command / tool call.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tool: str = Field(..., description="The semantic tool name, e.g. filter_data, group_data, forecast_data")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Semantic arguments for the tool")
    inferred_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    is_dataset_wide: bool = False


SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "1.1"
PROTOCOL_NAME = "minimal_semantic_command"


class LLMSemanticCommand(BaseModel):
    """
    The top-level container emitted by the LLM.
    Supports either a single tool call or a sequence of tool calls.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    protocol: str = Field(default=PROTOCOL_NAME, description="Protocol identifier")
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    prompt_version: str = Field(default=PROMPT_VERSION, description="Prompt version")
    tool: Optional[str] = Field(None, description="Primary tool name if single tool call")
    arguments: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arguments if single tool call")
    commands: Optional[List[ToolCall]] = Field(default_factory=list, description="Sequence of commands if multi-step")
    clarification_reason: Optional[str] = Field(None, description="Reason if query is ambiguous or underspecified")


# Minimal JSON Schema definition for Ollama / LLM structured output enforcement
MINIMAL_COMMAND_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": [
                "filter_data",
                "group_data",
                "sort_data",
                "correlate_data",
                "forecast_data",
                "clean_data",
                "detect_anomalies",
                "eda_data",
                "train_model",
                "cluster_data",
                "statistical_test",
                "pca_analysis",
                "autocorrelation",
                "feature_engineering",
                "clarification_needed",
                "general_chat"
            ]
        },
        "arguments": {
            "type": "object",
            "description": "Key-value arguments such as column, condition, value, group_by, metric, aggregation, horizon, etc."
        },
        "commands": {
            "type": "array",
            "description": "Optional list of sequential tool calls for multi-step requests",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"}
                },
                "required": ["tool", "arguments"]
            }
        },
        "clarification_reason": {
            "type": ["string", "null"],
            "description": "Explain why the query is ambiguous if clarification_needed is chosen"
        }
    }
}
